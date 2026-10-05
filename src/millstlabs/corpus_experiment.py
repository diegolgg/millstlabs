"""CPU predator-prey experiment: optional grounded or free-form corpus notes."""
import argparse
import copy
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .batch import freeze, run_batch
from .config import load_config


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    result = []
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise
    return result


def variants(base, styles):
    result = {}
    for style in styles:
        cfg = copy.deepcopy(base)
        cfg.training.note_style = style
        cfg.validate()
        result[style] = cfg
    return result


def plan(base, styles):
    t = base.training
    maps_per_arm = (3 if t.note_memory == "tactics" else 2)*(sum(c < t.decisions for c in t.checkpoints)*t.evaluation_maps+t.final_evaluation_maps)
    return {"styles": styles, "profile": "prosocial", "method": "r_adult", "seeds": base.seeds,
            "backend": t.backend, "physical_actions": 7, "automatic_context": True,
            **({"action_policy": t.action_policy, "note_memory": t.note_memory,
                "knowledge_energy_cost": t.corpus_tool_cost, "feed_imitation_weight": t.feed_retention}
               if t.action_policy == "lm_token" else {}),
            "decisions_per_arm": t.decisions, "total_training_decisions": t.decisions*len(styles)*len(base.seeds),
            "warmstart_decisions": t.warmstart_transitions*len(styles)*len(base.seeds),
            "evaluation_decision_upper_bound": maps_per_arm*t.evaluation_policies*t.evaluation_horizon*len(styles)*len(base.seeds),
            "generation_limit_per_note": t.note_max_tokens, "minimum_ticks_between_notes": t.note_interval,
            "note": ("Zero task demonstrations; compare pretrained baseline, lineage-private and no-tactic access at each checkpoint. " if t.action_policy == "lm_token" else
                     "Frozen peer-access ablations at every checkpoint. Each style has its own demonstration-trained baseline. ")+
                     "Hours are a resumable soft invocation limit, not a completion estimate."}


def paired(shared, private):
    control = {r["map_seed"]: r for r in private["episodes"]}
    keys = ["restricted_mean_lifetime", "survival_fraction", "food_per_alive_decision", "food_consumed"]
    if set(control) != {r["map_seed"] for r in shared["episodes"]}:
        raise ValueError("Evaluation map banks differ")
    differences = [{"map_seed": r["map_seed"], **{k: r[k]-control[r["map_seed"]][k] for k in keys}}
                   for r in shared["episodes"]]
    return {"per_map": differences, "mean": {k: float(np.mean([r[k] for r in differences])) for k in keys}}


def report(root):
    base = load_config(root/"config.yaml")
    deployment = json.loads((root/"deployment.json").read_text())
    runs = []
    for style in deployment["styles"]:
        for seed in base.seeds:
            directory = root/style/f"prosocial-r_adult-s{seed}"
            summaries = json.loads((directory/"summary.json").read_text()) if (directory/"summary.json").exists() else {}
            ecology = read_rows(directory/"ecology.jsonl")
            knowledge = read_rows(directory/"knowledge.jsonl")
            evaluations = {r["checkpoint"]: r for r in read_rows(directory/"evaluation.jsonl")}
            private = {r["checkpoint"]: r for r in read_rows(directory/"evaluation_private_corpus.jsonl")}
            empty = {r["checkpoint"]: r for r in read_rows(directory/"evaluation_without_tactics.jsonl")}
            probes = read_rows(directory/"note_probe.jsonl")
            committed = json.loads((directory/"checkpoint-status.json").read_text()) if (directory/"checkpoint-status.json").exists() else None
            prepared = (0 in evaluations and 0 in private and any(r["checkpoint"] == 0 for r in probes)
                        and (root/style/f"warm-s{seed}.pt").exists() and committed is not None)
            collected = ecology[-1]["decisions"] if ecology else (committed or {}).get("decisions", 0)
            status = ("complete" if summaries.get("finished") else "prepared; PPO not started" if prepared and not collected
                      else "incomplete; process liveness not inferred")
            item = {"style": style, "seed": seed, "finished": summaries.get("finished", False),
                    "prepared": prepared,
                    "last_logged_decisions": ecology[-1]["decisions"] if ecology else 0,
                    "budget": base.training.decisions,
                    "committed_checkpoint": committed,
                    "status": status,
                    "latest_knowledge": knowledge[-1] if knowledge else None,
                    "latest_cue_probe": probes[-1] if probes else None,
                    "evaluations": {str(c): {
                        "restricted_mean_lifetime": e["restricted_mean_lifetime"],
                        "survival_fraction": e["survival_fraction"],
                        "mean_food_consumed": float(np.mean([r["food_consumed"] for r in e["episodes"]])),
                        "hungry_feed_opportunities": sum(int(r["hungry_feed_opportunities"]) for r in e["episodes"]),
                        "hungry_feed_actions": sum(int(r["hungry_feed_actions"]) for r in e["episodes"])}
                        for c, e in evaluations.items()},
                    "peer_access_benefit": {str(c): paired(e, private[c]) for c, e in evaluations.items() if c in private}}
            if base.training.note_memory == "tactics":
                item["tactic_access_benefit"] = {str(c): paired(e, empty[c]) for c, e in evaluations.items() if c in empty}
            last = max(evaluations, default=0)
            if last and 0 in evaluations:
                # Final may use a larger map bank: compare its common subset only.
                initial_seeds = {r["map_seed"] for r in evaluations[0]["episodes"]}
                common_final = {**evaluations[last], "episodes": [r for r in evaluations[last]["episodes"] if r["map_seed"] in initial_seeds]}
                label = "change_from_pretrained_start" if base.training.action_policy == "lm_token" else "change_from_own_warmstart"
                item[label] = paired(common_final, evaluations[0])
            runs.append(item)
    return {"runs": runs, "prepared": sum(r["prepared"] for r in runs),
            "complete": sum(r["finished"] for r in runs), "total": len(runs),
            "interpretation": "Credit measures novel information delivery, not useful teaching. Claim benefit only from food/survival gains and positive frozen-access ablations. One training seed is exploratory; cue sensitivity alone is not competence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/corpus-notes.yaml")
    parser.add_argument("--mode", choices=["plan", "smoke", "prepare", "run", "report"], default="plan")
    parser.add_argument("--style", nargs="+", choices=["grounded", "prose"])
    parser.add_argument("--output")
    parser.add_argument("--hours", type=float, default=12)
    args = parser.parse_args()
    cfg = load_config(args.config)
    direct = cfg.training.action_policy == "lm_token"
    args.style = args.style or ["grounded", "prose"]
    name = "autonomous-llm" if direct else "corpus-notes"
    root = Path(args.output or (f"runs/{name}-smoke" if args.mode == "smoke" else f"runs/{name}")).resolve()
    if args.mode == "report":
        print(json.dumps(report(root), indent=2))
        return
    if cfg.training.backend != "smollm" or cfg.training.corpus_interface != "notes":
        parser.error("Deployment requires an LLM with the notes interface")
    if list(cfg.profiles) != ["prosocial"] or cfg.methods != ["r_adult"] or cfg.training.corpus_mode != "shared":
        parser.error("Use the shared-corpus cooperative R-adult configuration")
    if args.hours <= 0 or len(set(args.style)) != len(args.style):
        parser.error("Hours must be positive and styles unique")
    if args.mode == "smoke":
        t = cfg.training
        t.decisions, t.warmstart_transitions = 128, 0 if direct else 64
        t.rollout_ticks, t.social_window, t.sequence_length = 8, 16, 4
        t.checkpoints = [0, 128]
        t.evaluation_maps = t.final_evaluation_maps = 1
        t.evaluation_horizon = 8
        t.note_max_tokens = 24
    cfg.validate()
    design = plan(cfg, args.style)
    if args.mode == "plan":
        print(json.dumps(design, indent=2))
        return
    freeze(cfg, root)
    with (root/".deployment.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another notes deployment owns this output directory")
        deployment = root/"deployment.json"
        if deployment.exists() and json.loads(deployment.read_text()) != design:
            parser.error("Saved design differs; choose a new output directory")
        deployment.write_text(json.dumps(design, indent=2))
        deadline = time.monotonic()+args.hours*3600
        for name, variant in variants(cfg, args.style).items():
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            directory = root/name
            frozen = freeze(variant, directory)
            with (directory/".runner.lock").open("w") as arm_lock:
                fcntl.flock(arm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.mode == "prepare":
                    # Full demonstrations and initial frozen assays, zero PPO
                    # collection. A later run command resumes these checkpoints.
                    command = [sys.executable, "-m", "millstlabs.cli", "--config", str(frozen)]
                    for seed in variant.seeds:
                        if time.monotonic() >= deadline:
                            break
                        warm = directory/f"warm-s{seed}.pt"
                        output = directory/f"prosocial-r_adult-s{seed}"
                        if not warm.exists():
                            subprocess.run(command+["warmstart", "--seed", str(seed), "--output", str(warm)], check=True)
                        initial_done = (any(r["checkpoint"] == 0 for r in read_rows(output/"evaluation.jsonl"))
                                        and any(r["checkpoint"] == 0 for r in read_rows(output/"evaluation_private_corpus.jsonl"))
                                        and any(r["checkpoint"] == 0 for r in read_rows(output/"note_probe.jsonl")))
                        if not initial_done:
                            latest = output/"latest.pt"
                            resume = ["--resume", str(latest)] if latest.exists() else []
                            subprocess.run(command+["train", "--profile", "prosocial", "--method", "r_adult",
                                "--seed", str(seed), "--warmstart", str(warm), "--output", str(output),
                                "--max-wall-seconds", "0.000001"]+resume, check=True)
                    continue
                result = run_batch(variant, directory, frozen, remaining/3600)
            if result["complete"] < result["total"]:
                break
        result = report(root)
        (root/"notes-report.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
