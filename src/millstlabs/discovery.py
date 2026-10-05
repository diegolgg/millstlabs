"""Paired LLM corpus experiment, explicit tools, fixed evaluation, resumable CPU deployment."""
import argparse
import copy
import fcntl
import json
import math
import time
from pathlib import Path

import numpy as np

from .batch import freeze, run_batch
from .config import load_config


def arms(base, full=False):
    result = {}
    variants = ["plain", "guided"] if not full else ["plain", "novelty", "curriculum", "guided"]
    for variant in variants:
        for mode in ["private", "shared"]:
            cfg = copy.deepcopy(base)
            t = cfg.training
            t.corpus_mode = mode
            t.intrinsic_critic = True  # Identical architecture and common warm checkpoint.
            t.novelty_beta_start = .05 if variant in {"novelty", "guided"} else 0.
            t.novelty_beta_end = 0.
            t.novelty_decay_decisions = max(1, t.decisions*3//4)
            t.predator_curriculum = ([{"decisions": 0, "temperature": 4.0},
                                      {"decisions": t.decisions//2, "temperature": 1.0}]
                                     if variant in {"curriculum", "guided"} else [])
            cfg.validate()
            result[f"{mode}_{variant}"] = cfg
    return result


def proposal(base, full=False):
    configs = arms(base, full)
    t = base.training
    checkpoints = sum(c < t.decisions for c in t.checkpoints) + 1
    eval_maps = (checkpoints-1)*t.evaluation_maps + t.final_evaluation_maps + checkpoints*t.development_maps
    total_eval_maps = len(configs)*eval_maps + len(configs)//2*t.final_evaluation_maps
    return {"base_digest": base.digest(), "conditions": list(configs), "backend": base.training.backend,
            "runs": len(configs)*len(base.seeds), "training_decisions": len(configs)*len(base.seeds)*base.training.decisions,
            "decisions_per_run": base.training.decisions, "warmstart_demonstrations": len(base.seeds)*base.training.warmstart_transitions,
            "evaluation_decision_upper_bound": len(base.seeds)*total_eval_maps*t.evaluation_policies*t.evaluation_horizon,
            "tools": ["none", "put_terrain", "put_food", "get_terrain", "get_food"],
            "primary_metric": "New first-hand verified facts per 1000 allocated held-out decision opportunities, excluding initial observations",
            "control": "Eight independent private policies in the same game; own deposits only vs peer-accessible deposits. All tools and costs matched.",
            "note": "Four-arm default compares corpus access within each guidance setting. --full isolates novelty and curriculum separately."}


def rows_at(path, checkpoint):
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return next((r for r in reversed(rows) if r["checkpoint"] == checkpoint), None)


def compare_rows(shared, private):
    other = {r["map_seed"]: r for r in private["episodes"]}
    if not other or set(other) != {r["map_seed"] for r in shared["episodes"]}:
        raise ValueError("Comparison requires identical nonempty held-out map banks")
    for row in shared["episodes"]:
        control = other[row["map_seed"]]
        if row["allocated_decision_opportunities"] != control["allocated_decision_opportunities"]:
            raise ValueError("Comparison requires identical allocated evaluation budgets")
    keys = ["discoveries_per_1000_opportunities", "new_observed_facts", "restricted_mean_lifetime",
            "survival_fraction", "food_per_alive_decision", "processed_tokens", "elapsed_seconds"]
    if all("mean_new_facts_over_time" in r for r in shared["episodes"]+private["episodes"]):
        keys.append("mean_new_facts_over_time")
    paired = []
    for row in shared["episodes"]:
        if row["map_seed"] in other:
            paired.append({"map_seed": row["map_seed"], **{k: row[k]-other[row["map_seed"]][k] for k in keys}})
    return {"paired_map_differences": paired,
            "mean_differences": {k: float(np.mean([r[k] for r in paired])) for k in keys} if paired else {}}


def report(base, root, full=False):
    results, evaluations, manifests, ablations, initial = [], {}, {}, {}, {}
    for name, cfg in arms(base, full).items():
        for seed in cfg.seeds:
            directory = root/name/f"prosocial-r_adult-s{seed}"
            summary = directory/"summary.json"
            row = {"condition": name, "seed": seed, "finished": False, "status": "not_started"}
            if summary.exists():
                row.update(json.loads(summary.read_text()))
                row["status"] = "complete" if row["finished"] else "paused"
            elif directory.exists():
                row["status"] = "running_or_interrupted"
            if row["finished"]:
                final = rows_at(directory/"evaluation.jsonl", cfg.training.decisions)
                if final is None:
                    raise ValueError(f"Completed run is missing final evaluation: {directory}")
                evaluations[name, seed] = final
                manifests[name, seed] = json.loads((directory/"manifest.jsonl").read_text().splitlines()[0])
                ablations[name, seed] = rows_at(directory/"evaluation_private_corpus.jsonl", cfg.training.decisions)
                initial[name, seed] = rows_at(directory/"evaluation.jsonl", 0)
                row["final_metrics"] = {k: float(np.mean([r[k] for r in final["episodes"]])) for k in [
                    "discoveries_per_1000_opportunities", "new_observed_facts", "restricted_mean_lifetime",
                    "survival_fraction", "food_per_alive_decision", "peer_records", "deposit_calls", "retrieve_calls"]}
                row["change_from_warmstart"] = compare_rows(final, initial[name, seed]) if initial[name, seed] else None
            results.append(row)
    comparisons = []
    for (name, seed), shared in evaluations.items():
        if not name.startswith("shared_"):
            continue
        private_name = name.replace("shared_", "private_", 1)
        if (private_name, seed) not in evaluations:
            continue
        if manifests[name, seed]["warmstart_sha256"] != manifests[private_name, seed]["warmstart_sha256"]:
            raise ValueError("Paired conditions used different warm checkpoints")
        private = evaluations[private_name, seed]
        comparison = compare_rows(shared, private)
        warm_contrast = (compare_rows(initial[name, seed], initial[private_name, seed])
                         if initial[name, seed] and initial[private_name, seed] else None)
        ablation = ablations[name, seed]
        causal = compare_rows(shared, ablation) if ablation else None
        s_rate = np.mean([r["discoveries_per_1000_opportunities"] for r in shared["episodes"]])
        p_rate = np.mean([r["discoveries_per_1000_opportunities"] for r in private["episodes"]])
        fraction = float(s_rate/p_rate-1) if p_rate > 0 else None
        checks = {"discovery_gain_at_least_20_percent": fraction is not None and fraction >= .2,
                  "lifetime_not_lower": comparison["mean_differences"].get("restricted_mean_lifetime", -1) >= 0,
                  "gains_on_75_percent_of_maps": sum(r["new_observed_facts"] > 0 for r in comparison["paired_map_differences"]) >= math.ceil(.75*len(comparison["paired_map_differences"])),
                  "peers_actually_retrieved": sum(r["peer_records"] for r in shared["episodes"]) > 0,
                  "frozen_access_ablation_positive": causal is not None and causal["mean_differences"].get("new_observed_facts", 0) > 0,
                  "not_a_smoke_test": base.training.decisions >= 16384 and base.training.final_evaluation_maps >= 4}
        comparisons.append({"variant": name.removeprefix("shared_"), "seed": seed, **comparison,
                            "relative_discovery_gain": fraction, "frozen_policy_access_ablation": causal,
                            "warmstart_corpus_contrast": warm_contrast,
                            "change_in_corpus_advantage": ({k: v-warm_contrast["mean_differences"][k]
                                                           for k, v in comparison["mean_differences"].items()}
                                                          if warm_contrast else None),
                            "checks": checks, "exploratory_poc_pass": all(checks.values())})
    result = {"runs": results, "comparisons": comparisons,
              "complete": sum(r["finished"] for r in results), "total": len(results),
              "note": "An exploratory pass tests useful corpus access, conditional on one trained seed. It is not an RL learning verdict: inspect changes from the common warm start separately. Map repetitions are not independent training replicates; no claim of free-form research discovery."}
    if root.exists():
        (root/"discovery-report.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/discovery.yaml")
    parser.add_argument("--mode", choices=["plan", "run", "report", "smoke"], default="plan")
    parser.add_argument("--output")
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--full", action="store_true", help="Eight-arm factorial instead of the four-arm deployment")
    parser.add_argument("--smoke-backend", choices=["smollm", "structured"], default="smollm")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if list(cfg.profiles) != ["prosocial"] or cfg.methods != ["r_adult"]:
        parser.error("Discovery deployment requires the single prosocial R-adult condition")
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if args.mode == "smoke":
        t = cfg.training
        t.backend = args.smoke_backend
        t.controller_architecture = "split" if t.backend == "smollm" else "legacy"
        t.cpu_threads = 1
        t.decisions, t.warmstart_transitions = 128, 64
        t.rollout_ticks, t.social_window, t.sequence_length = 8, 16, 4
        t.checkpoints, t.evaluation_maps, t.final_evaluation_maps, t.evaluation_horizon = [0, 128], 1, 1, 8
        t.development_maps = 0
    cfg.validate()
    root = Path(args.output or ("runs/discovery-smoke" if args.mode == "smoke" else "runs/discovery-proof")).resolve()
    plan = proposal(cfg, args.full)
    if args.mode == "plan":
        result = plan
    elif args.mode == "report":
        saved = load_config(root/"config.yaml")
        saved_plan = json.loads((root/"deployment.json").read_text())
        result = report(saved, root, len(saved_plan["conditions"]) == 8)
    else:
        freeze(cfg, root)
        with (root/".deployment.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("Another discovery deployment uses this directory")
            path = root/"deployment.json"
            if path.exists() and json.loads(path.read_text()) != plan:
                parser.error("Deployment design differs; use a new output directory")
            path.write_text(json.dumps(plan, indent=2))
            deadline = time.monotonic()+args.hours*3600
            warmdir = root/"warm"
            warmdir.mkdir(exist_ok=True)
            for name, arm in arms(cfg, args.full).items():
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    break
                directory = root/name
                frozen = freeze(arm, directory)
                with (directory/".runner.lock").open("w") as arm_lock:
                    fcntl.flock(arm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    status = run_batch(arm, directory, frozen, remaining/3600, warmstart_dir=warmdir)
                if status["complete"] < status["total"]:
                    break
            result = report(cfg, root, args.full)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
