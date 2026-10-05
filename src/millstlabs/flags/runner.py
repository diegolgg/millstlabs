"""Serial, resumable GPT flag-game baseline and controlled corpus/RL comparison."""
import argparse
import copy
import fcntl
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .controller import Controllers
from .engine import feature_width, run_episode
from .provider import BudgetExhausted, MockProvider, OpenAIProvider
from .task import catalog_images, digest, prepare_trials

ARMS = ["paper_reference", "private_frozen", "shared_frozen", "private_rl", "shared_rl"]
ABLATION = "shared_rl_private_access"
GAPS = [
    "The paper does not enumerate its 28-country catalog or provide the original trial assignments.",
    "The exact GPT-4o snapshot is unspecified; this runner pins gpt-4o-2024-08-06.",
    "The numeric kappa and general probe cadence are unspecified. We choose 10 rounds, N directed exchanges per round, then one probe per agent.",
    "Crop positions are sampled uniformly at integer pixel offsets; the paper does not fully specify this distribution.",
    "Speaker and listener are distinct in this reconstruction; self-contact handling is unspecified for the empirical protocol.",
    "Invalid-response repair and tie handling are unspecified. We score invalid answers incorrect, do not retry them, and use lexical country order for ties.",
    "The public demo differs from Appendix D (including image detail and token cap); its defaults are not substituted for the PDF settings.",
]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    os.replace(temporary, path)


def plan(cfg):
    catalog, images = catalog_images(cfg.catalog, cfg)
    calls = cfg.population*(1+2*cfg.interaction_rounds)
    return {"backend": cfg.backend, "model": cfg.model, "config_digest": cfg.digest(),
            "protocol": "pairwise", "population": cfg.population, "arms": ARMS, "ablation": ABLATION,
            "train_trials_per_rl_arm": cfg.train_trials, "evaluation_trials_per_arm": cfg.evaluation_trials,
            "api_call_upper_bound_per_trial": calls,
            "api_call_upper_bound_total": calls*(2*cfg.train_trials+6*cfg.evaluation_trials),
            "training_tool_decisions_upper_bound": 2*cfg.train_trials*2*cfg.population*cfg.interaction_rounds,
            "country_count": len(images), "catalog_digest": digest(catalog),
            "catalog_provenance": catalog.get("provenance", "unspecified"),
            "exact_paper_reproduction": False, "reproduction_gaps": GAPS,
            "note": "Maximum calls before early stopping or cache reuse; paid use requires OPENAI_API_KEY and an explicit cumulative --max-usd limit. No GPU needed."}


def sized_config(cfg, size):
    cfg = copy.deepcopy(cfg)
    if size == "smoke":
        cfg.train_trials, cfg.evaluation_trials, cfg.interaction_rounds = 1, 2, 1
    elif size == "pilot":
        cfg.train_trials, cfg.evaluation_trials = 8, 6
    elif size != "full":
        raise ValueError("Unknown experiment size")
    cfg.validate()
    return cfg


def paired(left, right):
    lookup = {r["trial_hash"]: r for r in right}
    differences = [{"trial_seed": r["trial_seed"], "trial_hash": r["trial_hash"],
                    "truth_mass_difference": r["terminal_truth_mass"]-lookup[r["trial_hash"]]["terminal_truth_mass"],
                    "social_uplift_difference": r["social_uplift"]-lookup[r["trial_hash"]]["social_uplift"]}
                   for r in left if r["trial_hash"] in lookup]
    values = np.asarray([r["truth_mass_difference"] for r in differences])
    interval = None
    if len(values) > 1:
        rng = np.random.default_rng(9043)
        means = rng.choice(values, (2000, len(values)), replace=True).mean(1)
        interval = np.quantile(means, [.025, .975]).tolist()
    return {"paired_trials": len(values), "mean_truth_mass_difference": float(values.mean()) if len(values) else None,
            "trial_bootstrap_95_interval": interval, "trial_differences": differences,
            "uncertainty_unit": "held-out trials, conditional on one trained controller seed"}


def report(root, state=None):
    manifest = json.loads((root/"manifest.json").read_text())
    if state is None:
        state = torch.load(root/"latest.pt", map_location="cpu", weights_only=False)
    results = {a: [r for r in state["results"] if r["phase"] == "evaluation" and r["arm"] == a]
               for a in ARMS+[ABLATION]}
    summaries = {}
    for arm, rows in results.items():
        keys = ["initial_accuracy", "initial_majority_correct", "terminal_truth_mass", "social_uplift",
                "final_majority_correct", "queries", "rounds", "peer_records", "novel_color_imports",
                "deposit_calls", "retrieve_calls", "invalid_responses"]
        summaries[arm] = {"trials": len(rows), **({k: float(np.mean([r[k] for r in rows])) for k in keys} if rows else {}),
                          "endpoint_counts": {category: sum(r["endpoint"] == category for r in rows)
                                              for category in ["correct_consensus", "wrong_consensus", "polarization", "fragmentation"]}}
    comparisons = {f"{a}_minus_{b}": paired(results[a], results[b]) for a, b in [
        ("shared_rl", "private_rl"), ("shared_rl", "shared_frozen"), ("private_rl", "private_frozen"),
        ("shared_frozen", "private_frozen"), ("shared_rl", "paper_reference"), ("shared_rl", ABLATION)]}
    fingerprints = sorted({f for r in state["results"] for f in r["system_fingerprints"] if f is not None})
    versions = sorted({v for r in state["results"] for v in r["model_versions"] if v is not None})
    result = {"finished": state["next_job"] == state["total_jobs"], "completed_jobs": state["next_job"],
              "total_jobs": state["total_jobs"], "backend": manifest["backend"],
              "exact_paper_reproduction": False, "summaries": summaries, "comparisons": comparisons,
              "model_versions": versions, "system_fingerprints": fingerprints,
              "backend_fingerprint_changed": len(fingerprints) > 1,
              "status": state.get("status", "in_progress"), "cost_guard_charged_usd": state.get("spent_usd", 0),
              "interpretation": "Partial pairs are provisional. Mock scores are wiring tests only. Corpus benefit, RL improvement and frozen access ablation are separate contrasts. Trial intervals do not establish replication across controller training seeds."}
    atomic_json(root/"report.json", result)
    return result


def deploy(cfg, root, max_usd, hours=12, max_jobs=None, provider_factory=None, retry_uncertain=False):
    cfg.validate()
    proposal = plan(cfg)
    catalog, images = catalog_images(cfg.catalog, cfg)
    trials = prepare_trials(cfg, images)
    assignments = {k: [asdict(t) for t in v] for k, v in trials.items()}
    implementation = digest({p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("*.py"))})
    binding = {"config_digest": cfg.digest(), "catalog_digest": digest(catalog), "trials_digest": digest(assignments),
               "implementation_digest": implementation}
    root.mkdir(parents=True, exist_ok=True)
    with (root/".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root/"manifest.json").exists():
            saved = json.loads((root/"manifest.json").read_text())
            if any(saved[k] != v for k, v in binding.items()):
                raise ValueError("Frozen config/assets/trials differ; choose a new output directory")
        else:
            if any(p.name != ".runner.lock" for p in root.iterdir()):
                raise ValueError("Output is nonempty without a manifest; choose a fresh directory")
            atomic_json(root/"manifest.json", {**proposal, **binding, "created_unix": time.time(),
                                               "policy": "independent local actor-critics for tools; GPT weights frozen"})
            atomic_json(root/"config.json", asdict(cfg))
            atomic_json(root/"catalog.json", catalog)
            atomic_json(root/"trials.json", assignments)
        jobs = [("train", arm, i) for i in range(cfg.train_trials) for arm in ["private_rl", "shared_rl"]]
        jobs += [("evaluation", arm, i) for i in range(cfg.evaluation_trials) for arm in ARMS+[ABLATION]]
        frozen = Controllers(cfg, feature_width(cfg, images), cfg.crop_height*cfg.crop_width+2)
        learners = {name: Controllers(cfg, feature_width(cfg, images), cfg.crop_height*cfg.crop_width+2)
                    for name in ["private_rl", "shared_rl"]}
        state = {"binding": binding, "next_job": 0, "total_jobs": len(jobs), "results": [],
                 "learners": {k: v.state() for k, v in learners.items()}, "status": "ready"}
        latest = root/"latest.pt"
        if latest.exists():
            state = torch.load(latest, map_location="cpu", weights_only=False)
            if state["binding"] != binding:
                raise ValueError("Checkpoint binding differs from the experiment")
            for k, learner in learners.items():
                learner.restore(state["learners"][k])

        def checkpoint():
            state["learners"] = {k: v.state() for k, v in learners.items()}
            temporary = latest.with_suffix(".tmp")
            torch.save(state, temporary)
            os.replace(temporary, latest)

        checkpoint()
        if provider_factory:
            provider = provider_factory()
        elif cfg.backend == "mock":
            provider = MockProvider(images)
        else:
            provider = OpenAIProvider(cfg, root/"api.sqlite3", max_usd, retry_uncertain=retry_uncertain)
        deadline = time.monotonic()+hours*3600
        completed_this_invocation = 0
        try:
            while state["next_job"] < len(jobs):
                if time.monotonic() >= deadline or (max_jobs is not None and completed_this_invocation >= max_jobs):
                    state["status"] = "paused_invocation_limit"
                    break
                phase, arm, index = jobs[state["next_job"]]
                trial = trials["train" if phase == "train" else "evaluation"][index]
                bank = learners["shared_rl"] if arm == ABLATION else learners.get(arm, frozen)
                mode = "off" if arm == "paper_reference" else "private" if arm.startswith("private") or arm == ABLATION else "shared"
                print(json.dumps({"phase": phase, "arm": arm, "trial": index+1, "charged_usd": provider.spent}), flush=True)
                before = copy.deepcopy(bank.state()) if phase == "train" else None
                try:
                    result, trajectories, correct = run_episode(cfg, trial, images, provider, bank, mode)
                    result.update(phase=phase, arm=arm)
                    if phase == "train":
                        result["updates"] = bank.learn(trajectories, correct)
                    directory = root/arm/phase
                    directory.mkdir(parents=True, exist_ok=True)
                    atomic_json(directory/f"trial-{index:04d}.json", result)
                except Exception:
                    if before is not None:
                        bank.restore(before)
                    raise
                state["results"].append({k: v for k, v in result.items() if k not in {"calls", "corpus_events"}})
                state["next_job"] += 1
                completed_this_invocation += 1
                state["spent_usd"] = provider.spent
                state["status"] = "complete" if state["next_job"] == len(jobs) else "in_progress"
                checkpoint()
            state["spent_usd"] = provider.spent
            checkpoint()
        except BudgetExhausted as error:
            state["status"] = "paused_cost_guard"
            state["spent_usd"] = provider.spent
            checkpoint()
            print(str(error), flush=True)
        except Exception:
            # Learners are updated only once an entire episode is available. Replaying
            # an interrupted episode reuses complete cached responses, not paid retries.
            state["status"] = "interrupted"
            state["spent_usd"] = provider.spent
            checkpoint()
            report(root, state)
            raise
        finally:
            provider.close()
        return report(root, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/flag-game.yaml")
    parser.add_argument("--mode", choices=["plan", "run", "smoke", "report"], default="plan")
    parser.add_argument("--backend", choices=["openai", "mock"])
    parser.add_argument("--size", choices=["full", "pilot", "smoke"], default="full")
    parser.add_argument("--output")
    parser.add_argument("--max-usd", type=float, help="Explicit cumulative cost ceiling; independent of API account balance")
    parser.add_argument("--hours", type=float, default=12)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--retry-uncertain", action="store_true", help="Explicitly retry ambiguous calls, retaining their old cost reservations")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backend:
        cfg.backend = args.backend
    size = "smoke" if args.mode == "smoke" else args.size
    cfg = sized_config(cfg, size)
    suffix = "" if size == "full" else "-"+size
    root = Path(args.output or f"runs/flag-game{suffix}-{cfg.backend}").resolve()
    if args.hours <= 0 or (args.max_usd is not None and args.max_usd <= 0):
        parser.error("Time/cost limits must be positive")
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error("--max-jobs must be positive")
    if args.mode == "plan":
        result = plan(cfg)
    elif args.mode == "report":
        result = report(root)
    else:
        if cfg.backend == "openai" and args.max_usd is None:
            parser.error("Paid execution requires an explicit --max-usd cumulative ceiling")
        result = deploy(cfg, root, args.max_usd or 0, args.hours, args.max_jobs, retry_uncertain=args.retry_uncertain)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
