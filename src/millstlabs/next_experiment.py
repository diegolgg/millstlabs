"""CPU architecture experiment and independently calibrated predator uncertainty axis."""
import argparse
import fcntl
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .batch import freeze, plan, report, run_batch
from .config import load_config


def calibrate_predator(cfg, root):
    from .experiments import evaluate_baselines

    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for temperature in cfg.training.predator_evaluation_temperatures:
        local = replace(cfg, environment=replace(cfg.environment, predator_temperature=temperature, message_symbols=1),
                        training=replace(cfg.training, evaluation_seed=cfg.training.development_seed,
                                         final_evaluation_maps=max(8, cfg.training.development_maps)))
        result = evaluate_baselines(local, root / f"temperature-{temperature:g}.json")
        rows.append({"temperature": temperature, "scores": result["scores"],
                     "mean_heuristic_lifetime": float(np.mean([result["scores"][m] for m in ["food", "vigilant"]]))})
    result = {"rows": rows, "easier_to_harder_for_these_heuristics":
              [r["temperature"] for r in sorted(rows, key=lambda r: -r["mean_heuristic_lifetime"])],
              "note": "Development maps only. This ordering is empirical and policy-dependent, not a universal difficulty ranking."}
    (root / "predator-calibration.json").write_text(json.dumps(result, indent=2))
    return result


def promotion_gate(directory):
    """Advisory gate; never tunes on test maps or silently mutates an active run."""
    cfg = load_config(directory / "config.yaml")
    path = directory / "development_evaluation.jsonl"
    rows = list({r["checkpoint"]: r for r in map(json.loads, path.read_text().splitlines())}.values()) if path.exists() else []
    rows.sort(key=lambda r: r["checkpoint"])
    trained = [r for r in rows if r["checkpoint"] > 0]
    tail = trained[-3:]
    h = cfg.training.evaluation_horizon
    competent = len(tail) == 3 and all(r["restricted_mean_lifetime"] >= .5*h and r["survival_fraction"] >= .25 for r in tail)
    plateau = len(tail) == 3 and np.ptp([r["restricted_mean_lifetime"] for r in tail]) <= .05*h
    initial = next((r for r in rows if r["checkpoint"] == 0), None)
    improved = bool(initial and tail and tail[-1]["restricted_mean_lifetime"] >= initial["restricted_mean_lifetime"] + .05*h)
    return {"ready_to_test_harder_stage": bool(competent and plateau and improved),
            "competent": bool(competent), "plateau": bool(plateau), "improved_from_warmstart": improved,
            "checkpoints_used": [r["checkpoint"] for r in tail],
            "criteria": "Three nonzero development checkpoints: lifetime >= 50% horizon, survival >= 25%, lifetime range <= 5% horizon, latest gain >= 5% horizon over warm start.",
            "note": "Advisory thresholds, not proof of saturation. Low-score stagnation does not pass. Inspect predator calibration and frozen-policy temperature curves before a new stage."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["core", "cooperative", "communication", "muted", "shared", "numeric"], default="core")
    parser.add_argument("--mode", choices=["plan", "benchmark", "calibrate", "run", "report", "gate"], default="plan")
    parser.add_argument("--config", help="Optional custom YAML; output must match its frozen config")
    parser.add_argument("--output")
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--run-dir", help="One completed/paused condition for the development-only gate")
    parser.add_argument("--warmstart-dir", help="Override warm checkpoint directory; communication/muted share one by default")
    args = parser.parse_args()
    cfg = load_config(args.config or f"configs/next-{args.phase}.yaml")
    if args.hours <= 0 or cfg.training.device != "cpu":
        parser.error("Use a positive --hours budget and device: cpu")
    root = Path(args.output or f"runs/next-{'numeric' if args.phase == 'numeric' else 'llm-' + args.phase}").resolve()
    if args.mode == "plan":
        result = plan(cfg)
    elif args.mode == "gate":
        if not args.run_dir:
            parser.error("--gate requires --run-dir")
        result = promotion_gate(Path(args.run_dir))
    else:
        frozen = freeze(cfg, root)
        with (root / ".runner.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("Another runner uses this output directory")
            if args.mode == "benchmark":
                from .experiments import benchmark
                result = benchmark(cfg, 128, root / "benchmark.json")
            elif args.mode == "calibrate":
                result = calibrate_predator(cfg, root / "calibration")
            elif args.mode == "run":
                warmdir = Path(args.warmstart_dir).resolve() if args.warmstart_dir else (
                    root.parent / "next-llm-communication-warm" if args.phase in {"communication", "muted"} else root)
                warmdir.mkdir(parents=True, exist_ok=True)
                # Also serialize the shared warm checkpoint across both communication arms.
                with (warmdir / ".warm.lock").open("w") as warm_lock:
                    try:
                        fcntl.flock(warm_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        parser.error("Another run uses this warm-start directory; run paired arms serially")
                    result = run_batch(cfg, root, frozen, args.hours, warmstart_dir=warmdir)
            else:
                result = report(cfg, root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
