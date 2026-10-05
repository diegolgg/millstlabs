"""Serial CPU sweep with paired warm starts and a resumable invocation time budget."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from .config import load_config


def plan(cfg, seeds=None):
    seeds = cfg.seeds if seeds is None else seeds
    if not set(seeds) <= set(cfg.seeds):
        raise ValueError("Requested seed is not in the frozen configuration")
    rows = []
    for seed in seeds:
        # Visit every profile for each method before moving to the next method.
        # Rotate method order when a configuration includes additional seeds.
        offset = cfg.seeds.index(seed) % len(cfg.methods)
        methods = cfg.methods[offset:] + cfg.methods[:offset]
        for method in methods:
            rows.extend({"profile": profile, "method": method, "seed": seed} for profile in cfg.profiles)
    return {"config_digest": cfg.digest(), "runs": rows,
            "training_decisions": len(rows) * cfg.training.decisions,
            "warmstart_demonstrations": len(seeds) * cfg.training.warmstart_transitions,
            "note": "Serial CPU pilot. The hour limit is soft: warm starts, updates and evaluations finish before pausing."}


def freeze(cfg, root):
    root.mkdir(parents=True, exist_ok=True)
    frozen = root / "config.yaml"
    if frozen.exists():
        if load_config(frozen).digest() != cfg.digest():
            raise ValueError("Output belongs to a different configuration; choose a new --output directory")
    else:
        if any(root.iterdir()):
            raise ValueError("Nonempty output has no frozen config; choose a new --output directory")
        frozen.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    return frozen


def run_batch(cfg, root, frozen, hours, seeds=None, warmstart_dir=None):
    """Use separate processes so each completed condition releases all model memory."""
    deadline = time.monotonic() + hours * 3600
    proposal = plan(cfg, seeds)
    (root / "plan.json").write_text(json.dumps(proposal, indent=2))
    command = [sys.executable, "-m", "millstlabs.cli", "--config", str(frozen)]
    if not (root / "baselines.json").exists():
        subprocess.run(command + ["baselines", "--output", str(root / "baselines.json")], check=True)
    for row in proposal["runs"]:
        output = root / f"{row['profile']}-{row['method']}-s{row['seed']}"
        summary = output / "summary.json"
        if summary.exists() and json.loads(summary.read_text())["finished"]:
            print(f"Already complete: {output.name}", flush=True)
            continue
        if time.monotonic() >= deadline:
            break
        warm = (Path(warmstart_dir) if warmstart_dir else root) / f"warm-s{row['seed']}.pt"
        if not warm.exists():
            print(f"Preparing shared warm start: seed {row['seed']}", flush=True)
            subprocess.run(command + ["warmstart", "--seed", str(row["seed"]), "--output", str(warm)], check=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        latest = output / "latest.pt"
        if output.exists() and any(output.iterdir()) and not latest.exists():
            raise ValueError(f"{output} has partial logs but no checkpoint; move it aside before retrying")
        args = command + ["train", "--profile", row["profile"], "--method", row["method"],
                          "--seed", str(row["seed"]), "--warmstart", str(warm), "--output", str(output),
                          "--max-wall-seconds", str(remaining)]
        if latest.exists():
            args += ["--resume", str(latest)]
        print(f"Running {output.name}; {remaining / 3600:.2f} invocation hours left", flush=True)
        subprocess.run(args, check=True)
        if not json.loads(summary.read_text())["finished"]:
            break
    return report(cfg, root)


def report(cfg, root):
    """Explicitly mark incomplete conditions; only compare initial/final on common maps."""
    results = []
    for row in plan(cfg)["runs"]:
        directory = root / f"{row['profile']}-{row['method']}-s{row['seed']}"
        summary = directory / "summary.json"
        item = {**row, "finished": False, "decisions": 0, "status": "not_started"}
        if summary.exists():
            saved = json.loads(summary.read_text())
            item.update(saved)
            item["status"] = "complete" if saved["finished"] else "paused"
        elif directory.exists():
            item["status"] = "in_progress_or_interrupted"
        evaluation = directory / "evaluation.jsonl"
        if evaluation.exists():
            rows = {r["checkpoint"]: r for r in map(json.loads, evaluation.read_text().splitlines())}
            if 0 in rows:
                item["initial_lifetime"] = rows[0]["restricted_mean_lifetime"]
            if cfg.training.decisions in rows and item["finished"]:
                final = rows[cfg.training.decisions]
                item["final_lifetime"] = final["restricted_mean_lifetime"]
                if 0 in rows:
                    before = {r["map_seed"]: r["restricted_mean_lifetime"] for r in rows[0]["episodes"]}
                    diffs = [r["restricted_mean_lifetime"] - before[r["map_seed"]]
                             for r in final["episodes"] if r["map_seed"] in before]
                    item["lifetime_change_on_common_maps"] = sum(diffs) / len(diffs)
        results.append(item)
    result = {"config_digest": cfg.digest(), "complete": sum(r["finished"] for r in results),
              "total": len(results), "runs": results}
    if root.exists():
        (root / "status.json").write_text(json.dumps(result, indent=2))
    if all(r["finished"] for r in results):
        from .experiments import analyze
        analyze(root, root / "analysis.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/overnight-cpu.yaml")
    parser.add_argument("--output", default="runs/overnight-profiles")
    parser.add_argument("--mode", choices=["plan", "benchmark", "run", "report"], default="plan")
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--warmstart-dir", default=None, help="Shared paired warm starts (architecture/environment must match)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    proposal = plan(cfg, args.seed)
    if args.hours <= 0:
        parser.error("--hours must be positive")
    if cfg.training.device != "cpu":
        parser.error("This runner requires device: cpu")
    root = Path(args.output).resolve()
    if args.mode == "plan":
        result = proposal
    else:
        frozen = freeze(cfg, root)
        with (root / ".runner.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                parser.error("Another runner is using this output directory")
            if args.mode == "benchmark":
                from .experiments import benchmark
                result = benchmark(cfg, 128, root / "benchmark.json")
            elif args.mode == "run":
                result = run_batch(cfg, root, frozen, args.hours, args.seed, args.warmstart_dir)
            else:
                result = report(cfg, root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
