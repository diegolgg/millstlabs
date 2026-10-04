import argparse
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Mill Street Labs ecological learning sandbox")
    parser.add_argument("--config", default="configs/main.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("demo", help="Run observation-limited heuristics without neural dependencies")
    p.add_argument("--ticks", type=int, default=256)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--render", action="store_true")
    p = sub.add_parser("calibrate")
    p.add_argument("--ticks", type=int, default=2048)
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    p.add_argument("--output", default="runs/calibration.json")
    p = sub.add_parser("matrix")
    p.add_argument("--output", default="runs/matrix.jsonl")
    p = sub.add_parser("warmstart")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("train")
    p.add_argument("--profile", default="individual")
    p.add_argument("--method", choices=["iteration", "r_adult", "r_initial"], default="r_adult")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--warmstart", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--max-wall-seconds", type=float, default=None,
                   help="Override invocation wall limit without changing checkpoint configuration")
    p = sub.add_parser("benchmark")
    p.add_argument("--ticks", type=int, default=32)
    p.add_argument("--output", default="runs/benchmark.json")
    p = sub.add_parser("analyze")
    p.add_argument("--root", default="runs")
    p.add_argument("--output", default="runs/analysis.json")
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("run-index", help="Run one deterministic manifest index (suitable for job arrays)")
    p.add_argument("--manifest", default="runs/matrix.jsonl")
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--warmstart-dir", default="checkpoints")
    p.add_argument("--output-root", default="runs/main")
    p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "demo":
        from .env import PopulationEnv
        from .observations import Heuristic
        env = PopulationEnv(cfg.environment)
        obs, _ = env.reset(seed=args.seed)
        helpers = {}
        for _ in range(args.ticks):
            if not env.agents:
                break
            for j, i in enumerate(env.agents):
                if i not in helpers:
                    helpers[i] = Heuristic(args.seed + j)
            obs, *_ = env.step({i: helpers[i].act(obs[i]) for i in env.agents})
        if args.render:
            print(env.render())
        result = env.metrics()
    elif args.command == "matrix":
        from .experiments import matrix
        result = matrix(cfg, args.output)
    elif args.command == "calibrate":
        from .experiments import calibrate
        result = calibrate(cfg, args.seeds, args.ticks, args.output)
    elif args.command == "warmstart":
        from .trainer import warmstart
        if Path(args.output).exists():
            parser.error("Warm-start checkpoint already exists; use a new filename to preserve pairing")
        p = warmstart(cfg, args.seed, args.output)
        result = {"path": args.output, "transitions": p["transitions"], "resources": p["resources"]}
    elif args.command == "train":
        from .trainer import Trainer
        runner = Trainer(cfg, args.profile, args.method, args.seed, args.output, args.warmstart, args.resume)
        result = runner.run(max_wall_seconds=args.max_wall_seconds)
    elif args.command == "run-index":
        from .trainer import Trainer
        rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines()]
        row = rows[args.index]
        if row["config_digest"] != cfg.digest():
            parser.error("Manifest and configuration differ; regenerate the matrix after calibration")
        output = Path(args.output_root) / f"{row['profile']}-{row['method']}-s{row['seed']}"
        checkpoint = output / "latest.pt"
        resume = checkpoint if args.resume and checkpoint.exists() else None
        result = Trainer(cfg, row["profile"], row["method"], row["seed"], output,
                         Path(args.warmstart_dir) / f"warm-s{row['seed']}.pt", resume).run()
    elif args.command == "benchmark":
        from .experiments import benchmark
        if args.ticks <= 0:
            parser.error("--ticks must be positive")
        result = benchmark(cfg, args.ticks, args.output)
    elif args.command == "analyze":
        from .experiments import analyze
        result = analyze(args.root, args.output)
    else:
        import uvicorn
        uvicorn.run("millstlabs.api:app", host=args.host, port=args.port, workers=1)
        return
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
