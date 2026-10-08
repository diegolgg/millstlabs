"""G1 and S1 pilots on the stub backend (zero model calls): pipeline check and wall-clock sizing for the hosted runs.

Runs configs/g1.yaml (with --generations / --seeds overrides), configs/g1_null.yaml (K = --null-seeds populations per
condition, same length as G1) and configs/s1.yaml (at --s1-generations, --s1-seeds), all in one process pool, longest
jobs first. Then writes docs/results/g1-stub-pilot.json, g1-stub-pilot.png and s1-stub-pilot.json and prints a summary.
Every run is resumable: rerunning the command skips finished runs (their checkpoints are at the last generation) and
redoes only the analysis.

usage: python scripts/g1_pilot.py [--out runs/g1s1-stub] [--generations 50] [--seeds 5] [--null-seeds 10]
                                  [--s1-generations 90] [--s1-seeds 3] [--processes 10] [--skip-run]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.run.config import from_dict  # noqa: E402
from culture.run.experiment import configs, load_spec  # noqa: E402
from culture.run.warmstart import build_warm_start, needs_authoring, warm_start_key  # noqa: E402

RESULTS = ROOT / "docs" / "results"


def _jobs(spec_path: Path, out: Path, generations: int, seeds: int) -> tuple[list, list]:
    """(warm-start builds, run jobs) for one spec with `generations` after the warm start and the first `seeds`
    population seeds."""
    spec = load_spec(spec_path)
    spec["population_seeds"] = list(spec["population_seeds"])[:seeds]
    spec["base"].setdefault("runner", {})["generations"] = generations + 1
    cfgs = configs(spec)
    builds = {}
    for _, _, cfg in cfgs:
        if needs_authoring(cfg) and not cfg.population.warm_start_set:
            builds.setdefault(warm_start_key(cfg), (cfg, out / "_warm_start"))
    jobs = []
    for cname, ps, cfg in cfgs:
        jobs.append((cfg, out / cname / f"p{ps}", warm_start_key(cfg)))
    return list(builds.items()), jobs


def _build(args):
    key, (cfg, root) = args
    return key, str(build_warm_start(cfg, root))


def _run(args):
    from culture.run.runner import run_config

    cfg, out = args
    t0 = time.time()
    run_config(cfg, out)
    return str(out), time.time() - t0


def run_all(a) -> dict:
    os.environ["PYTHONHASHSEED"] = "0"
    out = Path(a.out)
    plan = {"g1": (ROOT / "configs" / "g1.yaml", out / "g1", a.generations, a.seeds),
            "g1_null": (ROOT / "configs" / "g1_null.yaml", out / "g1_null", a.generations, a.null_seeds),
            "s1": (ROOT / "configs" / "s1.yaml", out / "s1", a.s1_generations, a.s1_seeds)}
    builds, jobs = [], []
    for name, (spec, d, gens, seeds) in plan.items():
        b, j = _jobs(spec, d, gens, seeds)
        builds += b
        jobs += [(name, gens, *x) for x in j]
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(min(a.processes, max(1, len(builds)))) as pool:
        sets = dict(pool.map(_build, builds))
    t_ws = time.time() - t0
    work = []
    for name, gens, cfg, rdir, key in jobs:
        if key in sets:
            cfg = from_dict({"population": {"warm_start_set": sets[key]}}, cfg)
        work.append((gens, name, cfg, rdir))
    rank = {n: i for i, n in enumerate(plan)}
    work.sort(key=lambda w: (-w[0], rank[w[1]]))  # longest runs first; among equals G1 before its null runs
    walls = {}
    t1 = time.time()
    with ctx.Pool(a.processes) as pool:
        for rdir, sec in pool.imap_unordered(_run, [(cfg, rdir) for _, _, cfg, rdir in work]):
            walls[rdir] = sec
            print(f"done {Path(rdir).relative_to(out)} in {sec:.0f}s ({len(walls)}/{len(work)})", flush=True)
    return {"warm_start_seconds": round(t_ws, 1), "runs_seconds": round(time.time() - t1, 1),
            "processes": a.processes, "job_wall_seconds": {str(Path(k).relative_to(out)): round(v, 1)
                                                            for k, v in sorted(walls.items())}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "runs" / "g1s1-stub"))
    ap.add_argument("--generations", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--null-seeds", type=int, default=10)
    ap.add_argument("--s1-generations", type=int, default=90)
    ap.add_argument("--s1-seeds", type=int, default=3)
    ap.add_argument("--processes", type=int, default=10)
    ap.add_argument("--skip-run", action="store_true", help="analyse existing runs only")
    a = ap.parse_args()
    out = Path(a.out)
    timing = {}
    if not a.skip_run:
        timing = run_all(a)
        (out / "pilot_timing.json").write_text(json.dumps(timing, indent=1, sort_keys=True) + "\n")
    elif (out / "pilot_timing.json").exists():
        timing = json.loads((out / "pilot_timing.json").read_text())
    from culture.analysis import accumulation as acc

    RESULTS.mkdir(parents=True, exist_ok=True)
    g1 = acc.g1_report(out / "g1", out / "g1_null", generations=a.generations, timing=timing)
    acc.save_json(g1, RESULTS / "g1-stub-pilot.json")
    acc.g1_figure(out / "g1", out / "g1_null", RESULTS / "g1-stub-pilot.png")
    s1 = acc.s1_report(out / "s1", period=30, burn_in=5, timing=timing)
    acc.save_json(s1, RESULTS / "s1-stub-pilot.json")
    print(acc.summary_text(g1, s1))


if __name__ == "__main__":
    main()
