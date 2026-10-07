"""Null-population calibration (fix round 1, step 11). Runs K populations of the null stub (revisions are the parent's
rule list with a fresh random stream: no hill climbing) and writes the null distribution of every generation-level
metric (population best, mean, between-group cross-play, HC statistic, diversity; retained innovations per run).

Usage: python scripts/null_calibration.py [--config configs/null.yaml] [--k 20] [--processes 8]
       [--runs runs/null] [--out docs/results/null-calibration.json]
Stub only: zero model calls."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.analysis.null import calibrate, run_null_populations, save  # noqa: E402
from culture.run.config import load_config  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "null.yaml"))
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--runs", default=str(ROOT / "runs" / "null"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "null-calibration.json"))
    a = ap.parse_args()
    t0 = time.time()
    base = load_config(a.config)
    dirs = run_null_populations(base, a.k, a.runs, a.processes)
    cal = calibrate(dirs)
    cal["config"] = str(Path(a.config).relative_to(ROOT)) if Path(a.config).is_relative_to(ROOT) else a.config
    cal["wall_seconds"] = round(time.time() - t0, 1)
    save(cal, a.out)
    print(f"{a.k} null populations, {len(cal['generation'])} generations -> {a.out}; {cal['coverage_note']}")
