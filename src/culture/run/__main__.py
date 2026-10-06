"""CLI: python -m culture.run --config configs/probe.yaml --out runs/probe [--set population.seed=2 ...]

Re-executes itself with PYTHONHASHSEED=0 so set iteration order (and therefore any bot that iterates a set of strings)
is identical across processes and runs."""

from __future__ import annotations

import argparse
import os
import sys

import yaml


def main() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, "-m", "culture.run", *sys.argv[1:]])
    from .config import apply_dotted, load_config
    from .runner import run_config

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], help="dotted override, e.g. population.seed=2")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.set:
        cfg = apply_dotted(cfg, {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in a.set)})
    run_config(cfg, a.out, a.generations, progress=not a.quiet)


if __name__ == "__main__":
    main()
