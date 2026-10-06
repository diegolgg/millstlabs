"""Run an experiment: one base config, named condition overrides, several population seeds; every run shares the
experiment seed (same deals) and, because seed tags omit the condition, the same warm start.

Spec file format (YAML):
    base: {...ExperimentConfig fields...}
    conditions: {name: {...overrides...}, ...}
    population_seeds: [0, 1, 2]
Runs land in <out>/<condition>/p<seed>/."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import yaml

from .config import from_dict
from .runner import run_config


def load_spec(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        spec = yaml.safe_load(f)
    for k in ("base", "conditions", "population_seeds"):
        if k not in spec:
            raise ValueError(f"experiment spec needs `{k}`")
    return spec


def configs(spec: dict[str, Any]) -> list[tuple[str, int, Any]]:
    base = from_dict(spec["base"])
    out = []
    for cname, over in spec["conditions"].items():
        cfg_c = from_dict(dict(over or {}, condition=cname), base)
        for ps in spec["population_seeds"]:
            out.append((cname, ps, from_dict({"population": {"seed": ps}}, cfg_c)))
    return out


def _one(args):
    cfg, out = args
    run_config(cfg, out)
    return str(out)


def run_experiment(spec_path: str | Path, out: str | Path, processes: int | None = None) -> list[str]:
    spec = load_spec(spec_path)
    out = Path(out)
    jobs = [(cfg, out / cname / f"p{ps}") for cname, ps, cfg in configs(spec)]
    os.environ["PYTHONHASHSEED"] = "0"
    processes = processes or min(len(jobs), max(1, (os.cpu_count() or 2) - 2))
    if processes <= 1:
        return [_one(j) for j in jobs]
    with mp.get_context("spawn").Pool(processes) as pool:
        return pool.map(_one, jobs)
