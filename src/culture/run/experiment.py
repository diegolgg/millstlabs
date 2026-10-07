"""Run an experiment: one base config, named condition overrides, several population seeds; every run shares the
experiment seed (same deals) and the same warm start. Generation 0 is authored once per population seed into a shared
warm-start set (run/warmstart.py) before any condition starts, and conditions fork from it without calling the
backend for generation 0.

Spec file format (YAML):
    base: {...ExperimentConfig fields...}
    conditions: {name: {...overrides...}, ...}
    population_seeds: [0, 1, 2]
Runs land in <out>/<condition>/p<seed>/; warm-start sets in <out>/_warm_start/. A hosted (paid) backend reserves
against one spend file for the whole experiment: `base.llm.spend_file` if set, else <out>/spend.sqlite."""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import yaml

from .config import from_dict
from .runner import run_config
from .warmstart import build_warm_start, needs_authoring, warm_start_key


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


def _build(args):
    cfg, root = args
    return str(build_warm_start(cfg, root))


def share_spend_file(cfgs: list[tuple[str, int, Any]], out: Path) -> list[tuple[str, int, Any]]:
    """A hosted (paid) backend with no llm.spend_file gets <out>/spend.sqlite in every job, so the llm.max_usd cap is
    cumulative across the whole experiment (warm-start authoring, conditions, population seeds and processes).
    spend_file is not in the config digest or the warm-start key, so this changes no pairing."""
    shared = str((Path(out) / "spend.sqlite").resolve())
    return [(c, ps, from_dict({"llm": {"spend_file": shared}}, cfg) if cfg.llm.hosted and not cfg.llm.spend_file
             else cfg) for c, ps, cfg in cfgs]


def run_experiment(spec_path: str | Path, out: str | Path, processes: int | None = None) -> list[str]:
    spec = load_spec(spec_path)
    out = Path(out)
    cfgs = share_spend_file(configs(spec), out)
    os.environ["PYTHONHASHSEED"] = "0"
    # 1. one warm-start set per distinct key (in practice: per population seed), built before any condition runs
    todo: dict[str, object] = {}
    for _, _, cfg in cfgs:
        if needs_authoring(cfg) and not cfg.population.warm_start_set:
            todo.setdefault(warm_start_key(cfg), cfg)
    procs = processes or min(max(len(cfgs), 1), max(1, (os.cpu_count() or 2) - 2))
    builds = [(cfg, out / "_warm_start") for cfg in todo.values()]
    if procs <= 1 or len(builds) <= 1:
        paths = [_build(b) for b in builds]
    else:
        with mp.get_context("spawn").Pool(min(procs, len(builds))) as pool:
            paths = pool.map(_build, builds)
    sets = dict(zip(todo, paths))
    # 2. conditions fork from the shared set
    jobs = []
    for cname, ps, cfg in cfgs:
        k = warm_start_key(cfg)
        if k in sets:
            cfg = from_dict({"population": {"warm_start_set": sets[k]}}, cfg)
        jobs.append((cfg, out / cname / f"p{ps}"))
    if procs <= 1:
        return [_one(j) for j in jobs]
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.map(_one, jobs)
