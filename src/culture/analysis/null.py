"""Null-population calibration (spec idea N; fix round 1, step 11).

A null population runs the full loop with the stub in `null` mode: every revision is its parent's rule list with a
fresh random stream, so nothing can improve in expectation, but selection still acts on noisy evaluations. K such
populations give, for every generation-level metric, the distribution a population produces with no real learning.
A real run's curve is evidence of learning only where it leaves this band.

Band construction (per metric, per generation, over the K null runs): conformal order statistics. With values sorted
x_(1) <= ... <= x_(K) and r = floor((K + 1) * alpha / 2), the band [x_(r), x_(K+1-r)] contains a new exchangeable draw
with probability (K + 1 - 2r) / (K + 1) >= 1 - alpha. If r < 1 (K < 2/alpha - 1, i.e. K < 39 at 95%) the band is the
envelope [min, max], whose coverage (K - 1) / (K + 1) is below the nominal level; the output says so (`coverage`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import load_run, retained_innovations, series

SERIES_METRICS = ("population_best", "population_mean", "between_offdiag", "hc", "distinct_incumbents",
                  "code_clusters")
LABELS = {"population_best": "population best self-play", "population_mean": "population mean self-play",
          "between_offdiag": "between-group cross-play", "hc": "Higher Criticism statistic",
          "distinct_incumbents": "diversity: distinct incumbents", "code_clusters": "diversity: code clusters"}


def null_band(values: np.ndarray, level: float = 0.95) -> dict[str, Any]:
    """Per-column band over rows (runs). NaNs are ignored column by column."""
    v = np.asarray(values, float)
    if v.ndim == 1:
        v = v[:, None]
    lo, hi, med, cov = [], [], [], []
    for col in v.T:
        x = np.sort(col[~np.isnan(col)])
        k = x.size
        if k == 0:
            lo.append(np.nan), hi.append(np.nan), med.append(np.nan), cov.append(np.nan)
            continue
        r = math.floor((k + 1) * (1 - level) / 2)
        if r >= 1:
            lo.append(x[r - 1]), hi.append(x[k - r]), cov.append((k + 1 - 2 * r) / (k + 1))
        else:
            lo.append(x[0]), hi.append(x[-1]), cov.append((k - 1) / (k + 1) if k > 1 else 0.0)
        med.append(float(np.median(x)))
    return {"lo": np.array(lo), "hi": np.array(hi), "median": np.array(med), "coverage": np.array(cov)}


def run_metrics(run_dir: str | Path) -> dict[str, Any]:
    run = load_run(run_dir)
    s = series(run)
    out = {m: s[m].tolist() for m in SERIES_METRICS}
    out["generation"] = s["generation"].tolist()
    out["retained_innovations"] = retained_innovations(run)
    return out


def calibrate(run_dirs: list[str | Path], level: float = 0.95) -> dict[str, Any]:
    """Null distribution of every generation-level metric over the given null runs (aligned by generation)."""
    per_run = [run_metrics(d) for d in run_dirs]
    g = min(len(r["generation"]) for r in per_run)
    out: dict[str, Any] = {"k": len(per_run), "level": level, "generation": per_run[0]["generation"][:g],
                           "metrics": {}, "run_level": {}}
    for m in SERIES_METRICS:
        vals = np.array([r[m][:g] for r in per_run], float)
        b = null_band(vals, level)
        out["metrics"][m] = {"label": LABELS[m], "lo": _l(b["lo"]), "hi": _l(b["hi"]), "median": _l(b["median"]),
                             "values": [_l(row) for row in vals]}
        out["coverage"] = float(np.nanmin(b["coverage"]))
    ri = np.array([r["retained_innovations"] for r in per_run], float)
    b = null_band(ri, level)
    out["run_level"]["retained_innovations"] = {"lo": float(b["lo"][0]), "hi": float(b["hi"][0]),
                                                "median": float(b["median"][0]), "values": ri.tolist()}
    out["coverage_note"] = (f"band = conformal order statistics over K={len(per_run)} null runs; guaranteed coverage "
                            f"of a new null run per point = {out['coverage']:.3f}"
                            + ("" if out["coverage"] >= level else f" (< {level}: K < 39, so the band is the envelope "
                                                                  f"[min, max]; use more null runs for a {level} band)"))
    return out


def inside(band: dict[str, Any], metric: str, values: list[float]) -> np.ndarray:
    lo, hi = np.array(band["metrics"][metric]["lo"], float), np.array(band["metrics"][metric]["hi"], float)
    v = np.array(values[: lo.size], float)
    return (v >= lo[: v.size] - 1e-9) & (v <= hi[: v.size] + 1e-9) | np.isnan(v) | np.isnan(lo[: v.size])


def save(cal: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cal, indent=1, sort_keys=True) + "\n")


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _l(a) -> list:
    return [None if (isinstance(x, float) and math.isnan(x)) else float(x) for x in np.asarray(a, float)]


# ---------------------------------------------------------------------------------------------- running null pops
def null_config(base, seed: int):
    """The base config switched to the null stub, for one population seed."""
    from ..run.config import from_dict

    if base.llm.backend != "stub":
        raise ValueError("null calibration runs on the stub backend only")
    return from_dict({"condition": "null", "population": {"seed": seed}, "llm": {"stub": {"mode": "null"}}}, base)


def _run_one(args) -> str:
    from ..run.runner import run_config

    cfg, out = args
    run_config(cfg, out)
    return str(out)


def run_null_populations(base, k: int, runs_dir: str | Path, processes: int = 1, first_seed: int = 0) -> list[str]:
    """Run K null populations (population seeds first_seed .. first_seed + K - 1); returns their run directories."""
    import multiprocessing as mp
    import os

    os.environ["PYTHONHASHSEED"] = "0"
    jobs = [(null_config(base, s), Path(runs_dir) / f"p{s}") for s in range(first_seed, first_seed + k)]
    if processes <= 1:
        return [_run_one(j) for j in jobs]
    with mp.get_context("spawn").Pool(min(processes, len(jobs))) as pool:
        return pool.map(_run_one, jobs)
