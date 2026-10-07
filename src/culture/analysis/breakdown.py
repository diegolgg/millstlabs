"""D1: breakdown point of execution-based verification against sabotaged messages
(files/prereg/D1-breakdown-point.md; fix round 1, step 16).

Unit: one group-generation run; outcome Y_G = population mean held-out self-play at the last generation.
Exposure contrast tau(eps, v) = mean Y_G(eps, v) - mean Y_G(0, v). Breakdown point eps*(v) = the smallest eps on the grid
with mean Y_G(eps, v) < 0.7 mean Y_G(0, v), linearly interpolated between grid points; None when the grid never
crosses (censored: eps* > max grid). Primary contrast eps*(on) - eps*(off) with a bootstrap over population seeds,
stratified by cell; censored values are set to the largest grid point (so the contrast is a lower bound).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def breakdown_point(eps: list[float], means: list[float], frac: float = 0.7) -> float | None:
    y0 = means[0]
    thr = frac * y0
    for i in range(1, len(eps)):
        if means[i] < thr:
            e0, e1, m0, m1 = eps[i - 1], eps[i], means[i - 1], means[i]
            return float(e0 + (e1 - e0) * (m0 - thr) / (m0 - m1)) if m0 != m1 else float(e1)
    return None


def summarize(results: dict[str, dict[float, list[float]]], frac: float = 0.7, n_boot: int = 2000,
              seed: int = 0) -> dict[str, Any]:
    """results[arm][eps] = list of Y_G over population seeds (same seeds in every cell)."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"arms": {}, "frac": frac}
    boot: dict[str, list[float]] = {}
    for arm, cells in results.items():
        eps = sorted(cells)
        means = [float(np.mean(cells[e])) for e in eps]
        star = breakdown_point(eps, means, frac)
        cap = max(eps)
        draws = []
        for _ in range(n_boot):
            m = [float(np.mean(rng.choice(cells[e], size=len(cells[e]), replace=True))) for e in eps]
            b = breakdown_point(eps, m, frac)
            draws.append(cap if b is None else b)
        boot[arm] = draws
        out["arms"][arm] = {
            "eps": eps, "mean_Y_G": means, "tau": [m - means[0] for m in means],
            "breakdown_point": star, "censored": star is None,
            "null_band": {str(e): {"min": float(np.min(cells[e])), "max": float(np.max(cells[e])),
                                   "median": float(np.median(cells[e])), "n": len(cells[e])} for e in eps}}
    if {"on", "off"} <= set(boot):
        d = np.array(boot["on"]) - np.array(boot["off"])
        on, off = out["arms"]["on"]["breakdown_point"], out["arms"]["off"]["breakdown_point"]
        cap = max(out["arms"]["on"]["eps"])
        out["primary_contrast"] = {
            "definition": "eps*(on) - eps*(off); censored eps* set to the largest grid value (a lower bound)",
            "point": (cap if on is None else on) - (cap if off is None else off),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
    return out
