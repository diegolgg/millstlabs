"""Statistics: IQM, (stratified) bootstrap, paired deltas, CI widths from sigma, Higher Criticism.

Spec section 9/12. Empirical-null Higher Criticism (Efron) lives in analysis/hc.py and builds on `higher_criticism`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


def iqm(x) -> float:
    """Interquartile mean (mean of the middle 50%, scipy trim_mean with 0.25 cut each side)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(sps.trim_mean(x, 0.25))


def bootstrap_ci(x, stat=iqm, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    vals = np.array([stat(x[i]) for i in idx]) if stat is not np.mean else x[idx].mean(axis=1)
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def iqm_bootstrap_ci(x, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Vectorized bootstrap CI for the IQM (same trimming as scipy trim_mean(x, 0.25))."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return (float("nan"), float("nan"))
    k = int(0.25 * n)
    rng = np.random.default_rng(seed)
    v = np.sort(x[rng.integers(0, n, size=(n_boot, n))], axis=1)[:, k:n - k]
    m = v.mean(axis=1)
    return (float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2)))


def stratified_bootstrap(strata: list, stat=iqm, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Resample within each stratum (e.g. population seed), apply `stat` to the pooled resample.
    Returns (point estimate, (lo, hi), bootstrap draws)."""
    strata = [np.asarray(s, dtype=float) for s in strata if len(s)]
    rng = np.random.default_rng(seed)
    point = stat(np.concatenate(strata))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = stat(np.concatenate([s[rng.integers(0, s.size, s.size)] for s in strata]))
    return float(point), (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))), draws


@dataclass
class PairedDelta:
    mean: float
    se: float
    ci: tuple[float, float]
    n: int


def paired_delta(after, before, alpha: float = 0.05) -> PairedDelta:
    """Mean of (after - before) over identical seeds, with a normal-approximation CI."""
    a = np.asarray(after, dtype=float)
    b = np.asarray(before, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired delta needs scores on identical seeds")
    d = a - b
    n = d.size
    if n == 0:
        return PairedDelta(float("nan"), float("nan"), (float("nan"), float("nan")), 0)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    z = sps.norm.ppf(1 - alpha / 2)
    return PairedDelta(mean, se, (mean - z * se, mean + z * se), n)


def ci_halfwidth_from_sigma(sigma: float, n: int, alpha: float = 0.05) -> float:
    return float(sps.norm.ppf(1 - alpha / 2) * sigma / np.sqrt(n))


def games_for_halfwidth(sigma: float, halfwidth: float, alpha: float = 0.05) -> int:
    z = sps.norm.ppf(1 - alpha / 2)
    return int(np.ceil((z * sigma / halfwidth) ** 2))


def z_to_p(z) -> np.ndarray:
    """One-sided p-values for 'improved' (large positive z)."""
    return sps.norm.sf(np.asarray(z, dtype=float))


def higher_criticism(pvalues, alpha0: float = 0.5, plus: bool = True) -> tuple[float, int]:
    """Donoho-Jin HC*: max over the smallest alpha0 fraction of sorted p-values of
    sqrt(n) (i/n - p_(i)) / sqrt(p_(i)(1 - p_(i))). `plus` (HC+, default) only uses p_(i) > 1/n, which keeps one
    near-zero p-value from dominating. Returns (HC, number of p-values up to the maximizing index)."""
    p = np.sort(np.clip(np.asarray(pvalues, dtype=float), 1e-300, 1 - 1e-15))
    n = p.size
    if n == 0:
        return float("nan"), 0
    i = np.arange(1, n + 1)
    hc = np.sqrt(n) * (i / n - p) / np.sqrt(p * (1 - p))
    k = max(1, int(np.floor(alpha0 * n)))
    mask = np.zeros(n, dtype=bool)
    mask[:k] = True
    if plus:
        mask &= p > 1.0 / n
    if not mask.any():
        return 0.0, 0
    hc = np.where(mask, hc, -np.inf)
    j = int(np.argmax(hc))
    return float(hc[j]), j + 1


def hc_null_threshold(n: int, level: float = 0.05, sims: int = 4000, alpha0: float = 0.5, seed: int = 0) -> float:
    """Monte-Carlo (1 - level) quantile of HC* under the global null with n independent uniform p-values."""
    rng = np.random.default_rng(seed)
    vals = np.array([higher_criticism(rng.uniform(size=n), alpha0)[0] for _ in range(sims)])
    return float(np.quantile(vals, 1 - level))
