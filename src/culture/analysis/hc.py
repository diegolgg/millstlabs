"""Higher Criticism with Efron's empirical null (spec section 10, item 1).

Each generation, every agent's paired score delta against the frozen previous generation is a z-score. Agents that
share teachers, opponents and seeds have correlated noise, which inflates (and can shift) the null distribution of
those z-scores; testing against N(0, 1) then fires on noise. Efron's remedy: estimate the null N(delta0, sigma0^2) from
the centre of the observed z distribution, standardize, and only then test. HC (HC+ form) detects that a sparse subset
improved before any single agent is significant at Bonferroni; `fraction_improved` estimates how many.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats as sps

from ..evaluate.stats import higher_criticism


def empirical_null(z, method: str = "robust", central: float = 0.5) -> tuple[float, float]:
    """(delta0, sigma0) of the null component.

    `robust` (default): median and IQR/1.349. `mle`: Efron's truncated-normal MLE on the central `central` fraction
    of z (locfdr "MLE" method without the pi0 term), bounded to [1/3, 3] times the robust scale and to the central
    interval for the location; unbounded, that likelihood is nearly flat in sigma and drifts. Both assume the non-null
    component is sparse and sits in a tail."""
    z = np.asarray(z, float)
    med = float(np.median(z))
    iqr_sd = float((np.quantile(z, 0.75) - np.quantile(z, 0.25)) / 1.349)
    if method == "robust" or z.size < 20:
        return med, max(iqr_sd, 1e-6)
    lo, hi = np.quantile(z, [(1 - central) / 2, (1 + central) / 2])
    zz = z[(z >= lo) & (z <= hi)]

    def nll(params):
        d, log_s = params
        s = np.exp(log_s)
        mass = sps.norm.cdf((hi - d) / s) - sps.norm.cdf((lo - d) / s)
        return -(np.sum(sps.norm.logpdf(zz, d, s)) - zz.size * np.log(max(mass, 1e-300)))

    s0 = max(iqr_sd, 1e-3)
    res = optimize.minimize(nll, x0=[med, np.log(s0)], method="L-BFGS-B",
                            bounds=[(float(lo), float(hi)), (np.log(s0 / 3), np.log(s0 * 3))])
    d, log_s = res.x
    return float(d), float(np.exp(log_s))


@dataclass
class HCResult:
    hc: float
    threshold: float
    detected: bool
    delta0: float
    sigma0: float
    fraction_improved: float
    k: int


def fraction_improved(p, alpha: float = 0.05) -> float:
    """Meinshausen-Rice style lower bound on the proportion of non-null (improved) agents:
    sup_t [F_n(t) - t - c sqrt(t (1 - t) / n)] / (1 - t), with c the (1 - alpha) bound of the standardized uniform
    empirical process (approximated by sqrt(2 log log n) + a small-n correction)."""
    p = np.sort(np.asarray(p, float))
    n = p.size
    if n == 0:
        return 0.0
    c = np.sqrt(2 * np.log(np.log(max(n, 16)))) + np.sqrt(np.log(1 / alpha) / 2)
    t = p[(p > 0) & (p < 1)]
    if t.size == 0:
        return 0.0
    F = np.searchsorted(p, t, side="right") / n
    val = (F - t - c * np.sqrt(t * (1 - t) / n)) / (1 - t)
    return float(max(0.0, val.max()))


def _null_threshold(n: int, level: float, sims: int, null: str, seed: int, method: str = "robust") -> float:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(sims):
        z = rng.normal(size=n)
        if null == "empirical":
            d, s = empirical_null(z, method)
            z = (z - d) / s
        vals.append(higher_criticism(sps.norm.sf(z))[0])
    return float(np.quantile(vals, 1 - level))


_THRESH_CACHE: dict[tuple, float] = {}


def hc_test(z, null: str = "empirical", level: float = 0.05, sims: int = 1000, method: str | None = None,
            seed: int = 0) -> HCResult:
    """HC+ on one-sided p-values of (standardized) z. `null`: 'empirical' (Efron) or 'theoretical' (N(0,1)).
    The threshold is the Monte-Carlo (1 - level) quantile of the same procedure under a global null."""
    z = np.asarray(z, float)
    n = z.size
    method = method or "robust"
    if null == "empirical":
        d, s = empirical_null(z, method)
    else:
        d, s = 0.0, 1.0
    p = sps.norm.sf((z - d) / s)
    hc, k = higher_criticism(p)
    key = (n, level, sims, null, seed, method)
    if key not in _THRESH_CACHE:
        _THRESH_CACHE[key] = _null_threshold(n, level, sims, null, seed, method)
    thr = _THRESH_CACHE[key]
    return HCResult(hc, thr, bool(hc > thr), d, s, fraction_improved(p), k)


def plateau(hc_series: list[HCResult | None], k: int = 5) -> int | None:
    """First generation index at which HC has stayed under its threshold for k consecutive generations (the
    stopping / reallocation signal), or None."""
    run = 0
    for i, r in enumerate(hc_series):
        run = run + 1 if (r is not None and not r.detected) else 0
        if run >= k:
            return i - k + 1
    return None


def delta_zscores(after: np.ndarray, before: np.ndarray) -> np.ndarray:
    """Per-agent paired z-scores from games x agents score matrices on shared seeds."""
    d = np.asarray(after, float) - np.asarray(before, float)
    sd = d.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = d.mean(axis=0) / (sd / np.sqrt(d.shape[0]))
    return np.where(sd == 0, np.sign(d.mean(axis=0)) * 8.0, z)
