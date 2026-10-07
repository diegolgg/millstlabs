"""B1: verification as a sequential test (research program, section B1; fix round 1, step 15).

An adoption decision is a paired test on per-deal differences d_k = candidate - incumbent (self-play scores on the
same deal), with mean delta and SD sigma. Hypotheses: H0 delta <= 0 (not an improvement) versus H1 delta >= delta1
(the effect worth adopting; delta1 = 1 point). Procedures compared at alpha = beta = 0.05:

- `fixed_n` with N deals, two rules: `mean>0` (the harness's current verification rule: adopt if the paired mean is
  positive) and `t-test` (one-sided paired t-test of H0 at level alpha);
- `sprt`: Wald's sequential probability ratio test for a normal mean, H0 delta = 0 vs H1 delta = delta1, with sigma
  estimated online after a burn-in. Log-likelihood ratio after n deals: (delta1 / s^2) (sum d - n delta1 / 2);
  adopt when it reaches log((1 - beta) / alpha), reject at log(beta / (1 - alpha)); truncated at `cap` deals, where
  it adopts iff the mean exceeds delta1 / 2. Wald's approximation for the expected deals under H1 is
  [(1 - beta) A + beta B] / (delta1^2 / (2 sigma^2)), about 2 sigma^2 log(1/alpha) / delta1^2.

Error rates are counted against the known delta of each pair (from 2,000 separate deals): false adoption = adopting
when delta <= 0; missed improvement = not adopting when delta >= delta1. Pairs with 0 < delta < delta1 are in the
indifference zone and count towards neither. `missed_near_delta1` is the miss rate for 1 <= delta <= 1.25 and
`deals_near_delta1` the mean deals for |delta - 1| <= 0.25: the operating point of the decision rule.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as st


def fixed_n(d: np.ndarray, n: int, rule: str = "t-test", alpha: float = 0.05) -> bool:
    x = np.asarray(d[:n], float)
    if rule == "mean>0":
        return bool(x.mean() > 0)
    sd = x.std(ddof=1)
    if sd == 0:
        return bool(x.mean() > 0)
    t = x.mean() / (sd / math.sqrt(x.size))
    return bool(t > st.t.ppf(1 - alpha, x.size - 1))


def sprt(d: np.ndarray, delta1: float = 1.0, alpha: float = 0.05, beta: float = 0.05, burn_in: int = 20,
         cap: int = 2000, sigma: float | None = None) -> tuple[bool, int]:
    """Returns (adopt, deals used)."""
    a, b = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    x = np.asarray(d[:cap], float)
    s = np.cumsum(x)
    s2 = np.cumsum(x * x)
    for n in range(max(burn_in, 2), x.size + 1):
        if sigma is not None:
            var = sigma * sigma
        else:
            mean = s[n - 1] / n
            var = max((s2[n - 1] - n * mean * mean) / (n - 1), 0.25)  # floor: a sd of 0.5 points
        llr = delta1 / var * (s[n - 1] - n * delta1 / 2)
        if llr >= a:
            return True, n
        if llr <= b:
            return False, n
    return bool(x.mean() > delta1 / 2), x.size


def wald_asn_h1(sigma: float, delta1: float = 1.0, alpha: float = 0.05, beta: float = 0.05) -> float:
    a, b = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    return ((1 - beta) * a + beta * b) / (delta1 ** 2 / (2 * sigma ** 2))


def operating_characteristics(trials: list[dict[str, Any]], delta1: float = 1.0) -> dict[str, Any]:
    """trials: {"procedure", "delta", "adopt", "deals"}. Error rates against the known delta, and deals used."""
    out: dict[str, Any] = {}
    for p in sorted({t["procedure"] for t in trials}):
        ts = [t for t in trials if t["procedure"] == p]
        h0 = [t for t in ts if t["delta"] <= 0]
        h1 = [t for t in ts if t["delta"] >= delta1]
        near = [t for t in ts if abs(t["delta"] - delta1) <= 0.25]
        near_h1 = [t for t in near if t["delta"] >= delta1]
        out[p] = {"false_adoption": float(np.mean([t["adopt"] for t in h0])) if h0 else None, "n_h0": len(h0),
                  "missed_improvement": float(np.mean([not t["adopt"] for t in h1])) if h1 else None, "n_h1": len(h1),
                  "deals_mean": float(np.mean([t["deals"] for t in ts])),
                  "deals_near_delta1": float(np.mean([t["deals"] for t in near])) if near else None,
                  "missed_near_delta1": float(np.mean([not t["adopt"] for t in near_h1])) if near_h1 else None,
                  "n_near_delta1": len(near)}
    return out


def decision_rule(oc: dict[str, Any], tol: float = 0.01, min_saving: float = 0.30) -> dict[str, Any]:
    """Research-program rule: adopt the SPRT as the default if it matches fixed-N error rates with at least 30% fewer
    deals at delta = 1. A fixed-N t-test 'matches' when it is at least as accurate as the SPRT: false adoption and
    miss rate near delta1 each no worse than the SPRT's (within tol). The comparison is against the smallest such N."""
    s = oc["SPRT"]
    fixed = {p: v for p, v in oc.items() if p.startswith("fixed") and "t-test" in p}
    matched = sorted((p for p, v in fixed.items()
                      if v["false_adoption"] is not None and v["missed_near_delta1"] is not None
                      and v["false_adoption"] <= s["false_adoption"] + tol
                      and v["missed_near_delta1"] <= s["missed_near_delta1"] + tol),
                     key=lambda p: fixed[p]["deals_mean"])
    smallest = matched[0] if matched else None
    saving = (1 - s["deals_near_delta1"] / fixed[smallest]["deals_mean"]) if smallest and s["deals_near_delta1"] else None
    return {"rule": "adopt SPRT as default if, at delta = 1, it uses >= 30% fewer deals than the smallest fixed-N "
                    "t-test that is as accurate (false adoption and miss rate near delta = 1 no worse than the SPRT's)",
            "fixed_n_as_accurate_as_sprt": matched, "smallest_as_accurate": smallest,
            "sprt_deals_near_delta1": s["deals_near_delta1"], "saving_vs_smallest": saving,
            "passes": bool(saving is not None and saving >= min_saving)}
