"""C2 analysis (files/prereg/C2-quarantine.md; Overnight 2): intent-to-treat effects under `verified_quarantine`
against C1's `verified` stratum, paired by replicate seed (same deals, same LLM sampling seed).

Per seed: tau_j for both strata from the exhaustive replays (analysis/credit.tau); primary quantity
tau_flawed(quarantine) - tau_flawed(verified); secondary tau_piers(quarantine) - tau_piers(verified). Aggregates: IQM with
a bootstrap 95% interval over seeds (the resampling unit; one stratum), mean and SD, a paired t interval, and the R for
80% power at the pre-registered 2-point minimum effect (noncentral t, two-sided alpha 0.05), capped at 29. The decision
rules of section 5 are evaluated literally; with fewer seeds than the implied R, or one model block, they are reported as
"not judged (pilot)".

Also a mechanical check: under quarantine a withheld message cannot reach the prompt, so with deterministic sampling
v(S) equals v(S restricted to messages that passed verification); `collapse` reports, per seed, the largest
|v(S) - v(S & passed)| (0 when the outcome depends only on the passed messages).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as st

from . import credit as C


def _v(rows: list[dict[str, Any]], stratum: str) -> dict[int, dict[int, float]]:
    out: dict[int, dict[int, float]] = {}
    for x in rows:
        if x["stratum"] == stratum:
            out.setdefault(int(x["replicate"]), {})[int(x["mask"])] = float(x["y"])
    return out


def _agg(x: list[float], mde: float | None = None, cap: int | None = None) -> dict[str, Any]:
    x = [float(a) for a in x]
    d = C._iqm_ci(x)
    d["mean"] = float(np.mean(x)) if x else None
    d["sd"] = float(np.std(x, ddof=1)) if len(x) > 1 else None
    if len(x) > 1 and d["sd"]:
        half = st.t.ppf(0.975, len(x) - 1) * d["sd"] / np.sqrt(len(x))
        d["t_interval"] = [d["mean"] - half, d["mean"] + half]
        d["t_test_p"] = float(st.ttest_1samp(x, 0.0).pvalue)
    else:
        d["t_interval"], d["t_test_p"] = None, None
    if mde is not None:
        r = C.power_r(d["sd"], mde=mde) if d["sd"] else None
        d["R_for_80pct_power"] = r
        d["R_capped"] = None if r is None else min(r, cap) if cap else r
        d["mde"] = mde
    return d


def c2_summary(rows: list[dict[str, Any]], teachers: list[str], q_stratum: str = "verified_quarantine",
               v_stratum: str = "verified", seeds: list[int] | None = None, mde: float = 2.0, cap: int = 29,
               model_blocks: int = 1) -> dict[str, Any]:
    k = len(teachers)
    fl, pi = teachers.index("flawed_persuasive"), teachers.index("piers")
    vq, vv = _v(rows, q_stratum), _v(rows, v_stratum)
    both = sorted(r for r in set(vq) & set(vv) if len(vq[r]) == 1 << k and len(vv[r]) == 1 << k)
    if seeds is not None:
        both = [r for r in both if r in seeds]
    passed: dict[int, int] = {}
    for x in rows:  # messages that passed verification in the quarantine stratum (adopted under replace_if_better)
        if x["stratum"] == q_stratum:
            for j in x["adopted"]:
                passed[int(x["replicate"])] = passed.get(int(x["replicate"]), 0) | (1 << j)
    per_seed, prim, sec, tfq, tfv, tpq, tpv = {}, [], [], [], [], [], []
    for r in both:
        tq, tv = C.tau(vq[r], k), C.tau(vv[r], k)
        sq, sv = C.shapley(vq[r], k), C.shapley(vv[r], k)
        pm = passed.get(r, 0)
        collapse = max(abs(vq[r][m] - vq[r][m & pm]) for m in range(1 << k))
        per_seed[str(r)] = {"tau_quarantine": tq.tolist(), "tau_verified": tv.tolist(),
                            "shapley_quarantine": sq.tolist(), "shapley_verified": sv.tolist(),
                            "primary": float(tq[fl] - tv[fl]), "secondary": float(tq[pi] - tv[pi]),
                            "v_quarantine": {str(m): vq[r][m] for m in sorted(vq[r])},
                            "v_verified": {str(m): vv[r][m] for m in sorted(vv[r])},
                            "passed_mask": pm, "collapse_max_abs": float(collapse)}
        prim.append(float(tq[fl] - tv[fl]))
        sec.append(float(tq[pi] - tv[pi]))
        tfq.append(float(tq[fl]))
        tfv.append(float(tv[fl]))
        tpq.append(float(tq[pi]))
        tpv.append(float(tv[pi]))
    P, S, TQ = _agg(prim, mde, cap), _agg(sec), _agg(tfq)
    enough = bool(P.get("R_capped") and len(both) >= P["R_capped"]) and model_blocks >= 2

    def excl0(d):
        return d["lo"] is not None and (d["lo"] > 0 or d["hi"] < 0)

    rule1 = bool(P["iqm"] is not None and P["iqm"] >= 2.0 and excl0(P) and TQ["iqm"] is not None
                 and -1.0 <= TQ["iqm"] <= 1.0)
    rule2 = bool(S["iqm"] is not None and not excl0(S) and S["iqm"] > -1.0)
    return {
        "seeds": both, "teachers": teachers, "per_seed": per_seed,
        "primary": {"definition": "tau_flawed(quarantine) - tau_flawed(verified), paired by seed", **P},
        "secondary": {"definition": "tau_piers(quarantine) - tau_piers(verified), paired by seed", **S},
        "tau_flawed_quarantine": TQ, "tau_flawed_verified": _agg(tfv),
        "tau_piers_quarantine": _agg(tpq), "tau_piers_verified": _agg(tpv),
        "collapse_max_abs_over_seeds": max((p["collapse_max_abs"] for p in per_seed.values()), default=None),
        "decision_rules": {
            "removes_influence": {"rule": "primary IQM >= 2 with bootstrap 95% interval excluding 0, and "
                                          "tau_flawed(quarantine) IQM in [-1, 1]; both model blocks",
                                  "met_on_these_seeds": rule1,
                                  "verdict": ("claim" if rule1 else "no claim") if enough else "not judged (pilot)"},
            "costless_for_good_teachers": {"rule": "secondary bootstrap 95% interval contains 0 and IQM > -1; both "
                                                   "model blocks", "met_on_these_seeds": rule2,
                                           "verdict": ("claim" if rule2 else "no claim") if enough
                                           else "not judged (pilot)"},
        },
    }
