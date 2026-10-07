"""Credit estimators for one student and k teacher messages, checked against an exact Shapley oracle
(files/prereg/C1-credit-estimators.md; fix round 1, step 13).

A value function v maps a delivered subset S of the k messages (a bitmask) to the student's held-out paired
improvement Y(S). From replays of the same student state under every subset (the exhaustive design) this module
computes, per replicate:

- `shapley`: exact Shapley value per message (the oracle); efficiency: sum = v(all) - v(none);
- `tau` (intent-to-treat main effect): mean over subsets D of the other messages of v(D + j) - v(D), i.e. the main
  effect under Bernoulli(1/2) delivery of the others (the Banzhaf value);
- `adopter_effect`: tau_j / P(adopt j | j delivered), None ("undefined") when j was never adopted;
- `leave_one_out`: v(all) - v(all - j)                                   (k + 1 replays);
- `equal_split`: v(all) split equally among the messages adopted in the all-delivered replay (1 replay);
- `ridge`: ridge regression of v on delivery indicators over Bernoulli(1/2) runs, pooled across replicates;
- `singles_pairs`: from v(none), v({j}), v({j, l}) fit a second-order game and return its exact Shapley value
  a_j + 1/2 sum_l b_jl                                                     (1 + k + C(k, 2) replays);
- `plackett_burman`: main effects from the 8-run Plackett-Burman design (k <= 7)  (8 replays).

Metrics per estimator against the oracle: RMSE over the k teachers, Spearman rank correlation, sign error on the
sabotaged teacher (defined when the oracle's |value| >= 0.5), and replays (= LLM revise calls) per seed.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
from scipy import stats as st

# ----------------------------------------------------------------------------------------------- subsets
def full(k: int) -> int:
    return (1 << k) - 1


def members(mask: int, k: int) -> list[int]:
    return [j for j in range(k) if mask >> j & 1]


def subsets(k: int) -> list[int]:
    return list(range(1 << k))


# ----------------------------------------------------------------------------------------------- oracle and exact
def shapley(v: dict[int, float], k: int) -> np.ndarray:
    phi = np.zeros(k)
    fact = [math.factorial(i) for i in range(k + 1)]
    for j in range(k):
        bit = 1 << j
        for s in range(1 << k):
            if s & bit:
                continue
            size = bin(s).count("1")
            phi[j] += fact[size] * fact[k - size - 1] / fact[k] * (v[s | bit] - v[s])
    return phi


def tau(v: dict[int, float], k: int) -> np.ndarray:
    out = np.zeros(k)
    for j in range(k):
        bit = 1 << j
        diffs = [v[s | bit] - v[s] for s in range(1 << k) if not s & bit]
        out[j] = float(np.mean(diffs))
    return out


def adopter_effect(tau_: np.ndarray, adopt_rate: list[float | None]) -> list[float | None]:
    return [None if not r else float(t / r) for t, r in zip(tau_, adopt_rate)]


def leave_one_out(v: dict[int, float], k: int) -> np.ndarray:
    a = full(k)
    return np.array([v[a] - v[a & ~(1 << j)] for j in range(k)], float)


def equal_split(v_all: float, adopted_in_all: list[int], k: int) -> np.ndarray:
    out = np.zeros(k)
    if adopted_in_all:
        out[adopted_in_all] = v_all / len(adopted_in_all)
    return out


def singles_pairs(v: dict[int, float], k: int) -> np.ndarray:
    a = np.array([v[1 << j] - v[0] for j in range(k)], float)
    phi = a.copy()
    for j, m in itertools.combinations(range(k), 2):
        b = v[(1 << j) | (1 << m)] - v[1 << j] - v[1 << m] + v[0]
        phi[j] += b / 2
        phi[m] += b / 2
    return phi


def singles_pairs_masks(k: int) -> list[int]:
    return [0] + [1 << j for j in range(k)] + [(1 << j) | (1 << m) for j, m in itertools.combinations(range(k), 2)]


# ----------------------------------------------------------------------------------------------- designed delivery
PB8_GENERATOR = [1, 1, 1, 0, 1, 0, 0]  # Plackett-Burman 8-run: cyclic shifts of this row, plus the all-low row


def pb8_masks(k: int) -> list[int]:
    if not 1 <= k <= 7:
        raise ValueError("the 8-run Plackett-Burman design has at most 7 factors")
    rows = [PB8_GENERATOR[-i:] + PB8_GENERATOR[:-i] for i in range(7)] + [[0] * 7]
    return [sum(1 << j for j in range(k) if row[j]) for row in rows]


def plackett_burman(v: dict[int, float], k: int) -> np.ndarray:
    masks = pb8_masks(k)
    y = np.array([v[m] for m in masks], float)
    out = np.zeros(k)
    for j in range(k):
        on = np.array([bool(m >> j & 1) for m in masks])
        out[j] = y[on].mean() - y[~on].mean()
    return out


def bernoulli_masks(k: int, runs: int, rng: np.random.Generator, p: float = 0.5) -> list[int]:
    return [int(sum(1 << j for j in range(k) if rng.random() < p)) for _ in range(runs)]


def ridge(rows: list[tuple[int, float]], k: int, lam: float = 1.0) -> np.ndarray:
    """Coefficients of y on delivery indicators with an unpenalized intercept (centered ridge)."""
    if not rows:
        return np.full(k, np.nan)
    X = np.array([[m >> j & 1 for j in range(k)] for m, _ in rows], float)
    y = np.array([val for _, val in rows], float)
    Xc, yc = X - X.mean(0), y - y.mean()
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(k), Xc.T @ yc)


# ----------------------------------------------------------------------------------------------- metrics
def rmse(est, oracle) -> float:
    est, oracle = np.asarray(est, float), np.asarray(oracle, float)
    return float(np.sqrt(np.mean((est - oracle) ** 2)))


def spearman(est, oracle) -> float | None:
    est, oracle = np.asarray(est, float), np.asarray(oracle, float)
    if np.ptp(est) == 0 or np.ptp(oracle) == 0:
        return None  # undefined for constant input
    return float(st.spearmanr(est, oracle).statistic)


def sign_error(est_j: float, oracle_j: float, min_abs: float = 0.5) -> bool | None:
    if abs(oracle_j) < min_abs:
        return None
    return bool(np.sign(est_j) != np.sign(oracle_j))


def power_r(sd: float, mde: float = 1.0, alpha: float = 0.05, power: float = 0.8, cap: int = 10_000) -> int | None:
    """Smallest number of replicate seeds R for a two-sided paired t-test of mean difference mde with between-seed SD
    sd to reach the given power (noncentral t)."""
    if not sd or sd <= 0 or not np.isfinite(sd):
        return None
    for r in range(2, cap + 1):
        df = r - 1
        tc = st.t.ppf(1 - alpha / 2, df)
        ncp = mde / (sd / math.sqrt(r))
        if 1 - st.nct.cdf(tc, df, ncp) + st.nct.cdf(-tc, df, ncp) >= power:
            return r
    return None


ESTIMATOR_CALLS = {
    "shapley_oracle": lambda k, cfg: 1 << k,
    "tau_itt": lambda k, cfg: 1 << k,
    "leave_one_out": lambda k, cfg: k + 1,
    "equal_split": lambda k, cfg: 1,
    "ridge_bernoulli": lambda k, cfg: cfg.get("ridge_runs", 8),
    "singles_pairs": lambda k, cfg: 1 + k + k * (k - 1) // 2,
    "plackett_burman": lambda k, cfg: 8,
}


def evaluate_replicate(v: dict[int, float], k: int, adopted: dict[int, list[int]], est_extra: dict[str, np.ndarray],
                       sabotaged: int | None) -> dict[str, Any]:
    """All estimators for one replicate (exhaustive v), plus metrics against its Shapley oracle."""
    phi = shapley(v, k)
    t = tau(v, k)
    delivered = {j: [m for m in v if m >> j & 1] for j in range(k)}
    rate = [float(np.mean([j in adopted.get(m, []) for m in delivered[j]])) if delivered[j] else None
            for j in range(k)]
    est = {"tau_itt": t, "leave_one_out": leave_one_out(v, k), "equal_split": equal_split(v[full(k)],
                                                                                         adopted.get(full(k), []), k),
           "singles_pairs": singles_pairs(v, k)}
    if k <= 7:
        est["plackett_burman"] = plackett_burman(v, k)
    est.update(est_extra)
    metrics = {}
    for name, e in est.items():
        metrics[name] = {"rmse": rmse(e, phi), "spearman": spearman(e, phi),
                         "sign_error": sign_error(float(e[sabotaged]), float(phi[sabotaged])) if sabotaged is not None
                         else None}
    return {"shapley": phi.tolist(), "tau": t.tolist(), "adopt_rate": rate,
            "adopter_effect": adopter_effect(t, rate), "efficiency_gap": float(phi.sum() - (v[full(k)] - v[0])),
            "estimates": {n: np.asarray(e, float).tolist() for n, e in est.items()}, "metrics": metrics}


# ----------------------------------------------------------------------------------------------- experiment summary
def _iqm_ci(x: list[float]) -> dict[str, Any]:
    from ..evaluate import stats as S

    x = [float(v) for v in x if v is not None and np.isfinite(v)]
    if not x:
        return {"iqm": None, "lo": None, "hi": None, "n": 0}
    lo, hi = S.iqm_bootstrap_ci(x, n_boot=2000, seed=0) if len(x) > 1 else (x[0], x[0])
    return {"iqm": float(S.iqm(x)), "lo": float(lo), "hi": float(hi), "n": len(x)}


def summarize(rows: list[dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    """Per stratum: per-replicate oracle, ITT effects and estimator metrics (needs the exhaustive design), the pooled
    ridge estimator, aggregates, the primary contrast and its power, and operational numbers (tokens, seconds,
    admissibility, adoption, generalization gap)."""
    k, teachers = spec["k"], spec["teachers"]
    sab = teachers.index("flawed_persuasive") if "flawed_persuasive" in teachers else None
    out: dict[str, Any] = {"k": k, "teachers": teachers, "strata": {}}
    for si, stratum in enumerate(spec["strata"]):
        srows = [r for r in rows if r["stratum"] == stratum]
        reps = sorted({r["replicate"] for r in srows})
        v = {r: {x["mask"]: x["y"] for x in srows if x["replicate"] == r} for r in reps}
        adopted = {r: {x["mask"]: x["adopted"] for x in srows if x["replicate"] == r} for r in reps}
        complete = [r for r in reps if len(v[r]) == 1 << k]
        # ridge on Bernoulli(1/2) runs drawn from each replicate's replays, pooled across replicates
        ridge_rows = []
        for r in complete:
            rng = np.random.default_rng([int(spec.get("ridge_seed", 0)), si, r])
            ridge_rows += [(m, v[r][m]) for m in bernoulli_masks(k, int(spec.get("ridge_runs", 8)), rng)]
        beta = ridge(ridge_rows, k, float(spec.get("ridge_lam", 1.0))) if ridge_rows else None
        per_rep = {}
        for r in complete:
            extra = {"ridge_bernoulli": beta} if beta is not None else {}
            per_rep[r] = evaluate_replicate(v[r], k, adopted[r], extra, sab)
        names = sorted({n for x in per_rep.values() for n in x["metrics"]})
        agg = {}
        for n in names:
            ms = [per_rep[r]["metrics"][n] for r in complete]
            se = [m["sign_error"] for m in ms if m["sign_error"] is not None]
            agg[n] = {"rmse": _iqm_ci([m["rmse"] for m in ms]), "spearman": _iqm_ci([m["spearman"] for m in ms]),
                      "sign_error_rate": float(np.mean(se)) if se else None, "sign_error_defined_in": len(se),
                      "replays_per_seed": ESTIMATOR_CALLS.get(n, lambda *_: None)(k, spec)}
        contrast = [per_rep[r]["metrics"]["ridge_bernoulli"]["rmse"] - per_rep[r]["metrics"]["leave_one_out"]["rmse"]
                    for r in complete if "ridge_bernoulli" in per_rep[r]["metrics"]]
        sd = float(np.std(contrast, ddof=1)) if len(contrast) > 1 else None
        calls = [c for x in srows for c in x.get("calls", [])]
        fresh = [c for c in calls if not c["cached"]]
        revised = [x for x in srows if x.get("admissible") is not None]
        rate_by_teacher = {}
        for j, t in enumerate(teachers):
            dl = [x for x in srows if j in members(x["mask"], k)]
            rate_by_teacher[f"T{j + 1}:{t}"] = float(np.mean([j in x["adopted"] for x in dl])) if dl else None
        gaps = [x["generalization_gap"] for x in srows if x.get("generalization_gap") is not None]
        out["strata"][stratum] = {
            "replicates": reps, "complete_replicates": complete,
            "per_replicate": {str(r): {"shapley": per_rep[r]["shapley"], "tau": per_rep[r]["tau"],
                                       "adopter_effect": per_rep[r]["adopter_effect"],
                                       "adopt_rate": per_rep[r]["adopt_rate"],
                                       "efficiency_gap": per_rep[r]["efficiency_gap"],
                                       "v": {str(m): v[r][m] for m in sorted(v[r])},
                                       "estimates": per_rep[r]["estimates"], "metrics": per_rep[r]["metrics"]}
                              for r in complete},
            "ridge_pooled": None if beta is None else beta.tolist(),
            "estimators": agg,
            "primary_contrast": {"definition": "RMSE(ridge_bernoulli) - RMSE(leave_one_out), per replicate",
                                 "per_replicate": contrast, "mean": float(np.mean(contrast)) if contrast else None,
                                 "sd_between_seeds": sd, "R_for_80pct_power_at_1pt": power_r(sd) if sd else None,
                                 "R_cap": 60},
            "operations": {
                "replays": len(srows), "llm_calls": len(calls), "backend_calls": len(fresh),
                "cached_calls": len(calls) - len(fresh),
                "tokens_per_call": float(np.mean([c["input_tokens"] + c["cache_read_input_tokens"] + c["output_tokens"]
                                                  for c in fresh])) if fresh else None,
                "output_tokens_per_call": float(np.mean([c["output_tokens"] for c in fresh])) if fresh else None,
                "seconds_per_call": float(np.mean([c["latency_s"] for c in fresh if c.get("latency_s") is not None]))
                if fresh else None,
                "admissible_rate": float(np.mean([x["admissible"] for x in revised])) if revised else None,
                "accepted_rate": float(np.mean([bool(x["accepted"]) for x in revised])) if revised else None,
                "adoption_rate_when_delivered": rate_by_teacher,
                "generalization_gap_mean": float(np.mean(gaps)) if gaps else None, "generalization_gap_n": len(gaps),
            },
        }
    return out


def table(summary: dict[str, Any]) -> str:
    lines = ["| stratum | estimator | RMSE IQM [95% CI] | Spearman IQM [95% CI] | sign error on T2 | replays/seed |",
             "|---|---|---|---|---|---|"]

    def f(d):
        return "n/a" if d["iqm"] is None else f"{d['iqm']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}] (n={d['n']})"

    for stratum, s in summary["strata"].items():
        for n, a in sorted(s["estimators"].items()):
            se = "n/a" if a["sign_error_rate"] is None else f"{a['sign_error_rate']:.2f} (of {a['sign_error_defined_in']})"
            lines.append(f"| {stratum} | {n} | {f(a['rmse'])} | {f(a['spearman'])} | {se} | {a['replays_per_seed']} |")
    return "\n".join(lines) + "\n"


def figure(summary: dict[str, Any]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figures import MUTED, SERIES, _style

    strata = list(summary["strata"])
    fig, axes = plt.subplots(1, max(len(strata), 1), figsize=(5.2 * max(len(strata), 1), 4.6), squeeze=False)
    for ax, stratum in zip(axes[0], strata):
        s = summary["strata"][stratum]
        names = sorted(s["estimators"])
        allv = []
        for i, n in enumerate(names):
            xs, ys = [], []
            for rep in s["per_replicate"].values():
                xs += rep["shapley"]
                ys += rep["estimates"][n]
            allv += xs + ys
            ax.scatter(xs, ys, s=22, color=SERIES[i % len(SERIES)], label=n, alpha=0.85, edgecolors="none")
        lim = (min(allv + [0]) - 1, max(allv + [0]) + 1) if allv else (-1, 1)
        ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), label="identity")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        _style(ax, f"{stratum}: estimate vs Shapley oracle", "Shapley credit (oracle, points)",
               "estimated credit (points)")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return fig
