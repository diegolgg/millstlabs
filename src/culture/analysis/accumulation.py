"""G1 (accumulation under organization) and S1 (switchback perturbation of levers): trajectories, calls to threshold,
paired condition contrasts, null bands, switchback contrasts, and the G1 figure.

Everything is read from a run's logs (generations.jsonl, ledger.jsonl, artifacts.jsonl); nothing is re-simulated.

Outcome fields, per generation g (run/generation.py `_finish`):
- population_mean / population_best: mean and max over agents of the incumbent's self-play mean on generation g's
  held-out evaluation deals (`evaluation.selfplay_games` deals, shared by every condition and seed);
- within_group: mean over agents of the incumbent's cross-play with its group's other incumbents (None for groups
  of one, so NaN for G1's isolated condition);
- between_group: off-diagonal mean of the cross-play matrix of group-best artifacts (None with one group, so NaN for
  G1's full condition); with singleton groups it is the population's mean pairwise cross-play;
- retained_innovations: metrics.retained_innovations on the run truncated at g (k = 10): artifacts that beat the archive
  best when first evaluated and have a descendant at least k generations later, as known at generation g;
- calls / tokens: cumulative over the ledger's non-refused rows with generation <= g (the run's own ledger: the
  warm-start authoring calls live in the shared warm-start set's ledger and are identical across conditions);
- refusals: budget refusals in generation g (G1's cap is designed never to bind, so these should be 0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluate import stats
from . import null as null_mod
from .metrics import load_run, retained_innovations, series

OUTCOMES = ("population_mean", "population_best", "within_group", "between_group")


# ---------------------------------------------------------------------------------------------------- trajectories
def _retained_series(run: dict[str, Any], gens: np.ndarray, k: int = 10) -> np.ndarray:
    arts = run["artifacts"]
    out = []
    for i, g in enumerate(gens):
        trunc = {"generations": run["generations"][: i + 1], "artifacts": [a for a in arts if a.get("generation", 0) <= g]}
        out.append(retained_innovations(trunc, k))
    return np.array(out, float)


def run_trajectory(run_dir: str | Path, retained: bool = True) -> dict[str, Any]:
    """Per-generation series of one run (see the module docstring for the fields)."""
    run = load_run(run_dir)
    s = series(run)
    gens = s["generation"]
    led = [r for r in run["ledger"] if not r.get("refused")]
    calls_g: dict[int, int] = {}
    tok_g: dict[int, int] = {}
    for r in led:
        g = int(r["generation"])
        calls_g[g] = calls_g.get(g, 0) + 1
        tok_g[g] = tok_g.get(g, 0) + sum(int(r.get(k, 0)) for k in ("input_tokens", "output_tokens",
                                                                      "cache_read_input_tokens",
                                                                      "cache_creation_input_tokens"))
    refused: dict[int, int] = {}
    for r in run["ledger"]:
        if r.get("refused"):
            refused[int(r["generation"])] = refused.get(int(r["generation"]), 0) + 1
    recs = run["generations"]
    cfg = run["config"]
    return {
        "dir": str(run_dir),
        "condition": cfg["condition"],
        "seed": int(cfg["population"]["seed"]),
        "name": cfg["name"],
        "generation": gens,
        "population_mean": s["population_mean"],
        "population_best": s["population_best"],
        "within_group": s["within_crossplay"],
        "between_group": s["between_offdiag"],
        "retained_innovations": _retained_series(run, gens) if retained else None,
        "calls": np.cumsum([calls_g.get(int(g), 0) for g in gens]).astype(float),
        "tokens": np.cumsum([tok_g.get(int(g), 0) for g in gens]).astype(float),
        "refusals": np.array([refused.get(int(g), 0) for g in gens], float),
        "wall_seconds": np.array([r.get("volatile", {}).get("wall_seconds", np.nan) for r in recs], float),
        "levers": [r.get("levers") for r in recs] if any("levers" in r for r in recs) else None,
        "ladder": {int(r["generation"]): r["ladder"] for r in recs if r.get("ladder")},
    }


def trajectories(run_dirs, retained: bool = True) -> dict[str, dict[int, dict[str, Any]]]:
    """{condition: {population seed: run_trajectory}} for the given run directories (condition and seed are read from
    each run's config)."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for d in run_dirs:
        t = run_trajectory(d, retained)
        out.setdefault(t["condition"], {})[t["seed"]] = t
    return {c: dict(sorted(v.items())) for c, v in sorted(out.items())}


def run_dirs_under(root: str | Path) -> list[Path]:
    """Every run directory (one with generations.jsonl) below `root`, i.e. <root>/<condition>/p<seed>."""
    return sorted(p.parent for p in Path(root).glob("*/p*/generations.jsonl"))


# ---------------------------------------------------------------------------------------------- calls to threshold
def calls_to_threshold(run, thresholds=(17.0, 20.0), key: str = "population_mean") -> dict[str, dict[str, Any]]:
    """Cumulative language-model calls at the first generation whose `key` reaches each threshold, censored at the
    run length.

    The design asks for the frozen-ladder evaluation, but the ladder is logged only every `ladder_every` generations
    and only for the population best (its cross-play with frozen snapshots and anchors, no self-play). So this uses the
    held-out self-play on each generation's evaluation deals (`population_mean` = mean over agents of the incumbent's
    self-play mean on those deals), which every generation record has. Deals are shared by all conditions and seeds of
    an experiment, so the comparison is paired on deals.

    `run`: a trajectory (run_trajectory) or a run directory. Returns {str(threshold): {generation, calls, censored,
    calls_at_end, last_generation}}; a censored threshold has generation and calls None."""
    t = run if isinstance(run, dict) else run_trajectory(run, retained=False)
    y, g, c = np.asarray(t[key], float), np.asarray(t["generation"]), np.asarray(t["calls"], float)
    out = {}
    for th in thresholds:
        hit = np.where(y >= th)[0]
        if hit.size:
            i = int(hit[0])
            out[str(th)] = {"generation": int(g[i]), "calls": float(c[i]), "censored": False,
                            "calls_at_end": float(c[-1]), "last_generation": int(g[-1])}
        else:
            out[str(th)] = {"generation": None, "calls": None, "censored": True, "calls_at_end": float(c[-1]),
                            "last_generation": int(g[-1])}
    return out


# ------------------------------------------------------------------------------------------- condition contrasts
def _value_at(t: dict[str, Any], key: str, generation: int) -> float:
    g = np.asarray(t["generation"])
    i = np.where(g == generation)[0]
    return float(np.asarray(t[key], float)[i[0]]) if i.size else float("nan")


def condition_contrast(trajs, a: str, b: str, generation: int = -1, key: str = "population_mean",
                       n_boot: int = 5000, seed: int = 0) -> dict[str, Any]:
    """Paired-by-seed difference a minus b in `key` at `generation` (-1: the last generation every run of both
    conditions reached). Point estimate: IQM of the per-seed differences (scipy trim_mean 0.25; with 5 seeds the mean
    of the middle 3); interval: 95% percentile bootstrap over seeds of the IQM (stats.bootstrap_ci). The seed is the
    unit (population seeds are the replicates); deals are identical across conditions, so each difference is paired on
    deals and warm start."""
    seeds = sorted(set(trajs[a]) & set(trajs[b]))
    if generation == -1:
        generation = int(min(int(np.asarray(trajs[c][s]["generation"])[-1]) for c in (a, b) for s in seeds))
    diffs = np.array([_value_at(trajs[a][s], key, generation) - _value_at(trajs[b][s], key, generation)
                      for s in seeds], float)
    ok = ~np.isnan(diffs)
    d = diffs[ok]
    lo, hi = stats.bootstrap_ci(d, stat=stats.iqm, n_boot=n_boot, seed=seed) if d.size else (math.nan, math.nan)
    return {"a": a, "b": b, "key": key, "generation": generation, "n": int(d.size),
            "seeds": [s for s, k in zip(seeds, ok) if k], "diffs": [round(float(x), 4) for x in d],
            "iqm": stats.iqm(d) if d.size else math.nan, "mean": float(d.mean()) if d.size else math.nan,
            "ci": [lo, hi]}


# ----------------------------------------------------------------------------------------------------- null band
def null_band(null_run_dirs, key: str = "population_mean", level: float = 0.95) -> dict[str, Any]:
    """Per-generation band of `key` over null-stub runs (analysis/null.py `null_band`: conformal order statistics,
    the [min, max] envelope when K < 2/(1 - level) - 1, with its actual per-point coverage). Runs are aligned by
    generation index and cut to the shortest. Also returns the null runs' mean cumulative calls per generation, so
    the band can be drawn on a calls axis."""
    ts = [d if isinstance(d, dict) else run_trajectory(d, retained=key == "retained_innovations")
          for d in null_run_dirs]
    n = min(len(t["generation"]) for t in ts)
    vals = np.array([np.asarray(t[key], float)[:n] for t in ts])
    b = null_mod.null_band(vals, level)
    calls = np.array([np.asarray(t["calls"], float)[:n] for t in ts])
    return {"key": key, "k": len(ts), "level": level, "generation": np.asarray(ts[0]["generation"])[:n],
            "lo": b["lo"], "hi": b["hi"], "median": b["median"], "coverage": b["coverage"],
            "calls": calls.mean(axis=0)}


# -------------------------------------------------------------------------------------------------- switchbacks
def lever_on_series(t: dict[str, Any], lever: str, on_generation: int = 1) -> np.ndarray:
    """1 where the lever's logged value equals its value at `on_generation` (S1 starts every lever ON at generation
    1), else 0; aligned with t["generation"]."""
    levers = t["levers"]
    if levers is None:
        raise ValueError(f"{t['dir']} has no lever schedule")
    g = list(np.asarray(t["generation"]))
    ref = levers[g.index(on_generation)][lever]
    return np.array([1 if (lv or {}).get(lever) == ref else 0 for lv in levers], int)


def _blocks(lever: np.ndarray) -> list[tuple[int, int, int]]:
    """Maximal runs of constant lever value: (start, end exclusive, value)."""
    out, start = [], 0
    for i in range(1, lever.size + 1):
        if i == lever.size or lever[i] != lever[start]:
            out.append((start, i, int(lever[start])))
            start = i
    return out


def mixing_time(y, max_lag: int | None = None) -> tuple[int, bool]:
    """First lag >= 1 at which the sample autocorrelation of y drops below 1/e; (lag, censored). Censored at the
    largest lag examined (n // 2 by default) if it never drops."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = y.size
    if n < 3:
        return 0, True
    x = y - y.mean()
    den = float((x * x).sum())
    if den == 0:
        return 0, False  # constant series: no memory to mix
    max_lag = max_lag or max(1, n // 2)
    for lag in range(1, max_lag + 1):
        if float((x[:-lag] * x[lag:]).sum()) / den < 1 / math.e:
            return lag, False
    return max_lag, True


def switchback_contrasts(run, lever_series, outcome_series, period: int = 30, burn_in: int = 5) -> dict[str, Any]:
    """On-minus-off contrast of a switchback run (S1), plus a mixing-time estimate and a bias bound.

    `lever_series` (1 = ON, 0 = OFF) and `outcome_series` are aligned per generation; pass generations >= 1 (the warm
    start is not part of the design). `run` is the run's trajectory (used only for generation numbers) or None.

    - Blocks are the maximal runs of constant lever value; the first `burn_in` generations of every block are dropped
      (the first block too: it follows the warm start). Block means use the remaining generations.
    - `contrast`: the mean over switches of (ON block mean - OFF block mean) for each pair of adjacent blocks. With
      alternating blocks this weights interior blocks twice and cancels a linear drift in the outcome (accumulation),
      which a pooled difference does not. `pooled` = mean over kept ON generations - mean over kept OFF generations
      (the difference-in-means of Wager eq. 15.26 with burn-in) is reported alongside.
    - `mixing_time`: first lag at which the autocorrelation of the outcome drops below 1/e, computed on the residual
      of a least-squares fit of the outcome on (1, lever, generation), so the lever's own square wave and a linear
      accumulation trend do not dominate the autocorrelation. (Per-block demeaning was tried and rejected: with
      30-generation blocks it biases the autocorrelation down and halves t0 on an AR(1) with known t0.) Curvature the
      linear trend misses inflates t0, which makes the bound larger, the conservative direction.
      `mixing_time_raw` is the same on the globally demeaned outcome.
    - `bias_bound` = 4 M lam (1 + t0) with lam = 1/period, t0 = mixing_time and M = the largest absolute change of the
      outcome between consecutive generations observed (as specified for S1). This has the form of Wager's Theorem
      15.5 (eq. 15.29), but the theorem's M bounds |Y_t| itself, its lam is the switch probability of a memoryless
      randomized switchback, its t0 is the state chain's total-variation mixing time, and it bounds the bias of the
      Horvitz-Thompson estimator; our design switches deterministically every `period` generations and uses
      difference-in-means with burn-in (the book refers fixed-length switchbacks to Hu and Wager 2026). So the bound
      is a heuristic scale, not a guarantee. `bias_bound_theorem_M` uses M = max |Y_t| as the theorem states.
    """
    lever = np.asarray(lever_series, int)
    y = np.asarray(outcome_series, float)
    if lever.shape != y.shape:
        raise ValueError("lever and outcome series must be aligned")
    gens = np.asarray(run["generation"])[-y.size:] if isinstance(run, dict) and "generation" in run else np.arange(y.size)
    blocks = _blocks(lever)
    rows = []
    for s, e, v in blocks:
        kept = y[s + burn_in:e]
        m = float(np.nanmean(kept)) if kept.size and not np.isnan(kept).all() else math.nan
        rows.append({"first_generation": int(gens[s]), "last_generation": int(gens[e - 1]), "on": bool(v),
                     "length": e - s, "kept": int(kept.size), "mean": m})
    per_switch = []
    for b0, b1 in zip(rows, rows[1:]):
        on, off = (b0, b1) if b0["on"] else (b1, b0)
        per_switch.append({"at": b1["first_generation"], "to": "on" if b1["on"] else "off",
                           "on_minus_off": on["mean"] - off["mean"]})
    c = np.array([p["on_minus_off"] for p in per_switch], float)
    keep = np.zeros(y.size, bool)
    for s, e, _ in blocks:
        keep[s + burn_in:e] = True
    on_k, off_k = y[keep & (lever == 1)], y[keep & (lever == 0)]
    pooled = (float(np.nanmean(on_k)) - float(np.nanmean(off_k))) if on_k.size and off_k.size else math.nan
    ok = ~np.isnan(y)
    X = np.column_stack([np.ones(y.size), lever, np.arange(y.size)])[ok]
    resid = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0] if ok.sum() > 3 else y[ok]
    t0, cens = mixing_time(resid)
    t0_raw, cens_raw = mixing_time(y)
    dy = np.abs(np.diff(y))
    M = float(np.nanmax(dy)) if dy.size and not np.isnan(dy).all() else math.nan
    M_abs = float(np.nanmax(np.abs(y))) if y.size else math.nan
    lam = 1.0 / period
    return {"period": period, "burn_in": burn_in, "blocks": rows, "switches": per_switch, "n_switches": len(per_switch),
            "contrast": float(np.nanmean(c)) if c.size and not np.isnan(c).all() else math.nan,
            "pooled": pooled, "mixing_time": t0, "mixing_time_censored": cens, "mixing_time_raw": t0_raw,
            "mixing_time_raw_censored": cens_raw, "M_max_step": M, "lam": lam,
            "bias_bound": 4 * M * lam * (1 + t0), "M_max_abs": M_abs,
            "bias_bound_theorem_M": 4 * M_abs * lam * (1 + t0)}


# ------------------------------------------------------------------------------------------------------ figure
def g1_figure(g1, null=None, path: str | Path | None = None, key: str = "population_mean",
              title: str = "G1 on the stub"):
    """Two panels. Left: population mean self-play against cumulative calls, per condition: the line is the median
    over seeds at each generation (x = mean cumulative calls over seeds), the band the seed range; the condition's
    null-stub band is drawn as a dotted outline of the same color. Right: between-group cross-play against generation
    (full has one group, so its within-group cross-play, the population's mean pairwise cross-play, is drawn dashed).

    `g1` and `null`: trajectories ({condition: {seed: trajectory}}) or directories holding <condition>/p<seed>."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .figures import SERIES, _style

    if not isinstance(g1, dict):
        g1 = trajectories(run_dirs_under(g1), retained=False)
    if null is not None and not isinstance(null, dict):
        null = trajectories(run_dirs_under(null), retained=False)
    order = [c for c in ("isolated", "full", "organized") if c in g1] + sorted(set(g1) - {"isolated", "full", "organized"})
    color = {c: SERIES[i] for i, c in enumerate(order)}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for c in order:
        runs = list(g1[c].values())
        n = min(len(t["generation"]) for t in runs)
        Y = np.array([np.asarray(t[key], float)[:n] for t in runs])
        X = np.array([np.asarray(t["calls"], float)[:n] for t in runs]).mean(axis=0)
        if null and c in null:
            nb = null_band(list(null[c].values()), key)
            m = min(n, len(nb["lo"]))
            for edge in (nb["lo"][:m], nb["hi"][:m]):
                ax1.plot(nb["calls"][:m], edge, color=color[c], linewidth=1, linestyle=(0, (1, 2)), alpha=0.9)
        ax1.fill_between(X, Y.min(axis=0), Y.max(axis=0), color=color[c], alpha=0.15, linewidth=0)
        ax1.plot(X, np.median(Y, axis=0), color=color[c], linewidth=2, label=f"{c} (median of {len(runs)} seeds)")
        ax1.text(X[-1], np.median(Y, axis=0)[-1], f" {c}", color="#0b0b0b", fontsize=8, va="center")
        G = np.asarray(runs[0]["generation"])[:n]
        B = np.array([np.asarray(t["between_group"], float)[:n] for t in runs])
        if np.isnan(B).all():
            B = np.array([np.asarray(t["within_group"], float)[:n] for t in runs])
            ls, lab = (0, (4, 2)), f"{c}: within its one group"
        else:
            ls, lab = "-", c
        ax2.fill_between(G, np.nanmin(B, axis=0), np.nanmax(B, axis=0), color=color[c], alpha=0.15, linewidth=0)
        ax2.plot(G, np.nanmedian(B, axis=0), color=color[c], linewidth=2, linestyle=ls, label=lab)
        ax2.text(G[-1], np.nanmedian(B, axis=0)[-1], f" {c}", color="#0b0b0b", fontsize=8, va="center")
    ax1.plot([], [], color="#52514e", linewidth=1, linestyle=(0, (1, 2)), label="null-stub band (dotted, per condition)")
    _style(ax1, f"{title}: population mean self-play vs calls", "cumulative language-model calls (run ledger)",
           "mean self-play score (held-out deals)")
    _style(ax2, "Between-group cross-play", "generation", "cross-play score of group bests")
    for ax in (ax1, ax2):
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


# ------------------------------------------------------------------------------------------------------ reports
def _r(x, nd: int = 3):
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) else round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_r(v, nd) for v in x.tolist()]
    if isinstance(x, dict):
        return {str(k): _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


def _wall(trajs) -> dict[str, Any]:
    per_gen = [w for c in trajs.values() for t in c.values() for w in np.asarray(t["wall_seconds"])[1:]]
    per_run = [float(np.nansum(t["wall_seconds"])) for c in trajs.values() for t in c.values()]
    by_cond = {c: round(float(np.nanmean([w for t in v.values() for w in np.asarray(t["wall_seconds"])[1:]])), 2)
               for c, v in trajs.items()}
    return {"seconds_per_generation_mean": round(float(np.nanmean(per_gen)), 2) if per_gen else None,
            "seconds_per_generation_by_condition": by_cond,
            "seconds_per_run_mean": round(float(np.mean(per_run)), 1) if per_run else None,
            "seconds_per_run_max": round(float(np.max(per_run)), 1) if per_run else None,
            "note": "generation wall time inside the run (volatile.wall_seconds), excluding generation 0; runs shared "
                    "the machine with the other pilot processes"}


def g1_report(g1_dir, null_dir=None, generations: int | None = None, timing: dict | None = None,
              thresholds=(17.0, 20.0)) -> dict[str, Any]:
    trajs = trajectories(run_dirs_under(g1_dir))
    nulls = trajectories(run_dirs_under(null_dir), retained=False) if null_dir and Path(null_dir).exists() else {}
    conds = {}
    for c, runs in trajs.items():
        per_seed = {}
        for s, t in runs.items():
            per_seed[s] = {"final_generation": int(t["generation"][-1]),
                           "population_mean": t["population_mean"][-1], "population_best": t["population_best"][-1],
                           "between_group": t["between_group"][-1], "within_group": t["within_group"][-1],
                           "retained_innovations": t["retained_innovations"][-1], "calls": t["calls"][-1],
                           "tokens": t["tokens"][-1], "refusals": float(np.sum(t["refusals"])),
                           "calls_to_threshold": calls_to_threshold(t, thresholds)}
        agg = {}
        for k in ("population_mean", "population_best", "between_group", "within_group", "retained_innovations",
                  "calls", "tokens", "refusals"):
            v = np.array([p[k] for p in per_seed.values()], float)
            v = v[~np.isnan(v)]
            agg[k] = {"iqm": stats.iqm(v) if v.size else None, "min": float(v.min()) if v.size else None,
                      "max": float(v.max()) if v.size else None}
        cross = {}
        for th in thresholds:
            hits = [p["calls_to_threshold"][str(th)] for p in per_seed.values()]
            reached = [h["calls"] for h in hits if not h["censored"]]
            cross[str(th)] = {"seeds_reached": len(reached), "seeds": len(hits),
                              "calls_median_if_reached": float(np.median(reached)) if reached else None,
                              "generation_median_if_reached": float(np.median([h["generation"] for h in hits
                                                                                 if not h["censored"]]))
                              if reached else None}
        conds[c] = {"final": agg, "calls_to_threshold": cross, "per_seed": per_seed}
    contrasts = []
    for a, b in (("organized", "isolated"), ("full", "isolated"), ("organized", "full")):
        if a in trajs and b in trajs:
            contrasts.append(condition_contrast(trajs, a, b, key="population_mean"))
    if "organized" in trajs and "isolated" in trajs:
        contrasts.append(condition_contrast(trajs, "organized", "isolated", key="between_group"))
    null_out = {}
    for c, runs in nulls.items():
        nb = null_band(list(runs.values()), "population_mean")
        nbb = null_band(list(runs.values()), "between_group")
        null_out[c] = {"k": nb["k"], "coverage_per_point": float(np.nanmin(nb["coverage"])),
                       "population_mean_final": [nb["lo"][-1], nb["hi"][-1]],
                       "between_group_final": [nbb["lo"][-1], nbb["hi"][-1]],
                       "population_mean_lo": nb["lo"], "population_mean_hi": nb["hi"],
                       "real_median_final_inside": None}
        if c in trajs:
            med = float(np.median([t["population_mean"][-1] for t in trajs[c].values()]))
            null_out[c]["real_median_final_inside"] = bool(nb["lo"][-1] - 1e-9 <= med <= nb["hi"][-1] + 1e-9)
    return _r({
        "scope": ("G1 stub pilot: pipeline check and wall-clock sizing only. Every agent is a perturbed copy of an "
                  "anchor rule list (stub backend, zero model calls); no number here is evidence about organization."),
        "generations": generations, "conditions": conds, "contrasts": contrasts, "null_band": null_out,
        "wall_clock": {"g1": _wall(trajs), "g1_null": _wall(nulls) if nulls else None, "pilot": timing or {}},
        "outcome_note": ("calls to threshold use held-out self-play on each generation's evaluation deals, not the "
                         "frozen ladder (logged every 10 generations for the population best only)"),
    })


S1_LEVERS = {"teaching": "routing", "verification": "verification", "quarantine": "quarantine_unverified",
             "migration": "topology", "selection": "selection"}


def s1_report(s1_dir, period: int = 30, burn_in: int = 5, timing: dict | None = None,
              outcomes=("population_mean", "population_best")) -> dict[str, Any]:
    trajs = trajectories(run_dirs_under(s1_dir), retained=False)
    out = {}
    for c, runs in trajs.items():
        lever = S1_LEVERS.get(c)
        rows = {}
        for key in outcomes:
            per_seed = {}
            for s, t in runs.items():
                g = np.asarray(t["generation"])
                sel = g >= 1
                on = lever_on_series(t, lever)[sel]
                sb = switchback_contrasts({"generation": g[sel]}, on, np.asarray(t[key], float)[sel], period, burn_in)
                per_seed[s] = sb
            cs = np.array([v["contrast"] for v in per_seed.values()], float)
            fin = cs[np.isfinite(cs)]
            lo, hi = stats.bootstrap_ci(fin, stat=np.mean, n_boot=5000, seed=0) if fin.size > 1 else (math.nan, math.nan)
            mean_cs = float(np.nanmean(cs)) if np.isfinite(cs).any() else math.nan  # nan: no switch in the run
            rows[key] = {"contrast_mean_over_seeds": mean_cs, "ci_seed_bootstrap": [lo, hi],
                         "per_seed": {s: {k: v[k] for k in ("contrast", "pooled", "n_switches", "mixing_time",
                                                             "mixing_time_censored", "mixing_time_raw", "M_max_step",
                                                             "bias_bound", "bias_bound_theorem_M", "blocks",
                                                             "switches")}
                                      for s, v in per_seed.items()}}
        refusals = float(sum(np.sum(t["refusals"]) for t in runs.values()))
        out[c] = {"lever": lever, "outcomes": rows, "refusals": refusals,
                  "calls_final": {s: t["calls"][-1] for s, t in runs.items()}}
    return _r({
        "scope": ("S1 stub pilot: pipeline check and wall-clock sizing only (stub backend, zero model calls); the "
                  "contrasts are those of perturbed anchor rule lists, not evidence about any lever."),
        "period": period, "burn_in": burn_in, "levers": out, "wall_clock": {"s1": _wall(trajs), "pilot": timing or {}},
        "bias_bound_note": ("bias_bound = 4 M lam (1 + t0), M = max per-generation change, lam = 1/period, t0 = "
                            "autocorrelation mixing time of the outcome's residual on (1, lever, generation). Same form as Wager Thm 15.5, "
                            "whose assumptions (memoryless random switching, HT estimator, M = sup |Y|, TV mixing) "
                            "do not hold here: a heuristic scale, not a guarantee."),
    })


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n")


def summary_text(g1: dict[str, Any], s1: dict[str, Any]) -> str:
    lines = ["G1 (stub) at generation %s" % g1.get("generations")]
    lines.append(f"{'condition':<10} {'mean':>6} {'best':>6} {'between':>8} {'within':>7} {'calls':>7} "
                 f"{'->17 (seeds, calls)':>20} {'->20 (seeds, calls)':>20} refusals")
    for c, v in g1["conditions"].items():
        f = v["final"]
        ct = v["calls_to_threshold"]

        def th(k):
            x = ct[k]
            return f"{x['seeds_reached']}/{x['seeds']}, {x['calls_median_if_reached']}"

        lines.append(f"{c:<10} {f['population_mean']['iqm']!s:>6} {f['population_best']['iqm']!s:>6} "
                     f"{f['between_group']['iqm']!s:>8} {f['within_group']['iqm']!s:>7} {f['calls']['iqm']!s:>7} "
                     f"{th('17.0'):>20} {th('20.0'):>20} {f['refusals']['max']}")
    for ct in g1["contrasts"]:
        lines.append(f"  {ct['a']} - {ct['b']} ({ct['key']}, gen {ct['generation']}): IQM {ct['iqm']} "
                     f"[{ct['ci'][0]}, {ct['ci'][1]}] n={ct['n']}")
    for c, v in g1.get("null_band", {}).items():
        lines.append(f"  null band {c} (K={v['k']}): final mean [{v['population_mean_final'][0]}, "
                     f"{v['population_mean_final'][1]}], real median inside: {v['real_median_final_inside']}")
    w = g1["wall_clock"]["g1"]
    lines.append(f"  wall: {w['seconds_per_generation_mean']} s/generation, {w['seconds_per_run_mean']} s/run (mean)")
    lines.append("S1 (stub): on-minus-off contrast of population mean self-play, mean over seeds")
    for c, v in s1["levers"].items():
        r = v["outcomes"]["population_mean"]
        b = [x["bias_bound"] for x in r["per_seed"].values()]
        t0 = [x["mixing_time"] for x in r["per_seed"].values()]
        lines.append(f"  {c:<12} {r['contrast_mean_over_seeds']!s:>7} [{r['ci_seed_bootstrap'][0]}, "
                     f"{r['ci_seed_bootstrap'][1]}]  t0 {t0}  bias bound {b}")
    w = s1["wall_clock"]["s1"]
    lines.append(f"  wall: {w['seconds_per_generation_mean']} s/generation, {w['seconds_per_run_mean']} s/run (mean)")
    return "\n".join(lines)
