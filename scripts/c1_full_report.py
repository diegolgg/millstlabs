"""Analyze the full C1 run (Overnight 2, step 3) per files/prereg/C1-credit-estimators.md (sections 7, 8, 10, 12).

Inputs: runs/c1-full (MLX, verified R = 29, unverified R = 8, the pilot's 5 seeds first) and runs/c1-null (null stub,
same design, K = 20). Outputs: docs/results/c1-full.json, docs/results/c1-full.png (estimate versus oracle per
estimator, identity line, null band), docs/results/c1-full-table.md.

Intervals: IQM over seeds with a 95% bootstrap over seeds (2,000 resamples, seed 0), each verification stratum analysed
separately (the "stratified" bootstrap of section 7: seeds are resampled within a stratum, never pooled across strata).
The pooled ridge estimator is fitted once on all seeds of a stratum, as in the pilot; a sensitivity interval refits it
inside every bootstrap resample. Decision rules (section 8) are applied literally to every estimator in each stratum;
acceptance also needs the Gemma replication block (section 6), which has not been run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.analysis import credit as C  # noqa: E402
from culture.evaluate import stats as S  # noqa: E402
from culture.run.single_student import load_spec  # noqa: E402

CONFIRMATORY = {"leave_one_out": "(i) leave-one-out", "equal_split": "(ii) paired_delta v0 (equal split)",
                "ridge_bernoulli": "(iii) ridge on Bernoulli(1/2) delivery, pooled"}
EXPLORATORY = {"tau_itt": "exhaustive ITT (8 replays: the oracle's own cost)",
               "plackett_burman": "Plackett-Burman 8-run (= full factorial at k = 3, identical to tau_itt)",
               "singles_pairs": "(iv) singles plus pairs (pre-registered for Arm 2, k = 7; shown at k = 3)"}
TEACHER_LABEL = {"piers": "T1 Piers", "flawed_persuasive": "T2 Flawed (persuasive)", "self": "T3 re-sent own bot"}


def load(d: Path):
    spec = load_spec(json.loads((d / "spec.json").read_text()))
    rows = [json.loads(x) for x in (d / "replays.jsonl").read_text().splitlines() if x.strip()]
    return spec, rows


def agg(x) -> dict:
    x = [float(a) for a in x if a is not None and np.isfinite(a)]
    d = C._iqm_ci(x)
    d["mean"] = float(np.mean(x)) if x else None
    d["sd"] = float(np.std(x, ddof=1)) if len(x) > 1 else None
    return d


def t_summary(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    half = st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "t_interval_95": [m - half, m + half],
            "t": float(m / (sd / np.sqrt(n))) if sd > 0 else None,
            "p_two_sided": float(st.ttest_1samp(x, 0.0).pvalue) if sd > 0 else None}


def refit_bootstrap(rows_s: list[dict], spec: dict, si: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """Primary contrast with the pooled ridge refitted inside each bootstrap resample of seeds (sensitivity)."""
    k = spec["k"]
    v: dict[int, dict[int, float]] = {}
    for x in rows_s:
        v.setdefault(x["replicate"], {})[x["mask"]] = x["y"]
    reps = sorted(r for r in v if len(v[r]) == 1 << k)
    bern = {}
    for r in reps:
        rng = np.random.default_rng([int(spec.get("ridge_seed", 0)), si, r])
        bern[r] = [(m, v[r][m]) for m in C.bernoulli_masks(k, int(spec.get("ridge_runs", 8)), rng)]
    phi = {r: C.shapley(v[r], k) for r in reps}
    loo = {r: C.rmse(C.leave_one_out(v[r], k), phi[r]) for r in reps}
    rng = np.random.default_rng(seed)
    means, iqms = [], []
    for _ in range(n_boot):
        samp = [reps[i] for i in rng.integers(0, len(reps), len(reps))]
        beta = C.ridge([row for r in samp for row in bern[r]], k, float(spec.get("ridge_lam", 1.0)))
        d = [C.rmse(beta, phi[r]) - loo[r] for r in samp]
        means.append(np.mean(d))
        iqms.append(S.iqm(d))
    return {"mean_interval_95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "iqm_interval_95": [float(np.quantile(iqms, 0.025)), float(np.quantile(iqms, 0.975))],
            "n_boot": n_boot}


def ridge_draws(rows_s: list[dict], spec: dict, si: int, sab: int, n_draws: int = 200) -> dict:
    """Sensitivity: the ridge estimator's metrics over n_draws different Bernoulli(1/2) 8-run subsets per seed (the
    pre-registered analysis uses one draw, rng seed (ridge_seed, stratum index, seed), as in the pilot report)."""
    k = spec["k"]
    v: dict[int, dict[int, float]] = {}
    for x in rows_s:
        v.setdefault(x["replicate"], {})[x["mask"]] = x["y"]
    reps = sorted(r for r in v if len(v[r]) == 1 << k)
    phi = {r: C.shapley(v[r], k) for r in reps}
    loo = {r: C.rmse(C.leave_one_out(v[r], k), phi[r]) for r in reps}
    rm, sp, sp_lo, se, con, ok = [], [], [], [], [], []
    for d in range(n_draws):
        rr = []
        for r in reps:
            rng = np.random.default_rng([d, si, r])
            rr += [(m, v[r][m]) for m in C.bernoulli_masks(k, int(spec.get("ridge_runs", 8)), rng)]
        beta = C.ridge(rr, k, float(spec.get("ridge_lam", 1.0)))
        e_rm = [C.rmse(beta, phi[r]) for r in reps]
        e_sp = [C.spearman(beta, phi[r]) for r in reps]
        e_se = [C.sign_error(float(beta[sab]), float(phi[r][sab])) for r in reps]
        e_se = [x for x in e_se if x is not None]
        a_rm, a_sp = C._iqm_ci(e_rm), C._iqm_ci(e_sp)
        rm.append(a_rm["iqm"])
        sp.append(a_sp["iqm"])
        sp_lo.append(a_sp["lo"])
        se.append(float(np.mean(e_se)) if e_se else None)
        con.append(float(np.mean([a - loo[r] for a, r in zip(e_rm, reps)])))
        ok.append(verdict({"spearman": a_sp, "rmse": a_rm,
                           "sign_error_rate": float(np.mean(e_se)) if e_se else None})["passes_this_block"])
    q = lambda x: [float(np.quantile(x, 0.05)), float(np.median(x)), float(np.quantile(x, 0.95))]  # noqa: E731
    return {"n_draws": n_draws, "rmse_iqm_q05_median_q95": q(rm), "spearman_iqm_q05_median_q95": q(sp),
            "spearman_lo_q05_median_q95": q([x if x is not None else np.nan for x in sp_lo]),
            "sign_error_rate_q05_median_q95": q([x for x in se if x is not None]) if any(x is not None for x in se)
            else None, "primary_contrast_mean_q05_median_q95": q(con),
            "fraction_of_draws_passing_all_rules": float(np.mean(ok)),
            "fraction_of_draws_rmse_le_2": float(np.mean([x <= 2.0 for x in rm]))}


def pooled_floor(reps: dict) -> dict:
    """The per-seed RMSE of the best seed-invariant estimator (the oracle's own mean over seeds): a floor for any
    pooled estimator such as ridge, set by between-seed heterogeneity of the oracle, not by estimation noise."""
    phi = np.array([r["shapley"] for r in reps.values()])
    m = phi.mean(0)
    r = [float(np.sqrt(np.mean((m - p) ** 2))) for p in phi]
    return {"definition": "per-seed RMSE of the oracle's seed-mean against each seed's oracle (IQM)",
            "rmse_iqm": S.iqm(r), "between_seed_sd_of_oracle": phi.std(0, ddof=1).tolist()}


def verdict(e: dict) -> dict:
    sp, rm, se = e["spearman"], e["rmse"], e["sign_error_rate"]
    checks = {
        "spearman_iqm_ge_0.9": sp["iqm"] is not None and sp["iqm"] >= 0.9,
        "spearman_lower_bound_ge_0.8": sp["lo"] is not None and sp["lo"] >= 0.8,
        "rmse_iqm_le_2.0": rm["iqm"] is not None and rm["iqm"] <= 2.0,
        "sign_error_T2_le_5pct": None if se is None else se <= 0.05,
    }
    ok = all(v is True for v in checks.values())
    return {"checks": checks, "passes_this_block": ok}


def null_summary(summ: dict, stratum: str) -> dict:
    s = summ["strata"][stratum]
    out = {}
    for n, e in s["estimators"].items():
        rm = [s["per_replicate"][r]["metrics"][n]["rmse"] for r in s["per_replicate"]]
        sp = [s["per_replicate"][r]["metrics"][n]["spearman"] for r in s["per_replicate"]]
        res = [a - b for r in s["per_replicate"].values() for a, b in zip(r["estimates"][n], r["shapley"])]
        out[n] = {"rmse": {**e["rmse"], "min": float(min(rm)), "max": float(max(rm))},
                  "spearman": {**e["spearman"], "min": None if all(x is None for x in sp) else
                               float(min(x for x in sp if x is not None))},
                  "sign_error_rate": e["sign_error_rate"], "sign_error_defined_in": e["sign_error_defined_in"],
                  "residual_envelope": [float(min(res)), float(max(res))]}
    return out


def effects(s: dict, teachers: list[str]) -> dict:
    reps = s["per_replicate"]
    out = {}
    for j, t in enumerate(teachers):
        ad = [r["adopter_effect"][j] for r in reps.values()]
        out[TEACHER_LABEL[t]] = {
            "tau_itt": agg([r["tau"][j] for r in reps.values()]),
            "shapley": agg([r["shapley"][j] for r in reps.values()]),
            "adopter_effect": agg([a for a in ad if a is not None]) if any(a is not None for a in ad) else
            {"iqm": None, "note": "undefined: never adopted in any seed"},
            "adopter_effect_defined_in": sum(a is not None for a in ad),
            "adoption_rate_when_delivered": agg([r["adopt_rate"][j] for r in reps.values()])["mean"]}
    return out


def exploratory(s: dict, teachers: list[str]) -> dict:
    k = len(teachers)
    pi, fl, me = teachers.index("piers"), teachers.index("flawed_persuasive"), teachers.index("self")
    inter, text_vs_placebo = [], []
    for r in s["per_replicate"].values():
        v = {int(m): y for m, y in r["v"].items()}
        a, b = 1 << pi, 1 << fl
        inter.append(float(np.mean([v[d | a | b] - v[d | a] - v[d | b] + v[d]
                                    for d in range(1 << k) if not d & (a | b)])))
        text_vs_placebo.append(r["tau"][fl] - r["tau"][me])
    return {"interaction_piers_x_flawed": {**agg(inter), "t": t_summary(inter) if len(inter) > 2 else None,
                                           "definition": "mean over deliveries D of the re-sent bot of "
                                                         "v(D+P+F) - v(D+P) - v(D+F) + v(D)"},
            "flawed_minus_placebo_tau": {**agg(text_vs_placebo),
                                         "t": t_summary(text_vs_placebo) if len(text_vs_placebo) > 2 else None,
                                         "definition": "tau_flawed - tau_self per seed (is Flawed's effect larger "
                                                       "than a generic text perturbation?)"},
            "abs_placebo_tau": agg([abs(r["tau"][me]) for r in s["per_replicate"].values()])}


def operations(rows: list[dict], stratum: str, pilot_seeds: set[int]) -> dict:
    out = {}
    for label, sel in (("all", lambda x: True), ("tonight", lambda x: x["replicate"] not in pilot_seeds),
                       ("pilot", lambda x: x["replicate"] in pilot_seeds)):
        sr = [x for x in rows if x["stratum"] == stratum and sel(x)]
        calls = [c for x in sr for c in x.get("calls", [])]
        fresh = [c for c in calls if not c["cached"]]
        lat = [c["latency_s"] for c in fresh if c.get("latency_s") is not None]
        out[label] = {"replays": len(sr), "llm_calls": len(calls), "backend_calls": len(fresh),
                      "cache_hits": len(calls) - len(fresh),
                      "repair_calls": sum(c["tag"] == "repair" for c in fresh),
                      "backend_seconds": round(float(sum(lat)), 1),
                      "replay_wall_seconds": round(float(sum(x.get("wall_seconds", 0) for x in sr)), 1),
                      "tokens_per_backend_call": float(np.mean([c["input_tokens"] + c["cache_read_input_tokens"]
                                                                + c["output_tokens"] for c in fresh])) if fresh else None}
    return out


def figure(summ: dict, null: dict | None, teachers: list[str], names: list[str]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from culture.analysis.figures import GRID, MUTED, SERIES, SURFACE, TEXT

    strata = list(summ["strata"])
    fig, axes = plt.subplots(len(strata), len(names), figsize=(2.75 * len(names), 2.9 * len(strata) + 0.6),
                             squeeze=False)
    markers = ["o", "s", "^"]
    for i, stratum in enumerate(strata):
        s = summ["strata"][stratum]
        reps = s["per_replicate"]
        allv = [x for r in reps.values() for n in names for x in r["estimates"][n] + r["shapley"]]
        lo, hi = min(allv + [0]) - 1.5, max(allv + [0]) + 1.5
        for c, n in enumerate(names):
            ax = axes[i][c]
            ax.set_facecolor(SURFACE)
            if null is not None and stratum in null and n in null[stratum]:
                a, b = null[stratum][n]["residual_envelope"]
                ax.fill_between([lo, hi], [lo + a, hi + a], [lo + b, hi + b], color=MUTED, alpha=0.13, linewidth=0,
                                label="null-stub envelope (K = 20)")
            ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (3, 3)), label="identity")
            for j, t in enumerate(teachers):
                xs = [r["shapley"][j] for r in reps.values()]
                ys = [r["estimates"][n][j] for r in reps.values()]
                ax.scatter(xs, ys, s=24, marker=markers[j], color=SERIES[j], edgecolors=SURFACE, linewidths=0.8,
                           label=TEACHER_LABEL[t], zorder=3)
            e = s["estimators"][n]
            ax.text(0.03, 0.97, f"RMSE {e['rmse']['iqm']:.2f}\nSpearman {e['spearman']['iqm']:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=7, color=TEXT)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(True, color=GRID, linewidth=0.5)
            ax.tick_params(colors=MUTED, labelsize=7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(GRID)
            if i == 0:
                ax.set_title(n.replace("_", " "), fontsize=9, color=TEXT, loc="left")
            if c == 0:
                ax.set_ylabel(f"{stratum} (R = {len(reps)})\nestimate (points)", fontsize=8, color=MUTED)
            if i == len(strata) - 1:
                ax.set_xlabel("Shapley oracle (points)", fontsize=8, color=MUTED)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=len(lab), frameon=False, fontsize=8, labelcolor=TEXT)
    fig.suptitle("C1: teacher credit estimates vs exact Shapley oracle (Qwen3.6-35B-A3B on MLX, one point per seed "
                 "and teacher)", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    return fig


def table(report: dict) -> str:
    lines = [report["scope"], "",
             "| stratum | estimator | role | replays/seed | RMSE IQM [95% CI] | Spearman IQM [95% CI] | sign error T2 "
             "| null RMSE IQM [min, max] | passes this block |", "|---|---|---|---|---|---|---|---|---|"]

    def f(d):
        return "n/a" if d["iqm"] is None else f"{d['iqm']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]"

    for stratum, s in report["strata"].items():
        for n in list(CONFIRMATORY) + list(EXPLORATORY):
            e = s["estimators"][n]
            nr = (s.get("null") or {}).get(n)
            nul = "n/a" if not nr else f"{nr['rmse']['iqm']:.2f} [{nr['rmse']['min']:.2f}, {nr['rmse']['max']:.2f}]"
            se = "n/a" if e["sign_error_rate"] is None else f"{e['sign_error_rate']:.2f} (of {e['sign_error_defined_in']})"
            role = "confirmatory" if n in CONFIRMATORY else "exploratory"
            lines.append(f"| {stratum} (R={s['R']}) | {n} | {role} | {e['replays_per_seed']} | {f(e['rmse'])} | "
                         f"{f(e['spearman'])} | {se} | {nul} | {'yes' if e['decision']['passes_this_block'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "c1-full"))
    ap.add_argument("--null", default=str(ROOT / "runs" / "c1-null"))
    ap.add_argument("--outdir", default=str(ROOT / "docs" / "results"))
    a = ap.parse_args()
    out = Path(a.outdir)
    spec, rows = load(Path(a.run))
    teachers = spec["teachers"]
    summ = C.summarize(rows, spec)
    null_summ = None
    if (Path(a.null) / "replays.jsonl").exists():
        nspec, nrows = load(Path(a.null))
        nspec = {**nspec, "strata": {k: v for k, v in nspec["strata"].items() if k in spec["strata"]}}
        null_summ = C.summarize(nrows, nspec)
    null = {st_: null_summary(null_summ, st_) for st_ in null_summ["strata"]} if null_summ else None
    pilot = {0, 1, 2, 3, 4}
    report: dict = {
        "scope": "Scope: C1 Arm 1 only (k = 3), one model block (Qwen3.6-35B-A3B 4-bit on MLX, seeded, temperature "
                 "0); the Gemma replication block and Arm 2 (k = 7) were not run, so no estimator can be accepted, "
                 "only rejected or passed in this block. One student, one generation, 300 held-out deals; R = "
                 + ", ".join(f"{s_} {len(summ['strata'][s_]['complete_replicates'])}" for s_ in summ["strata"])
                 + " seeds (the pilot's 5 included). Null band from the null stub (K = 20), not from a model.",
        "design": "k = 3 (Piers; Flawed with persuasive prose and honest evidence; the student's own bot re-sent), "
                  "exhaustive delivery (8 replays per seed per stratum), strata verification on (selfplay n = 200, "
                  "replace_if_better) and off; outcome = held-out paired self-play improvement on 300 deals; oracle = "
                  "exact Shapley over the 8 replays; ridge uses a Bernoulli(1/2) 8-run subset of each seed's replays, "
                  "pooled over the stratum's seeds.",
        "teachers": teachers, "confirmatory_estimators": CONFIRMATORY, "exploratory_estimators": EXPLORATORY,
        "strata": {}}
    names_fig = list(CONFIRMATORY) + list(EXPLORATORY)
    order = [x for x in ("verified", "unverified") if x in summ["strata"]] + \
        [x for x in summ["strata"] if x not in ("verified", "unverified")]
    sab = teachers.index("flawed_persuasive")
    for stratum in order:
        s = summ["strata"][stratum]
        si = list(spec["strata"]).index(stratum)  # the index summarize used for the ridge draw (sorted spec.json)
        est = {}
        for n, e in s["estimators"].items():
            est[n] = {**e, "decision": verdict(e)}
        pc = s["primary_contrast"]
        x = pc["per_replicate"]
        primary = {"definition": pc["definition"], "per_replicate": x, **t_summary(x), "iqm": agg(x),
                   "ridge_refit_bootstrap": refit_bootstrap([r for r in rows if r["stratum"] == stratum], spec, si),
                   "mde": 1.0, "R_for_80pct_power_at_1pt_from_this_run": C.power_r(float(np.std(x, ddof=1)))}
        primary["detects_difference_at_alpha_0.05"] = bool(primary["p_two_sided"] is not None
                                                           and primary["p_two_sided"] < 0.05)
        reps = s["per_replicate"]
        report["strata"][stratum] = {
            "R": len(s["complete_replicates"]), "seeds": s["complete_replicates"],
            "estimators": est, "primary_contrast": primary, "effects_per_teacher": effects(s, teachers),
            "exploratory": exploratory(s, teachers),
            "per_seed": {r: {"shapley": [round(v, 3) for v in p["shapley"]], "tau_itt": [round(v, 3) for v in p["tau"]],
                             "adopter_effect": [None if v is None else round(v, 3) for v in p["adopter_effect"]],
                             "v": p["v"]} for r, p in reps.items()},
            "pooled_estimator_floor": pooled_floor(reps),
            "ridge_pooled": s["ridge_pooled"], "ridge_draw_stratum_index": si,
            "ridge_draw_sensitivity": ridge_draws([r for r in rows if r["stratum"] == stratum], spec, si, sab),
            "operations": {**s["operations"], "by_source": operations(rows, stratum, pilot)},
            "null": null.get(stratum) if null else None,
        }
    report["verdicts"] = {}
    for n in names_fig:
        per = {st_: report["strata"][st_]["estimators"][n]["decision"]["passes_this_block"] for st_ in report["strata"]}
        role = "confirmatory" if n in CONFIRMATORY else "exploratory"
        if all(per.values()):
            v = "passes the Qwen block in both strata; acceptance requires the Gemma block (not run)"
        elif any(per.values()):
            v = ("passes in " + ", ".join(k for k, ok in per.items() if ok) + " only; fails in "
                 + ", ".join(k for k, ok in per.items() if not ok) + ": not accepted")
        else:
            v = "fails in both strata: not accepted"
        report["verdicts"][n] = {"role": role, "passes_by_stratum": per, "verdict": v}
    passing = [n for n in CONFIRMATORY if all(report["verdicts"][n]["passes_by_stratum"].values())]
    report["overall"] = ("no confirmatory estimator passes" if not passing else
                         "confirmatory estimators passing this block: " + ", ".join(passing))
    report["section_8_reading"] = (
        "Applied literally: no confirmatory estimator (i, ii, iii) passes in the Qwen block, so none can be accepted. "
        "Section 8's fallback ('if no estimator passes at R = 60, causal credit is not available at this cost') is "
        "conditioned on R = 60; this run used the power-rule R (29 verified, 8 unverified), so the fallback is not "
        "formally triggered. Evidence that more seeds would not change the verdicts: leave-one-out and equal split "
        "do not pool, so their per-seed RMSE does not shrink with R; ridge pools, but its per-seed RMSE has a floor "
        "set by between-seed heterogeneity of the oracle (pooled_estimator_floor: the oracle's own seed-mean scores "
        "2.19 verified and 3.20 unverified), and 0 of 200 alternative Bernoulli draws pass (ridge_draw_sensitivity). "
        "The exploratory exhaustive ITT estimator (8 replays per seed, the oracle's own cost at k = 3) passes in both "
        "strata of this block.")
    report["multiplicity"] = (
        "Section 10 counts nine confirmatory tests (three statistics x three estimators) for BH at q = 0.1, but the "
        "section 8 rules are thresholds on IQMs and bootstrap bounds with no pre-specified test statistic, so no "
        "p-values exist for them and BH was not applied. The one pre-specified test is the primary contrast (paired "
        "two-sided t-test, alpha 0.05): p values per stratum are in strata.*.primary_contrast.")
    report["notes"] = [
        "The pilot's 5 seeds are the first 5 of each stratum: their 80 replay rows were copied in after a strict "
        "cache replay with the current code reproduced all 80 bit-identically (y, adoption, acceptance, request keys).",
        "The ridge's Bernoulli(1/2) 8-run subset per seed is drawn with rng seed (ridge_seed 0, stratum index, seed); "
        "the stratum index follows the sorted spec.json (unverified 0, verified 1), as in the pilot report "
        "(docs/results/c1-pilot.json). ridge_draw_sensitivity shows the verdict does not depend on the draw.",
        "The machine idle-slept 03:50-03:56 and 03:56-04:12 EDT (display off) while a replay was in flight; the "
        "request completed after wake, and latencies (perf_counter) exclude the sleep. No effect on outputs; about "
        "22 minutes of wall clock lost. caffeinate was started for the rest of the night."]
    (out / "c1-full.json").write_text(json.dumps(report, indent=1) + "\n")
    figure({**summ, "strata": {x: summ["strata"][x] for x in order}}, null, teachers, names_fig).savefig(
        out / "c1-full.png", dpi=150)
    (out / "c1-full-table.md").write_text(table(report))
    print(table(report))
    print(json.dumps(report["verdicts"], indent=1))
    for st_, s in report["strata"].items():
        p = s["primary_contrast"]
        print(st_, "primary", round(p["mean"], 3), [round(v, 3) for v in p["t_interval_95"]], p["p_two_sided"],
              p["ridge_refit_bootstrap"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
