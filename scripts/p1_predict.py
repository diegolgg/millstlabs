"""P1 draft prediction of G1 from the single-student micro-parameters (zero model calls).

Reads docs/results/p1-micro-params.json (scripts/p1_params.py), runs the accumulation model
(src/culture/analysis/accumulation_model.py) for the three G1 organizations, N = 8, 100 generations, 1,000 Monte Carlo
seeds, and writes docs/results/p1-prediction.json and docs/results/p1-prediction.png.

Usage: .venv/bin/python scripts/p1_predict.py [--runs 1000] [--generations 100]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.analysis.accumulation_model import (G1_ORGANIZATIONS, Fidelity, Innovation, Organization,  # noqa: E402
                                                 Verification, predicted_order, simulate, summarize)

RES = ROOT / "docs" / "results"
PARAMS = RES / "p1-micro-params.json"
OUT = RES / "p1-prediction.json"
FIG = RES / "p1-prediction.png"
N = 8
SEEDS_G1 = 5
MEDIUM = "both"  # the harness sends bot.py and conventions.md
MARGIN = "0.5"
COLORS = {"isolated": "#2a78d6", "full": "#eb6834", "organized": "#1baf7a"}


def build(P: dict, *, medium=MEDIUM, p_s=None, eps_scale=None, eps_override=None, margin=MARGIN, verif="current",
          fid_mode="mixture", beyond="flat", loss_floor=None):
    f = P["fidelity"][medium]
    if fid_mode == "mixture":
        fid = Fidelity(p_s=f["p_s"]["raw"] if p_s is None else p_s, mu_loss=f["loss_in_success_cluster"]["mean"],
                       sd_loss=f["loss_in_success_cluster"]["sd"])
    elif fid_mode == "em":
        up = f["em_mixture_sensitivity"]["upper_component"]
        fid = Fidelity(p_s=up["weight"], mu_loss=f["v_T"] - up["mean"], sd_loss=up["sd"])
    elif fid_mode == "empirical":
        fid = Fidelity(mode="empirical", losses=f["all_losses"])
    else:
        fid = Fidelity(mode="exact")
    fid.loss_floor = loss_floor

    V = P["verification"]
    if verif == "current":
        v = V["current_rule_selfplay_n200_mean_gt_0"]
        ver = Verification(v["false_adoption"], v["miss_near_delta1"])
    elif verif == "sprt":
        v = V["sprt"]
        ver = Verification(v["false_adoption"], v["miss_near_delta1"])
    elif verif == "current_oc":
        c = V["current_rule_selfplay_n200_mean_gt_0"]["oc_curve"]
        ver = Verification(oc_delta=[x["delta_mid"] for x in c], oc_p=[x["p_adopt"] for x in c])
    else:
        ver = Verification(0.0, 0.0)

    I = P["innovation"]
    levels = [I["weak"]["level"]["mean"], I["piers"]["level"]["mean"]]
    eps = []
    gains = []
    for lv in ("weak", "piers"):
        e = I[lv]["epsilon"][margin]
        if eps_override is not None:
            eps.append(e[eps_override])
        else:
            eps.append(e["raw"] * (eps_scale or 1.0))
        g = I[lv]["gain_given_improvement"][margin]
        gains.append(g.get("values") or [0.0])
    inn = Innovation(levels=levels, eps=eps, gains=gains, beyond=beyond)
    return fid, ver, inn, levels


def run_all(P, init, runs, gens, seed=0, orgs=G1_ORGANIZATIONS, **kw):
    fid, ver, inn, _ = build(P, **kw)
    res = {o.name: simulate(o, init, fid, ver, inn, generations=gens, runs=runs, seed=seed) for o in orgs}
    return res, (fid, ver, inn)


def short(res):
    po = predicted_order(res, SEEDS_G1)
    p30 = predicted_order(res, SEEDS_G1, generation=30)
    return {"mean_at_30": {k: round(float(r.mean[:, 30].mean()), 2) for k, r in res.items()},
            "order_at_30": p30["order"], f"p_order_at_30_mean_of_{SEEDS_G1}_seeds":
                round(p30[f"p_order_mean_of_{SEEDS_G1}_seeds"], 3),
            "final_mean": {k: round(float(r.mean[:, -1].mean()), 2) for k, r in res.items()},
            "final_one_seed_interval": {k: [round(float(np.quantile(r.mean[:, -1], q)), 2) for q in (0.025, 0.975)]
                                        for k, r in res.items()},
            "order": po["order"], "p_order_one_seed": round(po["p_order_one_seed"], 3),
            f"p_order_mean_of_{SEEDS_G1}_seeds": round(po[f"p_order_mean_of_{SEEDS_G1}_seeds"], 3),
            "gen_to_17_full": first_gen(res["full"].mean.mean(0), 17.0),
            "gen_to_17_isolated": first_gen(res["isolated"].mean.mean(0), 17.0)}


def gap(r, base, k=SEEDS_G1):
    """Paired (common random numbers) difference in population mean against isolated, per generation."""
    d = r.mean - base.mean
    R = d.shape[0]
    blk = d[: (R // k) * k].reshape(-1, k, d.shape[1]).mean(1)
    e = d.mean(0)
    return {"expected": e.tolist(), "one_seed_lo": np.quantile(d, 0.025, 0).tolist(),
            "one_seed_hi": np.quantile(d, 0.975, 0).tolist(),
            f"mean_of_{k}_seeds_lo": np.quantile(blk, 0.025, 0).tolist(),
            f"mean_of_{k}_seeds_hi": np.quantile(blk, 0.975, 0).tolist(),
            "peak": float(e.max()), "peak_generation": int(e.argmax()),
            "at_100": float(e[-1]), f"p_ge_2_at_100_mean_of_{k}_seeds": float((blk[:, -1] >= 2).mean()),
            "last_generation_ge_2": (int(np.nonzero(e >= 2)[0][-1]) if (e >= 2).any() else None)}


def first_gen(traj, t):
    idx = np.nonzero(np.asarray(traj) >= t)[0]
    return int(idx[0]) if idx.size else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--generations", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    P = json.loads(PARAMS.read_text())

    weak_level = P["innovation"]["weak"]["level"]["mean"]
    init = np.full(N, weak_level)
    res, (fid, ver, inn) = run_all(P, init, a.runs, a.generations, a.seed)
    checkpoints = [g for g in (10, 30) if g < a.generations]
    summaries = {k: summarize(r, SEEDS_G1, checkpoints=checkpoints) for k, r in res.items()}
    order = predicted_order(res, SEEDS_G1)
    orders_at = {str(g): predicted_order(res, SEEDS_G1, generation=g) for g in checkpoints}
    gaps = {k: gap(res[k], res["isolated"]) for k in ("full", "organized")}

    f = P["fidelity"][MEDIUM]
    I = P["innovation"]
    sens_rows = []

    def add(label, what, **kw):
        r, _ = run_all(P, kw.pop("init", init), a.runs, a.generations, a.seed, **kw)
        g_o = gap(r["organized"], r["isolated"])
        sens_rows.append({"row": label, "changed": what, **short(r),
                          "organized_minus_isolated": {"peak": round(g_o["peak"], 2),
                                                       "peak_generation": g_o["peak_generation"],
                                                       "at_100": round(g_o["at_100"], 2),
                                                       "last_generation_ge_2": g_o["last_generation_ge_2"]}})

    add("baseline", "none")
    sens_gap = {}
    add("p_s low", f"p_s = {f['p_s']['lo']:.3f} (95% lower bound)", p_s=f["p_s"]["lo"])
    add("p_s high", f"p_s = {f['p_s']['hi']:.3f} (95% upper bound)", p_s=f["p_s"]["hi"])
    add("epsilon low", "epsilon at both anchors at the 95% lower bound "
        f"({I['weak']['epsilon'][MARGIN]['lo']:.3f}, {I['piers']['epsilon'][MARGIN]['lo']:.3f})", eps_override="lo")
    add("epsilon high", "epsilon at both anchors at the 95% upper bound "
        f"({I['weak']['epsilon'][MARGIN]['hi']:.3f}, {I['piers']['epsilon'][MARGIN]['hi']:.3f})", eps_override="hi")
    add("margin 1.0", "innovation margin 1.0 (no Piers-level improvement observed: epsilon = 0 at 17, gain sample empty)",
        margin="1.0")
    add("epsilon falls to 0 at cap", "beyond Piers, epsilon falls linearly to 0 at 25 instead of staying flat",
        beyond="linear_to_cap")
    add("copies never exceed teacher", "copy loss floored at 0: switches off the copy-noise ratchet", loss_floor=0.0)
    add("conservative", "epsilon falls to 0 at the cap and copies never exceed the teacher",
        beyond="linear_to_cap", loss_floor=0.0)
    add("copy exact (harness)", "copying is replace_if_better adoption of the sender's artifact: p_s = 1, loss 0",
        fid_mode="exact")
    add("copy EM mixture", "fidelity from the EM upper component (weight, v_T - mean, sd)", fid_mode="em")
    add("copy empirical losses", "copy loss resampled from all 30 A1 losses (failed copies included, then verified)",
        fid_mode="empirical")
    add("medium code", "fidelity of medium code", medium="code")
    add("medium prose", "fidelity of medium prose", medium="prose")
    add("verification SPRT", "SPRT rates (false adoption 0.003, miss 0.10)", verif="sprt")
    add("verification OC curve", "current rule as the B1 P(adopt | delta) curve instead of two rates", verif="current_oc")
    add("verification perfect", "no false adoptions, no misses", verif="perfect")
    add("initial 3.17", "all agents start at 3.17 (single_student.py calibration deals)", init=np.full(N, 3.17))
    orgs_copy = (G1_ORGANIZATIONS[0], G1_ORGANIZATIONS[1],
                 Organization("organized", "organized", migration_rate=0.25, migration_mode="copy"))
    add("migration as copying", "organized: each agent sees the other group's best with probability 0.25 per "
        "generation, no swaps", orgs=orgs_copy)

    eps_info = {"anchor_levels": list(map(float, inn.levels)), "epsilon_at_anchors": list(map(float, inn.eps)),
                "beyond": inn.beyond, "gain_samples": [list(map(float, q)) for q in inn._q]}
    out = {
        "scope": "DRAFT prediction, not pre-registered until Enrico signs off. Built from one model block (Qwen3.6-35B-"
                 "A3B 4-bit, local MLX, temperature 0) and single-student data only (A1 copies, C1 replays, B1 engine "
                 "tests); G1 is planned on a different, hosted model. Zero model calls. The generation-100 levels "
                 "depend on an unmeasured extrapolation of the innovation rate above the Piers level (see "
                 "p1-micro-params.json 'missing').",
        "generated_by": "scripts/p1_predict.py; model src/culture/analysis/accumulation_model.py",
        "inputs": {"params": str(PARAMS.relative_to(ROOT)), "medium": MEDIUM, "innovation_margin": float(MARGIN),
                   "fidelity": {"mode": fid.mode, "p_s": fid.p_s, "mu_loss": fid.mu_loss, "sd_loss": fid.sd_loss},
                   "verification": {"rule": "selfplay n = 200, adopt if paired mean > 0 (current harness rule)",
                                    "false_adoption": ver.false_adoption, "miss": ver.miss},
                   "innovation": eps_info},
        "design": {"N": N, "generations": a.generations, "monte_carlo_runs": a.runs, "seed": a.seed,
                   "seeds_per_G1_condition": SEEDS_G1,
                   "organizations": {o.name: {"kind": o.kind, "groups": o.groups if o.kind == "organized" else 1,
                                              "migration": (f"swap, rate {o.migration_rate}, every "
                                                            f"{o.migration_interval} generations")
                                              if o.kind == "organized" else "none"} for o in G1_ORGANIZATIONS},
                   "initial_skills": {"value": float(weak_level), "source": "no authored G1 warm start exists on disk; "
                                      "all 8 agents start at the C1 student's mean held-out self-play (29 seeds, "
                                      "runs/c1-full/replays.jsonl s0_score), the level at which the weak-level "
                                      "innovation rate was measured (3.17 on single_student.py's calibration deals; "
                                      "sensitivity row 'initial 3.17')"},
                   "common_random_numbers": "organizations share innovation draws per Monte Carlo seed"},
        "prediction": {
            "generation_final_mean": {k: s["final_mean"] for k, s in summaries.items()},
            "generation_final_best": {k: s["final_best"] for k, s in summaries.items()},
            "order": order,
            "mean_at_checkpoints": {k: s["mean_at"] for k, s in summaries.items()},
            "order_at_checkpoints": orders_at,
            "paired_gap_vs_isolated": {k: {kk: v for kk, v in g.items() if not isinstance(v, list)}
                                       for k, g in gaps.items()},
            "reading": None,
            "sign_vs_isolated": {k: ("+" if summaries[k]["final_mean"]["expected"]
                                     > summaries["isolated"]["final_mean"]["expected"] + 1e-9 else
                                     ("0" if abs(summaries[k]["final_mean"]["expected"]
                                                 - summaries["isolated"]["final_mean"]["expected"]) < 1e-9 else "-"))
                                 for k in summaries if k != "isolated"},
            "generations_to_mean_17": {k: first_gen(s["mean_trajectory"]["expected"], 17.0)
                                       for k, s in summaries.items()},
            "generations_to_mean_20": {k: first_gen(s["mean_trajectory"]["expected"], 20.0)
                                       for k, s in summaries.items()},
            "adoptions_per_generation": {k: s["adoptions_per_generation"] for k, s in summaries.items()},
        },
        "trajectories": {k: {"mean": s["mean_trajectory"], "best": s["best_trajectory"]} for k, s in summaries.items()},
        "gap_trajectories_vs_isolated": gaps,
        "sensitivity": sens_rows,
        "interval_definitions": {
            "mc_se": "Monte Carlo standard error of the expected population mean",
            "one_seed_interval": "2.5% and 97.5% quantiles of one population's generation-final mean over runs",
            f"mean_of_{SEEDS_G1}_seeds_interval": f"same for the mean of {SEEDS_G1} populations (disjoint blocks of "
                                                  "runs), the quantity G1 reports",
            "trajectory lo/hi": "pointwise 2.5% and 97.5% quantiles over runs (one population)"},
        "seconds": None,
    }
    sat = all(s["final_mean"]["expected"] > 24.5 for s in summaries.values())
    out["prediction"]["reading"] = (
        "Under the baseline (epsilon held at its Piers-level value above 17, additive copy noise that can exceed the "
        "teacher) every organization reaches the cap of 25 well before generation 100, so the generation-100 means "
        "and order are ties at the ceiling and P1's generation-100 rule cannot discriminate. The informative "
        "predictions are the order and gaps at generations 10 and 30 and the generations to reach 17 and 20; the "
        "'conservative' sensitivity row shows what happens when the two unmeasured upward forces are switched off."
        if sat else "Generation-100 means are below the cap; see order and intervals.")
    out["seconds"] = round(time.perf_counter() - t0, 2)
    OUT.write_text(json.dumps(out, indent=1))
    plot(summaries, gaps, a.generations)

    print(f"P1 draft prediction (N = {N}, {a.generations} generations, {a.runs} Monte Carlo runs, "
          f"{out['seconds']} s)")
    print(f"{'organization':10} {'gen-100 mean':>12} {'one-seed 95%':>16} {'5-seed 95%':>16} {'best':>6} "
          f"{'gen to 17':>9}")
    for k, s in summaries.items():
        fm = s["final_mean"]
        i1, i5 = fm["one_seed_interval"], fm[f"mean_of_{SEEDS_G1}_seeds_interval"]
        print(f"{k:10} {fm['expected']:12.2f} [{i1[0]:5.2f}, {i1[1]:5.2f}] [{i5[0]:5.2f}, {i5[1]:5.2f}] "
              f"{s['final_best']['expected']:6.2f} {str(out['prediction']['generations_to_mean_17'][k]):>9}")
    print(f"order: {' > '.join(order['order'])}  (one seed {order['p_order_one_seed']:.2f}, "
          f"5-seed mean {order[f'p_order_mean_of_{SEEDS_G1}_seeds']:.2f})")
    for g in checkpoints:
        o = orders_at[str(g)]
        print(f"generation {g}: " + ", ".join(f"{k} {summaries[k]['mean_at'][str(g)]['expected']:.2f} "
                                             f"{summaries[k]['mean_at'][str(g)][f'mean_of_{SEEDS_G1}_seeds_interval']}"
                                             for k in summaries)
              + f"; order {' > '.join(o['order'])} (5-seed {o[f'p_order_mean_of_{SEEDS_G1}_seeds']:.2f})")
    print("generations to mean 20:", out["prediction"]["generations_to_mean_20"])
    for k, g in gaps.items():
        print(f"{k} - isolated: peak {g['peak']:.2f} at generation {g['peak_generation']}, at 100 {g['at_100']:.2f}, "
              f"last generation with gap >= 2: {g['last_generation_ge_2']}")
    print("\nSensitivity (mean at generation 30 and 100: isolated / full / organized; order at 30, 5-seed P(order))")
    for r in sens_rows:
        f30, fmv = r["mean_at_30"], r["final_mean"]
        print(f"{r['row']:28} g30 {f30['isolated']:6.2f} {f30['full']:6.2f} {f30['organized']:6.2f} | g100 "
              f"{fmv['isolated']:6.2f} {fmv['full']:6.2f} {fmv['organized']:6.2f} | "
              f"{' > '.join(r['order_at_30'])} p5={r[f'p_order_at_30_mean_of_{SEEDS_G1}_seeds']:.2f} | org-iso peak "
              f"{r['organized_minus_isolated']['peak']:.2f}@{r['organized_minus_isolated']['peak_generation']}, "
              f"at 100 {r['organized_minus_isolated']['at_100']:.2f}")
    print(f"\nWrote {OUT.relative_to(ROOT)} and {FIG.relative_to(ROOT)}")
    return 0


def plot(summaries, gaps, gens):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    x = np.arange(gens + 1)
    for k in ("isolated", "full", "organized"):
        s = summaries[k]["mean_trajectory"]
        c = COLORS[k]
        ax.fill_between(x, s["lo"], s["hi"], color=c, alpha=0.15, linewidth=0)
        ax.plot(x, s["expected"], color=c, linewidth=2, label=k)
    ax.axhline(17.0, color="#999999", linewidth=0.8, linestyle="--")
    ax.text(gens * 0.55, 16.0, "Piers level (17)", fontsize=8, color="#555555")
    ax.set_xlabel("generation")
    ax.set_ylabel("population mean self-play (points)")
    ax.set_ylim(0, 25.5)
    ax.set_xlim(0, gens)
    ax.set_title("Predicted mean skill (bands: 95% of one population)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    k5 = f"mean_of_{SEEDS_G1}_seeds"
    for k in ("full", "organized"):
        g = gaps[k]
        c = COLORS[k]
        bx.fill_between(x, g[f"{k5}_lo"], g[f"{k5}_hi"], color=c, alpha=0.15, linewidth=0)
        bx.plot(x, g["expected"], color=c, linewidth=2, label=f"{k} minus isolated")
    bx.axhline(2.0, color="#999999", linewidth=0.8, linestyle="--")
    bx.text(gens * 0.72, 2.2, "G1 minimum effect (2 points)", fontsize=8, color="#555555")
    bx.axhline(0.0, color="#cccccc", linewidth=0.8)
    bx.set_xlabel("generation")
    bx.set_ylabel("paired difference in mean self-play (points)")
    bx.set_xlim(0, gens)
    bx.set_title(f"Predicted contrast vs isolated (bands: 95% of a {SEEDS_G1}-seed mean)", fontsize=9)
    bx.legend(frameon=False, fontsize=8, loc="upper right")
    for a_ in (ax, bx):
        a_.grid(axis="y", color="#e5e5e5", linewidth=0.6)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    fig.text(0.01, 0.01, "P1 DRAFT, not pre-registered. One model block (local Qwen), single-student parameters; "
             "epsilon above 17 is extrapolated; skill capped at 25.", fontsize=7, color="#666666")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
