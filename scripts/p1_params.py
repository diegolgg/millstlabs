"""P1 micro-parameters from data already on disk (zero model calls).

Reads
  docs/results/a1-params.json   fidelity mixture per medium (30 copies each, teacher Piers)
  docs/results/c1-full.json     v(none) per seed: innovation at the weak-student level, feedback-only context
  runs/c1-full/replays.jsonl    per-replay incumbent and final scores: innovation at the Piers level (Piers delivered
                                and adopted; the gain is final minus the adopted Piers score). runs/ is gitignored; the
                                y column is cross-checked against c1-full.json.
  docs/results/b1-sprt.json     verification operating characteristics
Writes docs/results/p1-micro-params.json and prints a table.

Usage: .venv/bin/python scripts/p1_params.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "docs" / "results"
A1 = RES / "a1-params.json"
C1 = RES / "c1-full.json"
C1_ROWS = ROOT / "runs" / "c1-full" / "replays.jsonl"
B1 = RES / "b1-sprt.json"
OUT = RES / "p1-micro-params.json"

SUCCESS_TOL = 1.0
MARGINS = (0.5, 1.0)
QUANTS = (0.1, 0.25, 0.5, 0.75, 0.9)


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def beta_post(k: int, n: int) -> dict:
    a, b = 1 + k, 1 + n - k
    return {"k": int(k), "n": int(n), "raw": k / n if n else None, "posterior": f"Beta({a}, {b})",
            "mean": a / (a + b), "lo": float(stats.beta.ppf(0.025, a, b)), "hi": float(stats.beta.ppf(0.975, a, b))}


def describe(x) -> dict:
    x = np.asarray(sorted(x), float)
    if len(x) == 0:
        return {"n": 0, "note": "no observations"}
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "quantiles": {str(q): float(np.quantile(x, q)) for q in QUANTS}, "values": [round(float(v), 4) for v in x]}


def em_two_gaussians(x, n_starts: int = 20, iters: int = 500, sd_floor: float = 0.1, seed: int = 0) -> dict:
    """Two-component Gaussian mixture by EM, best of several starts; a variance floor stops the identical-score
    point mass (12 code copies at exactly 16.7) from collapsing a component to zero variance."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    best = None
    for s in range(n_starts):
        if s == 0:
            mu = np.array([np.quantile(x, 0.1), np.quantile(x, 0.8)])
        else:
            mu = np.sort(rng.choice(x, 2, replace=False)) + rng.normal(0, 0.1, 2)
        sd = np.array([x.std(), x.std()]) + sd_floor
        w = np.array([0.5, 0.5])
        ll_old = -np.inf
        for _ in range(iters):
            dens = w * stats.norm.pdf(x[:, None], mu, sd)
            tot = dens.sum(1, keepdims=True) + 1e-300
            r = dens / tot
            nk = r.sum(0) + 1e-12
            w = nk / len(x)
            mu = (r * x[:, None]).sum(0) / nk
            sd = np.maximum(np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / nk), sd_floor)
            ll = float(np.log(tot).sum())
            if abs(ll - ll_old) < 1e-9:
                break
            ll_old = ll
        if best is None or ll > best[0]:
            best = (ll, w.copy(), mu.copy(), sd.copy(), r.copy())
    ll, w, mu, sd, r = best
    hi = int(np.argmax(mu))
    lo = 1 - hi
    return {"method": f"EM, 2 Gaussians, {n_starts} starts, sd floor {sd_floor}", "loglik": ll,
            "upper_component": {"weight": float(w[hi]), "mean": float(mu[hi]), "sd": float(sd[hi])},
            "lower_component": {"weight": float(w[lo]), "mean": float(mu[lo]), "sd": float(sd[lo])},
            "n_assigned_upper": int((r[:, hi] > 0.5).sum())}


def fidelity() -> dict:
    a1 = json.loads(A1.read_text())
    out = {}
    for medium, m in a1["mlx"]["media"].items():
        vT = float(m["v_T"])
        x = np.asarray(m["V_SS"]["values"], float)
        ok = (vT - x) <= SUCCESS_TOL
        loss = vT - x[ok]
        out[medium] = {
            "n": int(len(x)), "v_T": vT, "v_S0": float(m["v_S0"]),
            "p_s": beta_post(int(ok.sum()), len(x)),
            "loss_in_success_cluster": {"mean": float(loss.mean()), "sd": float(loss.std(ddof=1)), "n": int(ok.sum()),
                                        "values": [round(float(v), 4) for v in sorted(loss)]},
            "failed_copies": {"n": int((~ok).sum()), "mean_score": float(x[~ok].mean()) if (~ok).any() else None,
                              "scores": [round(float(v), 4) for v in sorted(x[~ok])]},
            "all_losses": [round(float(v), 4) for v in sorted(vT - x)],
            "em_mixture_sensitivity": em_two_gaussians(x),
            "provenance": {"file": rel(A1), "field": f"mlx.media.{medium}.V_SS.values (copy self-play on the common 300 "
                           f"deals) and mlx.media.{medium}.v_T", "observations": int(len(x))},
        }
    return out


def innovation() -> dict:
    c1 = json.loads(C1.read_text())
    # weak level: v(none) = mask 0; the unverified stratum's seeds 0-7 repeat the verified ones (same prompt when
    # nothing is delivered), so de-duplicate by seed and check that duplicates agree.
    vnone: dict[int, float] = {}
    dup_check = []
    for st in ("verified", "unverified"):
        for sd, rec in c1["strata"][st]["per_seed"].items():
            y = float(rec["v"]["0"])
            s = int(sd)
            if s in vnone:
                dup_check.append(abs(vnone[s] - y))
            else:
                vnone[s] = y
    yw = np.array([vnone[s] for s in sorted(vnone)])

    rows = [json.loads(l) for l in C1_ROWS.read_text().splitlines() if l.strip()]
    # cross-check the weak level against the replay rows, and get the starting level
    s0 = {}
    for r in rows:
        s0[r["replicate"]] = r["s0_score"]
        if r["mask"] == 0:
            assert abs(r["y"] - vnone[r["replicate"]]) < 1e-9
    s0v = np.array([s0[s] for s in sorted(s0)])

    def block(gains, n_note):
        return {"epsilon": {str(m): beta_post(int((gains > m).sum()), len(gains)) for m in MARGINS},
                "gain_given_improvement": {str(m): describe(gains[gains > m]) for m in MARGINS},
                "all_gains": describe(gains), "n_note": n_note}

    weak = block(yw, "29 unique seeds: 29 verified + 8 unverified, the 8 being the same prompts as verified seeds 0-7")
    weak.update({
        "context": "feedback only (nothing delivered), weak student",
        "level": {"mean": float(s0v.mean()), "sd": float(s0v.std(ddof=1)), "n": int(len(s0v)),
                  "note": "student's starting self-play on each seed's 300 held-out deals; single_student.py documents "
                          "3.17 on its 200 calibration deals"},
        "rejected_outright": int((yw == 0).sum()),
        "duplicate_seeds_max_abs_diff": max(dup_check) if dup_check else None,
        "provenance": {"file": rel(C1), "field": "strata.{verified,unverified}.per_seed[seed].v['0'] (y for the empty "
                       "delivery; y = final minus starting score, held-out 300 deals)", "observations": int(len(yw)),
                       "level_from": f"{rel(C1_ROWS)} s0_score (gitignored run output; y cross-checked equal)"},
    })

    def piers_level(strata, mask):
        rs = [r for r in rows if r["mask"] == mask and r["stratum"] in strata and 0 in r["adopted"]]
        g = np.array([r["final_score"] - r["incumbent_score"] for r in rs])
        lev = np.array([r["incumbent_score"] for r in rs])
        acc = np.array([bool(r["accepted"]) for r in rs])
        return rs, g, lev, acc

    rs, g, lev, acc = piers_level(("verified",), 1)
    piers = block(g, "29 verified seeds, Piers delivered alone and adopted on every seed")
    piers.update({
        "context": "feedback plus a received good artifact (Piers adopted, its message in the revision prompt)",
        "level": {"mean": float(lev.mean()), "sd": float(lev.std(ddof=1)), "n": int(len(lev)),
                  "note": "adopted Piers' held-out self-play per seed (A1's common deals give 16.7)"},
        "revisions_accepted": int(acc.sum()),
        "accepted_but_worse_on_heldout": int(((g < 0) & acc).sum()),
        "provenance": {"file": rel(C1_ROWS), "field": "rows with stratum verified, mask 1, adopted [0]: final_score - "
                       "incumbent_score (incumbent = adopted Piers); gitignored run output, its y column equals "
                       "c1-full.json per_seed v['1']", "observations": int(len(g))},
    })
    # sensitivity: both strata (distinct prompts: the verification line differs), and Piers + re-sent own bot
    _, g2, _, _ = piers_level(("verified", "unverified"), 1)
    _, g5, _, _ = piers_level(("verified",), 5)
    piers["sensitivity"] = {
        "both_strata_mask1": {**{f"epsilon_{m}": beta_post(int((g2 > m).sum()), len(g2)) for m in MARGINS},
                              "note": "37 replays; the 8 unverified share seeds with verified 0-7 (correlated)"},
        "verified_piers_plus_self_mask5": {**{f"epsilon_{m}": beta_post(int((g5 > m).sum()), len(g5)) for m in MARGINS},
                                           "gains": describe(g5[g5 > 0.5])},
    }
    return {"weak": weak, "piers": piers,
            "rule": "epsilon(margin) = P(gain > margin) with a Beta(1,1) prior; gain = held-out (300 deals) self-play "
                    "after the generation minus before. A rejected or not-better revision gives gain 0 (harness "
                    "accept-if-not-worse on feedback deals)."}


def verification() -> dict:
    b1 = json.loads(B1.read_text())
    oc = b1["operating_characteristics"]
    pick = {"current_rule_selfplay_n200_mean_gt_0": "fixed N=200 (mean>0)", "sprt": "SPRT",
            "fixed_n200_t_test": "fixed N=200 (t-test)"}
    out = {}
    for key, name in pick.items():
        o = oc[name]
        out[key] = {"procedure": name, "false_adoption": o["false_adoption"], "miss_near_delta1": o["missed_near_delta1"],
                    "missed_improvement_delta_ge_1": o["missed_improvement"], "deals_mean": o["deals_mean"],
                    "deals_near_delta1": o["deals_near_delta1"], "n_h0": o["n_h0"], "n_h1": o["n_h1"],
                    "n_near_delta1": o["n_near_delta1"],
                    "oc_curve": [{"delta_mid": c["delta_mid"], "p_adopt": c["p_adopt"], "n": c["n"]}
                                 for c in b1["curves"][name]],
                    "provenance": {"file": rel(B1), "field": f"operating_characteristics['{name}'] and curves['{name}']",
                                   "observations": f"{o['n_h0']} tests with delta <= 0, {o['n_h1']} with delta >= 1, "
                                                   f"{o['n_near_delta1']} near delta = 1 (360 engine-bot pairs x 10 "
                                                   f"blocks of 600 deals)"}}
    out["caveat"] = ("B1 is engine-only: anchor-family rule bots, not model-written candidates; median sigma 3.6. "
                     "'False adoption' pools all delta <= 0, most of which are far below 0; the OC curve gives "
                     "P(adopt | delta) by delta bin.")
    return out


MISSING = [
    {"quantity": "epsilon at the Piers level, feedback only (no message in the prompt)",
     "why": "C1 measures the Piers level only with Piers' own message in the revision prompt; an isolated agent at "
            "that level revises with feedback only",
     "run": "30 revisions of a Piers-level incumbent with nothing delivered, local MLX model, seeds 300-329; about 30 "
            "model calls, 25 to 30 minutes, $0"},
    {"quantity": "epsilon and gain distribution at intermediate levels (5 to 15) and above Piers (17 to 20)",
     "why": "the model interpolates between two measured levels and holds the Piers-level value above 17; every "
            "generation-100 prediction depends on that extrapolation",
     "run": "a level ladder: 30 revisions each from incumbents at about 8, 12 and 18 (Piers mutants or IGGI-family "
            "bots from the B1 bank), feedback only; about 90 calls, 75 minutes, $0"},
    {"quantity": "copy fidelity for teachers other than Piers, and with several exemplars in the prompt",
     "why": "A1 sends one teacher at 16.7 whose rule functions already exist in the student's template; the loss may "
            "scale with the teacher's level and with the number of messages",
     "run": "A1 with a second teacher (IGGI, 15.9) and a weak teacher (about 8), 15 replicates each, medium both; about "
            "30 calls, 25 minutes, $0"},
    {"quantity": "verification operating characteristics on model-written candidates",
     "why": "B1's bank is engine bots; model candidates may have a different sigma and delta distribution",
     "run": "no new model calls: rescore the 90 A1 copies and the C1 candidates against their incumbents in 10 "
            "blocks of 200 deals each with the B1 code (engine only, minutes)"},
    {"quantity": "every parameter on the model G1 will actually run (hosted open-weight model) and a second block",
     "why": "these numbers come from one local model (Qwen3.6-35B-A3B 4-bit) and one student template",
     "run": "repeat the C1 'none' and 'Piers only' cells (2 x 30 calls) and A1 'both' (30 calls) on the hosted model; "
            "about 90 calls, cents"},
    {"quantity": "G1 warm-start skill distribution (generation 0 authored by the model)",
     "why": "no authored warm start for 8 agents exists on disk (phase-4 warm starts are stub runs)",
     "run": "author the G1 generation-0 set for 3 population seeds (24 calls) on the G1 model; about 20 minutes"},
]


def main() -> int:
    fid = fidelity()
    inn = innovation()
    ver = verification()
    out = {
        "scope": "Draft P1 micro-parameters from existing single-student runs on one model block (Qwen3.6-35B-A3B "
                 "4-bit on MLX, temperature 0). Zero model calls were made to produce this file. Not pre-registered.",
        "generated_by": "scripts/p1_params.py",
        "rules": {
            "copy_success": f"a copy succeeds if v_T - score <= {SUCCESS_TOL} (one-sided: copies above the teacher are "
                            "successes with negative loss); p_s posterior Beta(1 + successes, 1 + failures)",
            "loss": "v_T - score within the success cluster",
            "innovation": "see innovation.rule",
            "intervals": "95% equal-tailed Beta posterior intervals",
        },
        "fidelity": fid,
        "innovation": inn,
        "verification": ver,
        "missing": MISSING,
    }
    OUT.write_text(json.dumps(out, indent=1))

    print(f"Fidelity (A1, success = within {SUCCESS_TOL} of v_T = {fid['code']['v_T']:.1f})")
    print(f"{'medium':8} {'p_s':>6} {'95% CI':>15} {'loss mean':>10} {'loss sd':>8} {'fail n':>7} {'fail mean':>10}"
          f" {'EM upper w':>11}")
    for m, f in fid.items():
        p = f["p_s"]
        print(f"{m:8} {p['mean']:6.3f} [{p['lo']:5.3f}, {p['hi']:5.3f}] {f['loss_in_success_cluster']['mean']:10.2f}"
              f" {f['loss_in_success_cluster']['sd']:8.2f} {f['failed_copies']['n']:7d} "
              f"{(f['failed_copies']['mean_score'] or float('nan')):10.2f} "
              f"{f['em_mixture_sensitivity']['upper_component']['weight']:11.2f}")
    print("\nInnovation (C1)")
    print(f"{'level':6} {'at':>6} {'margin':>6} {'k/n':>7} {'eps':>6} {'95% CI':>15} {'gain|imp mean':>14} {'sd':>6}")
    for lv in ("weak", "piers"):
        b = inn[lv]
        for m in MARGINS:
            e = b["epsilon"][str(m)]
            gd = b["gain_given_improvement"][str(m)]
            print(f"{lv:6} {b['level']['mean']:6.2f} {m:6.1f} {e['k']:3d}/{e['n']:<3d} {e['mean']:6.3f} "
                  f"[{e['lo']:5.3f}, {e['hi']:5.3f}] {gd.get('mean', float('nan')):14.2f} "
                  f"{(gd.get('sd') or float('nan')):6.2f}")
    print("\nVerification (B1, engine)")
    for k in ("current_rule_selfplay_n200_mean_gt_0", "sprt"):
        v = ver[k]
        print(f"{v['procedure']:22} false adoption {v['false_adoption']:.3f}  miss near delta=1 "
              f"{v['miss_near_delta1']:.2f}  deals near delta=1 {v['deals_near_delta1']:.0f}")
    print(f"\nMissing: {len(MISSING)} items (see JSON). Wrote {rel(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
