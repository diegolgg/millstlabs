"""B1 on anchors, engine only (fix round 1, step 15): fixed-N verification versus Wald's SPRT.

Bank: (1) for each base (Piers, IGGI), 24 variants with one or two rule perturbations (the stub's mutation operator,
seeded); pairs are (incumbent = base, candidate = variant) and the reverse; (2) two graded families, all
ordered pairs within each: Piers with the play_probably_safe threshold in 0.3..0.9 and the extra-lives switch (14
bots), and IGGI with play_probably_safe(t) inserted after play_safe_card, t in 0.62..0.95 (10 bots), so delta spans 0
to about 2 smoothly (the random mutations mostly change a lot or nothing). Each
pair's true delta and sigma come from 2,000 deals (experiment seed 900). Tests run on separate deals (experiment seed
901): 10 replications per pair, each on its own block of 600 deals; fixed N uses the first N deals of the block, the
SPRT reads the block sequentially (capped at 600).

Writes docs/results/b1-sprt.json and docs/results/b1-sprt.png. Zero model calls.
Usage: python scripts/b1_sprt.py [--workers 10] [--variants 24]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from culture.analysis.sprt import decision_rule, fixed_n, operating_characteristics, sprt, wald_asn_h1  # noqa: E402
from culture.bots.anchors import RULE_LISTS, rulebot_source  # noqa: E402
from culture.bots.runner import BotSpec  # noqa: E402
from culture.evaluate.pool import Evaluator  # noqa: E402
from culture.evaluate.selfplay import selfplay  # noqa: E402
from culture.game.hanabi import HanabiParams  # noqa: E402
from culture.game.seeds import generation_seeds  # noqa: E402
from culture.llm.stub_backend import mutate  # noqa: E402

NS = (50, 100, 200, 400)
DELTA1, ALPHA, BETA = 1.0, 0.05, 0.05
REPS, BLOCK = 10, 600


GRADED_T = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
IGGI_T = (0.62, 0.64, 0.66, 0.68, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


def graded() -> dict[str, list]:
    """Piers with play_probably_safe threshold t and the extra-lives switch: small, graded behaviour changes, so the
    ordered pairs of this family span delta smoothly through 0 and 1."""
    out = {}
    for t in GRADED_T:
        for lives in (True, False):
            rules = [[n, dict(p)] for n, p in RULE_LISTS["piers"]]
            for r in rules:
                if r[0] == "play_probably_safe":
                    r[1] = {"threshold": t, "require_extra_lives": lives}
            out[f"graded t={t} lives={lives}"] = rules
    for t in IGGI_T:  # IGGI with play_probably_safe(t) after play_safe_card: steep in 0.6-0.7, so delta passes 1
        out[f"graded iggi+pps t={t}"] = (RULE_LISTS["iggi"][:2] + [["play_probably_safe", {"threshold": t}]]
                                         + [[n, dict(p)] for n, p in RULE_LISTS["iggi"][2:]])
    return out


def bank(n_variants: int) -> dict[str, list]:
    bots = {}
    for base in ("piers", "iggi"):
        bots[base] = [[n, dict(p)] for n, p in RULE_LISTS[base]]
        for i in range(n_variants):
            rng = random.Random(f"b1/{base}/{i}")
            rules = [[n, dict(p)] for n, p in RULE_LISTS[base]]
            what = []
            for _ in range(rng.choice([1, 2])):
                rules, w = mutate(rules, rng)
                what.append(w)
            if rules != RULE_LISTS[base] and rules not in bots.values():
                bots[f"{base}~{i}: " + "; ".join(what)] = rules
    return bots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--variants", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "b1-sprt.json"))
    a = ap.parse_args()
    t0 = time.time()
    bots = bank(a.variants)
    family = graded()
    bots.update(family)
    truth_seeds = generation_seeds(900, 0, 2000, "eval").seeds
    test_seeds = generation_seeds(901, 0, REPS * BLOCK, "eval").seeds
    ev = Evaluator(HanabiParams(), workers=a.workers, chunk=50)
    scores = {}
    try:
        for name, rules in bots.items():
            spec = BotSpec("b1:" + name, rulebot_source(rules))
            scores[name] = (np.array([r.score for r in selfplay(ev, spec, truth_seeds)], float),
                            np.array([r.score for r in selfplay(ev, spec, test_seeds)], float))
            ev.clear_memo()
    finally:
        ev.close()
    pairs, trials = [], []
    ordered = []
    for name in bots:
        if name in ("piers", "iggi") or name in family:
            continue
        base = name.split("~")[0]
        ordered += [(base, name, "mutation"), (name, base, "mutation")]
    fam_of = {n: ("iggi" if "iggi+pps" in n else "piers") for n in family}  # pairs within each graded family
    ordered += [(i, c, "graded") for i in family for c in family if i != c and fam_of[i] == fam_of[c]]
    for inc, cand, kind in ordered:
        d_truth = scores[cand][0] - scores[inc][0]
        delta, sigma = float(d_truth.mean()), float(d_truth.std(ddof=1))
        pairs.append({"incumbent": inc, "candidate": cand, "kind": kind, "delta": delta, "sigma": sigma,
                      "se": sigma / np.sqrt(d_truth.size)})
        d_test = scores[cand][1] - scores[inc][1]
        for rep in range(REPS):
            block = d_test[rep * BLOCK:(rep + 1) * BLOCK]
            for n in NS:
                for rule in ("mean>0", "t-test"):
                    trials.append({"procedure": f"fixed N={n} ({rule})", "delta": delta,
                                   "adopt": fixed_n(block, n, rule, ALPHA), "deals": n})
            adopt, used = sprt(block, DELTA1, ALPHA, BETA, burn_in=20, cap=BLOCK)
            trials.append({"procedure": "SPRT", "delta": delta, "adopt": adopt, "deals": used})
    oc = operating_characteristics(trials, DELTA1)
    rule = decision_rule(oc)
    sig = float(np.median([p["sigma"] for p in pairs]))
    bins = np.arange(-6.0, 6.5, 0.5)
    curves = {}
    for p in oc:
        ts = [t for t in trials if t["procedure"] == p]
        rows = []
        for lo in bins:
            inb = [t for t in ts if lo <= t["delta"] < lo + 0.5]
            if inb:
                rows.append({"delta_mid": float(lo + 0.25), "p_adopt": float(np.mean([t["adopt"] for t in inb])),
                             "deals": float(np.mean([t["deals"] for t in inb])), "n": len(inb)})
        curves[p] = rows
    out = {"design": {"bases": ["piers", "iggi"], "bots": len(bots), "pairs": len(pairs), "truth_deals": 2000,
                      "replications_per_pair": REPS, "block_deals": BLOCK, "delta1": DELTA1, "alpha": ALPHA,
                      "beta": BETA, "fixed_n": list(NS), "sprt": "Wald, online sigma after 20 deals, cap 600"},
           "pairs_summary": {"n_h0 (delta<=0)": sum(p["delta"] <= 0 for p in pairs),
                             "n_h1 (delta>=1)": sum(p["delta"] >= DELTA1 for p in pairs),
                             "n_indifference": sum(0 < p["delta"] < DELTA1 for p in pairs),
                             "median_sigma": sig, "wald_asn_h1_at_median_sigma": wald_asn_h1(sig, DELTA1, ALPHA, BETA)},
           "operating_characteristics": oc,
           "decision_rule": rule,
           "curves": curves, "pairs": pairs, "wall_seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    figure(out, Path(a.out).with_suffix(".png"))
    print(json.dumps({k: out[k] for k in ("pairs_summary", "decision_rule")}, indent=1))
    for p, v in oc.items():
        print(f"{p:26s} false adoption {v['false_adoption']}  missed {v['missed_improvement']}  missed near "
              f"delta1 {v['missed_near_delta1']}  deals {v['deals_mean']:.0f} (near delta1: {v['deals_near_delta1']})")
    return 0


def figure(out: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from culture.analysis.figures import MUTED, SERIES, _style

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    show = ["SPRT"] + [f"fixed N={n} (t-test)" for n in NS] + ["fixed N=200 (mean>0)"]
    for i, p in enumerate(show):
        c = out["curves"][p]
        x = [r["delta_mid"] for r in c]
        style = dict(color=SERIES[i % len(SERIES)], linewidth=2.4 if p == "SPRT" else 1.4, label=p)
        axes[0].plot(x, [r["p_adopt"] for r in c], **style)
        axes[1].plot(x, [r["deals"] for r in c], **style)
    for ax in axes:
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.axvline(out["design"]["delta1"], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    _style(axes[0], "Probability of adopting the candidate", "true delta (points, from 2,000 deals)", "P(adopt)")
    _style(axes[1], "Deals used per decision", "true delta (points)", "deals")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
