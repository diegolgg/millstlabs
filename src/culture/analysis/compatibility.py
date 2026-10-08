"""Prose vs code copies: compatibility with the teacher, not score (proposed idea 10). Zero model calls.

The copy-fidelity run (`culture.run.transmission`, runs/a1-params) stored, per copy, its self-play V(S', S') and its
cross-play with the teacher V(S', T) on one common set of 300 evaluation deals, but only as means. This module
rebuilds every copy from the run's call cache (the cached revise response parsed against the student's bot exactly
as `agents.agent.produce` does), checks that the rebuilt artifact id equals the recorded one, replays self-play and
cross-play on the same deals with per-deal scores (engine only), and summarizes per medium:

- rho = mean V(S', T) / v_T (already in a1-params.json; recomputed as a check);
- compatible = |V(S', S') - V(S', T)| <= 1, with a Wilson interval;
- the paired difference d = V(S', S') - V(S', T), mean with a percentile bootstrap interval over copies;
- the skill-adjusted gap G = (V(S', S') + v_T) / 2 - V(S', T): what the pair loses relative to the average of its two
  self-plays. d alone is negative for any weak copy (the teacher carries the pair), so d mixes skill with
  compatibility; G is 0 for a behavioural clone and for any copy whose conventions mesh with the teacher's under an
  additive seat model, and positive when the two bots' conventions clash;
- all of the above split by whether the copy's rule list is identical to the teacher's (a clone), and restricted to
  successful copies (self-play within 1 point of the teacher).

    python -m culture.analysis.compatibility --run runs/a1-params --out docs/results
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts.schema import artifact_id
from ..bots.anchors import anchor_conventions, anchor_source, conventions_for, parse_rules, rulebot_source
from ..bots.runner import BotSpec, SandboxLimits
from ..evaluate.crossplay import crossplay
from ..evaluate.pool import Evaluator
from ..evaluate.selfplay import selfplay
from ..game.hanabi import HanabiParams
from ..game.seeds import generation_seeds
from ..llm.parsing import ParseError, parse_artifact, tag
from ..run.config import from_dict
from ..run.single_student import STUDENT_RULES

MEDIA = ("code", "prose", "both")


# ---------------------------------------------------------------------------------------------------- rebuild
def load_run(run: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = json.loads((run / "spec.json").read_text())
    rows = [json.loads(x) for x in (run / "rows.jsonl").read_text().splitlines() if x.strip()]
    return spec, rows


def rebuild_copies(run: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each row gains `code`, `conventions`, `wrote_conventions`, `id_match`. A copy whose revision failed would be S0;
    the A1 run had none, and a mismatch is reported, never silently accepted."""
    s0_code, s0_conv = rulebot_source(STUDENT_RULES), conventions_for(STUDENT_RULES)
    out = []
    for r in rows:
        revise = [c for c in r["calls"] if c["tag"] == "revise"]
        if len(revise) != 1 or any(c["tag"] == "repair" for c in r["calls"]):
            raise ValueError(f"replicate {r['replicate']} {r['medium']}: expected exactly one revise call, no repair")
        key = revise[0]["request_key"]
        text = json.loads((run / "llm_cache" / key[:2] / f"{key}.json").read_text())["response"]["text"]
        try:
            code, conv, _ = parse_artifact(text, s0_code, s0_conv)
        except ParseError:
            code, conv = s0_code, s0_conv
        out.append({**r, "code": code, "conventions": conv, "wrote_conventions": tag(text, "conventions") is not None,
                    "id_match": artifact_id(code, conv) == r["copy"]})
    return out


# ---------------------------------------------------------------------------------------------------- replay
def replay(spec: dict[str, Any], copies: list[dict[str, Any]], workers: int = 8) -> dict[str, Any]:
    """Per-deal self-play of every copy and of the teacher, and per-deal cross-play copy x teacher, on the run's
    common evaluation deals (seats alternate by deal parity, as in `evaluate.crossplay`)."""
    base = from_dict(spec["base"])
    seeds = generation_seeds(int(spec["eval_experiment_seed"]), int(spec["generation"]),
                             int(spec["eval_games"]), "eval").seeds
    ev = Evaluator(HanabiParams(**asdict(base.game)), SandboxLimits(**asdict(base.sandbox)), workers=workers)
    try:
        # the teacher's key must be its artifact id in the run: cross-play seats are assigned by key order
        t_code = anchor_source(spec["teacher"])
        teacher = BotSpec(artifact_id(t_code, anchor_conventions(spec["teacher"])), t_code)
        uniq: dict[str, BotSpec] = {c["copy"]: BotSpec(c["copy"], c["code"]) for c in copies}
        jobs = [((teacher, teacher), seeds)]
        for s in uniq.values():
            jobs.append(((s, s), seeds))
        ev.run(jobs)  # warm the memo in one parallel batch
        t_ss = np.array([g.score for g in selfplay(ev, teacher, seeds)], float)
        ss, st = {}, {}
        for aid, s in uniq.items():
            ss[aid] = np.array([g.score for g in selfplay(ev, s, seeds)], float)
            st[aid] = np.array([g.score for g in crossplay(ev, s, teacher, seeds)], float)
    finally:
        ev.close()
    return {"seeds": seeds, "teacher_selfplay": t_ss, "selfplay": ss, "crossplay": st}


# ---------------------------------------------------------------------------------------------------- statistics
def boot_ci(x: np.ndarray, f=np.mean, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def _block(c: list[dict[str, Any]], v_t: float, n_boot: int) -> dict[str, Any]:
    if not c:
        return {"n": 0}
    ss = np.array([x["V_SS"] for x in c])
    st = np.array([x["V_ST"] for x in c])
    d = ss - st
    g = (ss + v_t) / 2 - st
    k = int(np.sum(np.abs(d) <= 1.0))
    both = np.column_stack([ss, st])
    rho_lo, rho_hi = boot_ci(both, lambda z: z[:, 1].mean() / v_t, n_boot) if len(c) > 1 else (None, None)
    return {
        "n": len(c),
        "selfplay_mean": float(ss.mean()), "selfplay_ci": boot_ci(ss, n_boot=n_boot),
        "crossplay_mean": float(st.mean()), "crossplay_ci": boot_ci(st, n_boot=n_boot),
        "rho": float(st.mean() / v_t), "rho_ci": (rho_lo, rho_hi),
        "compatible": k, "compatible_frac": k / len(c), "compatible_wilson": wilson(k, len(c)),
        "d_selfplay_minus_crossplay": float(d.mean()), "d_ci": boot_ci(d, n_boot=n_boot),
        "gap_G": float(g.mean()), "gap_G_ci": boot_ci(g, n_boot=n_boot),
        "gap_G_median": float(np.median(g)),
    }


def summarize(copies: list[dict[str, Any]], v_t: float, n_boot: int = 2000) -> dict[str, Any]:
    out: dict[str, Any] = {"v_T": v_t, "media": {}, "contrasts": {}}
    for mu in MEDIA:
        c = [x for x in copies if x["medium"] == mu]
        if not c:
            continue
        clone = [x for x in c if x["same_rule_list_as_teacher"]]
        nonclone = [x for x in c if not x["same_rule_list_as_teacher"]]
        good = [x for x in c if abs(x["V_SS"] - v_t) <= 1.0]
        good_nc = [x for x in good if not x["same_rule_list_as_teacher"]]
        out["media"][mu] = {"all": _block(c, v_t, n_boot), "clone": _block(clone, v_t, n_boot),
                            "nonclone": _block(nonclone, v_t, n_boot), "successful": _block(good, v_t, n_boot),
                            "successful_nonclone": _block(good_nc, v_t, n_boot),
                            "wrote_conventions": int(sum(x["wrote_conventions"] for x in c)),
                            "copy_illegal_rate_mean": float(np.mean([x["copy_illegal_rate"] for x in c]))}
    # prose minus code, paired by replicate seed (the same feedback traces), percentile bootstrap over replicates
    by = {(x["medium"], x["replicate"]): x for x in copies}
    for a, b in (("prose", "code"), ("both", "code"), ("prose", "both")):
        reps = sorted({r for (m, r) in by if m == a} & {r for (m, r) in by if m == b})
        if not reps:
            continue
        res = {}
        for name, f in (("crossplay", lambda x: x["V_ST"]), ("selfplay", lambda x: x["V_SS"]),
                        ("d", lambda x: x["V_SS"] - x["V_ST"]),
                        ("gap_G", lambda x: (x["V_SS"] + v_t) / 2 - x["V_ST"]),
                        ("compatible", lambda x: float(abs(x["V_SS"] - x["V_ST"]) <= 1.0))):
            diff = np.array([f(by[(a, r)]) - f(by[(b, r)]) for r in reps])
            res[name] = {"point": float(diff.mean()), "ci": boot_ci(diff, n_boot=n_boot)}
        out["contrasts"][f"{a}_minus_{b}"] = {"n_pairs": len(reps), **res}
    return out


def per_copy_noise(copies: list[dict[str, Any]], rep: dict[str, Any]) -> dict[str, Any]:
    """Deal-level standard error of each copy's paired difference self-play minus cross-play (300 shared deals):
    how far a 1-point threshold sits above the noise of one copy's estimate."""
    se = []
    for x in copies:
        d = rep["selfplay"][x["copy"]] - rep["crossplay"][x["copy"]]
        se.append(float(d.std(ddof=1) / math.sqrt(len(d))))
    se = np.array(se)
    return {"median_se": float(np.median(se)), "max_se": float(se.max()), "min_se": float(se.min())}


# ---------------------------------------------------------------------------------------------------- figure
def figure(copies: list[dict[str, Any]], v_t: float, summary: dict[str, Any], path: Path) -> None:
    """Top row: every copy. Bottom row: the successful region, where clones sit on the teacher's point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from .figures import GRID, MUTED, SERIES, SURFACE, TEXT

    colors = {"code": SERIES[0], "prose": SERIES[1], "both": SERIES[2]}
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.6))
    fig.set_facecolor(SURFACE)
    for row, (lo, hi) in enumerate(((0.0, 19.0), (12.5, 18.2))):
        xs = np.linspace(lo, hi, 50)
        for ax, mu in zip(axes[row], MEDIA):
            c = [x for x in copies if x["medium"] == mu]
            ax.set_facecolor(SURFACE)
            ax.fill_between(xs, xs - 1, xs + 1, color=MUTED, alpha=0.10, linewidth=0)
            ax.plot(xs, xs, color=MUTED, linewidth=1)
            ax.plot(xs, (xs + v_t) / 2, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
            cl = [x for x in c if x["same_rule_list_as_teacher"]]
            nc = [x for x in c if not x["same_rule_list_as_teacher"]]
            ax.scatter([x["V_SS"] for x in nc], [x["V_ST"] for x in nc], s=38, facecolors="none",
                       edgecolors=colors[mu], linewidths=1.6, zorder=3)
            ax.scatter([v_t], [v_t], marker="*", s=150, color=TEXT, zorder=4)
            if cl:
                ax.scatter([x["V_SS"] for x in cl], [x["V_ST"] for x in cl], s=38, color=colors[mu],
                           edgecolors=SURFACE, linewidths=1.5, zorder=5)
                if row == 1:
                    ax.annotate(f"{len(cl)} rule-list clone{'s' if len(cl) > 1 else ''} on the teacher's point",
                                (v_t, v_t), xytext=(12.8, 17.75), color=MUTED, fontsize=8,
                                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
            s = summary["media"][mu]["all"]
            if row == 0:
                ax.set_title(f"{mu}: {s['compatible']}/{s['n']} within 1 point; gap G {s['gap_G']:.2f}",
                             loc="left", color=TEXT, fontsize=10.5)
            else:
                g = summary["media"][mu]["successful"]
                ax.set_title(f"{mu}, zoom: {g['compatible']}/{g['n']} successful copies within 1 pt",
                             loc="left", color=TEXT, fontsize=10)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(True, color=GRID, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel("copy's self-play (mean score, 300 deals)", color=MUTED, fontsize=8.5)
        axes[row][0].set_ylabel("copy x teacher cross-play (same deals)", color=MUTED, fontsize=8.5)
    handles = [Patch(color=MUTED, alpha=0.10, label="within 1 point of self-play (counted compatible)"),
               Line2D([], [], color=MUTED, linewidth=1, label="cross-play = self-play"),
               Line2D([], [], color=MUTED, linewidth=1, linestyle=(0, (3, 3)),
                      label="cross-play = average of the two self-plays (G = 0)"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=MUTED,
                      markersize=7, label="rewritten copy"),
               Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=7,
                      label="rule list identical to teacher"),
               Line2D([], [], marker="*", linestyle="", color=TEXT, markersize=11, label=f"teacher ({v_t:.1f})")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5, labelcolor=MUTED)
    fig.suptitle("Copies of a 17-point bot taught by code, prose or both: self-play vs cross-play with the teacher "
                 "(30 copies per medium)", x=0.01, ha="left", color=TEXT, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="runs/a1-params")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    spec, rows = load_run(run)
    copies = rebuild_copies(run, rows)
    rep = replay(spec, copies, a.workers)
    v_t_replay = float(rep["teacher_selfplay"].mean())
    check = []
    for x in copies:
        ss, st = float(rep["selfplay"][x["copy"]].mean()), float(rep["crossplay"][x["copy"]].mean())
        check.append({"replicate": x["replicate"], "medium": x["medium"], "id_match": x["id_match"],
                      "V_SS_stored": x["V_SS"], "V_SS_replay": ss, "V_ST_stored": x["V_ST"], "V_ST_replay": st})
    max_dev = max(max(abs(c["V_SS_stored"] - c["V_SS_replay"]), abs(c["V_ST_stored"] - c["V_ST_replay"]))
                  for c in check)
    v_t = float(np.mean([x["v_T"] for x in copies]))
    summary = summarize(copies, v_t)
    rules_t = parse_rules(anchor_source(spec["teacher"]))
    report = {
        "scope": "Zero model calls. Copies rebuilt from runs/a1-params/llm_cache (one cached revise response per copy, "
                 "parsed against the student's bot as agents.agent.produce does); self-play and cross-play with the "
                 "teacher (Piers) replayed on the run's 300 common evaluation deals (experiment seed 299, "
                 "generation 1, purpose eval); seats alternate by deal parity in cross-play.",
        "reproduction": {"ids_matched": int(sum(c["id_match"] for c in check)), "n": len(check),
                         "teacher_selfplay_replay": v_t_replay, "teacher_selfplay_stored": v_t,
                         "max_abs_deviation_from_stored_means": max_dev},
        "per_copy_noise": per_copy_noise(copies, rep),
        "definitions": {
            "compatible": "|V(S',S') - V(S',T)| <= 1 point",
            "d": "V(S',S') - V(S',T), mean over copies, percentile bootstrap (2,000) over copies",
            "gap_G": "(V(S',S') + v_T)/2 - V(S',T): cross-play shortfall against the average of the two self-plays",
            "clone": "rule list parsed from the copy's bot.py equals the teacher's: " + json.dumps(rules_t),
            "successful": "|V(S',S') - v_T| <= 1 point",
            "contrasts": "paired by replicate seed (same feedback traces in the prompt)"},
        "summary": summary,
        "copies": [{k: x[k] for k in ("replicate", "medium", "copy", "V_SS", "V_ST", "same_rule_list_as_teacher",
                                      "copy_illegal_rate", "wrote_conventions")} for x in copies],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "prose-vs-code-compatibility.json").write_text(json.dumps(report, indent=1) + "\n")
    figure(copies, v_t, summary, out / "prose-vs-code-compatibility-scatter.png")
    print(json.dumps(report["reproduction"], indent=1))
    print(json.dumps(report["per_copy_noise"], indent=1))
    for mu, m in summary["media"].items():
        s = m["all"]
        print(mu, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    print(json.dumps(summary["contrasts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
