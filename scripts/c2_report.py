"""Assemble the C2 pilot report (Overnight 2, step 4): `verified_quarantine` against C1's `verified` stratum on the
same seeds, on the stub (pipeline proof) and on MLX, with the null-stub band. Writes docs/results/c2-pilot.json.

Usage: python scripts/c2_report.py [--stub runs/c2_stub] [--mlx-q runs/c2-pilot] [--mlx-v runs/c1-full]
       [--null runs/c1-null] [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.analysis.quarantine import c2_summary  # noqa: E402

SCOPE = ("Scope: C2 pilot only (files/prereg/C2-quarantine.md section 4): one student, k = 3 fixed teachers, 5 seeds, "
         "one model (Qwen3.6-35B-A3B 4-bit on MLX, seeded, temperature 0), one generation, 300 held-out deals. The "
         "decision rules need the full R and two model blocks, so no claim is judged here; the stub section only "
         "proves the pipeline.")


def rows_of(d: Path) -> list[dict]:
    p = Path(d) / "replays.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def compact(s: dict) -> dict:
    keep = ("seeds", "primary", "secondary", "tau_flawed_quarantine", "tau_flawed_verified", "tau_piers_quarantine",
            "tau_piers_verified", "collapse_max_abs_over_seeds", "decision_rules")
    out = {k: s[k] for k in keep}
    out["per_seed"] = {r: {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]
                               if isinstance(v, list) else v)
                           for k, v in p.items() if k not in ("v_quarantine", "v_verified")}
                       for r, p in s["per_seed"].items()}
    out["per_seed_v"] = {r: {"quarantine": p["v_quarantine"], "verified": p["v_verified"]}
                         for r, p in s["per_seed"].items()}
    return out


def operations(rows: list[dict], stratum: str) -> dict:
    sr = [x for x in rows if x["stratum"] == stratum]
    calls = [c for x in sr for c in x.get("calls", [])]
    # a backend call carries its latency; a hit on an entry recorded by the same run id is ledgered as not cached
    # (the crash-resume convention, llm/cache.py) but has no latency: count it as a within-run cache hit
    fresh = [c for c in calls if not c["cached"] and c.get("latency_s") is not None]
    lat = [c["latency_s"] for c in fresh]
    return {"replays": len(sr), "llm_calls": len(calls), "backend_calls": len(fresh),
            "cache_hits_within_run": sum(not c["cached"] and c.get("latency_s") is None for c in calls),
            "cache_hits_other_runs": sum(bool(c["cached"]) for c in calls), "backend_seconds": round(sum(lat), 1),
            "seconds_per_backend_call": round(sum(lat) / len(lat), 1) if lat else None,
            "admissible_rate": (sum(bool(x["admissible"]) for x in sr if x.get("admissible") is not None)
                                / max(1, sum(x.get("admissible") is not None for x in sr))),
            "withheld_by_teacher": {t: sum(t in x.get("withheld", []) for x in sr) for t in ("T1", "T2", "T3")}}


def mlx_extras(q: list[dict], v: list[dict], s: dict, seeds: list[int]) -> dict:
    """The mechanical note, the cross-session reproducibility check and exploratory outcome contrasts."""
    import numpy as np

    from culture.analysis import credit as C
    from culture.analysis.quarantine import _agg

    cq = {(x["replicate"], x["mask"]): x for x in q if x["stratum"] == "verified_quarantine"}
    cv = {(x["replicate"], x["mask"]): x for x in v if x["stratum"] == "verified"}
    repro = {str(r): {str(m): {"same_candidate_artifact": cq[(r, m)]["candidate"] == cv[(r, m)]["candidate"],
                               "same_y": cq[(r, m)]["y"] == cv[(r, m)]["y"]} for m in (0, 1)} for r in seeds}
    all_same = all(z["same_candidate_artifact"] and z["same_y"] for d in repro.values() for z in d.values())
    full = 7
    d_all = [cq[(r, full)]["y"] - cv[(r, full)]["y"] for r in seeds]
    d_avg = [float(np.mean([cq[(r, m)]["y"] for m in range(8)]) - np.mean([cv[(r, m)]["y"] for m in range(8)]))
             for r in seeds]
    # projection: under the collapse identity the primary quantity equals -tau_flawed(verified) on any seed set
    allv = {}
    for x in v:
        if x["stratum"] == "verified":
            allv.setdefault(x["replicate"], {})[x["mask"]] = x["y"]
    proj = [-float(C.tau(allv[r], 3)[1]) for r in sorted(allv) if len(allv[r]) == 8]
    return {
        "mechanical_note": (
            "At temperature 0 with seeded sampling, quarantine withholds every message that did not pass "
            "verification, so the revision prompt, and hence v(S), depends only on the delivered messages that "
            "passed. Flawed never passes, so tau_flawed(quarantine) = 0 by construction (collapse_max_abs_over_seeds "
            "= 0.0: on every seed v(S) = v(S restricted to Piers)). The primary quantity is therefore minus C1's "
            "tau_flawed(verified) on the same seeds, and the pre-registered rule tau_flawed(quarantine) in [-1, 1] "
            "cannot fail. The empirical content is C1's Flawed-versus-placebo contrast (docs/results/c1-full.json, "
            "strata.verified.exploratory.flawed_minus_placebo_tau: -2.44, t interval [-3.69, -1.19], p = 0.0004, "
            "29 seeds, exploratory): a rejected message's text lowers the student more than a generic text "
            "perturbation does, and quarantine removes that channel by construction."),
        "cross_session_reproducibility": {
            "what": "C2 ran under a different run name, so no C1 cache entry could be reused; its fresh calls for "
                    "nothing delivered (mask 0) and Piers only (mask 1) are byte-identical requests to C1's. Same "
                    "candidate artifact id means the model wrote the same code and conventions.",
            "all_identical": all_same, "per_seed": repro},
        "projection_primary_from_c1_29_seeds": {
            "note": "not a C2 run: under the collapse identity the primary quantity equals -tau_flawed(verified), "
                    "so C1's 29 verified seeds show what the full C2 would report in this model block",
            **_agg(proj)},
        "exploratory_outcome_contrasts": {
            "note": "Exploratory, not pre-registered. The secondary quantity is negative because tau_piers(verified) "
                    "averages over deliveries of the others and includes Piers' interaction with Flawed's text "
                    "(C1 exploratory interaction +3.76 [2.15, 5.37]): adopting Piers partly undoes the harm of "
                    "Flawed's prose, a harm quarantine removes. The student's outcomes themselves:",
            "v_all_quarantine_minus_verified": {**_agg(d_all), "values": d_all},
            "mean_v_over_8_subsets_quarantine_minus_verified": {**_agg(d_avg), "values": d_avg}},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", default=str(ROOT / "runs" / "c2_stub"))
    ap.add_argument("--mlx-q", default=str(ROOT / "runs" / "c2-pilot"))
    ap.add_argument("--mlx-v", default=str(ROOT / "runs" / "c1-full"))
    ap.add_argument("--null", default=str(ROOT / "runs" / "c1-null"))
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "c2-pilot.json"))
    a = ap.parse_args()
    teachers = ["piers", "flawed_persuasive", "self"]
    report: dict = {"scope": SCOPE,
                    "design": "k = 3 (Piers; Flawed with persuasive prose and honest evidence; the student's own bot "
                              "re-sent), exhaustive delivery (8 replays per seed), stratum verified_quarantine "
                              "(selfplay n = 200, replace_if_better, org.quarantine_unverified) against C1's verified "
                              "stratum on the same seeds (same deals, same sampling seed). Primary: tau_flawed(q) - "
                              "tau_flawed(v); secondary: tau_piers(q) - tau_piers(v); implied R at a 2-point minimum "
                              "effect, 80% power, capped at 29."}
    stub = rows_of(Path(a.stub))
    if stub:
        report["stub"] = compact(c2_summary(stub, teachers, seeds=a.seeds))
        report["stub"]["operations"] = operations(stub, "verified_quarantine")
    q, v = rows_of(Path(a.mlx_q)), rows_of(Path(a.mlx_v))
    if q and v:
        s = c2_summary(q + [x for x in v if x["stratum"] == "verified"], teachers, seeds=a.seeds)
        report["mlx"] = compact(s)
        report["mlx"]["operations"] = {"verified_quarantine": operations(q, "verified_quarantine")}
        report["mlx"].update(mlx_extras(q, v, s, a.seeds))
    null = rows_of(Path(a.null))
    if null:
        ns = c2_summary(null, teachers)
        report["null_stub"] = {"seeds": ns["seeds"], "primary": ns["primary"], "secondary": ns["secondary"],
                               "tau_flawed_quarantine": ns["tau_flawed_quarantine"],
                               "per_seed_primary": [p["primary"] for p in ns["per_seed"].values()],
                               "per_seed_secondary": [p["secondary"] for p in ns["per_seed"].values()],
                               "band_primary": [min(p["primary"] for p in ns["per_seed"].values()),
                                                max(p["primary"] for p in ns["per_seed"].values())],
                               "band_secondary": [min(p["secondary"] for p in ns["per_seed"].values()),
                                                  max(p["secondary"] for p in ns["per_seed"].values())]}
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: (v if k in ("scope",) else bool(v)) for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
