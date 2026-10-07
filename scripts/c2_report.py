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
    fresh = [c for c in calls if not c["cached"]]
    lat = [c["latency_s"] for c in fresh if c.get("latency_s") is not None]
    return {"replays": len(sr), "llm_calls": len(calls), "backend_calls": len(fresh),
            "cache_hits": len(calls) - len(fresh), "backend_seconds": round(sum(lat), 1),
            "seconds_per_backend_call": round(sum(lat) / len(lat), 1) if lat else None,
            "admissible_rate": (sum(bool(x["admissible"]) for x in sr if x.get("admissible") is not None)
                                / max(1, sum(x.get("admissible") is not None for x in sr))),
            "withheld_by_teacher": {t: sum(t in x.get("withheld", []) for x in sr) for t in ("T1", "T2", "T3")}}


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
