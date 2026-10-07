"""Assemble the C1 pilot report (fix round 1, step 14) from the single-student runs: the stub pilot (pipeline proof) and
the MLX pilot (Qwen3.6-35B-A3B 4-bit on mlx_lm.server, temperature 0, seed per replicate). Writes
docs/results/c1-pilot.json, the per-backend figure and table.

Usage: python scripts/c1_report.py [--stub runs/c1_stub] [--mlx runs/c1_mlx]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.analysis import credit as C  # noqa: E402
from culture.run.single_student import load_spec  # noqa: E402


def compact(summary: dict) -> dict:
    out = {"teachers": summary["teachers"], "strata": {}}
    for stratum, s in summary["strata"].items():
        out["strata"][stratum] = {
            "per_seed": {r: {"shapley": [round(x, 3) for x in v["shapley"]], "tau_itt": [round(x, 3) for x in v["tau"]],
                             "adopter_effect": [None if x is None else round(x, 3) for x in v["adopter_effect"]],
                             "v(none)": v["v"]["0"], "v(all)": v["v"][str((1 << len(summary['teachers'])) - 1)]}
                         for r, v in s["per_replicate"].items()},
            "estimators": s["estimators"], "primary_contrast": s["primary_contrast"], "operations": s["operations"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", default=str(ROOT / "runs" / "c1_stub"))
    ap.add_argument("--mlx", default=str(ROOT / "runs" / "c1_mlx"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "c1-pilot.json"))
    a = ap.parse_args()
    report = {"design": "k = 3 (Piers; Flawed with persuasive prose and honest evidence; the student's own bot re-sent), "
                        "exhaustive delivery (8 replays per seed per stratum), 5 replicate seeds, strata verification "
                        "on (selfplay n=200, replace_if_better) and off; outcome = held-out paired self-play "
                        "improvement on 300 deals. Oracle = exact Shapley over the 8 replays."}
    res = Path(a.out).parent
    for name, d in (("stub", a.stub), ("mlx", a.mlx)):
        d = Path(d)
        if not (d / "replays.jsonl").exists():
            report[name] = None
            continue
        spec = load_spec(json.loads((d / "spec.json").read_text()))
        rows = [json.loads(x) for x in (d / "replays.jsonl").read_text().splitlines() if x.strip()]
        summ = C.summarize(rows, spec)
        report[name] = compact(summ)
        C.figure(summ).savefig(res / f"c1-pilot-{name}.png", dpi=150)
        (res / f"c1-pilot-{name}-table.md").write_text(C.table(summ))
    Path(a.out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: (v if k == "design" else bool(v)) for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
