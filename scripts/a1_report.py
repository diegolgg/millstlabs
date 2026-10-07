"""Assemble the A1 parameter report (Overnight 2, step 5): copy loss alpha, copy dispersion beta, ratio rho and the
accumulation prediction delta(N = 4) per medium, from runs/a1-params (MLX) and runs/a1-stub (pipeline proof).
Writes docs/results/a1-params.json.

Usage: python scripts/a1_report.py [--mlx runs/a1-params] [--stub runs/a1-stub]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.run.transmission import load_spec, summarize  # noqa: E402


def load(d: Path):
    if not (d / "rows.jsonl").exists():
        return None
    spec = load_spec(json.loads((d / "spec.json").read_text()))
    rows = [json.loads(x) for x in (d / "rows.jsonl").read_text().splitlines() if x.strip()]
    s = summarize(rows, spec)
    for mu, m in s["media"].items():
        rs = [r for r in rows if r["medium"] == mu]
        m["seeds"] = sorted(r["replicate"] for r in rs)
        m["repair_calls"] = sum(c["tag"] == "repair" and not c["cached"] for r in rs for c in r["calls"])
        m["copy_illegal_rate_mean"] = sum(r["copy_illegal_rate"] for r in rs) / len(rs)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", default=str(ROOT / "runs" / "a1-params"))
    ap.add_argument("--stub", default=str(ROOT / "runs" / "a1-stub"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "a1-params.json"))
    a = ap.parse_args()
    mlx, stub = load(Path(a.mlx)), load(Path(a.stub))
    n = {mu: m["n"] for mu, m in (mlx or {"media": {}})["media"].items()}
    report = {
        "scope": "Scope: A1 parameters only (files/prereg/A1-transmission-fidelity.md sections 2 and 3, amendments "
                 "A1-1 and A1-2); the check populations (section 4) were not run, so the accumulation prediction is "
                 "not tested and the model-support rule is not judged. One student, teacher Piers, one revision per "
                 f"replicate, one model block (Qwen3.6-35B-A3B 4-bit on MLX, seeded, temperature 0), replicates per "
                 f"medium {n}; student and teacher share one rule-bot template, so copying is easier than between "
                 "unrelated codebases. The stub section only proves the pipeline.",
        "design": "Media code (bot.py only), prose (conventions.md only), both. Content-free cover note and the "
                  "harness's evidence for Piers in every medium. Verification none, adoption never; S' = the one "
                  "revision's artifact (S0 if the revision failed). Replicate r: experiment seed 300 + r (feedback "
                  "traces in the prompt), sampling seed r. All copies, Piers and S0 scored on one common set of 300 "
                  "evaluation deals (experiment seed 299). alpha = v_T - E[V(S',S')]; beta = sd(V(S',S')) sqrt(6)/pi; "
                  "rho = E[V(S',T)] / v_T; delta(N) = -alpha + beta (0.5772 + ln N), N = 4; percentile bootstrap over "
                  "replicates (2,000 resamples).",
        "verdicts": {
            "medium_ordering": "judged (section 5 does not require two blocks for this rule): see "
                               "mlx.medium_ordering.met; false means no claim that code transmits with less loss "
                               "than prose",
            "model_support": "not judged: the check populations (section 4) were not run"},
        "caveat_distribution": (
            "The model's accumulation prediction assumes Gumbel (light-tailed, unimodal) copy noise. The observed "
            "V(S', S') is a point mass near v_T (behaviourally exact copies) plus a heavy left tail of copies that "
            "score near the weak incumbent (2 to 5 points). beta is computed from the SD, so the left tail inflates "
            "it, and a larger beta raises delta(N): the positive predicted signs lean on copies that selection would "
            "discard. Read delta(N = 4) as an upper bound until the check populations test it."),
        "mlx": mlx, "stub": stub}
    Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    if mlx:
        for mu, m in mlx["media"].items():
            p = m["parameters"]
            print(mu, m["n"], "fail", m["failed_revisions"], "same rules", m["same_rule_list_as_teacher"],
                  {k: (round(v["point"], 2), round(v["lo"], 2), round(v["hi"], 2)) for k, v in p.items()
                   if isinstance(v, dict) and "point" in v}, p["predicted_sign_at_N"])
        print(mlx.get("medium_ordering"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
