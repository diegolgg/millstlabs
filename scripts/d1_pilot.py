"""D1 pilot on the null stub (fix round 1, step 16): breakdown point of verification against sabotaged messages.

Grid: epsilon in {0, 0.1, 0.2, 0.3, 0.5} x verification on (selfplay n=200, replace_if_better) / off x population seeds
0-4, all on the same deals. Y_G = population mean held-out self-play at the last generation. Writes
docs/results/d1-pilot.json and d1-pilot.png. Zero model calls (null stub).
Usage: python scripts/d1_pilot.py [--processes 8] [--seeds 5]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.analysis.breakdown import summarize  # noqa: E402
from culture.run.config import from_dict, load_config  # noqa: E402

EPS = (0.0, 0.1, 0.2, 0.3, 0.5)
ARMS = {"on": {"verification": {"name": "selfplay", "n": 200}}, "off": {"verification": "none"}}


def _run(job):
    from culture.run.runner import run_config

    arm, eps, seed, cfg, out = job
    recs = run_config(cfg, out)
    last = json.loads((Path(out) / "generations.jsonl").read_text().splitlines()[-1])
    sab = sum(json.loads(x).get("teaching", {}).get("sabotaged", 0) for x in (Path(out) / "generations.jsonl").read_text().splitlines())
    sab_ad = sum(json.loads(x).get("teaching", {}).get("sabotaged_adopted", 0)
                 for x in (Path(out) / "generations.jsonl").read_text().splitlines())
    return arm, eps, seed, last["population_mean"], sab, sab_ad, len(recs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--runs", default=str(ROOT / "runs" / "d1_pilot"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "d1-pilot.json"))
    a = ap.parse_args()
    t0 = time.time()
    os.environ["PYTHONHASHSEED"] = "0"
    base = load_config(ROOT / "configs" / "d1_pilot.yaml")
    jobs = []
    for arm, over in ARMS.items():
        for eps in EPS:
            for s in range(a.seeds):
                cfg = from_dict({"condition": f"{arm}_eps{eps}", "org": over, "sabotage": {"epsilon": eps},
                                 "population": {"seed": s}}, base)
                jobs.append((arm, eps, s, cfg, Path(a.runs) / arm / f"eps{eps}" / f"p{s}"))
    with mp.get_context("spawn").Pool(a.processes) as pool:
        rows = pool.map(_run, jobs)
    results: dict = {arm: {e: [] for e in EPS} for arm in ARMS}
    exposure: dict = {arm: {str(e): {"sabotaged": 0, "sabotaged_adopted": 0} for e in EPS} for arm in ARMS}
    for arm, eps, seed, y, sab, sab_ad, _ in sorted(rows, key=lambda r: (r[0], r[1], r[2])):
        results[arm][eps].append(y)
        exposure[arm][str(eps)]["sabotaged"] += sab
        exposure[arm][str(eps)]["sabotaged_adopted"] += sab_ad
    summ = summarize(results)
    summ["exposure"] = exposure
    summ["per_seed_Y_G"] = {arm: {str(e): v for e, v in cells.items()} for arm, cells in results.items()}
    summ["design"] = {"students": 4, "generations": 10, "eps": list(EPS), "seeds": a.seeds,
                      "arms": {"on": "selfplay(n=200), replace_if_better", "off": "none (adopt what is delivered)"},
                      "backend": "stub, null mode (revisions keep the rule list)"}
    summ["wall_seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(summ, indent=1, sort_keys=True) + "\n")
    figure(summ, Path(a.out).with_suffix(".png"))
    print(json.dumps({arm: {k: v for k, v in s.items() if k != "null_band"} for arm, s in summ["arms"].items()},
                     indent=1))
    print("primary contrast", summ.get("primary_contrast"))
    return 0


def figure(summ: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from culture.analysis.figures import MUTED, SERIES, _style

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (arm, s) in enumerate(sorted(summ["arms"].items())):
        col = SERIES[i]
        for e in s["eps"]:
            ys = summ["per_seed_Y_G"][arm][str(e)]
            ax.scatter([e + (i - 0.5) * 0.01] * len(ys), ys, s=14, color=col, alpha=0.6, edgecolors="none")
        ax.plot(s["eps"], s["mean_Y_G"], color=col, linewidth=2, label=f"verification {arm}")
        ax.axhline(0.7 * s["mean_Y_G"][0], color=col, linewidth=0.8, linestyle=(0, (3, 3)))
    _style(ax, "Population skill at generation 10 vs sabotage rate (null stub)", "sabotage probability epsilon",
           "population mean self-play")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.0, ax.get_ylim()[0], " dashed: 70% of the epsilon = 0 mean", color=MUTED, fontsize=7, va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
