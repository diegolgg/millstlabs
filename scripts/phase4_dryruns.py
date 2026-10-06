"""Phase 4: a 20-generation stub dry run per new mechanism, each with a mechanical check that the mechanism actually
acted (not just that the run finished). Writes docs/results/phase4-dryruns.json.

usage: python scripts/phase4_dryruns.py runs/phase4"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.analysis.metrics import load_run  # noqa: E402
from culture.run.config import from_dict  # noqa: E402
from culture.run.runner import run_config  # noqa: E402

BASE = {
    "population": {"groups": 3, "agents_per_group": 4, "seed": 0},
    "evaluation": {"selfplay_games": 20, "crossplay_games": 8, "anchor_games": 8, "between_group_games": 10,
                   "ladder_every": 0, "workers": 0},
    "org": {"routing": "broadcast_group", "verification": {"name": "selfplay", "n": 20}},
    "runner": {"generations": 21},
}
BUDGET = {"budget": {"unit": "calls", "per_group_per_generation": 6}}

RUNS = {
    "islands": {"org": {"topology": {"name": "islands", "migration_rate": 0.25, "interval": 5}}},
    "best_to_neighbor": {"org": {"migration": {"name": "best_to_neighbor", "interval": 4}}},
    "bernoulli": {"org": {"delivery": {"name": "bernoulli", "p": 0.5}}},
    "critical_social_learning": {"org": {"verification": {"name": "critical_social_learning", "theta": 12.0, "n": 20}}},
    "lineage_decay": {"org": {"credit": {"name": "lineage_decay", "gamma": 0.5}}},
    "datamodel_regression": {"org": {"delivery": {"name": "bernoulli", "p": 0.5},
                                     "credit": {"name": "datamodel_regression", "lam": 1.0}}},
    "softmax_floor": {**BUDGET, "org": {"allocation": {"name": "softmax_floor", "T": 1.0, "eps": 0.2}}},
    "nash_relative": {**BUDGET, "org": {"allocation": {"name": "nash_relative", "eps": 0.2}}},
    "shinka_weighted": {"org": {"selection": {"name": "shinka_weighted", "lam": 10.0}}},
    "hgm_clade_ts": {"org": {"selection": {"name": "hgm_clade_ts", "alpha": 0.6}}},
    "variant_switch": {"org": {"environment": {"name": "variant_switch", "at_generation": 10, "variant": "hand4_clues6"}}},
}


def merge(a, b):
    out = json.loads(json.dumps(a))
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def _run(args):
    name, out = args
    cfg = from_dict(merge(BASE, dict(RUNS[name], name=f"p4_{name}", condition=name)))
    run_config(cfg, out)
    return name


def check(name: str, run: dict) -> dict:
    R = run["generations"]
    t = [r.get("teaching", {}) for r in R]
    if name in ("islands", "best_to_neighbor"):
        moves = [len(r.get("migrations", [])) for r in R]
        sizes = [sorted({a: v["group"] for a, v in r["agents"].items()}.values()) for r in R]
        const = all(sorted(np.unique(s, return_counts=True)[1].tolist()) == [4, 4, 4] for s in sizes)
        return {"migrating_generations": [r["generation"] for r, m in zip(R, moves) if m], "moves": int(sum(moves)),
                "agents_constant": all(len(r["agents"]) == 12 for r in R), "group_sizes_constant": const,
                "ok": sum(moves) > 0 and const}
    if name == "bernoulli":
        d = sum(x.get("delivered", 0) for x in t)
        u = sum(x.get("undelivered", 0) for x in t)
        rate = d / max(d + u, 1)
        return {"delivered": d, "undelivered": u, "rate": rate, "ok": abs(rate - 0.5) < 0.1}
    if name == "critical_social_learning":
        rech = sum(x.get("rechecked", 0) for x in t)
        rev = sum(x.get("reverted", 0) for x in t)
        ad = sum(x.get("adopted", 0) for x in t)
        return {"adopted_first": ad, "rechecked": rech, "reverted": rev, "ok": ad > 0 and rech > 0 and rev > 0}
    if name == "lineage_decay":
        indirect = 0
        for r in R:
            direct = {s for w in r.get("credit_windows", []) for s in w.get("teachers", [])}
            indirect += len(set(r.get("credit_increments", {})) - direct)
        return {"indirect_credit_events": indirect, "ok": indirect > 0}
    if name == "datamodel_regression":
        betas = [w["beta"] for r in R for w in r.get("credit_windows", []) if "beta" in w]
        last = betas[-1] if betas else {}
        return {"fits": len(betas), "teachers_in_last_fit": len(last), "last_beta_sample": dict(list(last.items())[:4]),
                "ok": len(betas) > 5 and len(last) >= 6}
    if name in ("softmax_floor", "nash_relative"):
        allocs = [r["allocation"] for r in R if r.get("allocation")]
        sums_ok = all(abs(sum(a.values()) - 18) < 1e-6 for a in allocs)
        floor_ok = all(min(a.values()) >= 0.2 / 3 * 18 - 1e-6 for a in allocs)
        uneven = sum(1 for a in allocs if max(a.values()) - min(a.values()) > 0.5)
        skipped = sum(v.get("skipped_for_budget", 0) for v in R[-1]["counters"].values())
        return {"allocations": len(allocs), "sums_to_budget": sums_ok, "floor_respected": floor_ok,
                "uneven_generations": uneven, "budget_skips": skipped, "last": allocs[-1] if allocs else None,
                "ok": bool(allocs) and sums_ok and floor_ok and uneven > 0}
    if name == "shinka_weighted":
        non_inc = sum(1 for r in R for v in r["agents"].values() if v.get("parent") and v["parent"] != v.get("start"))
        return {"revisions_from_non_incumbent_parent": non_inc, "ok": non_inc > 0}
    if name == "hgm_clade_ts":
        revised = [sum(1 for v in r["agents"].values() if v.get("candidate")) for r in R[1:]]
        return {"revisions_per_generation": revised, "skipped_steps": int(12 * len(revised) - sum(revised)),
                "ok": 0 < sum(revised) < 12 * len(revised)}
    if name == "variant_switch":
        hs = [r["game_params"]["hand_size"] for r in R]
        return {"hand_size_by_generation": hs, "population_mean_before": float(np.mean([r["population_mean"] for r in R[5:10]])),
                "population_mean_after": float(np.mean([r["population_mean"] for r in R[10:15]])),
                "ok": hs[9] == 5 and all(h == 4 for h in hs[10:])}
    return {"ok": False}


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/phase4")
    os.environ["PYTHONHASHSEED"] = "0"
    jobs = [(n, out / n) for n in RUNS if not (out / n / "checkpoint.json").exists()]
    with mp.get_context("spawn").Pool(min(len(RUNS), 11)) as pool:
        pool.map(_run, jobs) if jobs else None
    res = {"scope": "Stub backend dry runs (20 generations after warm start, 3 groups of 4); each check shows the "
                    "mechanism acted. Not evidence about any mechanism's effect on learning.",
           "runs": {n: check(n, load_run(out / n)) for n in RUNS}}
    res["all_ok"] = all(v["ok"] for v in res["runs"].values())
    (ROOT / "docs" / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "results" / "phase4-dryruns.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
