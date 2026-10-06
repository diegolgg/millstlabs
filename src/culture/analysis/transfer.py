"""Stage-1 analysis (queue item 1): student score versus cumulative tokens per condition, and paired deltas.

Student = the agent with the lower generation-0 score. Warm starts are paired across conditions (same seed tags), so
the student is the same agent, holding the same artifact, in every condition of a population seed; `pairing_checks`
verifies that and that final evaluations used the same deals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..evaluate import stats
from .metrics import load_run


def _final_evals(run_dir: Path) -> dict[str, Any]:
    ck = json.loads((run_dir / "checkpoint.json").read_text())["state"]
    return {"evals": ck["evals"], "agents": ck["agents"]}


def load_experiment(root: str | Path) -> dict[str, dict[int, dict[str, Any]]]:
    root = Path(root)
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        for pdir in sorted(cdir.glob("p*")):
            run = load_run(pdir)
            run["final"] = _final_evals(pdir)
            out.setdefault(cdir.name, {})[int(pdir.name[1:])] = run
    return out


def student_of(run: dict[str, Any]) -> str:
    g0 = run["generations"][0]["agents"]
    return min(sorted(g0), key=lambda a: g0[a]["score"])


def pairing_checks(exp: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    conds = sorted(exp)
    seeds = sorted(exp[conds[0]])
    same_warm, same_student, same_deals = True, True, True
    for ps in seeds:
        ref = exp[conds[0]][ps]
        w0 = {a: v["incumbent"] for a, v in ref["generations"][0]["agents"].items()}
        for c in conds[1:]:
            run = exp[c][ps]
            same_warm &= {a: v["incumbent"] for a, v in run["generations"][0]["agents"].items()} == w0
            same_student &= student_of(run) == student_of(ref)
            fa = run["final"]["evals"][run["final"]["agents"][student_of(run)]["incumbent"]]
            fr = ref["final"]["evals"][ref["final"]["agents"][student_of(ref)]["incumbent"]]
            same_deals &= fa["seed_base"] == fr["seed_base"] and fa["n_games"] == fr["n_games"]
    return {"warm_starts_identical": bool(same_warm), "student_identical": bool(same_student),
            "final_deals_identical": bool(same_deals), "conditions": conds, "population_seeds": seeds}


def curves(exp, n_boot: int = 2000) -> dict[str, dict[str, np.ndarray]]:
    """Per condition: x = cumulative population tokens (mean over seeds), y = IQM over seeds of the student's score,
    band = bootstrap over population seeds."""
    out = {}
    for c, runs in exp.items():
        ys, xs = [], []
        for ps, run in sorted(runs.items()):
            st = student_of(run)
            ys.append([r["agents"][st]["score"] for r in run["generations"]])
            xs.append([r["cost"]["tokens"] for r in run["generations"]])
        Y, X = np.array(ys, float), np.array(xs, float)
        iqm = np.array([stats.iqm(Y[:, g]) for g in range(Y.shape[1])])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, Y.shape[0], size=(n_boot, Y.shape[0]))
        boots = np.array([[stats.iqm(Y[i, g]) for g in range(Y.shape[1])] for i in idx])
        out[c] = {"tokens": X.mean(axis=0), "iqm": iqm, "lo": np.quantile(boots, 0.025, axis=0),
                  "hi": np.quantile(boots, 0.975, axis=0), "per_seed": Y, "per_seed_tokens": X}
    return out


def paired_deltas(exp, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Game-level paired difference of the students' final artifacts on the identical final-generation deals,
    pooled over population seeds with a bootstrap stratified by population seed. Also the per-seed difference in the
    area under the student-score curve (mean over generations)."""
    rows = []
    for a, b in pairs:
        strata, auc = [], []
        for ps in sorted(exp[a]):
            ra, rb = exp[a][ps], exp[b][ps]
            sa, sb = student_of(ra), student_of(rb)
            ea = ra["final"]["evals"][ra["final"]["agents"][sa]["incumbent"]]
            eb = rb["final"]["evals"][rb["final"]["agents"][sb]["incumbent"]]
            strata.append(np.array(ea["selfplay_scores"], float) - np.array(eb["selfplay_scores"], float))
            auc.append(np.mean([r["agents"][sa]["score"] for r in ra["generations"]])
                       - np.mean([r["agents"][sb]["score"] for r in rb["generations"]]))
        point, (lo, hi), _ = stats.stratified_bootstrap(strata, stat=np.mean, n_boot=2000)
        tok_a = np.mean([exp[a][ps]["generations"][-1]["cost"]["tokens"] for ps in exp[a]])
        tok_b = np.mean([exp[b][ps]["generations"][-1]["cost"]["tokens"] for ps in exp[b]])
        rows.append({"label": f"{a} - {b}", "mean": point, "lo": lo, "hi": hi,
                     "n_games": int(sum(s.size for s in strata)), "per_seed_mean": [float(s.mean()) for s in strata],
                     "auc_delta_per_seed": [float(x) for x in auc], "tokens_ratio": float(tok_a / tok_b)})
    return rows
