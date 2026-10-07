"""P1 epsilon ladder probe: the innovation rate at the IGGI and Piers levels (Design A) and how far one agent climbs
above Piers under accept-if-not-worse (Design B). Config: configs/p1_eps_probe.yaml (design described there).

A parameter measurement for the P1 model, not a confirmatory test. One model block (local Qwen on MLX).

Runs are sequential in this one process, so model calls go one at a time. Resumable: a finished Design A replicate is
skipped (rows.jsonl), a partial run resumes from its checkpoint (its LLM cache replays finished calls), and the chain
continues from its last checkpointed generation. Partial results are written after every replicate and generation.

Usage:
  .venv/bin/python scripts/p1_eps_probe.py --backend stub --replicates 3 --chain-generations 4   # pipeline check, $0
  caffeinate -i -s .venv/bin/python scripts/p1_eps_probe.py                                       # MLX run
  .venv/bin/python scripts/p1_eps_probe.py --summarize-only                                       # rebuild the JSON
"""

from __future__ import annotations

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":  # same determinism rule as `python -m culture.run`
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from culture.analysis.innovation import beta_posterior  # noqa: E402
from culture.artifacts.store import ArtifactStore  # noqa: E402
from culture.game.seeds import generation_seeds  # noqa: E402
from culture.run.config import from_dict  # noqa: E402
from culture.run.context import RunContext  # noqa: E402
from culture.run.runner import Runner  # noqa: E402

QUANTS = (0.1, 0.25, 0.5, 0.75, 0.9)
SCOPE = ("Scope: P1 epsilon ladder probe, one model block (Qwen3.6-35B-A3B 4-bit on the local MLX server, seeded, "
         "temperature 0), a parameter measurement for the P1 accumulation model, not pre-registered as a confirmatory "
         "test. Design A: independent single revisions of a fixed anchor incumbent (Piers, IGGI), replicates vary the "
         "experiment seed (feedback traces); Design B: one agent's 40-generation chain from Piers under "
         "accept-if-not-worse. Context: feedback only (no messages, no corpus).")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def describe(x) -> dict[str, Any]:
    x = np.sort(np.asarray(x, float))
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else None,
            "quantiles": {str(q): float(np.quantile(x, q)) for q in QUANTS}, "values": [round(float(v), 4) for v in x]}


def check_server(base_url: str) -> None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=10) as r:
            body = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"MLX server at {base_url} does not answer /models ({e}); not starting it. Stopping.")
    log(f"server answers: {[m.get('id') for m in body.get('data', [])]}")


# ------------------------------------------------------------------------------------------------ configs
def run_cfg(spec: dict, condition: str, experiment_seed: int, anchor: str, generations: int, pop_seed: int = 0):
    return from_dict({
        "name": spec["name"], "condition": condition, "experiment_seed": experiment_seed,
        "population": {"groups": 1, "agents_per_group": 1, "seed": pop_seed, "warm_start": anchor},
        "evaluation": copy.deepcopy(spec["evaluation"]), "llm": copy.deepcopy(spec["llm"]),
        "org": {"routing": "none", "verification": "none"}, "corpus": {"enabled": False},
        "runner": {"generations": generations},
    })


class Scorer:
    """Scores artifacts of any run on one common deal set (engine only, no model)."""

    def __init__(self, spec: dict):
        ce = spec["common_eval"]
        self.seeds = generation_seeds(int(ce["experiment_seed"]), int(ce["generation"]), int(ce["games"]), "eval").seeds
        self.ctx = RunContext(from_dict({"name": "p1_eps_score", "llm": {"backend": "stub", "cache_mode": "off"},
                                         "evaluation": {"workers": spec["evaluation"].get("workers", 0)}}), None)
        self.memo: dict[str, float] = {}

    def score(self, run_dir: Path, aid: str | None) -> float | None:
        if not aid:
            return None
        if aid not in self.memo:
            self.ctx.add_artifact(ArtifactStore(run_dir / "artifacts").get(aid))
            self.memo[aid] = round(self.ctx.selfplay(aid, self.seeds)[0], 4)
        return self.memo[aid]

    def close(self):
        self.ctx.close()


def gen_rows(run_dir: Path) -> list[dict]:
    p = run_dir / "generations.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def revision_row(rec: dict) -> dict[str, Any]:
    (a, v), = rec["agents"].items()
    cand = v.get("candidate")
    adm = cand is not None and v.get("candidate_score") is not None and v.get("parent_score") is not None
    delta = float(v["candidate_score"]) - float(v["parent_score"]) if adm else 0.0
    gen = (rec.get("generalization") or {}).get(cand, {}) if cand else {}
    return {"generation": rec["generation"], "parent": v.get("start"), "parent_score": v.get("parent_score"),
            "candidate": cand, "candidate_score": v.get("candidate_score"), "admissible": adm,
            "delta": round(delta, 4), "accepted": bool((rec.get("accepted") or {}).get(a)),
            "incumbent": v["incumbent"], "incumbent_score": v["score"],
            "candidate_feedback_score": gen.get("feedback"), "context": (v.get("revise_context") or {}),
            "calls_cumulative": rec["cost"]["calls"], "wall_seconds": rec.get("volatile", {}).get("wall_seconds")}


# ------------------------------------------------------------------------------------------------ designs
def design_a(spec, out: Path, scorer: Scorer, replicates: list[int], results_cb) -> None:
    A = spec["design_a"]
    rows_p = out / "A" / "rows.jsonl"
    rows_p.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["anchor"], r["replicate"]) for r in _jsonl(rows_p)}
    for r in replicates:  # replicate-major, so a partial run stays balanced across anchors
        for anchor in A["anchors"]:
            if (anchor, r) in done:
                continue
            d = out / "A" / anchor / f"r{r}"
            cfg = run_cfg(spec, f"a_{anchor}_r{r}", int(A["experiment_seed"]) + r, anchor, int(A["generations"]))
            t0 = time.perf_counter()
            with Runner(cfg, d) as R:
                R.run(int(A["generations"]))
            recs = gen_rows(d)
            g1 = [x for x in recs if x["generation"] == 1]
            if not g1:
                raise RuntimeError(f"{d}: no generation-1 record")
            row = {"anchor": anchor, "replicate": r, "experiment_seed": cfg.experiment_seed, **revision_row(g1[0])}
            row["anchor_heldout_g0"] = recs[0]["agents"][next(iter(recs[0]["agents"]))]["score"]
            row["parent_common"] = scorer.score(d, row["parent"])
            row["candidate_common"] = scorer.score(d, row["candidate"])
            row["delta_common"] = (round(row["candidate_common"] - row["parent_common"], 4)
                                   if row["candidate_common"] is not None else 0.0)
            row["run_seconds"] = round(time.perf_counter() - t0, 1)
            with open(rows_p, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
            log(f"A {anchor} r{r}: parent {row['parent_score']} candidate {row['candidate_score']} delta "
                f"{row['delta']:+.2f} (common {row['delta_common']:+.2f}) admissible {row['admissible']} "
                f"({row['run_seconds']}s)")
            results_cb()


def design_b(spec, out: Path, scorer: Scorer, generations: int, results_cb) -> None:
    B = spec["design_b"]
    for i, es in enumerate(B["experiment_seeds"]):
        d = out / "B" / f"s{es}"
        cfg = run_cfg(spec, f"b_chain_s{es}", int(es), B["anchor"], generations, pop_seed=i)
        with Runner(cfg, d) as R:
            while R.ctx.generation + 1 < generations:
                g = R.ctx.generation + 1
                t0 = time.perf_counter()
                R.run(g + 1)
                rec = gen_rows(d)[-1]
                if g >= 1:
                    rw = revision_row(rec)
                    log(f"B s{es} gen {g}: parent {rw['parent_score']} candidate {rw['candidate_score']} delta "
                        f"{rw['delta']:+.2f} accepted {rw['accepted']} incumbent {rw['incumbent_score']:.2f} "
                        f"({time.perf_counter() - t0:.0f}s)")
                results_cb()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


# ------------------------------------------------------------------------------------------------ summary
def eps_block(deltas, margins) -> dict[str, Any]:
    d = np.asarray(deltas, float)
    out = {"n": int(d.size), "deltas": describe(d)}
    for m in margins:
        out[str(m)] = {"epsilon": beta_posterior(int((d > m).sum()), int(d.size)) if d.size else None,
                       "gain_given_gain": describe(d[d > m]),
                       "loss_given_loss": describe(d[d < -m])}
    out["loss_given_any_loss"] = describe(d[d < 0])
    return out


def summarize(spec, out: Path, scorer: Scorer | None, backend: str) -> dict[str, Any]:
    margins = spec["margins"]
    res: dict[str, Any] = {"scope": SCOPE, "backend": backend, "generated_by": "scripts/p1_eps_probe.py",
                           "config": "configs/p1_eps_probe.yaml", "run_dir": str(out.relative_to(ROOT)),
                           "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "rules": {"delta": "candidate minus parent self-play on the generation's held-out deals "
                                              "(paired, 300 deals); an inadmissible revision counts as delta 0",
                                     "epsilon": "P(delta > margin), Beta(1,1) prior, 95% equal-tailed interval",
                                     "common": "candidate and parent re-scored on one common 300-deal set "
                                               f"(experiment seed {spec['common_eval']['experiment_seed']}, "
                                               "generation 1), so levels compare across replicates"}}
    rows = _jsonl(out / "A" / "rows.jsonl")
    A = {}
    for anchor in spec["design_a"]["anchors"]:
        rs = sorted((r for r in rows if r["anchor"] == anchor), key=lambda r: r["replicate"])
        if not rs:
            continue
        lev = [r["parent_score"] for r in rs if r["parent_score"] is not None]
        A[anchor] = {
            "n_replicates": len(rs), "n_admissible": sum(r["admissible"] for r in rs),
            "n_accepted": sum(r["accepted"] for r in rs),
            "level_heldout": describe(lev), "level_common": rs[0]["parent_common"],
            "heldout": eps_block([r["delta"] for r in rs], margins),
            "common_deals_sensitivity": eps_block([r["delta_common"] for r in rs], margins),
            "candidate_common_scores": describe([r["candidate_common"] for r in rs if r["candidate_common"] is not None]),
            "rows": rs}
    res["design_a"] = A

    B = {}
    bins = [float(x) for x in spec["design_b"]["level_bins"]]
    labels = [f"<{bins[0]:g}"] + [f"{bins[i]:g}-{bins[i + 1]:g}" for i in range(len(bins) - 1)] + [f">={bins[-1]:g}"]
    pooled: list[dict] = []
    for es in spec["design_b"]["experiment_seeds"]:
        d = out / "B" / f"s{es}"
        recs = gen_rows(d)
        if not recs:
            continue
        traj = []
        for rec in recs:
            (a, v), = rec["agents"].items()
            item = {"generation": rec["generation"], "incumbent": v["incumbent"], "incumbent_heldout": v["score"]}
            if scorer is not None:
                item["incumbent_common"] = scorer.score(d, v["incumbent"])
            if rec["generation"] >= 1:
                rw = revision_row(rec)
                item.update({k: rw[k] for k in ("parent_score", "candidate_score", "delta", "accepted", "admissible")})
                pooled.append({"seed": es, **item})
            traj.append(item)
        com = [t.get("incumbent_common") for t in traj if t.get("incumbent_common") is not None]
        B[str(es)] = {"generations_done": recs[-1]["generation"], "trajectory": traj,
                      "start_common": com[0] if com else None, "final_common": com[-1] if com else None,
                      "max_common": max(com) if com else None,
                      "accepted_revisions": sum(1 for t in traj if t.get("accepted"))}
    binned = {}
    for key in ("parent_heldout", "parent_common"):
        groups: dict[str, list[dict]] = {lab: [] for lab in labels}
        for p in pooled:
            lv = p["parent_score"] if key == "parent_heldout" else None
            if key == "parent_common":
                # the parent is the previous generation's incumbent; use its common score
                prev = [t for t in B[str(p["seed"])]["trajectory"] if t["generation"] == p["generation"] - 1]
                lv = prev[0].get("incumbent_common") if prev else None
            if lv is None:
                continue
            idx = int(np.searchsorted(bins, lv, side="right"))
            groups[labels[idx]].append({**p, "level": lv})
        binned[key] = {lab: {"n": len(g), "level_mean": float(np.mean([x["level"] for x in g])) if g else None,
                             **eps_block([x["delta"] for x in g], margins)} for lab, g in groups.items()}
    res["design_b"] = {"chains": B, "binned_epsilon": binned,
                       "binning_note": "primary: the parent's held-out score in that generation (as specified); "
                                       "sensitivity: the parent's common-deal score. Revisions within a chain are "
                                       "dependent (same lineage), so the Beta intervals are optimistic."}
    res["ladder"] = ladder(res, margins)
    return res


def ladder(res: dict, margins) -> list[dict]:
    """Points for the P1 epsilon function: Design A anchors, then the chain's bins at or above 17 (primary binning)."""
    pts = []
    for anchor, a in res.get("design_a", {}).items():
        if a["level_heldout"].get("n"):
            pts.append({"source": f"design_a:{anchor}", "level": a["level_heldout"]["mean"], "n": a["n_replicates"],
                        **{str(m): {"epsilon": a["heldout"][str(m)]["epsilon"],
                                    "gains": a["heldout"][str(m)]["gain_given_gain"].get("values", [])}
                           for m in margins}})
    for lab, b in res.get("design_b", {}).get("binned_epsilon", {}).get("parent_heldout", {}).items():
        if b["n"] and b["level_mean"] is not None and b["level_mean"] >= 17.0:
            pts.append({"source": f"design_b:{lab}", "level": b["level_mean"], "n": b["n"],
                        **{str(m): {"epsilon": b[str(m)]["epsilon"],
                                    "gains": b[str(m)]["gain_given_gain"].get("values", [])} for m in margins}})
    return sorted(pts, key=lambda p: p["level"])


def write_results(spec, out, scorer, backend, path: Path) -> None:
    res = summarize(spec, out, scorer, backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "p1_eps_probe.yaml"))
    ap.add_argument("--backend", choices=("mlx", "stub"), default="mlx")
    ap.add_argument("--out", default=None, help="run directory (default runs/p1_eps_probe[_stub])")
    ap.add_argument("--results", default=None, help="results JSON (default docs/results/p1-eps-ladder.json; stub: "
                                                    "inside the run directory)")
    ap.add_argument("--replicates", type=int, default=None, help="use only the first k Design A replicates")
    ap.add_argument("--chain-generations", type=int, default=None, help="override Design B generations (incl. 0)")
    ap.add_argument("--designs", default="AB")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()

    spec = yaml.safe_load(Path(a.config).read_text())
    if a.backend == "stub":
        spec["llm"] = {"backend": "stub", "cache_mode": "record"}
        spec["name"] = spec["name"] + "_stub"
    out = Path(a.out) if a.out else ROOT / "runs" / ("p1_eps_probe" + ("_stub" if a.backend == "stub" else ""))
    out = out.resolve()
    results = Path(a.results) if a.results else (out / "p1-eps-ladder.json" if a.backend == "stub"
                                                 else ROOT / "docs" / "results" / "p1-eps-ladder.json")
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    scorer = Scorer(spec)
    try:
        if not a.summarize_only:
            if a.backend == "mlx":
                check_server(spec["llm"]["base_url"])
            reps = spec["design_a"]["replicates"][: a.replicates] if a.replicates else spec["design_a"]["replicates"]
            cb = lambda: write_results(spec, out, scorer, a.backend, results)  # noqa: E731
            if "A" in a.designs:
                log(f"Design A: {len(reps)} replicates x {spec['design_a']['anchors']}")
                design_a(spec, out, scorer, reps, cb)
            if "B" in a.designs:
                gens = a.chain_generations or int(spec["design_b"]["generations"])
                log(f"Design B: {spec['design_b']['experiment_seeds']} x {gens - 1} revise generations")
                design_b(spec, out, scorer, gens, cb)
        write_results(spec, out, scorer, a.backend, results)
        log(f"wrote {results}")
        res = json.loads(results.read_text())
        for anchor, x in res["design_a"].items():
            e = x["heldout"]["0.5"]["epsilon"]
            log(f"A {anchor}: level {x['level_heldout'].get('mean', float('nan')):.2f}, eps(0.5) {e['successes']}/{e['n']}"
                f" = {e['mean']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]")
        for lab, b in res["design_b"]["binned_epsilon"]["parent_heldout"].items():
            if b["n"]:
                e = b["0.5"]["epsilon"]
                log(f"B bin {lab}: n {b['n']}, eps(0.5) {e['successes']}/{e['n']} = {e['mean']:.3f}")
    finally:
        scorer.close()
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
