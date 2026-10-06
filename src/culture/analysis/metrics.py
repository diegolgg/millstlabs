"""Per-generation metrics from a run's JSONL logs and ledger (no re-simulation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def load_run(run_dir: str | Path) -> dict[str, Any]:
    d = Path(run_dir)
    return {
        "dir": str(d),
        "config": json.loads((d / "config.json").read_text())["config"],
        "manifest": json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {},
        "generations": _jsonl(d / "generations.jsonl"),
        "ledger": _jsonl(d / "ledger.jsonl"),
        "messages": _jsonl(d / "messages.jsonl"),
        "artifacts": _jsonl(d / "artifacts.jsonl"),
        "touch": _jsonl(d / "touch.jsonl"),
    }


def series(run: dict[str, Any]) -> dict[str, np.ndarray]:
    recs = run["generations"]
    g = np.array([r["generation"] for r in recs])
    out: dict[str, Any] = {
        "generation": g,
        "tokens": np.array([r["cost"]["tokens"] for r in recs], float),
        "spend_usd": np.array([r["cost"]["spend_usd"] for r in recs], float),
        "population_best": np.array([r["population_best"] for r in recs], float),
        "population_mean": np.array([r["population_mean"] for r in recs], float),
        "anchor_mean": np.array([np.mean([a["anchor_score"] for a in r["agents"].values()]) for r in recs], float),
        "distinct_incumbents": np.array([r["diversity"]["distinct_incumbents"] for r in recs], float),
        "code_clusters": np.array([r["diversity"]["code_clusters"] for r in recs], float),
        "hc": np.array([np.nan if r["hc"]["hc"] is None else r["hc"]["hc"] for r in recs], float),
    }
    bg = [r.get("between_group") for r in recs]
    out["between_offdiag"] = np.array([b["offdiag_mean"] if b else np.nan for b in bg], float)
    out["between_diag"] = np.array([b["diag_mean"] if b else np.nan for b in bg], float)
    out["within_crossplay"] = np.array([np.nan if r["within_group_crossplay"] is None else r["within_group_crossplay"]
                                       for r in recs], float)
    tags = sorted({t for r in recs for t in r["cost"]["by_tag"]})
    out["tokens_by_tag"] = {t: np.array([r["cost"]["by_tag"].get(t, {}).get("tokens", 0) for r in recs], float)
                            for t in tags}
    teach = [r.get("teaching", {}) for r in recs]
    for k in ("sent", "delivered", "verified", "passed", "adopted", "merged", "rejected", "reverted"):
        out[f"teach_{k}"] = np.array([t.get(k, 0) for t in teach], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["adoption_rate"] = out["teach_adopted"] / out["teach_delivered"]
        out["verification_pass_rate"] = out["teach_passed"] / out["teach_verified"]
    groups = sorted(recs[0]["groups"]) if recs else []
    out["group_best"] = {gn: np.array([r["groups"].get(gn, {}).get("best", np.nan) for r in recs], float) for gn in groups}
    agents = sorted(recs[-1]["agents"]) if recs else []
    out["agent_score"] = {a: np.array([r["agents"].get(a, {}).get("score", np.nan) for r in recs], float) for a in agents}
    out["agent_credit"] = {a: np.array([r["agents"].get(a, {}).get("credit", np.nan) for r in recs], float) for a in agents}
    return out


def slope(y: np.ndarray, x: np.ndarray | None = None) -> float:
    y = np.asarray(y, float)
    x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, float)
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def changepoints(y: np.ndarray, max_k: int = 3, min_seg: int = 5, penalty: float | None = None) -> list[int]:
    """Binary segmentation on the mean with a BIC-style penalty (indices where a new segment starts)."""
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    n = y.size
    if n < 2 * min_seg:
        return []
    pen = penalty if penalty is not None else 2 * np.log(n) * max(np.var(y), 1e-9)

    def cost(a, b):
        s = y[a:b]
        return float(((s - s.mean()) ** 2).sum())

    segs, cps = [(0, n)], []
    for _ in range(max_k):
        best = None
        for a, b in segs:
            base = cost(a, b)
            for t in range(a + min_seg, b - min_seg + 1):
                gain = base - cost(a, t) - cost(t, b)
                if gain > pen and (best is None or gain > best[0]):
                    best = (gain, a, b, t)
        if best is None:
            break
        _, a, b, t = best
        segs.remove((a, b))
        segs += [(a, t), (t, b)]
        cps.append(t)
    return sorted(cps)


def trajectory_shape(s: dict[str, Any], key: str = "population_mean", threshold: float | None = None) -> dict[str, Any]:
    y = s[key]
    n = y.size
    third = max(n // 3, 2)
    out = {"slope_first_third": slope(y[:third]), "slope_last_third": slope(y[-third:]),
           "changepoints": changepoints(y)}
    if threshold is not None:
        hit = np.where(s["between_offdiag"] >= threshold)[0]
        out["first_gen_between_above_threshold"] = int(s["generation"][hit[0]]) if hit.size else None
    return out


def adoption_edges(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Edges sender -> receiver for adopted or merged messages, with generation and groups."""
    out = []
    for m in run["messages"]:
        if m.get("decision") in ("adopted", "merged"):
            out.append({"sender": m["sender"], "receiver": m["receiver"], "generation": m["generation"],
                        "sender_group": m.get("sender_group"), "receiver_group": m.get("receiver_group")})
    return out


def retained_innovations(run: dict[str, Any], k: int = 10) -> int:
    """Artifacts that beat the archive best when first evaluated and still have descendants k generations later.
    Scores come from the generation records (candidate and incumbent self-play on that generation's deals)."""
    arts = {a["id"]: a for a in run["artifacts"]}
    children: dict[str, list[str]] = {}
    for a in run["artifacts"]:
        for p in a.get("parents", []):
            children.setdefault(p, []).append(a["id"])
    first_score: dict[str, tuple[int, float]] = {}
    for r in run["generations"]:
        for v in r["agents"].values():
            for aid, sc in ((v.get("candidate"), v.get("candidate_score")), (v.get("incumbent"), v.get("score"))):
                if aid and sc is not None and aid not in first_score:
                    first_score[aid] = (r["generation"], sc)
    best, count = -np.inf, 0
    for aid, (g, score) in sorted(first_score.items(), key=lambda kv: (kv[1][0], kv[0])):
        if score > best:
            best = score
            stack, late = list(children.get(aid, [])), False
            while stack:
                c = stack.pop()
                if arts.get(c, {}).get("generation", -1) >= g + k:
                    late = True
                    break
                stack += children.get(c, [])
            count += late
    return count
