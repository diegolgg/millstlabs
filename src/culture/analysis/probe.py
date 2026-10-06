"""Probe (spec sections 13-14): one generation per model tier, two independent agents, no teaching.

Reports self-play score, illegal rate, lines, tokens and dollars per call, a conventions diff against H-group
vocabulary, cross-play between the two independent agents, and the three probe gates. Tonight only the stub backend
exists, so the numbers exercise the pipeline and say nothing about any real model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..agents import agent as A
from ..run.config import load_config
from ..run.runner import Runner

# H-group convention vocabulary (keyword list; an embedding comparison needs a real rulebook text and model)
HGROUP_KEYWORDS = ["finesse", "chop", "5 save", "2 save", "critical save", "play clue", "save clue", "prompt", "bluff",
                   "trash", "chop move", "good touch", "focus", "delayed play", "fix clue", "early game", "sarcastic",
                   "positional", "double discard"]


def keyword_hits(text: str) -> dict[str, int]:
    t = text.lower()
    return {k: len(re.findall(re.escape(k), t)) for k in HGROUP_KEYWORDS if k in t}


def bow_cosine(a: str, b: str) -> float:
    def vec(s):
        v: dict[str, int] = {}
        for w in re.findall(r"[a-z]+", s.lower()):
            v[w] = v.get(w, 0) + 1
        return v

    va, vb = vec(a), vec(b)
    num = sum(va[w] * vb.get(w, 0) for w in va)
    den = (sum(x * x for x in va.values()) ** 0.5) * (sum(x * x for x in vb.values()) ** 0.5)
    return num / den if den else 0.0


def run_probe(config_path: str | Path, out_dir: str | Path, model: str) -> dict[str, Any]:
    cfg = load_config(config_path, {"llm": {"model": model}, "condition": model})
    out = Path(out_dir) / model
    with Runner(cfg, out) as r:
        recs = r.run()
        ctx = r.ctx
        agents = sorted(ctx.agents)
        last = recs[-1] if recs else None
        # one measured teach call (the probe itself has no teaching)
        a0 = ctx.agents[agents[0]]
        art0 = ctx.store.get(a0.incumbent)
        ctx.start_budgets(None)
        A.teach(ctx, a0, [agents[1]], art0, ctx.evals.get(a0.incumbent), cfg.runner.generations)
        rows = ctx.ledger.rows
        per_tag: dict[str, dict[str, float]] = {}
        for t in sorted({x["tag"] for x in rows}):
            rs = [x for x in rows if x["tag"] == t]
            per_tag[t] = {"calls": len(rs),
                          "tokens_per_call": float(np.mean([x["input_tokens"] + x["output_tokens"] + x["cache_read_input_tokens"]
                                                            + x["cache_creation_input_tokens"] for x in rs])),
                          "usd_per_call": float(np.mean([x["spend_usd"] or 0.0 for x in rs]))}
        ids = [ctx.agents[a].incumbent for a in agents]
        seeds = ctx.seeds(cfg.runner.generations, "probe", cfg.evaluation.selfplay_games)
        self_scores = [ctx.selfplay(i, seeds)[0] for i in ids]
        cross = ctx.crossplay(ids[0], ids[1], seeds)
        conv = [ctx.store.get(i).conventions for i in ids]
        res = {
            "model": model,
            "backend": cfg.llm.backend,
            "agents": {a: {"incumbent": i, "selfplay": s, "illegal_rate": ctx.selfplay(i, seeds)[1],
                           "lines": len(ctx.store.get(i).code.splitlines()),
                           "hgroup_keywords": keyword_hits(c)}
                       for a, i, s, c in zip(agents, ids, self_scores, conv)},
            "crossplay_between_independent_agents": cross,
            "conventions_similarity_between_agents": bow_cosine(conv[0], conv[1]),
            "cost_per_call": per_tag,
            "gates": {
                "headroom_selfplay_below_20": bool(max(self_scores) < 20),
                "cost_measured": "revise" in per_tag and "teach" in per_tag,
                "null_alive_crossplay_well_below_selfplay": bool(cross < 0.8 * float(np.mean(self_scores))),
            },
            "generation_records": len(recs),
            "population_best_last": last["population_best"] if last else None,
        }
    return res
