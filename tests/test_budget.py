"""Fix round 1, step 3: the soft budget cap reserves a call's estimated cost before sending, so a group never
overspends its per-generation budget."""

import json

from culture.run.config import from_dict
from culture.run.context import RunContext
from culture.run.runner import run_config

TINY = {
    "name": "budget_cap",
    "population": {"groups": 1, "agents_per_group": 2, "seed": 1, "warm_start": "piers"},
    "evaluation": {"selfplay_games": 8, "crossplay_games": 0, "anchor_games": 4, "between_group_games": 0,
                   "ladder_every": 0, "workers": 0},
    "corpus": {"enabled": False},
    "budget": {"unit": "calls", "per_group_per_generation": 1},
    "runner": {"generations": 4},
}


def _ledger(run_dir):
    return [json.loads(x) for x in (run_dir / "ledger.jsonl").read_text().splitlines() if x.strip()]


def test_group_of_two_makes_exactly_one_revise_call_per_generation(tmp_path):
    run_config(TINY, tmp_path / "r")
    rows = _ledger(tmp_path / "r")
    calls = rows_by = {}
    for r in rows:
        if r["tag"] == "revise" and not r.get("refused"):
            calls.setdefault(r["generation"], 0)
            calls[r["generation"]] += 1
    # generations 1..3 (gen 0 is the unbudgeted warm start) each make exactly one accepted revise call
    assert calls == {1: 1, 2: 1, 3: 1}, calls
    # the second agent's attempt each budgeted generation is refused and logged with a reason
    refusals = [r for r in rows if r.get("refused")]
    assert len(refusals) >= 3 and all("group budget" in r["refusal_reason"] for r in refusals)


def test_totals_exclude_refusals(tmp_path):
    run_config(TINY, tmp_path / "r")
    rows = _ledger(tmp_path / "r")
    real = [r for r in rows if not r.get("refused")]
    # ledger totals count only real calls
    from culture.llm.backend import CostLedger

    led = CostLedger(tmp_path / "r" / "ledger.jsonl")
    assert led.totals()["calls"] == len(real) < len(rows)


def test_reserve_refuses_when_estimate_exceeds_pool():
    cfg = from_dict(dict(TINY, budget={"unit": "calls", "per_group_per_generation": 1}))
    ctx = RunContext(cfg, None)
    try:
        g0 = ctx.pop.groups[0]
        a0, a1 = sorted(ctx.agents)[:2]
        ctx.start_budgets({g0: 1.0})
        from culture.llm.backend import Request

        def req(aid):
            return Request(system=[{"type": "text", "text": "x"}], messages=[{"role": "user", "content": "y"}],
                           model=cfg.llm.model, tag="revise", seed_tag="t",
                           meta={"run": "r", "group": ctx.agents[aid].group, "agent": aid, "generation": 1})

        ok0, _ = ctx.reserve(ctx.agents[a0], req(a0))
        assert ok0
        from culture.llm.backend import Response, Usage
        ctx.charge(ctx.agents[a0], Response(text="", usage=Usage(), cost_usd=0.0, model=cfg.llm.model, backend="stub"))
        ok1, reason = ctx.reserve(ctx.agents[a1], req(a1))  # pool now 0
        assert not ok1 and "group budget" in reason
    finally:
        ctx.close()
