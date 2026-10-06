"""Learning smoke test (spec section 12), through the real generation loop with LLM revision off.

Oracle teacher g0a0 holds Canaan Piers (about 17); student g0a2 holds RandomAgent (about 0). In the sabotage cases a
second sender g0a1 holds Flawed and its message arrives after the oracle's (messages are processed in id order).
- verification on, no saboteur: the student adopts Piers and scores about 17;
- verification off, saboteur: the student adopts Piers, then blindly adopts Flawed, and drops to 0;
- verification on, saboteur: the student adopts Piers and rejects Flawed.
Deterministic: engine-only decisions, stub teach text, no cache."""

import pytest

from culture.run.config import from_dict
from culture.run.runner import Runner


def run(tmp_path, verification, saboteur: bool):
    overrides = {"g0a0": "piers", "g0a1": "flawed" if saboteur else "random", "g0a2": "random"}
    cfg = from_dict({
        "name": "smoke_learning",
        "population": {"groups": 1, "agents_per_group": 3, "warm_start_overrides": overrides},
        "evaluation": {"selfplay_games": 100, "crossplay_games": 0, "anchor_games": 0, "between_group_games": 0,
                       "ladder_every": 0, "workers": 0},
        "org": {"routing": "broadcast_group", "verification": verification, "credit": "paired_delta"},
        "corpus": {"enabled": False},
        "llm": {"cache_mode": "off"},
        "runner": {"generations": 2, "revise": False},
    })
    with Runner(cfg, tmp_path / f"{verification if isinstance(verification, str) else 'v'}_{saboteur}") as r:
        recs = r.run()
        msgs = [m for m in (r.out / "messages.jsonl").read_text().splitlines()]
        return recs, msgs, r.ctx


def decisions_for(msgs, receiver):
    import json

    return {json.loads(m)["sender"]: json.loads(m)["decision"] for m in msgs if json.loads(m)["receiver"] == receiver}


def test_verified_transfer_lifts_student_to_piers(tmp_path):
    recs, msgs, ctx = run(tmp_path, {"name": "selfplay", "n": 100}, saboteur=False)
    assert recs[0]["agents"]["g0a2"]["score"] == 0.0
    assert recs[1]["agents"]["g0a2"]["score"] == pytest.approx(17.0, abs=1.0)
    assert decisions_for(msgs, "g0a2")["g0a0"] == "adopted"
    assert ctx.agents["g0a0"].credit > 15  # paired-delta credit goes to the oracle


def test_unverified_sabotage_is_adopted_and_student_drops_to_zero(tmp_path):
    recs, msgs, ctx = run(tmp_path, "none", saboteur=True)
    d = decisions_for(msgs, "g0a2")
    assert d["g0a0"] == "adopted" and d["g0a1"] == "adopted"
    assert recs[1]["agents"]["g0a2"]["score"] == 0.0
    assert ctx.store.get(ctx.agents["g0a2"].incumbent).author == "anchor:flawed"


def test_verified_sabotage_is_rejected(tmp_path):
    recs, msgs, ctx = run(tmp_path, {"name": "selfplay", "n": 100}, saboteur=True)
    d = decisions_for(msgs, "g0a2")
    assert d["g0a0"] == "adopted" and d["g0a1"] == "rejected"
    assert recs[1]["agents"]["g0a2"]["score"] == pytest.approx(17.0, abs=1.0)
    assert ctx.store.get(ctx.agents["g0a2"].incumbent).author == "anchor:piers"
