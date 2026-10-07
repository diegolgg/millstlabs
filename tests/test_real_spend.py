"""Fix round 1, step 8: `real_spend_usd` excludes same-run replays, which `new_spend_usd` counts as new spend."""

import json

from culture.llm.backend import CostLedger
from culture.run.config import from_dict
from culture.run.runner import run_config

TINY = {
    "name": "spend",
    "population": {"groups": 1, "agents_per_group": 2, "seed": 4},
    "evaluation": {"selfplay_games": 6, "crossplay_games": 2, "anchor_games": 2, "between_group_games": 0,
                   "ladder_every": 0, "workers": 0},
    "corpus": {"enabled": False},
    "runner": {"generations": 2},
}


def test_replaying_own_calls_is_not_real_spend(tmp_path):
    cache = tmp_path / "cache"
    first = from_dict(dict(TINY, llm={"cache_mode": "replay", "cache_dir": str(cache)}))
    run_config(first, tmp_path / "a")
    a = CostLedger(tmp_path / "a" / "ledger.jsonl").totals()
    assert a["real_spend_usd"] == a["new_spend_usd"] > 0  # every call was sent and paid
    # same config, same run id, new directory, replay_strict: every call is a same-run replay, nothing is sent
    again = from_dict(dict(TINY, llm={"cache_mode": "replay_strict", "cache_dir": str(cache)}))
    run_config(again, tmp_path / "b")
    b = CostLedger(tmp_path / "b" / "ledger.jsonl").totals()
    assert b["calls"] == a["calls"] and b["new_spend_usd"] == a["new_spend_usd"]  # the curve matches the original
    assert b["real_spend_usd"] == 0.0  # but no money was spent
    rec = [json.loads(x) for x in (tmp_path / "b" / "generations.jsonl").read_text().splitlines()][-1]
    assert rec["cost"]["real_spend_usd"] == 0.0 and rec["cost"]["new_spend_usd"] > 0


def test_report_uses_real_spend(tmp_path):
    from culture.analysis.report import run_report

    run_config(from_dict(TINY), tmp_path / "r")
    assert "money actually paid" in run_report(tmp_path / "r")
