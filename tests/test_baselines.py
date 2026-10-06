"""OpenEvolve adapter, stub-tested only: the evaluator contract on our protocol, and a short OpenEvolve run whose LLM
is our stub backend (zero API calls; every call ledgered)."""

import json

import pytest

from culture.baselines import openevolve_adapter as oe
from culture.bots.anchors import anchor_source


def test_evaluator_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("CULTURE_OE_GAMES", "40")
    p = tmp_path / "bot.py"
    p.write_text(anchor_source("piers"))
    m = oe.evaluate(str(p))
    assert m["valid"] == 1.0 and abs(m["combined_score"] - 17) < 1.0 and m["illegal_rate"] == 0
    assert set(m) >= {"combined_score", "anchor_piers", "anchor_iggi"}
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps([{"id": "flawed_partner", "code": anchor_source("flawed")}]))
    monkeypatch.setenv("CULTURE_OE_SNAPSHOT", str(snap))
    assert oe.evaluate(str(p))["crossplay_snapshot"] < 5  # population snapshot fetched out of band
    bad = tmp_path / "bad.py"
    bad.write_text("import socket\nclass Bot: pass\n")
    assert oe.evaluate(str(bad))["valid"] == 0.0


def test_openevolve_runs_with_stub_llm(tmp_path):
    pytest.importorskip("openevolve")
    result = oe.run_openevolve(anchor_source("iggi"), iterations=4, out_dir=tmp_path / "oe", games=20)
    assert result.best_score is not None and result.best_score >= 10
    rows = [json.loads(x) for x in (tmp_path / "oe" / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) >= 4 and all(r["backend"] == "stub" for r in rows)
