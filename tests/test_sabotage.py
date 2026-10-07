"""Fix round 1, step 16: the D1 sabotage treatment and the breakdown-point analysis."""

import json

import pytest

from culture.analysis.breakdown import breakdown_point, summarize
from culture.bots.anchors import RULE_LISTS, anchor_conventions, anchor_source, parse_rules
from culture.artifacts.schema import artifact_id
from culture.run.runner import run_config

BASE = {
    "name": "sab_test",
    "population": {"groups": 1, "agents_per_group": 3, "seed": 2, "warm_start": "piers"},
    "evaluation": {"selfplay_games": 12, "crossplay_games": 0, "anchor_games": 0, "between_group_games": 0,
                   "ladder_every": 0, "workers": 0},
    "org": {"routing": "broadcast_group", "verification": "none", "credit": "none"},
    "corpus": {"enabled": False},
    "llm": {"stub": {"mode": "null"}},
    "runner": {"generations": 3},
}
FLAWED = artifact_id(anchor_source("flawed"), anchor_conventions("flawed"))


def _messages(d):
    return [json.loads(x) for x in (d / "messages.jsonl").read_text().splitlines() if x.strip()]


def _run(tmp_path, name, **over):
    cfg = json.loads(json.dumps(BASE))
    for k, v in over.items():
        cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    run_config(cfg, tmp_path / name)
    return tmp_path / name


def test_full_sabotage_without_verification_collapses(tmp_path):
    d = _run(tmp_path, "e1", sabotage={"epsilon": 1.0})
    msgs = [m for m in _messages(d) if m.get("delivered")]
    assert msgs and all(m["sabotaged"] and m["artifact_id"] == FLAWED for m in msgs)
    last = json.loads((d / "generations.jsonl").read_text().splitlines()[-1])
    # everyone ends on the saboteur's policy (null revisions keep its rule list with a new random stream) and scores 0
    for a in last["agents"].values():
        code = json.loads((d / "artifacts" / a["incumbent"][:2] / f"{a['incumbent']}.json").read_text())["code"]
        assert parse_rules(code) == RULE_LISTS["flawed"]
    assert last["population_mean"] == 0.0 and last["teaching"]["sabotaged_adopted"] >= 1


def test_verification_rejects_sabotage(tmp_path):
    d = _run(tmp_path, "v1", sabotage={"epsilon": 1.0}, org={**BASE["org"], "verification": {"name": "selfplay", "n": 12}})
    msgs = [m for m in _messages(d) if m.get("sabotaged")]
    assert msgs and all(m["decision"] == "rejected" for m in msgs)  # Piers holders never take Flawed


def test_no_sabotage_at_zero_and_nested_sets(tmp_path):
    assert not any(m.get("sabotaged") for m in _messages(_run(tmp_path, "e0", sabotage={"epsilon": 0.0})))
    lo = {m["id"] for m in _messages(_run(tmp_path, "e3", sabotage={"epsilon": 0.3})) if m.get("sabotaged")}
    hi = {m["id"] for m in _messages(_run(tmp_path, "e6", sabotage={"epsilon": 0.6})) if m.get("sabotaged")}
    assert lo <= hi  # common random numbers: the sabotaged set grows with epsilon


def test_breakdown_point_known_answers():
    eps = [0.0, 0.1, 0.2, 0.3, 0.5]
    assert breakdown_point(eps, [10, 9, 8, 6, 2]) == pytest.approx(0.25)  # crosses 7 halfway between 0.2 and 0.3
    assert breakdown_point(eps, [10, 10, 10, 10, 10]) is None
    s = summarize({"on": {e: [10.0, 10.0] for e in eps}, "off": {0.0: [10.0, 10.0], 0.1: [5.0, 5.0], 0.2: [1.0, 1.0],
                                                              0.3: [0.0, 0.0], 0.5: [0.0, 0.0]}}, n_boot=200)
    assert s["arms"]["on"]["censored"] and s["arms"]["off"]["breakdown_point"] == pytest.approx(0.06)
    assert s["primary_contrast"]["point"] == pytest.approx(0.5 - 0.06)  # a lower bound: eps*(on) is censored
