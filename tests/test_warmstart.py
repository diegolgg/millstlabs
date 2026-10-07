"""Fix round 1, step 5: generation 0 is authored once per population seed into a shared warm-start set; conditions
fork from it and never call the backend for generation 0."""

import json

import pytest
import yaml

from culture.run.config import from_dict
from culture.run.experiment import run_experiment
from culture.run.runner import run_config
from culture.run.warmstart import WarmStartMismatch, build_warm_start

SPEC = {
    "base": {"name": "ws_test", "experiment_seed": 5, "population": {"groups": 1, "agents_per_group": 2},
             "evaluation": {"selfplay_games": 8, "crossplay_games": 4, "anchor_games": 4, "between_group_games": 0,
                            "ladder_every": 0, "workers": 0},
             "corpus": {"enabled": False}, "runner": {"generations": 2}},
    "conditions": {"solo": {"org": {"routing": "none", "verification": "none"}},
                   "blind": {"org": {"routing": "best_to_all", "verification": "none"}},
                   "verified": {"org": {"routing": "best_to_all", "verification": {"name": "selfplay", "n": 8}}}},
    "population_seeds": [0, 1],
}


def _jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_conditions_fork_from_one_warm_start_per_seed(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(SPEC))
    run_experiment(spec_path, tmp_path / "out", processes=1)
    sets = sorted((tmp_path / "out" / "_warm_start").glob("p*/warm_start.json"))
    assert len(sets) == 2  # one per population seed, shared by all three conditions
    for ps in (0, 1):
        store_bytes = []
        for cname in SPEC["conditions"]:
            run = tmp_path / "out" / cname / f"p{ps}"
            g0 = _jsonl(run / "generations.jsonl")[0]
            ids = {a: v["incumbent"] for a, v in g0["agents"].items()}
            store_bytes.append({a: (run / "artifacts" / i[:2] / f"{i}.json").read_bytes() for a, i in ids.items()})
            # no condition calls the backend for generation 0's warm start
            assert not [r for r in _jsonl(run / "ledger.jsonl")
                        if r["generation"] == 0 and r["tag"] in ("author", "repair")], cname
            assert g0["warm_start_set"]["calls"] >= 2
        assert all(b == store_bytes[0] for b in store_bytes)  # byte-identical generation-0 incumbents
        # the set itself holds the authoring calls (one per agent, plus any repairs)
        sset = [s for s in sets if s.parent.name.startswith(f"p{ps}-")][0]
        led = _jsonl(sset.parent / "ledger.jsonl")
        assert sum(r["tag"] == "author" for r in led) == 2


def test_set_is_reused_and_mismatch_is_refused(tmp_path):
    cfg = from_dict(dict(SPEC["base"]))
    p1 = build_warm_start(cfg, tmp_path / "ws")
    mtime = p1.stat().st_mtime_ns
    assert build_warm_start(cfg, tmp_path / "ws") == p1 and p1.stat().st_mtime_ns == mtime  # reused, not rebuilt
    other = from_dict({"population": {"seed": 7, "warm_start_set": str(p1)}}, cfg)
    with pytest.raises(WarmStartMismatch):
        run_config(other, tmp_path / "r")


def test_forked_run_equals_in_run_authoring(tmp_path):
    """With the deterministic stub, forking from the set gives the same generation-0 incumbents as authoring in-run."""
    cfg = from_dict(dict(SPEC["base"]))
    run_config(cfg, tmp_path / "inrun", generations=1)
    p = build_warm_start(cfg, tmp_path / "ws")
    run_config(from_dict({"population": {"warm_start_set": str(p)}}, cfg), tmp_path / "forked", generations=1)
    a = _jsonl(tmp_path / "inrun" / "generations.jsonl")[0]["agents"]
    b = _jsonl(tmp_path / "forked" / "generations.jsonl")[0]["agents"]
    assert {k: v["incumbent"] for k, v in a.items()} == {k: v["incumbent"] for k, v in b.items()}
