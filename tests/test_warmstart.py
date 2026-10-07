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


def test_canonical_warm_start_pairs_conditions_with_different_group_layouts(tmp_path):
    """G1 prep: with population.warm_start_canonical, 4 x 1, 1 x 4 and 2 x 2 fork from one set per seed and get the
    same artifacts by agent position (relabelled with the run's own agent and group)."""
    spec = {
        "base": {"name": "ws_canon", "experiment_seed": 5, "population": {"warm_start_canonical": True},
                 "evaluation": {"selfplay_games": 6, "crossplay_games": 2, "anchor_games": 2, "between_group_games": 2,
                                "ladder_every": 0, "workers": 0},
                 "org": {"routing": "none", "verification": "none", "credit": "none"},
                 "corpus": {"enabled": False}, "runner": {"generations": 1}},
        "conditions": {"singletons": {"population": {"groups": 4, "agents_per_group": 1}},
                       "one": {"population": {"groups": 1, "agents_per_group": 4}},
                       "two": {"population": {"groups": 2, "agents_per_group": 2}}},
        "population_seeds": [0],
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec))
    run_experiment(spec_path, tmp_path / "out", processes=1)
    sets = sorted((tmp_path / "out" / "_warm_start").glob("p*/warm_start.json"))
    assert len(sets) == 1 and json.loads(sets[0].read_text())["canonical"] is True
    by_position = {}
    for cname in spec["conditions"]:
        run = tmp_path / "out" / cname / "p0"
        g0 = _jsonl(run / "generations.jsonl")[0]
        order = sorted(g0["agents"], key=lambda a: (int(a[1:].split("a")[0]), int(a.split("a")[1])))
        by_position[cname] = [g0["agents"][a]["incumbent"] for a in order]
        arts = {a["id"]: a for a in _jsonl(run / "artifacts.jsonl")}
        for a in order:  # relabelled: author and group are the run's own
            meta = arts[g0["agents"][a]["incumbent"]]
            assert meta["group"] == g0["agents"][a]["group"]
            assert meta["author"] == a or meta["author"].startswith("anchor:")
    assert by_position["singletons"] == by_position["one"] == by_position["two"]
    assert len(set(by_position["one"])) > 1  # not one artifact copied four times
    # the flag stays out of the digest while off, and is refused with per-agent overrides
    from culture.run.config import ConfigError

    assert from_dict({}).digest() == "f875065a1af48815"
    with pytest.raises(ConfigError):
        from_dict({"population": {"warm_start_canonical": True, "warm_start_overrides": {"g0a0": "piers"}}})


def test_spec_extends_deep_merges_the_parent(tmp_path):
    parent = tmp_path / "parent.yaml"
    parent.write_text(yaml.safe_dump(SPEC))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"extends": "parent.yaml", "base": {"name": "child", "llm": {"stub": {"mode": "null"}}},
                                     "population_seeds": [3]}))
    from culture.run.experiment import load_spec

    s = load_spec(child)
    assert s["base"]["name"] == "child" and s["base"]["llm"] == {"stub": {"mode": "null"}}
    assert s["base"]["evaluation"] == SPEC["base"]["evaluation"] and s["conditions"] == SPEC["conditions"]
    assert s["population_seeds"] == [3] and "extends" not in s
