"""S1 lever schedule (org/schedule.py): organization policies switched at fixed generations inside one run.

A 12-generation stub run toggles four levers every 3 generations (routing, quarantine, selection, and migration via
the islands rate). The policies in effect change exactly at the switch generations, the generation record logs the
lever values, and a run killed across a switch resumes to a byte-identical state."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from culture.org.migration import RandomMigration
from culture.org.routing import BroadcastGroup, NoRouting
from culture.org.selection import KeepBestK, ShinkaWeighted
from culture.run.config import ConfigError, from_dict
from culture.run.runner import Runner, state_digest

ROOT = Path(__file__).resolve().parents[1]

OFF = {"routing": "none", "quarantine_unverified": False, "selection": {"name": "shinka_weighted", "lam": 10.0},
       "topology": {"name": "islands", "migration_rate": 0.0, "interval": 1}}
ON = {"routing": "broadcast_group", "quarantine_unverified": True, "selection": {"name": "keep_best_k", "k": 5},
      "topology": {"name": "islands", "migration_rate": 0.5, "interval": 1}}
CFG = {
    "name": "sched",
    "population": {"groups": 2, "agents_per_group": 2, "seed": 3},
    "evaluation": {"selfplay_games": 4, "crossplay_games": 2, "anchor_games": 2, "between_group_games": 2,
                   "ladder_every": 0, "workers": 0, "feedback_games": 2, "failure_traces": 1},
    "org": {"routing": "broadcast_group", "verification": {"name": "selfplay", "n": 4}, "credit": "none",
            "quarantine_unverified": True, "selection": {"name": "keep_best_k", "k": 5},
            "topology": {"name": "islands", "migration_rate": 0.5, "interval": 1},
            "schedule": [{"at": 3, "set": OFF}, {"at": 6, "set": ON}, {"at": 9, "set": OFF}]},
    "runner": {"generations": 12},
}


def on_at(g: int) -> bool:
    return (g // 3) % 2 == 0  # ON for 0-2 and 6-8, OFF for 3-5 and 9-11


def _jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def test_policies_switch_exactly_at_the_scheduled_generations(tmp_path):
    with Runner(from_dict(CFG), tmp_path / "r") as r:
        archive_sizes = {}
        for g in range(12):
            r.run(generations=g + 1)
            assert r.ctx.generation == g
            on = on_at(g)
            pol = r.ctx.pol
            assert isinstance(pol["routing"], BroadcastGroup if on else NoRouting), g
            assert type(pol["selection"]) is (KeepBestK if on else ShinkaWeighted), g
            assert isinstance(pol["migration"], RandomMigration) and pol["migration"].rate == (0.5 if on else 0.0), g
            assert r.ctx.quarantine is on, g
            archive_sizes[g] = sum(len(v) for v in pol["selection"].archive.values())
    # the selection archive is carried across a switch (never restarts from empty)
    assert all(archive_sizes[g] > 0 for g in range(12))
    recs = _jsonl(tmp_path / "r" / "generations.jsonl")
    assert [x["generation"] for x in recs] == list(range(12))
    for rec in recs:
        g, on = rec["generation"], on_at(rec["generation"])
        lv = rec["levers"]
        assert set(lv) == {"routing", "quarantine_unverified", "selection", "topology"}
        assert lv["routing"]["name"] == ("broadcast_group" if on else "none")
        assert lv["quarantine_unverified"] is on
        assert lv["selection"]["name"] == ("keep_best_k" if on else "shinka_weighted")
        assert lv["topology"]["migration_rate"] == (0.5 if on else 0.0)
        # mechanics follow the levers: messages are sent and agents migrate only while ON
        assert (rec["teaching"]["sent"] > 0) is on, g
        if g > 0:
            assert bool(rec["migrations"]) is on, g
            withheld = [c.get("withheld") for c in rec["revise_context"].values()]
            assert all((w is not None) is on for w in withheld), g


def test_schedule_validation_and_digest():
    base = from_dict({})
    assert base.digest() == "f875065a1af48815"  # empty schedule: digest unchanged
    s = from_dict({"org": {"schedule": [{"at": 5, "set": {"routing": "none"}}]}})
    assert s.digest() != base.digest()
    bad = [
        [{"at": 0, "set": {"routing": "none"}}],  # generation 0 is the warm start
        [{"at": 5, "set": {"routing": "none"}}, {"at": 5, "set": {"routing": "broadcast_group"}}],  # not increasing
        [{"at": 5, "set": {"budget": 3}}],  # not a lever
        [{"at": 5, "set": {"routing": "nonexistent"}}],  # unknown policy
        [{"at": 5, "set": {"verification": {"name": "selfplay", "bogus": 1}}}],  # bad parameter
        [{"at": 5, "set": {"quarantine_unverified": "yes"}}],
        [{"at": 5}],
    ]
    for sched in bad:
        with pytest.raises(ConfigError):
            from_dict({"org": {"schedule": sched}})


def _cli(cfg_path, out, *sets):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    args = [sys.executable, "-m", "culture.run", "--config", str(cfg_path), "--out", str(out), "--quiet"]
    for s in sets:
        args += ["--set", s]
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    d = tmp_path_factory.mktemp("sched_ref")
    cfg_path = d / "sched.yaml"
    cfg_path.write_text(yaml.safe_dump(CFG))
    ref = _cli(cfg_path, d / "ref")
    assert ref.returncode == 0, ref.stderr
    return cfg_path, d / "ref"


@pytest.mark.parametrize("kill_gen,step", [(3, "receive"), (4, "revise")])
def test_kill_and_resume_across_a_switch_is_byte_identical(tmp_path, reference, kill_gen, step):
    """Killed inside the switch generation (3: the checkpoint is before the switch, resume applies it) and inside the
    generation after it (4: the checkpoint is at an OFF generation while the configured base is ON, so resume must
    re-apply the switch before loading the policies' state)."""
    cfg_path, ref_dir = reference
    killed = _cli(cfg_path, tmp_path / "crash", f"debug.kill_at_generation={kill_gen}", f"debug.kill_after_step={step}")
    assert killed.returncode == 17, killed.stderr
    ck = json.loads((tmp_path / "crash" / "checkpoint.json").read_text())
    assert ck["state"]["generation"] == kill_gen - 1
    resumed = _cli(cfg_path, tmp_path / "crash")
    assert resumed.returncode == 0, resumed.stderr
    assert state_digest(ref_dir) == state_digest(tmp_path / "crash")
    recs = _jsonl(tmp_path / "crash" / "generations.jsonl")
    assert [r["levers"]["routing"]["name"] for r in recs] == [
        "broadcast_group" if on_at(g) else "none" for g in range(12)]
