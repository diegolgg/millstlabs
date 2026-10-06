"""Runner: config freeze, deployment lock, kill-and-resume identity, long dry run with bounded logs, replay_strict
end-to-end run from a committed cache fixture."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from culture.run.config import from_dict, load_config
from culture.run.runner import ConfigMismatch, Runner, RunLocked, run_config, state_digest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CACHE = ROOT / "tests" / "fixtures" / "llm_cache_smoke"

TINY = {
    "name": "tiny",
    "population": {"groups": 2, "agents_per_group": 2, "seed": 1},
    "evaluation": {"selfplay_games": 8, "crossplay_games": 4, "anchor_games": 4, "between_group_games": 4,
                   "ladder_every": 2, "ladder_games": 2, "workers": 0},
    "org": {"routing": "broadcast_group", "verification": {"name": "selfplay", "n": 8}},
    "runner": {"generations": 4},
}


def test_config_digest_refuses_changed_config(tmp_path):
    run_config(dict(TINY, runner={"generations": 1}), tmp_path / "r")
    changed = json.loads(json.dumps(TINY))
    changed["evaluation"]["selfplay_games"] = 9
    with pytest.raises(ConfigMismatch):
        Runner(from_dict(changed), tmp_path / "r")
    # generations / workers / debug are excluded from the digest: extending a run is allowed
    longer = json.loads(json.dumps(TINY))
    longer["runner"]["generations"] = 2
    longer["evaluation"]["workers"] = 0
    recs = run_config(longer, tmp_path / "r")
    assert [r["generation"] for r in recs] == [1]


def test_deployment_lock_refuses_concurrent_access(tmp_path):
    r1 = Runner(from_dict(TINY), tmp_path / "r")
    try:
        with pytest.raises(RunLocked):
            Runner(from_dict(TINY), tmp_path / "r")
    finally:
        r1.close()
    Runner(from_dict(TINY), tmp_path / "r").close()  # released


def _cli(cfg_path, out, *sets):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    args = [sys.executable, "-m", "culture.run", "--config", str(cfg_path), "--out", str(out), "--quiet"]
    for s in sets:
        args += ["--set", s]
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=600)


@pytest.mark.parametrize("step", ["receive", "verify", "revise", "evaluate"])
def test_killed_mid_generation_resumes_to_identical_state(tmp_path, step):
    import yaml

    cfg_path = tmp_path / "tiny.yaml"
    cfg_path.write_text(yaml.safe_dump(TINY))
    ref = _cli(cfg_path, tmp_path / "ref")
    assert ref.returncode == 0, ref.stderr
    killed = _cli(cfg_path, tmp_path / "crash", "debug.kill_at_generation=2", f"debug.kill_after_step={step}")
    assert killed.returncode == 17, killed.stderr
    ck = json.loads((tmp_path / "crash" / "checkpoint.json").read_text())
    assert ck["state"]["generation"] == 1  # died inside generation 2
    resumed = _cli(cfg_path, tmp_path / "crash")
    assert resumed.returncode == 0, resumed.stderr
    assert state_digest(tmp_path / "ref") == state_digest(tmp_path / "crash")


def test_thousand_generation_dry_run_logs_bounded(tmp_path):
    cfg = {
        "name": "dry1000",
        "population": {"groups": 2, "agents_per_group": 1, "seed": 0, "warm_start": "piers"},
        "evaluation": {"selfplay_games": 2, "crossplay_games": 0, "anchor_games": 0, "between_group_games": 0,
                       "ladder_every": 0, "workers": 0, "anchors": []},
        "org": {"routing": "none", "verification": "none", "credit": "none"},
        "corpus": {"enabled": False},
        "runner": {"generations": 1000, "revise": False},
    }
    out = tmp_path / "dry"
    run_config(cfg, out)
    lines = (out / "generations.jsonl").read_text().splitlines()
    assert len(lines) == 1000
    sizes = [len(x) for x in lines]
    assert max(sizes[500:]) <= 1.2 * max(sizes[1:100])  # per-generation log size does not grow
    assert (out / "checkpoint.json").stat().st_size < 50_000
    assert json.loads((out / "checkpoint.json").read_text())["state"]["generation"] == 999


def test_end_to_end_replay_strict_from_committed_fixture(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml",
                      {"name": "smoke-fixture", "llm": {"cache_mode": "replay_strict", "cache_dir": str(FIXTURE_CACHE)}})
    recs = run_config(cfg, tmp_path / "e2e")
    assert len(recs) == cfg.runner.generations
    ledger = [json.loads(x) for x in (tmp_path / "e2e" / "ledger.jsonl").read_text().splitlines()]
    assert ledger and all(r["cached"] or r.get("replayed_within_run") for r in ledger)  # every call served from cache
