import json
from dataclasses import replace

import pytest

from millstlabs.batch import freeze, plan, report, run_batch
from millstlabs.config import ExperimentConfig, TrainingConfig, load_config
from millstlabs.experiments import evaluate_baselines


def test_overnight_budget_and_pair_order():
    cfg = load_config("configs/overnight-cpu.yaml")
    proposal = plan(cfg)
    assert len(proposal["runs"]) == 12
    assert proposal["training_decisions"] == 196608
    assert proposal["warmstart_demonstrations"] == 4096
    assert {r["seed"] for r in proposal["runs"]} == {11}
    assert [r["profile"] for r in proposal["runs"][:4]] == list(cfg.profiles)
    assert {r["method"] for r in proposal["runs"][:4]} == {"r_adult"}
    assert {(r["profile"], r["method"]) for r in proposal["runs"]} == {
        (profile, method) for profile in cfg.profiles for method in cfg.methods}
    with pytest.raises(ValueError):
        plan(cfg, [99])


def test_serial_runner_finishes_and_reuses_completed_run(tmp_path, monkeypatch):
    cfg = ExperimentConfig(training=TrainingConfig(
        backend="tiny", device="cpu", cpu_threads=1, decisions=64, warmstart_transitions=16,
        rollout_ticks=8, social_window=16, sequence_length=4, checkpoints=[0, 64],
        evaluation_maps=1, final_evaluation_maps=1, evaluation_horizon=4),
        profiles={"prosocial": {"social_preference": 0.5, "vigilance_bias": 0}},
        methods=["r_adult"], seeds=[11])
    root = tmp_path / "sweep"
    frozen = freeze(cfg, root)
    first = run_batch(cfg, root, frozen, hours=1)
    assert first["complete"] == first["total"] == 1
    assert "lifetime_change_on_common_maps" in first["runs"][0]
    assert (root / "analysis.json").exists()
    monkeypatch.setattr("millstlabs.batch.subprocess.run", lambda *a, **k: pytest.fail("Completed run reran"))
    assert run_batch(cfg, root, frozen, hours=1) == first
    cfg.training.decisions += 1
    with pytest.raises(ValueError, match="different configuration"):
        freeze(cfg, root)


def test_report_does_not_promote_partial_final_evaluation(tmp_path):
    cfg = ExperimentConfig(profiles={"prosocial": {}}, seeds=[11], methods=["iteration"])
    run = tmp_path / "prosocial-iteration-s11"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"finished": False, "decisions": 12}))
    (run / "evaluation.jsonl").write_text(json.dumps({"checkpoint": cfg.training.decisions,
                                                     "restricted_mean_lifetime": 999}) + "\n")
    result = report(cfg, tmp_path)
    assert result["complete"] == 0
    assert "final_lifetime" not in result["runs"][0]
    assert not (tmp_path / "analysis.json").exists()


def test_baselines_match_heldout_maps_and_are_reproducible(tmp_path):
    cfg = ExperimentConfig(training=replace(TrainingConfig(), evaluation_horizon=8,
                                           final_evaluation_maps=2, evaluation_seed=2000000))
    first = evaluate_baselines(cfg, tmp_path / "a.json")
    assert first == evaluate_baselines(cfg, tmp_path / "b.json")
    assert len(first["episodes"]) == 6
    assert {r["map_seed"] for r in first["episodes"]} == {2000000, 2000001}
    assert all(0 <= r["restricted_mean_lifetime"] <= 8 for r in first["episodes"])
