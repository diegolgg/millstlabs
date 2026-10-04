import copy
from dataclasses import replace

import pytest
import torch

from millstlabs.config import EnvironmentConfig, ExperimentConfig, TrainingConfig
from millstlabs.env import PopulationEnv
from millstlabs.evaluation import calibration_histories
from millstlabs.policy import PolicyBank
from millstlabs.ppo import Transition, advantages, update
from millstlabs.trainer import Trainer, warmstart


def config():
    return ExperimentConfig(environment=replace(EnvironmentConfig(), max_individuals=1000), training=TrainingConfig(
        backend="tiny", device="cpu", warmstart_transitions=16, decisions=128,
        rollout_ticks=8, social_window=16, sequence_length=4, evaluation_maps=1,
        final_evaluation_maps=1, evaluation_horizon=4, checkpoints=[128], cpu_threads=1))


def fixture_bank():
    cfg = config()
    env = PopulationEnv(cfg.environment)
    obs, _ = env.reset(seed=1)
    obs = next(iter(obs.values()))
    bank = PolicyBank(cfg.training, obs)
    bank.add("a")
    bank.add("b", bank.state("a"))
    return cfg, bank, obs


def test_private_updates_do_not_change_neighbor_or_backbone():
    cfg, bank, obs = fixture_bank()
    neighbor = copy.deepcopy(bank.state("b"))
    base = copy.deepcopy(bank.base.state_dict())
    original = copy.deepcopy(bank.state("a"))
    h = bank.zero_hidden()
    a, lp, vp, vs, _ = bank.act("a", obs, h)
    row = Transition(obs, h, a, lp, vp, vs, 1, 0.5, False, False)
    update(bank, "a", [row] * 4, 0.5)
    assert any(not torch.equal(v, bank.state("a")["controller"][k]) for k, v in original["controller"].items())
    for section in neighbor:
        for k, v in neighbor[section].items():
            assert torch.equal(v, bank.state("b")[section][k])
    for k, v in base.items():
        assert torch.equal(v, bank.base.state_dict()[k])


def test_post_death_social_tail_not_erased_and_window_closes():
    cfg = TrainingConfig(gamma=0.9, gae_lambda=1)
    row = Transition({}, torch.zeros(1, 64), 6, 0, 0, 0, 0, 0.25, True, False)
    ap, ass = advantages([row], cfg, personal_bootstrap=100, social_bootstrap=0.5 + 0.9 * 0.5)
    assert ap[0] == 0
    assert ass[0] == pytest.approx(0.25 + 0.9 * (0.5 + 0.9 * 0.5))
    row.social_terminal = True
    assert advantages([row], cfg, social_bootstrap=100)[1][0] == 0.25


def test_bootstrap_across_live_rollout_boundary():
    cfg = TrainingConfig(gamma=0.9, gae_lambda=1)
    row = Transition({}, torch.zeros(1, 64), 6, 0, 2, 3, 1, 0.25, False, False)
    a, b = advantages([row], cfg, 5, 6)
    assert a[0] == pytest.approx(1 + 0.9 * 5 - 2)
    assert b[0] == pytest.approx(0.25 + 0.9 * 6 - 3)


def test_inheritance_mutation_kl_and_exact_unchanged_components():
    cfg, bank, _ = fixture_bank()
    source = bank.state("a")
    histories = calibration_histories(cfg.environment)
    result = bank.inherit("child", source, histories, torch.Generator().manual_seed(1))
    assert result["mutation_kl"] <= cfg.training.mutation_kl
    child = bank.state("child")
    for k, v in source["adapter"].items():
        assert torch.equal(v, child["adapter"][k])
    for k, v in source["controller"].items():
        if k not in {"action.weight", "watch_bias"}:
            assert torch.equal(v, child["controller"][k])
    assert len(bank.optimizers["child"].state) == 0


def test_checkpoint_resume_preserves_world_rng_policy_and_pending_credit(tmp_path):
    cfg = config()
    warm = tmp_path / "warm.pt"
    warmstart(cfg, 11, warm)
    a = Trainer(cfg, "prosocial", "r_adult", 11, tmp_path / "a", warm)
    for _ in range(5):
        a.step()
    a.checkpoint()
    # Simulate a partial next rollout written after the committed checkpoint.
    with (tmp_path / "a/ecology.jsonl").open("a") as f:
        f.write('{"uncommitted": true}\n')
    b = Trainer(cfg, "prosocial", "r_adult", 11, tmp_path / "b", warm, tmp_path / "a/latest.pt")
    assert "uncommitted" not in (tmp_path / "b/ecology.jsonl").read_text()
    for _ in range(7):
        a.step()
        b.step()
    assert a.decisions == b.decisions
    assert a.env.render() == b.env.render()
    assert a.env.metrics() == b.env.metrics()
    for i in a.bank.controllers:
        for k, v in a.bank.state(i)["controller"].items():
            assert torch.equal(v, b.bank.state(i)["controller"][k])


def test_extinction_recovery_finishes_terminal_updates(tmp_path):
    cfg = config()
    warm = tmp_path / "warm.pt"
    warmstart(cfg, 11, warm)
    t = Trainer(cfg, "individual", "iteration", 11, tmp_path / "r", warm)
    for p in t.env.prey.values():
        p.energy = 0.001
    # No prey near food, so every possible policy action dies by starvation.
    t.env.station_positions = [(19, 19)] * 3
    for j, p in enumerate(t.env.prey.values()):
        p.pos = (j, 0)
    t.obs = {i: t.env.observe(i) for i in t.env.agents}
    t.step()
    assert t.restarts == 1 and t.trial == 1
    assert len(t.env.agents) == 8 and not t.dead
    assert all(p.energy == 60 and p.age == 128 for p in t.env.prey.values())
    assert all(len(o.state) == 0 for o in t.bank.optimizers.values())
    assert all(torch.count_nonzero(h) == 0 for h in t.hidden.values())
    assert t.bank.gradient_updates > 0


def test_all_methods_end_to_end(tmp_path):
    cfg = config()
    warm = tmp_path / "warm.pt"
    warmstart(cfg, 11, warm)
    for method in cfg.methods:
        trainer = Trainer(cfg, "competitive", method, 11, tmp_path / method, warm)
        summary = trainer.run()
        assert summary["finished"]
        assert 0 <= summary["unused_decisions"] < cfg.environment.population_cap
        assert (tmp_path / method / "evaluation.jsonl").exists()
        assert trainer.bank.gradient_updates > 0


@pytest.mark.parametrize("method", ["r_adult", "r_initial"])
def test_real_birth_uses_correct_committed_source(tmp_path, method):
    cfg = config()
    cfg.training.mutation_rms = 0
    cfg.training.mutation_watch_std = 0
    warm = tmp_path / "warm.pt"
    warmstart(cfg, 11, warm)
    t = Trainer(cfg, "vigilance", method, 11, tmp_path / method, warm)
    parent = t.env.agents[0]
    t.env.obstacles = set()
    t.env.predator = (19, 19)
    t.env.attack_ready = 10000
    for j, p in enumerate(t.env.prey.values()):
        p.pos = (j + 2, 5)
    t.env.prey[parent].energy = 90
    with torch.no_grad():
        t.bank.controllers[parent].action.weight.add_(0.2)
    source = t.bank.state(parent) if method == "r_adult" else t.initial
    t.obs = {i: t.env.observe(i) for i in t.env.agents}
    t.step()
    birth = next(e for e in t.env.events if e["type"] == "birth" and e["parent"] == parent)
    child = birth["id"]
    for section in source:
        for key, value in source[section].items():
            assert torch.equal(value, t.bank.state(child)[section][key])
    assert not t.buffers[child]
    assert not t.bank.optimizers[child].state
    assert torch.count_nonzero(t.hidden[child]) == 0
    before = t.bank.state(child)
    with torch.no_grad():
        t.bank.controllers[parent].action.weight.add_(1)
    assert torch.equal(before["controller"]["action.weight"], t.bank.state(child)["controller"]["action.weight"])
