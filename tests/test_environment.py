from dataclasses import replace

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from millstlabs.config import EnvironmentConfig
from millstlabs.env import EAST, FEED, REST, WATCH, WEST, PopulationEnv
from millstlabs.observations import observation_text


def world(**kwargs):
    cfg = replace(EnvironmentConfig(), max_individuals=1000, **kwargs)
    env = PopulationEnv(cfg)
    env.reset(seed=11)
    return env


def safe(env):
    env.predator = (19, 19)
    env.attack_ready = 99999
    env.obstacles = set()


def test_parallel_api():
    env = world()
    parallel_api_test(env, num_cycles=200)


def test_seed_and_space():
    a, b = world(), world()
    assert a.render() == b.render()
    assert a._connected({(x, y) for x in range(20) for y in range(20)} - a.obstacles)
    for i in a.agents:
        assert a.observation_space(i).contains(a.observe(i))
        assert abs(a.prey[i].pos[0] - a.predator[0]) + abs(a.prey[i].pos[1] - a.predator[1]) >= 8
    for _ in range(20):
        act = dict.fromkeys(a.agents, WATCH)
        a.step(act)
        b.step(act)
        assert a.render() == b.render()


def test_equal_feeding_redistributes_capacity():
    env = world(founders=3, metabolism=0, replenishment_rates=[0])
    safe(env)
    env.station_positions = [(5, 5), (10, 10), (15, 15)]
    env.stock[:] = [10, 0, 0]
    for i, pos, energy in zip(env.agents, [(5, 5), (4, 5), (6, 5)], [99, 50, 50]):
        env.prey[i].pos, env.prey[i].energy = pos, energy
        env.prey[i].cooldown = 100
    env.step(dict.fromkeys(env.agents, FEED))
    assert [p.energy for p in env.prey.values()] == [100, 54.5, 54.5]
    assert env.stock[0] == pytest.approx(0)


def test_movement_conflict_random_and_swap():
    wins = set()
    for seed in range(25):
        env = world(founders=2)
        safe(env)
        a, b = env.agents
        env.prey[a].pos, env.prey[b].pos = (3, 3), (5, 3)
        env.rng = np.random.default_rng(seed)
        env.step({a: EAST, b: WEST})
        assert len({p.pos for p in env.prey.values()}) == 2
        wins.add(next(i for i, p in env.prey.items() if p.pos == (4, 3)))
    assert wins == {a, b}
    env.prey[a].pos, env.prey[b].pos = (3, 3), (4, 3)
    env.step({a: EAST, b: WEST})
    assert (env.prey[a].pos, env.prey[b].pos) == ((4, 3), (3, 3))


@pytest.mark.parametrize("mode", ["immediate", "gestation"])
def test_birth_energy_cooldown_identity_and_next_tick(mode):
    env = world(founders=1, metabolism=0, reproduction_mode=mode)
    safe(env)
    parent = env.agents[0]
    env.prey[parent].pos, env.prey[parent].energy = (5, 5), 90
    before = sum(p.energy for p in env.prey.values())
    obs, _, terms, _, _ = env.step({parent: REST})
    if mode == "gestation":
        assert env.prey[parent].energy == before - 40
        assert env.prey[parent].cooldown == 0 and env.total_births == 0
        assert env.prey[parent].pregnancy.completion_tick == 16
        for _ in range(15):
            env.step({parent: REST})
            assert env.total_births == 0
        obs, _, terms, _, _ = env.step({parent: REST})
        birth = next(e for e in env.events if e["type"] == "birth")
        assert birth["birth_tick"] == 16 and birth["conception_tick"] == 0
    child = next(i for i in env.agents if i != parent)
    assert sum(p.energy for p in env.prey.values()) == before - 10
    assert env.prey[child].age == 0 and env.prey[child].energy == 30
    assert env.prey[child].generation == 1 and env.prey[child].parent == parent
    assert env.prey[parent].cooldown == 128
    assert child in obs and not terms[child]
    env.step(dict.fromkeys(env.agents, REST))
    assert env.prey[child].age == 1


def test_cap_defers_without_cost():
    env = world(founders=1, population_cap=1, metabolism=0)
    safe(env)
    p = env.prey[env.agents[0]]
    p.energy = 90
    env.step({p.id: REST})
    assert p.energy == 90 and p.cooldown == 0


def test_watch_alarm_and_opportunity_cost():
    env = world(founders=2, metabolism=0)
    safe(env)
    a, b = env.agents
    env.prey[a].pos, env.prey[b].pos = (5, 5), (5, 9)
    env.predator = (8, 5)
    assert (env.observe(a)["threats"] == -1).all()
    env.step({a: WATCH, b: REST})
    assert env.prey[a].energy == pytest.approx(59.9)
    assert env.prey[a].alert_until == env.prey[b].alert_until == 1
    assert (env.observe(b)["threats"][0] == (8, 5)).all()
    env.step({a: REST, b: REST})
    assert env.observe(a)["self"][7] == 0


def test_no_hidden_state_leak_and_los():
    env = world(founders=1)
    safe(env)
    i = env.agents[0]
    env.prey[i].pos = (5, 5)
    env.predator = (7, 5)
    env.obstacles = {(6, 5)}
    obs = env.observe(i)
    assert obs["local"][2, 3] == 1 and obs["local"][2, 4] == -1
    assert (obs["threats"] == -1).all()
    before = observation_text(obs)
    env.predator = (18, 18)
    env.rates[:] = 999
    env.obstacles.add((15, 15))
    assert before == observation_text(env.observe(i))


def test_rewards_after_death_and_birth_fixed_denominator():
    env = world(founders=2, metabolism=1, social_denominator=16)
    safe(env)
    a, b = env.agents
    env.prey[a].energy = 0.5
    _, _, terms, _, infos = env.step({a: REST, b: REST})
    assert terms[a] and not terms[b]
    assert infos[a]["social_reward"] == 1 / 16
    assert infos[b]["social_reward"] == 0
    assert a not in env.agents
    with pytest.raises(ValueError):
        env.step({a: REST, b: REST})


def test_fork_coupled_randomness():
    env = world()
    fork = env.fork()
    actions = dict.fromkeys(env.agents, REST)
    env.step(actions)
    fork.step(actions)
    assert env.render() == fork.render()
    assert np.array_equal(env.stock, fork.stock)


def test_attack_probability_and_cooldown():
    class FixedRng:
        def __init__(self):
            self.draws = iter([0.95, 0.30, 0.95])  # stay, attack draw, stay again

        def random(self):
            return next(self.draws)

        def integers(self, *args):
            return 0

        def choice(self, values):
            return values[0]

    for alert in [False, True]:
        env = world(founders=1)
        safe(env)
        p = env.prey[env.agents[0]]
        p.pos, env.predator = (5, 5), (5, 6)
        p.alert_until = 1 if alert else -1
        env.attack_ready = 0
        env.rng = FixedRng()
        env._predator()
        assert (p.id in env.prey) == alert  # 0.30 < 0.45, but 0.30 > 0.15
        assert env.attack_ready == 8
        if alert:
            env.tick = 1
            env._predator()
            assert sum(e["type"] == "attack" for e in env.events) == 1


def test_pending_birth_writes_once_and_does_not_recheck_energy():
    env = world(founders=1, metabolism=0, gestation_ticks=2)
    safe(env)
    p = env.prey[env.agents[0]]
    p.pos, p.energy = (5, 5), 90
    calls = []
    def writer(history, inherited):
        calls.append((history, inherited))
        return "Watch for predators. Seek observed food.", 8
    env.step({p.id: REST}, note_writer=writer)
    env.obstacles = set(env.neighbors(p.pos))
    for _ in range(4):
        env.step({p.id: REST}, note_writer=writer)
    assert len(calls) == 1 and len(calls[0][0]) == 3
    assert p.energy == 50 and p.cooldown == 0
    assert p.pregnancy.remaining(env.tick) == 0
    assert env.total_births == 0
    fork = env.fork()
    for candidate in (env, fork):
        candidate.obstacles = set()
        candidate.step({p.id: REST}, note_writer=writer)
        child = next(q for q in candidate.prey.values() if q.parent)
        assert child.survival_note == "Watch for predators. Seek observed food."
        assert candidate.observe(child.id)["survival_note"] == child.survival_note
        assert candidate.observation_space(child.id).contains(candidate.observe(child.id))
        assert candidate.total_conceptions == candidate.total_births == 1
        assert candidate.birth_wait_ticks == 3
    assert len(calls) == 1


@pytest.mark.parametrize("cause", ["starvation", "predation"])
def test_parent_death_loses_pregnancy_without_refund(cause):
    env = world(founders=1, metabolism=0, gestation_ticks=2)
    safe(env)
    p = env.prey[env.agents[0]]
    p.energy = 90
    env.step({p.id: REST})
    env.events = []
    env._kill(p.id, cause)
    assert p.energy == 50 and env.total_births == 0
    loss = next(e for e in env.events if e["type"] == "pregnancy_loss")
    assert loss["conception_tick"] == 0 and loss["gestation_completion_tick"] == 2
    assert env.pregnancy_losses == 1


def test_capacity_blocks_birth_but_not_multiple_pregnancies():
    env = world(founders=2, population_cap=3, metabolism=0, gestation_ticks=1)
    safe(env)
    for j, p in enumerate(env.prey.values()):
        p.energy, p.pos = 90, (5 + j * 5, 5)
    env.step(dict.fromkeys(env.agents, REST))
    assert env.total_conceptions == 2
    env.step(dict.fromkeys(env.agents, REST))
    assert env.total_births == 1 and len(env.prey) == 3
    waiting = next(p for p in env.prey.values() if p.pregnancy)
    assert waiting.energy == 50 and waiting.cooldown == 0
    child = next(p for p in env.prey.values() if p.parent)
    env._kill(child.id, "predation")
    env.agents = list(env.prey)
    env.step(dict.fromkeys(env.agents, REST))
    assert env.total_births == 2 and waiting.energy == 50


def test_iteration_disables_pregnancy_and_observes_own_state_only():
    env = PopulationEnv(EnvironmentConfig(founders=2), reproduction=False)
    env.reset(seed=11)
    safe(env)
    a, b = env.agents
    env.prey[a].energy = 90
    env.prey[a].survival_note = "Private advice: café food may be scarce."
    env.step(dict.fromkeys(env.agents, REST))
    assert env.total_conceptions == env.total_births == 0
    assert env.prey[a].pregnancy is None
    assert env.observe(a)["self"][10:].tolist() == [0, 0]
    assert env.observe(b)["survival_note"] == ""
    assert env.observation_space(a).contains(env.observe(a))


@pytest.mark.parametrize("mode", ["immediate", "gestation"])
def test_no_adjacent_space_only_blocks_conception_in_immediate_mode(mode):
    env = world(founders=1, metabolism=0, reproduction_mode=mode)
    safe(env)
    p = env.prey[env.agents[0]]
    p.pos, p.energy = (5, 5), 90
    env.obstacles = set(env.neighbors(p.pos))
    env.step({p.id: REST})
    assert p.energy == (90 if mode == "immediate" else 50)
    assert (p.pregnancy is not None) == (mode == "gestation")
    assert env.observe(p.id)["self"].shape == ((10,) if mode == "immediate" else (12,))


def test_pending_pregnancy_serialization_and_private_history_bound():
    import pickle

    env = world(founders=1, metabolism=0, gestation_ticks=130)
    safe(env)
    p = env.prey[env.agents[0]]
    p.pos, p.energy = (5, 5), 90
    env.step({p.id: REST})
    env.cfg = replace(env.cfg, population_cap=1)
    for _ in range(130):
        env.step({p.id: REST}, note_writer=lambda h, n: ("Keep watching.", 3))
    restored = pickle.loads(pickle.dumps(env))
    q = restored.prey[p.id]
    assert len(q.history) == 128
    assert q.history[0].startswith("tick=3 ") and q.history[-1].startswith("tick=130 ")
    assert q.pregnancy.survival_note == "Keep watching."
    restored.cfg = replace(restored.cfg, population_cap=2)
    restored.step({q.id: REST}, note_writer=lambda h, n: pytest.fail("Note was already prepared"))
    child = next(p for p in restored.prey.values() if p.parent)
    assert child.survival_note == "Keep watching." and not child.history
    assert child.pregnancy is None


def test_reproduction_config_validation_and_digest():
    from millstlabs.config import ExperimentConfig

    cfg = ExperimentConfig()
    original = cfg.digest()
    cfg.environment.gestation_ticks = 8
    assert cfg.digest() != original
    cfg.environment.reproduction_mode = "immediate"
    legacy = cfg.digest()
    cfg.environment.gestation_ticks = 16
    assert cfg.digest() == legacy
    cfg.environment.reproduction_mode = "unknown"
    with pytest.raises(AssertionError):
        cfg.validate()
    cfg.environment.reproduction_mode = "gestation"
    cfg.environment.gestation_ticks = 0
    with pytest.raises(AssertionError):
        cfg.validate()
