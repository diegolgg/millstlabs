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


def test_birth_energy_cooldown_identity_and_next_tick():
    env = world(founders=1, metabolism=0)
    safe(env)
    parent = env.agents[0]
    env.prey[parent].pos, env.prey[parent].energy = (5, 5), 90
    before = sum(p.energy for p in env.prey.values())
    obs, _, terms, _, _ = env.step({parent: REST})
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
