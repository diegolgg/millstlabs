import copy
from dataclasses import replace

import numpy as np
import torch

from .env import FEED, WATCH, PopulationEnv


def calibration_histories(env_cfg, seed=8675309):
    """Fixed observation histories, independent of train and held-out evaluation maps."""
    from .observations import Heuristic
    env = PopulationEnv(replace(env_cfg, max_individuals=1000), reproduction=False)
    obs, _ = env.reset(seed=seed)
    histories = {i: [] for i in env.agents[:4]}
    helpers = {i: Heuristic(seed + j) for j, i in enumerate(env.agents)}
    for _ in range(8):
        for i in histories:
            if i in env.agents:
                histories[i].append(copy.deepcopy(obs[i]))
        obs, *_ = env.step({i: helpers[i].act(obs[i]) for i in env.agents})
    return list(histories.values())


def evaluate(bank, policies, env_cfg, train_cfg, checkpoint, final=False):
    """Same unseen environments, body resets, and action RNG seeds for all conditions."""
    count = train_cfg.final_evaluation_maps if final else train_cfg.evaluation_maps
    rng = np.random.default_rng(train_cfg.evaluation_seed)
    selected = rng.choice(policies, train_cfg.evaluation_policies, replace=len(policies) < train_cfg.evaluation_policies).tolist()
    rows = []
    for episode in range(count):
        seed = train_cfg.evaluation_seed + episode
        env = PopulationEnv(replace(env_cfg, founders=len(selected), max_individuals=1000), reproduction=False)
        observations, _ = env.reset(seed=seed)
        assignment = dict(zip(env.agents, selected))
        hidden = {i: bank.zero_hidden() for i in env.agents}
        generators = {i: torch.Generator().manual_seed(seed * 100 + j) for j, i in enumerate(env.agents)}
        lifetimes = dict.fromkeys(env.agents, train_cfg.evaluation_horizon)
        watch_count = feed_count = decisions = 0
        for _ in range(train_cfg.evaluation_horizon):
            if not env.agents:
                break
            actions = {}
            for i in env.agents:
                a, _, _, _, hidden[i] = bank.act(assignment[i], observations[i], hidden[i], generators[i])
                actions[i] = a
                watch_count += a == WATCH
                feed_count += a == FEED
            decisions += len(actions)
            observations, _, _, _, _ = env.step(actions)
            for event in env.events:
                if event["type"] == "death":
                    lifetimes[event["id"]] = env.tick
        rows.append({"map_seed": seed, "restricted_mean_lifetime": float(np.mean(list(lifetimes.values()))),
                     "survival_fraction": len(env.agents) / len(selected), "lifetimes": lifetimes,
                     "starvation": env.deaths["starvation"], "predation": env.deaths["predation"],
                     "decisions": decisions, "watch_rate": watch_count / max(decisions, 1),
                     "feed_rate": feed_count / max(decisions, 1)})
    return {"checkpoint": checkpoint, "sampled_policies": selected, "episodes": rows,
            "restricted_mean_lifetime": float(np.mean([r["restricted_mean_lifetime"] for r in rows])),
            "survival_fraction": float(np.mean([r["survival_fraction"] for r in rows]))}


def probe_histories(histories):
    """Counterfactual observation probes, not transplanted recurrent states.

    These deliberately controlled sensory inputs do not claim to be native episodes.
    Their final observation varies; each policy replays the common preceding history.
    """
    history = histories[0]
    variants = {}
    for name in ["self_threat", "companion_threat", "scarce_food_weak_peer", "others_watching", "safe_abundance"]:
        h = copy.deepcopy(history)
        o = h[-1]
        x, y = (int(v) for v in o["self"][2:4])
        o["self"][0] = 70
        o["self"][7] = 0
        o["local"][:] = 0
        o["companions"][:] = -1
        o["threats"][:] = -1
        o["action_mask"][:] = 1
        o["stations"][0] = [x, y, 60, o["self"][6]]
        if name == "self_threat":
            o["threats"][0] = [min(x + 1, int(o["self"][9]) - 1), y]
            o["local"][2, 3] = 4
        if name in {"companion_threat", "others_watching", "scarce_food_weak_peer"}:
            px = x + (1 if x < int(o["self"][9]) - 2 else -1)
            o["companions"][0] = [px, y, 5, name == "others_watching"]
            o["local"][2, 3 if px > x else 1] = 3
        if name in {"companion_threat", "others_watching"}:
            o["threats"][0] = [px + (1 if px > x else -1), y]
        if name == "scarce_food_weak_peer":
            o["stations"][:, 2] = 1
        variants[name] = h
    return variants


def probe(bank, policies, histories):
    variants = probe_histories(histories)
    return {name: np.mean([bank.probabilities(i, [h])[-1].numpy() for i in policies], axis=0).tolist()
            for name, h in variants.items()}
