import copy
import time
from dataclasses import replace

import numpy as np
import torch

from .env import FEED, WATCH, PopulationEnv
from .knowledge import execute_tools, make_corpus, next_observations
from .notes import publish_notes


def calibration_histories(env_cfg, seed=8675309, training=None):
    """Fixed observation histories, independent of train and held-out evaluation maps."""
    from .observations import Heuristic
    env = PopulationEnv(replace(env_cfg, max_individuals=1000), reproduction=False)
    obs, _ = env.reset(seed=seed)
    corpus = None
    if training is not None:
        corpus, obs = make_corpus(training, obs, "private" if training.corpus_mode != "off" else "off")
    histories = {i: [] for i in env.agents[:4]}
    helpers = {i: Heuristic(seed + j) for j, i in enumerate(env.agents)}
    for _ in range(8):
        for i in histories:
            if i in env.agents:
                histories[i].append(copy.deepcopy(obs[i]))
        actions = {i: helpers[i].act(obs[i]) for i in env.agents}
        physical = execute_tools(corpus, actions, obs, env, training)
        obs, *_ = env.step(physical)
        obs = next_observations(corpus, obs, env.agents)
    return list(histories.values())


def evaluate(bank, policies, env_cfg, train_cfg, checkpoint, final=False, deliver_messages=None):
    """Same unseen environments, body resets, and action RNG seeds for all conditions."""
    count = train_cfg.final_evaluation_maps if final else train_cfg.evaluation_maps
    rng = np.random.default_rng(train_cfg.evaluation_seed)
    selected = rng.choice(policies, train_cfg.evaluation_policies, replace=len(policies) < train_cfg.evaluation_policies).tolist()
    rows = []
    for episode in range(count):
        start = time.perf_counter()
        tokens_before = bank.tokens
        seed = train_cfg.evaluation_seed + episode
        delivery = train_cfg.deliver_messages if deliver_messages is None else deliver_messages
        env = PopulationEnv(replace(env_cfg, founders=len(selected), max_individuals=1000), reproduction=False,
                            deliver_messages=delivery)
        observations, _ = env.reset(seed=seed)
        corpus, observations = make_corpus(train_cfg, observations)
        discovery_curve = [{"tick": 0, "decisions": 0, **corpus.metrics()}] if corpus is not None else []
        discovery_area = 0
        discovery_100_tick = None
        assignment = dict(zip(env.agents, selected))
        hidden = {i: bank.zero_hidden() for i in env.agents}
        generators = {i: torch.Generator().manual_seed(seed * 100 + j) for j, i in enumerate(env.agents)}
        lifetimes = dict.fromkeys(env.agents, train_cfg.evaluation_horizon)
        watch_count = feed_count = decisions = 0
        hungry_opportunities = hungry_feeds = 0
        for _ in range(train_cfg.evaluation_horizon):
            if not env.agents:
                break
            actions, writes = {}, {}
            for i in env.agents:
                a, _, _, _, hidden[i], publication = bank.act(assignment[i], observations[i], hidden[i], generators[i],
                                                train_cfg.evaluation_temperature, include_note=True)
                writes[i] = publication[0]
                actions[i] = a
                watch_count += a % 7 == WATCH
                feed_count += a % 7 == FEED
                opportunity = bool(observations[i]["self"][0] < 80 and observations[i]["action_mask"][FEED])
                hungry_opportunities += opportunity
                hungry_feeds += opportunity and a % 7 == FEED
            decisions += len(actions)
            publish_notes(corpus, writes, observations, bank, env, train_cfg, assignment)
            physical = execute_tools(corpus, actions, observations, env, train_cfg)
            observations, _, _, _, _ = env.step(physical)
            observations = next_observations(corpus, observations, env.agents)
            if corpus is not None:
                corpus.events.clear()
                discovered = corpus.metrics()["new_observed_facts"]
                discovery_area += discovered
                if discovered >= 100 and discovery_100_tick is None:
                    discovery_100_tick = env.tick
                if env.tick % 32 == 0 or not env.agents:
                    discovery_curve.append({"tick": env.tick, "decisions": decisions, **corpus.metrics()})
            for event in env.events:
                if event["type"] == "death":
                    lifetimes[event["id"]] = env.tick
        knowledge = {}
        if corpus is not None:
            if not discovery_curve or discovery_curve[-1]["tick"] != env.tick:
                discovery_curve.append({"tick": env.tick, "decisions": decisions, **corpus.metrics()})
            # Cumulative discoveries stay fixed after extinction; the time budget stays fixed too.
            discovery_area += (train_cfg.evaluation_horizon-env.tick)*corpus.metrics()["new_observed_facts"]
            knowledge = {**corpus.metrics(), "discovery_curve": discovery_curve,
                         "mean_new_facts_over_time": discovery_area/train_cfg.evaluation_horizon,
                         "ticks_to_100_new_facts": discovery_100_tick,
                         "allocated_decision_opportunities": train_cfg.evaluation_horizon*len(selected),
                         "discoveries_per_1000_opportunities": 1000*corpus.metrics()["new_observed_facts"]/(train_cfg.evaluation_horizon*len(selected)),
                         "discoveries_per_1000_actual_decisions": 1000*corpus.metrics()["new_observed_facts"]/max(1, decisions),
                         "tool_energy_cost": train_cfg.corpus_tool_cost*(corpus.counts["deposit_calls"]+corpus.counts["retrieve_calls"])}
        rows.append({"map_seed": seed, "restricted_mean_lifetime": float(np.mean(list(lifetimes.values()))),
                     **knowledge, "elapsed_seconds": time.perf_counter()-start, "processed_tokens": bank.tokens-tokens_before,
                     "survival_fraction": len(env.agents) / len(selected), "lifetimes": lifetimes,
                     "starvation": env.deaths["starvation"], "predation": env.deaths["predation"],
                     "decisions": decisions, "watch_rate": watch_count / max(decisions, 1),
                     "feed_rate": feed_count / max(decisions, 1), "food_consumed": env.total_consumption,
                     "food_per_alive_decision": env.total_consumption / max(1, decisions),
                     "hungry_feed_opportunities": hungry_opportunities, "hungry_feed_actions": hungry_feeds,
                     "hungry_feed_fraction": hungry_feeds / hungry_opportunities if hungry_opportunities else None,
                     "messages_sent": env.messages_sent, "messages_delivered": env.messages_delivered,
                     "predator_move_entropy": env.predator_entropy_sum / max(1, env.predator_kernel_ticks)})
    return {"checkpoint": checkpoint, "sampled_policies": selected, "episodes": rows,
            "policy_temperature": train_cfg.evaluation_temperature, "predator_temperature": env_cfg.predator_temperature,
            "deliver_messages": delivery,
            "corpus_mode": train_cfg.corpus_mode,
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
    return {name: np.mean([bank.probabilities(i, [h])[-1].numpy().reshape(-1, 7).sum(0) for i in policies], axis=0).tolist()
            for name, h in variants.items()}
