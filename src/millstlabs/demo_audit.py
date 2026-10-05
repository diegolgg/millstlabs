"""Audit whether note demonstrations depend on peer information; no LLM required.

Replay the observation-limited grounded teacher on its training trajectories.
Counterfactuals share the same physical observation and tie-breaking RNG. This
measures label dependence, not the survival of a separately simulated population.
"""
import argparse
import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .env import ACTION_NAMES, PopulationEnv
from .knowledge import make_corpus, next_observations
from .notes import publish_notes
from .observations import Heuristic


def compare_teacher(helper, private_helper, observation, agent):
    without = copy.deepcopy(observation)
    without["corpus"][without["corpus"][:, 5] == 1] = -1
    without["notes"] = [n for n in without["notes"] if n["author"] == agent]
    immediate = copy.deepcopy(helper).act(without)
    # The private helper retains only firsthand/own-note walls. Synchronizing
    # tie-breaking randomness isolates information from arbitrary route ties.
    private_helper.rng.bit_generator.state = copy.deepcopy(helper.rng.bit_generator.state)
    private_action = private_helper.act(without)
    action = helper.act(observation)
    return action, immediate, private_action


def audit(cfg, seed, limit=None):
    if cfg.training.corpus_interface != "notes":
        raise ValueError("This audit requires the notes interface")
    training = replace(cfg.training, note_style="grounded")
    budget = training.warmstart_transitions if limit is None else limit
    if budget <= 0:
        raise ValueError("Demonstration count must be positive")
    env = PopulationEnv(cfg.environment, reproduction=False)
    observations, _ = env.reset(seed=seed)
    corpus, observations = make_corpus(training, observations)
    helpers = {i: Heuristic(seed+j) for j, i in enumerate(env.agents)}
    private_helpers = copy.deepcopy(helpers)
    counts, examples = Counter(), []
    decisions = trial = 0
    while decisions < budget:
        actions = {}
        for agent in env.agents:
            obs = observations[agent]
            action, immediate, private_action = compare_teacher(helpers[agent], private_helpers[agent], obs, agent)
            actions[agent] = action
            if decisions >= budget:
                continue
            counts[ACTION_NAMES[action]] += 1
            has_peer = any(n["author"] != agent for n in obs["notes"])
            counts["with_peer_context"] += int(has_peer)
            counts["teacher_action_changed_by_current_peer_note"] += int(action != immediate)
            counts["teacher_action_changed_by_any_peer_memory"] += int(action != private_action)
            counts["movement_decisions"] += int(action < 4)
            counts["peer_context_movement_decisions"] += int(action < 4 and has_peer)
            if action != private_action and len(examples) < 5:
                examples.append({"trial": trial, "tick": env.tick, "position": obs["self"][2:4].tolist(),
                                 "action_with": ACTION_NAMES[action], "action_without": ACTION_NAMES[private_action],
                                 "notes": obs["notes"]})
            decisions += 1
        publish_notes(corpus, {i: int(observations[i]["note_opportunity"]) for i in env.agents},
                      observations, None, env, training)
        observations, *_ = env.step(actions)
        observations = next_observations(corpus, observations, env.agents)
        corpus.events.clear()
        if not env.agents:
            trial += 1
            observations, _ = env.reset(seed=seed+trial*1009)
            corpus, observations = make_corpus(training, observations)
            helpers = {i: Heuristic(seed+trial*1009+j) for j, i in enumerate(env.agents)}
            private_helpers = copy.deepcopy(helpers)
    return {"config_digest": cfg.digest(), "seed": seed, "demonstrations": decisions,
            "counts": dict(counts), "example_action_changes": examples,
            "interpretation": "Grounded teacher replay, not an LLM run or separate survival rollout. Immediate removal retains prior imported wall memory; the private-history check excludes all peer wall memory. Both use identical physical observations and tie-breaking RNG. Zero differences expose absent action-dependent supervision on these trajectories, not impossibility of useful communication."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/corpus-notes/grounded/config.yaml")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--decisions", type=int)
    parser.add_argument("--output", default="runs/corpus-notes-demo-audit.json")
    args = parser.parse_args()
    result = audit(load_config(args.config), args.seed, args.decisions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
