"""Asynchronous pairwise flag game; probes are synchronous, with fixed ex ante contacts."""
import hashlib
import time

import numpy as np

from .corpus import VerifiedCorpus
from .task import crops_for, endpoint


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:4], "little") % (2**31-1)


def observation(crop, memory, last_country, countries, fraction, probe, mask):
    # Only the agent's own crop, received transcript, last response and own tool mask.
    # The controller sees neither target label, global crop location nor unseen deposits.
    country_features = np.zeros((2, len(countries)+1), np.float32)
    country_features[0, countries.index(last_country) if last_country in countries else -1] = 1
    for line in memory:
        label = line.split(" | ")[0].split(": ")[-1]
        if label in countries:
            country_features[1, countries.index(label)] += 1/max(1, len(memory))
    return np.concatenate([crop.flatten()/255, country_features.flatten(), np.asarray(mask, float),
                           [fraction, float(probe), len(memory)/8,
                            sum(s.startswith("Verified sensor fact:") for s in memory)/8]]).astype(np.float32)


def feature_width(cfg, countries):
    return cfg.crop_height*cfg.crop_width*3 + 2*(len(countries)+1) + cfg.crop_height*cfg.crop_width+2 + 4


def run_episode(cfg, trial, images, provider, controllers, mode):
    started = time.perf_counter()
    countries = list(images)
    crops = crops_for(trial, images, cfg)
    corpus = VerifiedCorpus(crops, mode, cfg.retrieval_records)
    slots = cfg.crop_width*cfg.crop_height
    memory = [[] for _ in crops]
    guesses = ["__invalid__"]*cfg.population
    policy_rng = [np.random.default_rng(seed_for(cfg.policy_seed, trial.seed, i, "tools")) for i in range(cfg.population)]
    trajectories = [[] for _ in crops]
    traces, tool_events, fingerprints, model_versions = [], [], set(), set()
    invalid = query_count = 0
    trace_calls = []
    queries_before = provider.requests
    hits_before = provider.cache_hits
    prompt_before, completion_before = provider.prompt_tokens, provider.completion_tokens

    def append(agent, text):
        memory[agent].append(text)
        memory[agent] = memory[agent][-cfg.memory_entries:]

    def query_group(agents, phase, index, bandwidth, tools=True):
        nonlocal invalid, query_count
        if tools and mode != "off":
            actions = {}
            for i in agents:
                mask = corpus.mask(i, slots)
                obs = observation(crops[i], memory[i], guesses[i], countries,
                                  min(index/(cfg.interaction_rounds*cfg.population), 1.), phase == "probe", mask)
                a, logp, value = controllers.act(i, obs, mask, policy_rng[i])
                actions[i] = a
                trajectories[i].append({"observation": obs, "mask": mask, "action": a, "logp": logp, "value": value})
            retrieved = corpus.execute(actions, slots)
            for i, facts in retrieved.items():
                for fact in facts:
                    append(i, fact.text())
            tool_events.extend({"phase": phase, "interaction": index, **e} for e in corpus.events)
            corpus.events.clear()
        reports = []
        for i in agents:
            response = provider.complete(crops[i], list(memory[i]), countries, bandwidth,
                                         seed_for(trial.seed, phase, index, i))
            query_count += 1
            invalid += not response["valid"]
            guesses[i] = response["country"]
            model_versions.add(response.get("model"))
            fingerprints.add(response.get("system_fingerprint"))
            trace_calls.append({"phase": phase, "interaction": index, "agent": i,
                                "transcript": list(memory[i]), **response})
            reports.append(response)
        return reports

    initial = [r["country"] for r in query_group(range(cfg.population), "initial", 0, 1, tools=False)]
    final = list(initial)
    consensus_streak = 0
    stop = "round_limit"
    for round_index in range(cfg.interaction_rounds):
        for j in range(cfg.population):
            index = round_index*cfg.population+j
            speaker, listener = trial.contacts[index]
            report = query_group([speaker], "interaction", index, cfg.bandwidth)[0]
            text = f"Agent {speaker}: {report['country']}"
            if cfg.bandwidth == 3:
                text += " | "+report["reason"]
            append(listener, text)
        final = [r["country"] for r in query_group(range(cfg.population), "probe",
                                                  (round_index+1)*cfg.population, 1)]
        traces.append({"round": round_index+1, "interactions": (round_index+1)*cfg.population,
                       "guesses": list(final), **endpoint(initial, final, trial.country, cfg), **corpus.metrics()})
        full = len(set(final)) == 1 and final[0] in countries
        consensus_streak = consensus_streak+1 if full else 0
        if consensus_streak >= cfg.consensus_patience:
            stop = "five_consecutive_full_consensus_probes" if cfg.consensus_patience == 5 else "consensus_patience"
            break
    result = {"trial_seed": trial.seed, "trial_hash": trial.hash, "country": trial.country,
              "mode": mode, "initial_guesses": initial, "final_guesses": final,
              **endpoint(initial, final, trial.country, cfg), **corpus.metrics(),
              "rounds": len(traces), "stop": stop, "queries": query_count,
              "new_requests": provider.requests-queries_before, "cache_hits": provider.cache_hits-hits_before,
              "prompt_tokens": provider.prompt_tokens-prompt_before,
              "completion_tokens": provider.completion_tokens-completion_before,
              "invalid_responses": invalid, "model_versions": sorted(model_versions, key=str),
              "system_fingerprints": sorted(fingerprints, key=str), "elapsed_seconds": time.perf_counter()-started,
              "curve": traces, "calls": trace_calls, "corpus_events": tool_events}
    return result, trajectories, [int(c == trial.country) for c in final]
