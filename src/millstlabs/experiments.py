import json
import math
import time
from pathlib import Path

import numpy as np

from .env import WATCH, PopulationEnv
from .observations import Heuristic


def matrix(cfg, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"index": j, "profile": p, "method": m, "seed": s, "decisions": cfg.training.decisions,
             "config_digest": cfg.digest()}
            for j, (p, m, s) in enumerate((p, m, s) for p in cfg.profiles for m in cfg.methods for s in cfg.seeds)]
    output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"runs": len(rows), "training_decisions": len(rows) * cfg.training.decisions,
            "warmstart_decisions": len(cfg.seeds) * cfg.training.warmstart_transitions,
            "manifest": str(output)}


def calibrate(cfg, seeds, ticks, output):
    rows = []
    for reproduction in [False, True]:
        for mode in ["random", "food", "vigilant"]:
            for seed in seeds:
                env = PopulationEnv(cfg.environment, reproduction=reproduction)
                obs, _ = env.reset(seed=seed)
                policies = {i: Heuristic(seed + j, vigilance=mode == "vigilant") for j, i in enumerate(env.agents)}
                rng = np.random.default_rng(seed + 7)
                initial = list(env.agents)
                lifetime = dict.fromkeys(initial, ticks)
                total_population = cap_ticks = watch = decisions = 0
                max_generation = 0
                newborn_deaths = 0
                for t in range(ticks):
                    if not env.agents:
                        break
                    for i in env.agents:
                        if i not in policies:
                            policies[i] = Heuristic(seed + env.next_id, vigilance=mode == "vigilant")
                    actions = {i: int(rng.choice(np.flatnonzero(obs[i]["action_mask"]))) if mode == "random"
                               else policies[i].act(obs[i]) for i in env.agents}
                    decisions += len(actions)
                    watch += sum(a == WATCH for a in actions.values())
                    obs, *_ = env.step(actions)
                    total_population += len(env.agents)
                    cap_ticks += len(env.agents) == cfg.environment.population_cap
                    max_generation = max(max_generation, env.metrics()["generation_max"])
                    for event in env.events:
                        if event["type"] == "death":
                            if event["id"] in lifetime:
                                lifetime[event["id"]] = env.tick
                            elif event["age"] < cfg.environment.maturity:
                                newborn_deaths += 1
                rows.append({"seed": seed, "mode": mode, "reproduction": reproduction, "ticks": env.tick,
                             "extinct": not env.agents, "restricted_mean_lifetime": float(np.mean(list(lifetime.values()))),
                             "survival_fraction": sum(i in env.agents for i in initial) / len(initial),
                             "mean_population": total_population / ticks, "cap_fraction": cap_ticks / ticks,
                             "watch_rate": watch / max(decisions, 1), "max_generation": max_generation,
                             "newborn_deaths": newborn_deaths, **env.metrics()})
    # Report observations, never automatically tune the frozen test distribution.
    standard = [r for r in rows if not r["reproduction"]]
    scores = {m: float(np.mean([r["restricted_mean_lifetime"] for r in standard if r["mode"] == m]))
              for m in ["random", "food", "vigilant"]}
    fertile = [r for r in rows if r["reproduction"] and r["mode"] != "random"]
    checks = {"food_outperforms_random": scores["food"] > scores["random"],
              "multiple_generations": any(r["max_generation"] >= 2 for r in fertile),
              "some_newborns_mature": any(r["matured"] > 0 for r in fertile),
              "some_newborns_die": any(r["newborn_deaths"] > 0 for r in fertile),
              "not_always_at_cap": all(r["cap_fraction"] < 0.9 for r in fertile),
              "not_universal_survival": any(r["survival_fraction"] < 1 for r in standard if r["mode"] != "random"),
              "vigilance_lifetime_improvement": scores["vigilant"] > scores["food"]}
    result = {"config_digest": cfg.digest(), "ticks": ticks, "seeds": seeds, "scores": scores,
              "checks": checks, "pilot_passed": all(checks.values()), "rows": rows,
              "note": "Heuristic comparison is descriptive. Confirm causal WATCH benefit/cost with controlled probes."}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    return {k: v for k, v in result.items() if k != "rows"}


def benchmark(cfg, ticks, output):
    import torch

    from .policy import PolicyBank
    from .ppo import Transition, update
    from .trainer import seed_all
    seed_all(101)
    env = PopulationEnv(cfg.environment, reproduction=False)
    obs, _ = env.reset(seed=101)
    bank = PolicyBank(cfg.training, next(iter(obs.values())))
    hidden, buffers = {}, {}
    for i in env.agents:
        bank.add(i)
        hidden[i] = bank.zero_hidden()
        buffers[i] = []
    initial = next(iter(bank.controllers))
    private = sum(p.numel() for g in bank.optimizers[initial].param_groups for p in g["params"])
    if bank.device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    decisions = 0
    for _ in range(ticks):
        if not env.agents:
            break
        actions, records = {}, {}
        for i in env.agents:
            h = hidden[i]
            a, lp, vp, vs, hidden[i] = bank.act(i, obs[i], h)
            actions[i] = a
            records[i] = (obs[i], h.cpu(), a, lp, vp, vs)
        obs, _, terms, _, info = env.step(actions)
        decisions += len(actions)
        for i, record in records.items():
            buffers[i].append(Transition(*record, info[i]["personal_reward"], info[i]["social_reward"], terms[i], False))
    for i, seq in buffers.items():
        update(bank, i, seq, 0)
    if bank.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    n_runs = len(cfg.profiles) * len(cfg.methods) * len(cfg.seeds)
    result = {"backend": cfg.training.backend, "device": str(bank.device), "decisions": decisions,
              "resolved_revision": bank.resolved_revision,
              "seconds": seconds, "decisions_per_second_with_ppo": decisions / seconds,
              "private_trainable_parameters_per_agent": private, "resources": bank.resource_metrics(),
              "estimated_training_hours_per_run": cfg.training.decisions / decisions * seconds / 3600,
              "estimated_training_hours_sweep": n_runs * cfg.training.decisions / decisions * seconds / 3600,
              "evaluation_decision_upper_bound_per_run": cfg.training.evaluation_policies * cfg.training.evaluation_horizon * (
                  cfg.training.evaluation_maps * len([c for c in cfg.training.checkpoints if c < cfg.training.decisions])
                  + cfg.training.final_evaluation_maps),
              "caveat": "Short sample estimate; excludes evaluation, warm starts, inheritance probes and I/O. Benchmark on target GPU."}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    return result


def analyze(root, output):
    """Aggregate each trained run first, then paired differences across training seeds."""
    groups, runs, ecological, newborn = {}, [], [], []
    paired_warmstarts = {}
    for file in Path(root).rglob("evaluation.jsonl"):
        manifest = file.parent / "manifest.jsonl"
        meta = json.loads(manifest.read_text().splitlines()[0])
        pair_key = (meta["config_digest"], meta["seed"])
        warm_hash = meta.get("warmstart_sha256")
        if pair_key in paired_warmstarts and paired_warmstarts[pair_key] != warm_hash:
            raise ValueError(f"Mismatched warm starts among paired runs for seed {meta['seed']}")
        paired_warmstarts[pair_key] = warm_hash
        evaluations = [json.loads(line) for line in file.read_text().splitlines()]
        # A resumed run can contain duplicate eval rows; take latest row per threshold.
        latest = {r["checkpoint"]: r for r in evaluations}
        def read_log(name):
            path = file.parent / f"{name}.jsonl"
            return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        ecology = read_log("ecology")
        events = read_log("events")
        if ecology:
            trial_ends = {row["trial"]: row for row in ecology}
            births = sum(row["type"] == "birth" for row in events)
            matured = sum(row["type"] == "maturation" for row in events)
            ecological.append({"profile": meta["profile"], "method": meta["method"], "seed": meta["seed"],
                               "config_digest": meta["config_digest"],
                               "mean_population": float(np.mean([r["population"] for r in ecology])),
                               "fraction_at_cap": float(np.mean([r["at_cap"] for r in ecology])),
                               "births_per_1000_ticks": births / len(ecology) * 1000,
                               "newborn_maturation_fraction": matured / births if births else None,
                               "births": births, "matured": matured,
                               "starvation": sum(r["starvation"] for r in trial_ends.values()),
                               "predation": sum(r["predation"] for r in trial_ends.values()),
                               "food_consumed": sum(r["consumption"] for r in trial_ends.values()),
                               "maximum_generation": max(r["generation_max"] for r in ecology),
                               "mean_living_lineages": float(np.mean([r["lineages"] for r in ecology])),
                               "extinction_times": [r["time_to_extinction"] for r in read_log("extinction")],
                               "external_restarts": len(read_log("recovery"))})
        generation_groups = {}
        for row in read_log("newborn_evaluation"):
            generation_groups.setdefault(row["generation"], []).append(row["restricted_mean_lifetime"])
        for generation, values in generation_groups.items():
            newborn.append({"profile": meta["profile"], "method": meta["method"], "seed": meta["seed"],
                            "generation": generation, "sampled_newborns": len(values),
                            "mean_frozen_newborn_lifetime": float(np.mean(values))})
        for threshold, row in latest.items():
            run = {"profile": meta["profile"], "method": meta["method"], "seed": meta["seed"],
                   "checkpoint": threshold, "score": row["restricted_mean_lifetime"],
                   "config_digest": meta["config_digest"]}
            runs.append(run)
            key = (run["config_digest"], run["profile"], run["method"], threshold)
            if run["seed"] in groups.get(key, {}):
                raise ValueError(f"Duplicate training replicate {key} seed={run['seed']}; choose one output root")
            groups.setdefault(key, {})[run["seed"]] = run["score"]
    def summarize(values):
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        # Run-level bootstrap. No pooling of evaluation episodes as independent replicates.
        rng = np.random.default_rng(419)
        draws = rng.choice(arr, (10_000, n), replace=True).mean(1)
        return {"n_training_seeds": n, "mean": float(arr.mean()),
                "standard_error": float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
                "bootstrap_95_percent_interval": np.quantile(draws, [0.025, 0.975]).tolist() if n > 1 else None}
    summaries = [{"config_digest": key[0], "profile": key[1], "method": key[2], "checkpoint": key[3],
                  **summarize(list(values.values()))} for key, values in groups.items()]
    contrasts = []
    for (digest, profile, method, checkpoint), adult in groups.items():
        if method != "r_adult":
            continue
        for control in ["r_initial", "iteration"]:
            other = groups.get((digest, profile, control, checkpoint), {})
            seeds = sorted(adult.keys() & other.keys())
            if seeds:
                contrasts.append({"config_digest": digest, "profile": profile, "checkpoint": checkpoint,
                                  "contrast": f"r_adult - {control}", "paired_seeds": seeds,
                                  **summarize([adult[s] - other[s] for s in seeds])})
    result = {"summaries": summaries, "paired_contrasts": contrasts, "ecological_runs": ecological,
              "newborn_generation_runs": newborn,
              "note": "Uncertainty unit is the independent training seed. Five seeds provide limited precision."}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2))
    return result
