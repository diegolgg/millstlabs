"""Collection, per-agent PPO, birth inheritance, recovery, and resumable checkpoints."""
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import EnvironmentConfig, TrainingConfig
from .env import ACTION_NAMES, PopulationEnv
from .evaluation import calibration_histories, evaluate, probe
from .knowledge import CountNovelty, execute_tools, make_corpus, next_observations, novelty_beta, teacher_tool
from .notes import publish_notes
from .observations import Heuristic
from .policy import PolicyBank, cpu_state
from .ppo import Transition, update
from .tactics import TacticCorpus, record_experience


def save_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def warmstart(cfg, seed, path):
    """Behavior cloning from observation-limited demonstrations; one checkpoint/seed."""
    started = time.perf_counter()
    seed_all(seed)
    env = PopulationEnv(cfg.environment, reproduction=False)
    obs, _ = env.reset(seed=seed)
    warm_mode = cfg.training.corpus_mode if cfg.training.corpus_interface == "notes" else ("private" if cfg.training.corpus_mode != "off" else "off")
    corpus, obs = make_corpus(cfg.training, obs, warm_mode)
    bank = PolicyBank(cfg.training, next(iter(obs.values())))
    bank.add("warm")
    helpers = {i: Heuristic(seed + j) for j, i in enumerate(env.agents)} if cfg.training.warmstart_transitions else {}
    hidden = {i: bank.zero_hidden() for i in env.agents}
    buffers = defaultdict(list)
    decisions, trial = 0, 0
    next_progress = 512

    def train_sequence(i):
        seq = buffers.pop(i, [])
        if not seq:
            return
        outputs, _ = bank.sequence("warm", [x[0] for x in seq], seq[0][2])
        # Demonstrations supervise physical movement only, not arbitrary message symbols.
        loss = -torch.cat([o[0].probs.reshape(1, -1, 7).sum(1)[:, seq[j][1]].clamp_min(1e-12).log()
                          for j, o in enumerate(outputs)]).mean() if cfg.environment.message_symbols > 1 else -torch.cat([
                              o[0].log_prob(torch.tensor(seq[j][1], device=bank.device)) for j, o in enumerate(outputs)]).mean()
        bank.optimizers["warm"].zero_grad(set_to_none=True)
        gpu_start = bank.gpu_start()
        loss.backward()
        if cfg.environment.message_symbols > 1:
            # The exact marginal has zero message-head gradient; avoid Adam
            # amplifying floating-point cancellation noise during demonstrations.
            for parameter in bank.controllers["warm"].message.parameters():
                parameter.grad = None
        torch.nn.utils.clip_grad_norm_([p for g in bank.optimizers["warm"].param_groups for p in g["params"]], 0.5)
        bank.optimizers["warm"].step()
        bank.gpu_end(gpu_start)
        bank.gradient_updates += 1

    while decisions < cfg.training.warmstart_transitions:
        actions = {i: helpers[i].act(obs[i]) for i in env.agents}
        if corpus is not None and cfg.training.corpus_interface == "tools":
            actions = {i: a + 7*teacher_tool(corpus, i, obs[i], j) for j, (i, a) in enumerate(actions.items())}
        for i in env.agents:
            # Exactly the requested demonstration count (unused final tick actions aren't trained).
            if decisions < cfg.training.warmstart_transitions:
                buffers[i].append((copy.deepcopy(obs[i]), actions[i], hidden[i].detach()))
                with torch.no_grad():
                    _, hidden[i] = bank.sequence("warm", [obs[i]], hidden[i])
                decisions += 1
                if len(buffers[i]) == cfg.training.sequence_length:
                    train_sequence(i)
        if cfg.training.corpus_interface == "notes":
            publish_notes(corpus, {i: int(obs[i]["note_opportunity"]) for i in env.agents}, obs, bank, env,
                          cfg.training, {i: "warm" for i in env.agents})
        physical = execute_tools(corpus, actions, obs, env, cfg.training)
        obs, _, terms, _, _ = env.step(physical)
        obs = next_observations(corpus, obs, env.agents)
        if decisions >= next_progress:
            print(json.dumps({"phase": "warmstart", "seed": seed, "demonstrations": decisions,
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}), flush=True)
            next_progress = (decisions // 512 + 1) * 512
        for i, dead in terms.items():
            if dead:
                train_sequence(i)
        if not env.agents:
            trial += 1
            obs, _ = env.reset(seed=seed + trial * 1009)
            corpus, obs = make_corpus(cfg.training, obs, warm_mode)
            helpers = {i: Heuristic(seed + trial * 1009 + j) for j, i in enumerate(env.agents)}
            hidden = {i: bank.zero_hidden() for i in env.agents}
    for i in list(buffers):
        train_sequence(i)
    payload = {"format": 1, "state": bank.state("warm"), "seed": seed,
               "config": asdict(cfg), "config_digest": cfg.digest(), "transitions": decisions,
               "resolved_revision": bank.resolved_revision, "resources": bank.resource_metrics(),
               "elapsed_seconds": time.perf_counter() - started}
    if cfg.training.backend == "tiny":
        payload["tiny_base"] = cpu_state(bank.base.state_dict())
    save_atomic(path, payload)
    return payload


class Trainer:
    def __init__(self, cfg, profile, method, seed, output, warm_path, resume=None):
        if profile not in cfg.profiles or method not in cfg.methods:
            raise ValueError("Unknown profile or method")
        self.cfg, self.profile, self.method, self.seed = cfg, profile, method, seed
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if not resume and (self.output / "latest.pt").exists():
            raise FileExistsError("Run exists; use --resume rather than mixing log histories")
        seed_all(seed)
        self.preference = cfg.profiles[profile]["social_preference"]
        self.env = PopulationEnv(replace(cfg.environment), reproduction=method != "iteration", social_preference=self.preference,
                                 deliver_messages=cfg.training.deliver_messages)
        self.obs, _ = self.env.reset(seed=seed)
        self.corpus, self.obs = make_corpus(cfg.training, self.obs)
        self.novelty = CountNovelty()
        self.bank = PolicyBank(cfg.training, next(iter(self.obs.values())))
        # Load only locally generated/trusted torch checkpoints (pickle contains simulator state).
        warm = torch.load(warm_path, map_location="cpu", weights_only=False)
        self.warmstart_sha256 = hashlib.sha256(Path(warm_path).read_bytes()).hexdigest()
        if warm["seed"] != seed:
            raise ValueError("Warm-start seed must match paired training seed")
        for field in ["backend", "model_id", "revision", "max_tokens", "warmstart_transitions"]:
            if warm["config"]["training"][field] != getattr(cfg.training, field):
                raise ValueError(f"Warm-start mismatch: {field}")
        if asdict(EnvironmentConfig(**warm["config"]["environment"])) != asdict(cfg.environment):
            raise ValueError("Warm-start environment differs from frozen configuration")
        prior = TrainingConfig(**warm["config"]["training"])
        if prior.controller_architecture != cfg.training.controller_architecture:
            raise ValueError("Warm-start controller architecture differs")
        if cfg.training.split_controller:
            for field in ["separate_critic", "value_scale", "intrinsic_critic", "corpus_slots",
                          "corpus_interface", "actor_grounding", "note_style", "action_policy", "note_memory"]:
                if getattr(prior, field) != getattr(cfg.training, field):
                    raise ValueError(f"Warm-start architecture mismatch: {field}")
        if (prior.corpus_mode == "off") != (cfg.training.corpus_mode == "off"):
            raise ValueError("Warm-start corpus observation layout differs")
        if cfg.training.backend == "tiny":
            self.bank.base.load_state_dict(warm["tiny_base"])
        self.initial = copy.deepcopy(warm["state"])
        # Profile prior applied only after the shared warm start.
        self.initial["controller"]["watch_bias"] = torch.tensor(float(cfg.profiles[profile]["vigilance_bias"]))
        self.histories = calibration_histories(cfg.environment, training=cfg.training)
        self.hidden, self.buffers, self.dead, self.archive = {}, defaultdict(list), {}, []
        self.death_serial = 0
        self.decisions = self.trial = self.world_ticks = self.restarts = 0
        self.completed_evaluations = []
        self.elapsed_seconds = 0.0
        self.action_rng = torch.Generator().manual_seed(seed + 13)
        self.mutation_rng = torch.Generator().manual_seed(seed + 29)
        self.newborn_tracking = {}
        self.births_total = 0
        self.evaluation_decisions = 0
        if resume:
            self.restore(resume)
        else:
            for i in self.env.agents:
                self.bank.add(i, self.initial)
                self.hidden[i] = self.bank.zero_hidden()
                self.env.prey[i].lineage = f"trial0:{i}"
            if isinstance(self.corpus, TacticCorpus):
                self.corpus.owners = {i: p.lineage for i, p in self.env.prey.items()}
        (self.output / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
        self.log("manifest", {"config_digest": cfg.digest(), "profile": profile, "method": method, "seed": seed,
                               "python": platform.python_version(), "torch": torch.__version__,
                               "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "pettingzoo", "transformers", "peft"]},
                               "warmstart_sha256": self.warmstart_sha256,
                               "device": str(self.bank.device), "resolved_revision": self.bank.resolved_revision,
                               "action_policy": cfg.training.action_policy,
                               "demonstrations": warm["transitions"],
                               "knowledge_energy_cost": cfg.training.corpus_tool_cost,
                               "resumed": bool(resume)})
        if not resume:
            self.checkpoint()  # Recovery exists even if the initial evaluation is interrupted.

    def log(self, kind, data):
        with (self.output / f"{kind}.jsonl").open("a") as f:
            f.write(json.dumps({"decisions": self.decisions, "trial": self.trial, **data}, default=str) + "\n")

    def checkpoint(self, name="latest.pt"):
        payload = {"format": 1, "config_digest": self.cfg.digest(), "profile": self.profile, "method": self.method,
                   "warmstart_sha256": self.warmstart_sha256,
                   "seed": self.seed, "env": self.env, "obs": self.obs, "initial": self.initial,
                   "corpus": self.corpus, "novelty": self.novelty,
                   "policies": {i: self.bank.state(i) for i in self.bank.controllers},
                   "optimizers": {i: copy.deepcopy(o.state_dict()) for i, o in self.bank.optimizers.items()},
                   "hidden": {i: h.cpu() for i, h in self.hidden.items()}, "buffers": self.buffers,
                   "dead": self.dead, "archive": self.archive, "death_serial": self.death_serial,
                   "decisions": self.decisions, "trial": self.trial, "world_ticks": self.world_ticks,
                   "restarts": self.restarts, "completed_evaluations": self.completed_evaluations,
                   "elapsed_seconds": self.elapsed_seconds, "resources": self.bank.resource_metrics(),
                   "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
                   "python_rng": random.getstate(), "action_rng": self.action_rng.get_state(),
                   "mutation_rng": self.mutation_rng.get_state(), "newborn_tracking": self.newborn_tracking,
                   "births_total": self.births_total, "evaluation_decisions": self.evaluation_decisions,
                   "log_offsets": {p.name: p.stat().st_size for p in self.output.glob("*.jsonl")}}
        if self.bank.device.type == "cuda":
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        if self.cfg.training.backend == "tiny":
            payload["tiny_base"] = cpu_state(self.bank.base.state_dict())
        save_atomic(self.output / name, payload)
        if name == "latest.pt":
            status = self.output / "checkpoint-status.json"
            temporary = status.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"decisions": self.decisions, "world_ticks": self.world_ticks,
                                             "trial": self.trial, "elapsed_seconds": self.elapsed_seconds}))
            os.replace(temporary, status)

    def restore(self, path):
        p = torch.load(path, map_location="cpu", weights_only=False)
        if p["config_digest"] != self.cfg.digest() or (p["profile"], p["method"], p["seed"]) != (self.profile, self.method, self.seed):
            raise ValueError("Resume requires exactly the saved config/profile/method/seed")
        if p.get("warmstart_sha256", self.warmstart_sha256) != self.warmstart_sha256:
            raise ValueError("Resume warm-start content differs from the original run")
        # Roll back uncommitted log suffixes after a crash, along with simulator state.
        # Copy the committed prefix when resuming into a separate output directory.
        offsets = p.get("log_offsets", {})
        for name, offset in offsets.items():
            if Path(name).name != name:
                raise ValueError("Invalid checkpoint log filename")
            source, destination = Path(path).parent / name, self.output / name
            if source.exists():
                if source.resolve() == destination.resolve():
                    with destination.open("r+b") as f:
                        f.truncate(offset)
                else:
                    with source.open("rb") as f:
                        destination.write_bytes(f.read(offset))
        for log_path in self.output.glob("*.jsonl"):
            if log_path.name not in offsets:
                log_path.write_text("")
        for key in ["env", "obs", "initial", "buffers", "dead", "archive", "death_serial", "decisions", "trial",
                    "world_ticks", "restarts", "completed_evaluations", "elapsed_seconds", "newborn_tracking",
                    "births_total", "evaluation_decisions"]:
            setattr(self, key, p[key])
        self.corpus = p.get("corpus")
        self.novelty = p.get("novelty", CountNovelty())
        if "tiny_base" in p:
            self.bank.base.load_state_dict(p["tiny_base"])
        for i, state in p["policies"].items():
            self.bank.add(i, state)
            self.bank.optimizers[i].load_state_dict(p["optimizers"][i])
        self.hidden = {i: h.to(self.bank.device) for i, h in p["hidden"].items()}
        for key, attribute in [("processed_tokens", "tokens"), ("encoder_forwards", "encoder_forwards"),
                               ("gradient_updates", "gradient_updates"), ("inference_seconds", "inference_seconds"),
                               ("observations_at_token_cap", "truncated_observations")]:
            setattr(self.bank, attribute, p["resources"][key])
        self.bank.cuda_stream_seconds = p["resources"].get("cuda_model_stream_seconds", 0.0)
        torch.set_rng_state(p["torch_rng"])
        if "cuda_rng" in p and self.bank.device.type == "cuda":
            torch.cuda.set_rng_state_all(p["cuda_rng"])
        np.random.set_state(p["numpy_rng"])
        random.setstate(p["python_rng"])
        self.action_rng.set_state(p["action_rng"])
        self.mutation_rng.set_state(p["mutation_rng"])

    def _learn(self, close_window=False, extinction=False):
        for i in list(self.bank.controllers):
            if i in self.dead:
                if not (close_window or extinction):
                    continue
                entry = self.dead[i]
                result = update(self.bank, i, self.buffers[i], self.preference, social_bootstrap=entry["tail"])
                self.log("updates", {"id": i, "samples": len(self.buffers[i]), "post_death": True,
                                     "social_tail": entry["tail"], **result})
                self.archive.append({"serial": entry["serial"], "state": self.bank.state(i), **entry["metadata"]})
                self.archive = sorted(self.archive, key=lambda a: a["serial"])[-self.cfg.environment.founders:]
                self.bank.remove(i)
                self.buffers.pop(i, None)
                self.hidden.pop(i, None)
                del self.dead[i]
            elif self.buffers[i]:
                with torch.no_grad():
                    outputs, _ = self.bank.sequence(i, [self.obs[i]], self.hidden[i])
                    _, vp, vs = outputs[0][:3]
                    vi = float(outputs[0][3]) if self.cfg.training.intrinsic_critic else 0.
                result = update(self.bank, i, self.buffers[i], self.preference, float(vp), 0.0 if close_window else float(vs), vi)
                if i in self.newborn_tracking:
                    self.log("newborn", {"id": i, **self.newborn_tracking.pop(i), "alive_at_first_update": True})
                self.log("updates", {"id": i, "samples": len(self.buffers[i]), **result})
                self.buffers[i] = []

    def _recover(self):
        if len(self.archive) < self.cfg.environment.founders:
            raise RuntimeError("Insufficient distinct death checkpoints for recovery")
        self.log("extinction", {"time_to_extinction": self.env.tick, "world_ticks": self.world_ticks})
        self.trial += 1
        self.restarts += 1
        self.obs, _ = self.env.reset(seed=self.seed + self.trial * 1009)
        previous_corpus = self.corpus
        self.corpus, self.obs = make_corpus(self.cfg.training, self.obs)
        for i, archived in zip(self.env.agents, self.archive):
            self.bank.add(i, archived["state"])
            self.hidden[i] = self.bank.zero_hidden()
            self.env.prey[i].lineage = archived["lineage"]
            self.env.prey[i].generation = archived["generation"]
        if isinstance(self.corpus, TacticCorpus):
            self.corpus.restore_memory(previous_corpus, {i: p.lineage for i, p in self.env.prey.items()})
            self.obs = self.corpus.augment(self.obs)
        self.log("recovery", {"restored_death_serials": [a["serial"] for a in self.archive]})

    def step(self):
        actions, records, writes, write_logps = {}, {}, {}, {}
        for stage in self.cfg.training.predator_curriculum:
            if self.decisions >= stage["decisions"]:
                target = stage["temperature"]
        if self.cfg.training.predator_curriculum and self.decisions >= self.cfg.training.predator_curriculum[0]["decisions"]:
            if self.env.cfg.predator_temperature != target:
                self.env.cfg = replace(self.env.cfg, predator_temperature=target)
                self.log("curriculum", {"predator_temperature": target, "world_ticks": self.world_ticks})
        temperature = self.bank.training_temperature(self.decisions)
        beta = novelty_beta(self.cfg.training, self.decisions)
        intrinsic_values = {}
        # All actions collected before any environmental effect or policy update.
        for i in self.env.agents:
            h = self.hidden[i]
            action, logp, vp, vs, self.hidden[i], intrinsic_values[i], publication = self.bank.act(
                i, self.obs[i], h, self.action_rng, temperature, include_intrinsic=True, include_note=True)
            writes[i], write_logps[i] = publication
            actions[i] = action
            records[i] = (copy.deepcopy(self.obs[i]), h.cpu(), action, logp, vp, vs)
        prior_metadata = {i: {"lineage": p.lineage, "generation": p.generation} for i, p in self.env.prey.items()}
        publication_rewards = publish_notes(self.corpus, writes, self.obs, self.bank, self.env, self.cfg.training)
        physical = execute_tools(self.corpus, actions, self.obs, self.env, self.cfg.training)
        before = self.obs
        self.obs, _, terms, _, infos = self.env.step(physical)
        record_experience(self.corpus, physical, before, self.obs, self.env)
        self.obs = next_observations(self.corpus, self.obs, self.env.agents)
        self.decisions += len(actions)
        self.world_ticks += 1
        close = self.env.tick % self.cfg.training.social_window == 0
        # Existing dead individuals receive social outcomes, but never generate actions.
        for entry in self.dead.values():
            entry["tail"] += entry["discount"] * len(self.env.agents) / self.cfg.environment.social_denominator
            entry["discount"] *= self.cfg.training.gamma
        for i, record in records.items():
            raw_novelty = self.novelty.reward(prior_metadata[i]["lineage"], self.obs[i]) if beta > 0 and not terms[i] else 0.
            self.buffers[i].append(Transition(*record, infos[i]["personal_reward"], infos[i]["social_reward"], terms[i], close,
                                             temperature, intrinsic_values[i], beta*raw_novelty,
                                             writes[i], write_logps[i], publication_rewards.get(i, 0.)))
            if self.cfg.training.corpus_interface == "notes":
                self.log("note_decisions", {"id": i, "tick": self.env.tick-1,
                    "position": record[0]["self"][2:4].tolist(), "energy": float(record[0]["self"][0]),
                    "action": actions[i], "action_probability": float(np.exp(record[3])),
                    "action_word": ACTION_NAMES[actions[i] % 7], "action_policy": self.cfg.training.action_policy,
                    "note_ids": [n["id"] for n in record[0]["notes"]],
                    "publication": writes[i], "publication_reward": publication_rewards.get(i, 0.)})
                if self.cfg.training.action_policy == "lm_token" and (self.world_ticks <= 2 or writes[i] == 1 or self.world_ticks % 64 == 0):
                    from .language_policy import bounded_prompt
                    self.log("llm_decisions", {"id": i, "tick": self.env.tick-1,
                        "prompt": bounded_prompt(self.bank.tokenizer, record[0], self.cfg.training.max_tokens),
                        "output": ACTION_NAMES[actions[i]], "probability": float(np.exp(record[3])),
                        "publication": writes[i]})
            if i in self.newborn_tracking:
                self.newborn_tracking[i]["ticks_before_first_update"] += 1
            if terms[i]:
                self.death_serial += 1
                self.dead[i] = {"tail": 0.0, "discount": 1.0, "serial": self.death_serial, "metadata": prior_metadata[i]}
                if i in self.newborn_tracking:
                    self.log("newborn", {"id": i, **self.newborn_tracking.pop(i), "alive_at_first_update": False})
        for event in self.env.events:
            if event["type"] == "birth":
                i, parent = event["id"], event["parent"]
                source = self.bank.state(parent) if self.method == "r_adult" else self.initial
                mutation = self.bank.inherit(i, source, self.histories, self.mutation_rng)
                self.hidden[i] = self.bank.zero_hidden()
                self.newborn_tracking[i] = {"generation": event["generation"], "ticks_before_first_update": 0}
                self.births_total += 1
                event.update(mutation)
                event["initial_probe"] = self.bank.probabilities(i, [self.histories[0]])[-1].tolist()
                every = self.cfg.training.newborn_evaluation_every
                if every and self.births_total % every == 0:
                    # Frozen, standardized cohort assay before the child's first action/update.
                    # Shorter horizon and periodic sampling bound evaluation overhead.
                    t = self.cfg.training
                    assay_cfg = replace(t, evaluation_maps=t.newborn_evaluation_maps,
                                        evaluation_horizon=t.newborn_evaluation_horizon,
                                        evaluation_seed=t.evaluation_seed + 100_000)
                    assay = evaluate(self.bank, [i], self.cfg.environment, assay_cfg, self.decisions)
                    self.evaluation_decisions += sum(r["decisions"] for r in assay["episodes"])
                    self.log("newborn_evaluation", {"id": i, "generation": event["generation"],
                                                   "birth_index": self.births_total, **assay})
            self.log("events", event)
        self.log("ecology", {**self.env.metrics(), "world_ticks": self.world_ticks, "policy_temperature": temperature,
                             "novelty_beta": beta,
                             "actions": {str(a): sum(v % 7 == a for v in actions.values()) for a in range(7)}})
        if self.corpus is not None:
            self.log("knowledge", {"tick": self.env.tick, **self.corpus.metrics()})
            for event in self.corpus.events:
                self.log("corpus_events", event)
            self.corpus.events.clear()
        if self.env.tick % self.cfg.training.rollout_ticks == 0 or not self.env.agents:
            self._learn(close_window=close, extinction=not self.env.agents)
        if not self.env.agents:
            self._recover()

    def assess(self, threshold, final=False):
        memory = {"initial_corpus": self.corpus, "policy_owners": {i: p.lineage for i, p in self.env.prey.items()}}
        metrics = evaluate(self.bank, self.env.agents, self.cfg.environment, self.cfg.training, threshold, final, **memory)
        self.evaluation_decisions += sum(r["decisions"] for r in metrics["episodes"])
        metrics["population"] = len(self.env.agents)
        metrics["ages"] = [p.age for p in self.env.prey.values()]
        self.log("evaluation", metrics)
        t = self.cfg.training
        assays = []
        if t.development_maps:
            assays.append(("development_evaluation", self.cfg.environment,
                           replace(t, evaluation_maps=t.development_maps, evaluation_seed=t.development_seed), False, None))
        if t.evaluation_ablations and self.cfg.environment.message_symbols > 1:
            assays.append(("evaluation_muted", self.cfg.environment, t, final, False))
        if t.corpus_mode == "shared" and (final or t.corpus_interface == "notes"):
            assays.append(("evaluation_private_corpus", self.cfg.environment, replace(t, corpus_mode="private"), final, None))
        if final:
            for temperature in t.predator_evaluation_temperatures:
                assays.append(("predator_evaluation", replace(self.cfg.environment, predator_temperature=temperature),
                               t, True, None))
        for kind, environment, training, last, delivery in assays:
            assay = evaluate(self.bank, self.env.agents, environment, training, threshold, last, delivery, **memory)
            self.evaluation_decisions += sum(r["decisions"] for r in assay["episodes"])
            self.log(kind, assay)
        if isinstance(self.corpus, TacticCorpus):
            assay = evaluate(self.bank, self.env.agents, self.cfg.environment, t, threshold, final,
                             without_tactics=True, **memory)
            self.evaluation_decisions += sum(r["decisions"] for r in assay["episodes"])
            self.log("evaluation_without_tactics", assay)
        self.log("probes", {"checkpoint": threshold, "probabilities": probe(self.bank, self.env.agents, self.histories)})
        if t.corpus_interface == "notes":
            from .note_probe import probe_notes
            self.log("note_probe", {"checkpoint": threshold,
                                   **probe_notes(self.bank, self.env.agents, next(iter(self.obs.values())))})
        if self.corpus is not None:
            stores = {"shared": self.corpus.shared} if t.corpus_mode == "shared" else self.corpus.private
            snapshot = {"trial": self.trial, "tick": self.env.tick, "checkpoint": threshold,
                        "mode": t.corpus_mode, "stores": {name: [vars(f) for f in store.values()] for name, store in stores.items()}}
            if t.corpus_interface == "notes":
                snapshot["notes"] = [asdict(n) for n in self.corpus.notes]
            (self.output / f"corpus-checkpoint-{threshold}.json").write_text(json.dumps(snapshot, indent=2))
        self.completed_evaluations.append(threshold)
        self.checkpoint(f"checkpoint-{threshold}.pt")

    def run(self, max_wall_seconds=None):
        start = time.perf_counter()
        initial_elapsed = self.elapsed_seconds
        cfg = self.cfg.training
        wall_limit = cfg.max_wall_seconds if max_wall_seconds is None else max_wall_seconds
        # Evaluate the actual starting cohort before any collection or PPO update.
        if 0 in cfg.checkpoints and 0 not in self.completed_evaluations:
            if self.decisions != 0:
                raise ValueError("Cannot create a missing initial evaluation after training has started")
            self.assess(0)
            self.elapsed_seconds = initial_elapsed + time.perf_counter() - start
            self.checkpoint()
        while self.decisions + len(self.env.agents) <= cfg.decisions:
            if wall_limit and time.perf_counter() - start >= wall_limit:
                break
            self.step()
            self.elapsed_seconds = initial_elapsed + time.perf_counter() - start
            for threshold in cfg.checkpoints:
                if threshold < cfg.decisions and self.decisions >= threshold and threshold not in self.completed_evaluations:
                    self.assess(threshold)
            if self.env.tick % cfg.rollout_ticks == 0:
                self.checkpoint()
                print(json.dumps({"decisions": self.decisions, "population": len(self.env.agents),
                                  "elapsed_seconds": round(self.elapsed_seconds, 2)}), flush=True)
        finished = self.decisions + len(self.env.agents) > cfg.decisions
        if finished:
            self._learn(close_window=self.env.tick % cfg.social_window == 0)
            if cfg.decisions not in self.completed_evaluations:
                self.assess(cfg.decisions, final=True)
        self.elapsed_seconds = initial_elapsed + time.perf_counter() - start
        self.checkpoint()
        summary = {"finished": finished, "decisions": self.decisions, "budget": cfg.decisions,
                   "unused_decisions": cfg.decisions - self.decisions, "world_ticks": self.world_ticks,
                   "restarts": self.restarts, "elapsed_seconds": self.elapsed_seconds,
                   "evaluation_decisions": self.evaluation_decisions, "births": self.births_total,
                   "pending_dead_social_updates": len(self.dead), "resources": self.bank.resource_metrics()}
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary
