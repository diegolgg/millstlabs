"""One frozen backbone, isolated adapters/controllers/optimizers per individual."""
import copy
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from .observations import observation_text, vector_observation
from .reproduction import SURVIVAL_PROMPT
from .structured import StructuredController
from .structured import features as structured_features


def cpu_state(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}


class Controller(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(width, 64)
        self.gru = nn.GRUCell(64, 64)
        self.action = nn.Linear(64, 7)
        self.watch_bias = nn.Parameter(torch.zeros(()))
        self.personal_value = nn.Linear(64, 1)
        self.social_value = nn.Linear(64, 1)

    def forward(self, features, hidden, mask, temperature=1.0):
        hidden = self.gru(torch.tanh(self.projection(features.float())), hidden)
        logits = self.action(hidden)
        logits = logits + torch.nn.functional.one_hot(torch.tensor(5, device=logits.device), 7) * self.watch_bias
        logits = (logits / temperature).masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits), self.personal_value(hidden).squeeze(-1), self.social_value(hidden).squeeze(-1), hidden


class PolicyBank:
    def __init__(self, cfg, example_observation):
        self.cfg = cfg
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device)
        torch.set_num_threads(cfg.cpu_threads)
        self.controllers = {}
        self.adapters = {}
        self.optimizers = {}
        self.tokens = 0
        self.encoder_forwards = 0
        self.gradient_updates = 0
        self.inference_seconds = 0.0
        self.truncated_observations = 0
        self.cuda_stream_seconds = 0.0
        self.cuda_events = []
        self.survival_note_tokens = 0
        self.survival_note_seconds = 0.0
        self.resolved_revision = None
        self.width = 64
        self.ground_width = len(self.physical_features(example_observation)) if cfg.actor_grounding else 0
        self.critic_width = len(structured_features(example_observation))
        self.message_symbols = len(example_observation["action_mask"]) // 7
        if cfg.backend == "structured":
            self.width = len(structured_features(example_observation))
        elif cfg.backend == "tiny":
            dim = len(vector_observation(example_observation))
            self.base = nn.Sequential(nn.Linear(dim, 64), nn.Tanh()).to(self.device)
            self.base.requires_grad_(False)
            self.base.eval()
        else:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
            base = AutoModel.from_pretrained(cfg.model_id, revision=cfg.revision, torch_dtype=dtype)
            self.resolved_revision = getattr(base.config, "_commit_hash", None)
            self.width = base.config.hidden_size
            layers = base.config.num_hidden_layers
            # AutoModel exposes root-level `layers.N`, unlike AutoModelForCausalLM's
            # `model.layers.N`. An explicit regex works for both PEFT name layouts.
            layer_pattern = "|".join(str(n) for n in range(layers - 4, layers))
            target_pattern = rf"(?:model\.)?layers\.({layer_pattern})\.self_attn\.(q_proj|v_proj)"
            self.lora_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=4, lora_alpha=4,
                                          lora_dropout=0.0, target_modules=target_pattern, bias="none")
            self.base = get_peft_model(base, self.lora_config, adapter_name="template").to(self.device)
            self.base.eval()  # deterministic dropout; eval() still permits gradients
            self.template = self._adapter_state("template")

    def _activate(self, i):
        if self.cfg.backend == "smollm":
            self.base.set_adapter(i)

    def _adapter_parameters(self, i):
        if self.cfg.backend == "structured":
            return []
        if self.cfg.backend == "tiny":
            return list(self.adapters[i].parameters())
        return [p for name, p in self.base.named_parameters() if f".{i}." in name and "lora_" in name]

    def _adapter_state(self, i):
        if self.cfg.backend == "structured":
            return {}
        if self.cfg.backend == "tiny":
            return cpu_state(self.adapters[i].state_dict())
        from peft import get_peft_model_state_dict
        return cpu_state(get_peft_model_state_dict(self.base, adapter_name=i))

    def add(self, i, state=None):
        if i in self.controllers:
            raise ValueError(f"duplicate policy {i}")
        if self.cfg.backend == "tiny":
            self.adapters[i] = nn.Sequential(nn.Linear(64, 4, bias=False), nn.Linear(4, 64, bias=False)).to(self.device)
            nn.init.zeros_(self.adapters[i][1].weight)
        elif self.cfg.backend == "smollm":
            self.base.add_adapter(i, copy.deepcopy(self.lora_config))
        self.controllers[i] = (StructuredController(self.width+self.ground_width, self.cfg, self.message_symbols, self.critic_width,
                                                    language=self.cfg.backend == "smollm")
                               if self.cfg.split_controller else Controller(self.width)).to(self.device)
        if state:
            self.load_state(i, state)
        elif self.cfg.backend == "smollm":
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.base, self.template, adapter_name=i)
        self._activate(i)
        if self.cfg.split_controller:
            controller = self.controllers[i]
            groups = [
                {"params": controller.actor_parameters(), "lr": self.cfg.controller_lr},
                {"params": controller.critic_parameters(), "lr": self.cfg.critic_lr},
            ]
            if self.cfg.backend == "smollm":
                groups.insert(0, {"params": self._adapter_parameters(i), "lr": self.cfg.adapter_lr})
            self.optimizers[i] = torch.optim.Adam(groups)
            return
        params = self._adapter_parameters(i)
        assert params, "adapter target selection matched no parameters"
        self.optimizers[i] = torch.optim.Adam([
            {"params": params, "lr": self.cfg.adapter_lr},
            {"params": self.controllers[i].parameters(), "lr": self.cfg.controller_lr},
        ])

    def load_state(self, i, state):
        self.controllers[i].load_state_dict(state["controller"])
        if self.cfg.backend == "tiny":
            self.adapters[i].load_state_dict(state["adapter"])
        elif self.cfg.backend == "smollm":
            from peft import set_peft_model_state_dict
            set_peft_model_state_dict(self.base, state["adapter"], adapter_name=i)

    def state(self, i):
        return {"controller": cpu_state(self.controllers[i].state_dict()), "adapter": self._adapter_state(i)}

    def remove(self, i):
        del self.optimizers[i]
        del self.controllers[i]
        if self.cfg.backend == "tiny":
            del self.adapters[i]
        elif self.cfg.backend == "smollm":
            self.base.set_adapter("template")
            self.base.delete_adapter(i)

    def encode(self, i, observations):
        self._activate(i)
        self.encoder_forwards += len(observations)
        if self.cfg.backend == "structured":
            return torch.as_tensor(np.stack([structured_features(o) for o in observations]), device=self.device)
        if self.cfg.backend == "tiny":
            v = torch.as_tensor(np.stack([vector_observation(o) for o in observations]), device=self.device)
            start = self.gpu_start()
            features = self.base(v)
            result = features + self.adapters[i](features)
            self.gpu_end(start)
            return result
        if not any(obs.get("survival_note") for obs in observations):
            enc = self.tokenizer([observation_text(o) for o in observations], padding=True,
                                 truncation=True, max_length=self.cfg.max_tokens, return_tensors="pt")
        else:
            # Reserve note tokens separately so a full observation cannot evict advice.
            token_ids = []
            for obs in observations:
                ids = self.tokenizer.encode(observation_text(obs, include_survival_note=False),
                                            truncation=True, max_length=self.cfg.max_tokens)
                if obs.get("survival_note"):
                    ids += self.tokenizer.encode(" Survival note: " + obs["survival_note"],
                                                 add_special_tokens=False, truncation=True, max_length=64)
                token_ids.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            enc = self.tokenizer.pad(token_ids, padding=True, return_tensors="pt")
        self.tokens += int(enc["attention_mask"].sum())
        # A max-length sequence may be exactly full, so this is a conservative counter.
        self.truncated_observations += int((enc["attention_mask"].sum(-1) >= self.cfg.max_tokens).sum())
        enc = {k: v.to(self.device) for k, v in enc.items()}
        start = self.gpu_start()
        out = self.base(**enc, use_cache=False).last_hidden_state
        last = enc["attention_mask"].sum(-1) - 1
        result = out[torch.arange(len(observations), device=self.device), last].float()
        self.gpu_end(start)
        return result

    def gpu_start(self):
        if self.device.type != "cuda":
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def gpu_end(self, start):
        if start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            self.cuda_events.append((start, end))

    def sequence(self, i, observations, hidden, temperature=1.0):
        features = self.encode(i, observations)
        if self.cfg.actor_grounding:
            ground = torch.as_tensor(np.stack([self.physical_features(o) for o in observations]), device=self.device)
            features = torch.cat([features, ground], dim=-1)
        critic_inputs = torch.as_tensor(np.stack([structured_features(o) for o in observations]), device=self.device) if (
            (self.cfg.backend == "smollm" and self.cfg.split_controller) or self.cfg.actor_grounding) else None
        output = []
        for j, obs in enumerate(observations):
            mask = torch.as_tensor(obs["action_mask"], device=self.device).unsqueeze(0)
            temp = temperature[j] if isinstance(temperature, (list, tuple)) else temperature
            kwargs = {"critic_inputs": critic_inputs[j:j+1]} if critic_inputs is not None else {}
            result = self.controllers[i](features[j:j+1], hidden, mask, temp, **kwargs)
            dist, vp, vs, hidden = result[:4]
            row = (dist, vp, vs) + result[4:]
            if self.cfg.corpus_interface == "notes":
                row += (Bernoulli(logits=self.controllers[i].publication(hidden[:, :64]).squeeze(-1)),)
            output.append(row)
        return output, hidden

    @torch.no_grad()
    def act(self, i, obs, hidden, generator=None, temperature=1.0, include_intrinsic=False, include_note=False):
        start = time.perf_counter()
        out, next_hidden = self.sequence(i, [obs], hidden, temperature)
        dist, vp, vs = out[0][:3]
        # Dedicated RNG keeps evaluation independent from collection randomness.
        action = torch.multinomial(dist.probs.cpu(), 1, generator=generator).squeeze()
        logp = dist.log_prob(action.to(self.device))
        self.inference_seconds += time.perf_counter() - start
        result = (int(action), float(logp), float(vp), float(vs), next_hidden.detach())
        if include_intrinsic:
            result += (float(out[0][3]) if self.cfg.intrinsic_critic else 0.0,)
        if include_note:
            write, write_logp = -1, 0.
            if self.cfg.corpus_interface == "notes" and obs.get("note_opportunity", False):
                distribution = out[0][-1]
                write = int(torch.rand((), generator=generator) < distribution.probs.cpu().squeeze())
                write_logp = float(distribution.log_prob(torch.tensor(float(write), device=self.device)))
            result += ((write, write_logp),)
        return result

    @staticmethod
    def physical_features(obs):
        # Direct grounding for energy, geometry and legal movement. Corpus
        # information still reaches the LLM actor exclusively through language.
        return structured_features({k: v for k, v in obs.items() if k not in {"corpus", "corpus_status"}})

    @torch.no_grad()
    def describe_note(self, i, evidence):
        """Bounded, greedy prose from the frozen tied-embedding language model.

        Selection is learned by the publication head. Wording is not PPO-trained.
        Disable action LoRA during writing so it cannot destroy language fluency.
        KV caching avoids repeatedly encoding the evidence for each output token.
        """
        if self.cfg.backend != "smollm":
            raise ValueError("Free-form notes require a real language model")
        return self._write_note(evidence, "Write one short useful note to another animal in a survival game. Preserve the observed coordinates, quantities and tick. Food stock is an amount now, not units per day. Do not infer trends, stability or hidden mechanics. Use only the supplied observations.", self.cfg.note_max_tokens, 384)

    @torch.no_grad()
    def write_survival_note(self, history, inherited_note):
        """Frozen, greedy writer; private inputs and compute are accounted separately.

        Offline backends use a short extract for simulator/training tests only.
        """
        if self.cfg.backend != "smollm":
            text = "Recent private observation: " + (history[-1] if history else "No experience.")
            text = " ".join(text.split()[:64])
            return text, len(text.split())
        started = time.perf_counter()
        before = self.tokens
        inference_before = self.inference_seconds
        limit = int(getattr(self.base.config, "max_position_embeddings", 2048)) - 64
        # Fit each of the latest 128 ticks into the context, retaining the inherited note.
        per_tick = max(1, (limit - 192) // max(1, len(history)))
        recent = [self.tokenizer.decode(self.tokenizer.encode(row, add_special_tokens=False)[:per_tick])
                  for row in history]
        evidence = "Inherited note: " + inherited_note + "\nPrivate history:\n" + "\n".join(recent)
        result = self._write_note(evidence, SURVIVAL_PROMPT, 64, limit)
        consumed = self.tokens - before
        self.tokens = before
        self.inference_seconds = inference_before
        self.survival_note_tokens += consumed
        self.survival_note_seconds += time.perf_counter() - started
        return result

    @torch.no_grad()
    def _write_note(self, evidence, instruction, max_tokens, input_limit):
        if not self.base.config.tie_word_embeddings:
            raise ValueError("Local writer requires tied input/output embeddings")
        prompt = self.tokenizer.apply_chat_template([
            {"role": "system", "content": instruction},
            {"role": "user", "content": evidence}], tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=input_limit)["input_ids"].to(self.device)
        self.tokens += ids.numel()
        generated, past = [], None
        started = time.perf_counter()
        with self.base.disable_adapter():
            weight = self.base.get_input_embeddings().weight
            for _ in range(max_tokens):
                output = self.base(input_ids=ids, past_key_values=past, use_cache=True)
                token = int((output.last_hidden_state[:, -1] @ weight.T).argmax(-1))
                self.encoder_forwards += 1
                self.tokens += 1
                if token == self.tokenizer.eos_token_id:
                    break
                generated.append(token)
                past = output.past_key_values
                ids = torch.tensor([[token]], device=self.device)
        self.inference_seconds += time.perf_counter()-started
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), len(generated)

    def zero_hidden(self):
        width = 128 if self.cfg.split_controller and self.cfg.separate_critic else 64
        return torch.zeros(1, width, device=self.device)

    @torch.no_grad()
    def probabilities(self, i, histories):
        rows = []
        for history in histories:
            outputs, _ = self.sequence(i, history, self.zero_hidden(), self.cfg.evaluation_temperature)
            rows.extend(o[0].probs.squeeze(0).cpu() for o in outputs)
        return torch.stack(rows)

    def training_temperature(self, decisions):
        t = self.cfg
        fraction = min(1.0, decisions / t.temperature_decay_decisions) if t.temperature_decay_decisions else 0.0
        return t.temperature_start + fraction * (t.temperature_end - t.temperature_start)

    @torch.no_grad()
    def inherit(self, child, source, histories, generator):
        self.add(child, source)
        base_state = self.state(child)
        parent_probs = self.probabilities(child, histories)
        controller = self.controllers[child]
        weight = controller.action.weight
        noise = torch.randn(weight.shape, generator=generator) * self.cfg.mutation_rms * float(weight.square().mean().sqrt())
        watch_noise = float(torch.randn((), generator=generator)) * self.cfg.mutation_watch_std
        accepted = 0.0
        scale = 1.0
        for _ in range(20):
            self.load_state(child, base_state)
            controller.action.weight.add_(noise.to(self.device) * scale)
            controller.watch_bias.add_(watch_noise * scale)
            child_probs = self.probabilities(child, histories)
            kl = (parent_probs * (parent_probs.clamp_min(1e-12).log() - child_probs.clamp_min(1e-12).log())).sum(-1).mean()
            if float(kl) <= self.cfg.mutation_kl:
                accepted = float(kl)
                break
            scale *= 0.5
        else:
            self.load_state(child, base_state)
            scale = 0.0
        return {"mutation_kl": accepted, "mutation_scale": scale}

    def resource_metrics(self):
        if self.cuda_events:
            torch.cuda.synchronize(self.device)
            self.cuda_stream_seconds += sum(a.elapsed_time(b) for a, b in self.cuda_events) / 1000
            self.cuda_events.clear()
        return {"processed_tokens": self.tokens, "encoder_forwards": self.encoder_forwards,
                "gradient_updates": self.gradient_updates, "inference_seconds": self.inference_seconds,
                "observations_at_token_cap": self.truncated_observations,
                "survival_note_tokens": self.survival_note_tokens,
                "survival_note_seconds": self.survival_note_seconds,
                "cuda_model_stream_seconds": self.cuda_stream_seconds,
                "cuda_peak_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0}
