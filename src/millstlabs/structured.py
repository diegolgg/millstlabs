"""Trainable, observation-only recurrent actor and critic for the CPU experiment."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def features(obs):
    s = obs["self"]
    size = float(s[9])
    own = s / np.asarray([100, 512, size, size, 128, 1, 512, 1, 2, size, 1, 16][:len(s)])
    own = np.clip(own, -1, 4)
    local = np.eye(7, dtype=np.float32)[obs["local"].astype(int) + 1].ravel()
    food = []
    for x, y, stock, seen in obs["stations"]:
        food.extend([(x-s[2])/size, (y-s[3])/size, max(stock, 0)/60,
                     min(max(s[6]-seen, 0)/128, 4) if seen >= 0 else 0, float(seen >= 0)])
    peers = []
    for x, y, energy, watching in obs["companions"]:
        peers.extend([(x-s[2])/size, (y-s[3])/size, energy/100, watching, 1] if x >= 0 else [0]*5)
    threats = []
    for x, y in obs["threats"]:
        threats.extend([(x-s[2])/size, (y-s[3])/size, 1] if x >= 0 else [0]*3)
    messages = []
    for x, y, symbol in obs.get("messages", []):
        # Eight bounded categories reserve the same feature layout for all vocabularies.
        messages.extend([(x-s[2])/size, (y-s[3])/size, 1, *np.eye(8)[int(symbol)]]
                        if symbol > 0 else [0]*11)
    corpus = []
    for kind, x, y, value, age, peer, valid in obs.get("corpus", []):
        corpus.extend([kind, (x-s[2])/size, (y-s[3])/size, value/(60 if kind == 1 else 1),
                       min(age/32, 4), peer, 1] if valid == 1 else [0]*7)
    return np.concatenate([own, local, food, peers, threats, obs["action_mask"][:7], messages,
                           corpus, np.asarray(obs.get("corpus_status", []))/4]).astype(np.float32)


class StructuredController(nn.Module):
    def __init__(self, width, cfg, message_symbols, critic_width=None, language=False):
        super().__init__()
        self.separate = cfg.separate_critic
        self.value_scale = cfg.value_scale
        self.message_symbols = message_symbols
        def encoder(input_width, normalize=False):
            layers = [nn.LayerNorm(input_width)] if normalize else []
            return nn.Sequential(*layers, nn.Linear(input_width, 128), nn.LayerNorm(128), nn.SiLU(),
                                 nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU())
        self.actor_encoder = encoder(width, language)
        self.actor_gru = nn.GRUCell(64, 64)
        self.action = nn.Linear(64, 7)
        self.watch_bias = nn.Parameter(torch.zeros(()))
        self.message = nn.Linear(64, message_symbols) if message_symbols > 1 else None
        self.publication = nn.Linear(64, 1) if cfg.corpus_interface == "notes" else None
        if self.publication is not None:
            nn.init.zeros_(self.publication.weight)
            nn.init.zeros_(self.publication.bias)
        self.critic_encoder = encoder(critic_width or width) if self.separate else None
        self.critic_gru = nn.GRUCell(64, 64) if self.separate else None
        self.personal_value = nn.Linear(64, 1)
        self.social_value = nn.Linear(64, 1)
        self.intrinsic_value = nn.Linear(64, 1) if cfg.intrinsic_critic else None
        nn.init.orthogonal_(self.action.weight, gain=0.01)
        nn.init.zeros_(self.action.bias)
        for head in [self.personal_value, self.social_value] + ([self.intrinsic_value] if cfg.intrinsic_critic else []):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.message is not None:
            nn.init.orthogonal_(self.message.weight, gain=0.01)
            nn.init.zeros_(self.message.bias)

    def actor_parameters(self):
        return [p for n, p in self.named_parameters()
                if not n.startswith(("critic_", "personal_value.", "social_value.", "intrinsic_value."))]

    def critic_parameters(self):
        return [p for n, p in self.named_parameters()
                if n.startswith(("critic_", "personal_value.", "social_value.", "intrinsic_value."))]

    def forward(self, inputs, hidden, mask, temperature=1.0, critic_inputs=None):
        ha = self.actor_gru(self.actor_encoder(inputs), hidden[:, :64])
        critic_inputs = inputs if critic_inputs is None else critic_inputs
        hc = self.critic_gru(self.critic_encoder(critic_inputs), hidden[:, 64:]) if self.separate else ha
        logits = self.action(ha)
        logits = logits + nn.functional.one_hot(torch.tensor(5, device=logits.device), 7) * self.watch_bias
        if self.message is not None:
            logits = (logits[:, None, :] + self.message(ha)[:, :, None]).flatten(1)
        logits = (logits / temperature).masked_fill(~mask.bool(), -1e9)
        result = (Categorical(logits=logits), self.personal_value(hc).squeeze(-1)*self.value_scale,
                self.social_value(hc).squeeze(-1)*self.value_scale,
                torch.cat([ha, hc], -1) if self.separate else ha)
        return result + (self.intrinsic_value(hc).squeeze(-1),) if self.intrinsic_value is not None else result
