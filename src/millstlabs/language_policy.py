"""Direct pretrained language-token actions; a separate critic never chooses moves."""
import copy
import json

import torch
from torch import nn

from .env import ACTION_NAMES
from .observations import observation_text

RULES = (
    "You are one prey animal in a predator-prey game. Choose your own action. "
    "Your objective is to stay alive and help the other animals survive. "
    "Energy sustains life. Movement and living consume energy. Feeding at an adjacent food station restores energy if food is available. "
    "Watching sends an alarm and reduces attack risk for nearby animals. Rest stays in place. "
    "Reproduction can also consume energy; the outcome says when you gave birth. "
    "Outcome situation flags mean [energy below 50, feeding legal, predator visible]. "
    "Outcome food_distance is [before, after] distance to a known station, whose current stock may be unknown. "
    "North decreases y; south increases y; east increases x; west decreases x. "
    "The grid is a row-major 5 by 5 local view centered on you: ? unknown, . floor, # wall, F food, p prey, x predator, a alerted prey. "
    "Knowledge reading and writing cost zero energy. Notes are unverified hypotheses; their observations are evidence, not proof of a general rule. "
    "Past-map coordinates do not apply to the current map."
)


def messages(obs, purpose="action"):
    physical = {k: v for k, v in obs.items() if k not in {"corpus", "corpus_status", "notes", "note_candidate"}}
    sections = ["Current permitted observation: "+observation_text(physical)]
    x, y = obs["self"][2:4]
    stations = [f"dx={int(s[0]-x)},dy={int(s[1]-y)},distance={int(abs(s[0]-x)+abs(s[1]-y))}"
                for s in obs["stations"] if s[0] >= 0]
    sections.append("Known station offsets from you (positive dx=east, positive dy=south): "+"; ".join(stations))
    sections.append("Your recent action outcomes: "+json.dumps(obs.get("recent_experience", []), separators=(",", ":")))
    for note in obs.get("notes", []):
        sections.append("Retrieved note (unverified): "+note["text"]+"\nEvidence: "+note.get("evidence", ""))
    if purpose == "publication":
        question = "Do you want to publish a short note about what your own recent experience suggests? Answer exactly yes or no."
    else:
        legal = [name for name, allowed in zip(ACTION_NAMES, obs["action_mask"][:7]) if allowed]
        question = "Choose your next action. Output exactly one legal word: "+", ".join(legal)+"."
    return [{"role": "system", "content": RULES}, {"role": "user", "content": "\n".join(sections)+"\n"+question}]


def bounded_prompt(tokenizer, observation, max_tokens, purpose="action"):
    """Drop old optional context before ever truncating sensors or the action request."""
    obs = copy.deepcopy(observation)
    while True:
        prompt = tokenizer.apply_chat_template(messages(obs, purpose), tokenize=False, add_generation_prompt=True)
        if len(tokenizer.encode(prompt, add_special_tokens=False)) <= max_tokens:
            return prompt
        if obs.get("recent_experience"):
            obs["recent_experience"] = obs["recent_experience"][1:]
        elif obs.get("notes"):
            obs["notes"] = obs["notes"][:-1]
        else:
            raise ValueError("Essential game prompt exceeds max_tokens; increase the context budget")


def single_token_ids(tokenizer, words):
    ids = [tokenizer.encode(word, add_special_tokens=False) for word in words]
    if any(len(row) != 1 for row in ids) or len({row[0] for row in ids}) != len(ids):
        raise ValueError("Direct action policy requires distinct single-token action words for this tokenizer")
    return [row[0] for row in ids]


class LanguageCritic(nn.Module):
    """No action projection. Actor probabilities come exclusively from LM logits."""
    def __init__(self, width, cfg):
        super().__init__()
        self.value_scale = cfg.value_scale
        self.watch_bias = nn.Parameter(torch.zeros(()), requires_grad=False)  # Profile prior, never an action head.
        self.critic_encoder = nn.Sequential(nn.Linear(width, 128), nn.LayerNorm(128), nn.SiLU(),
                                            nn.Linear(128, 64), nn.LayerNorm(64), nn.SiLU())
        self.critic_gru = nn.GRUCell(64, 64)
        self.personal_value = nn.Linear(64, 1)
        self.social_value = nn.Linear(64, 1)
        for head in [self.personal_value, self.social_value]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def actor_parameters(self):
        return []

    def critic_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def value_step(self, inputs, hidden):
        h = self.critic_gru(self.critic_encoder(inputs), hidden[:, 64:])
        return (self.personal_value(h).squeeze(-1)*self.value_scale,
                self.social_value(h).squeeze(-1)*self.value_scale, torch.cat([torch.zeros_like(h), h], -1))
