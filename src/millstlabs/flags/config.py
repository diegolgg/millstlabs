from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import yaml


@dataclass
class FlagConfig:
    model: str = "gpt-4o-2024-08-06"
    population: int = 8
    width: int = 24
    height: int = 16
    crop_width: int = 6
    crop_height: int = 4
    render_scale: int = 25
    memory_entries: int = 8
    bandwidth: int = 3
    temperature: float = .2
    top_p: float = 1.
    max_completion_tokens: int = 200
    image_detail: str = "high"
    interaction_rounds: int = 10
    consensus_patience: int = 5
    consensus_threshold: float = .85
    polarization_threshold: float = .25
    retrieval_records: int = 2
    train_trials: int = 64
    evaluation_trials: int = 60
    train_seed: int = 11000
    evaluation_seed: int = 6000000
    policy_seed: int = 11
    cooperation_weight: float = .5
    tool_cost: float = .01
    learning_rate: float = .0003
    gamma: float = .99
    gae_lambda: float = .95
    ppo_epochs: int = 2
    clip: float = .2
    entropy: float = .01
    catalog: str = "configs/flag-catalog.json"
    trial_manifest: str | None = None
    backend: str = "openai"

    def validate(self):
        if self.backend not in {"openai", "mock"}:
            raise ValueError("backend must be openai or explicitly labelled mock")
        if self.population < 2 or min(self.train_trials, self.evaluation_trials, self.interaction_rounds) < 1:
            raise ValueError("Positive trials/rounds and at least two agents are required")
        if not (0 < self.crop_width < self.width and 0 < self.crop_height < self.height):
            raise ValueError("Each crop must be smaller than the flag")
        if self.bandwidth not in {1, 3} or self.image_detail != "high":
            raise ValueError("This reconstruction supports paper m=1 or m=3 and high image detail")
        if self.memory_entries < 1 or not 1 <= self.retrieval_records <= self.memory_entries:
            raise ValueError("Retrieved facts must fit the same bounded transcript memory")
        if not 0 <= self.tool_cost or not 0 <= self.cooperation_weight:
            raise ValueError("Reward weights must be nonnegative")
        if self.render_scale < 1 or self.consensus_patience < 1 or self.max_completion_tokens < 1:
            raise ValueError("Scale, patience and token cap must be positive")
        if not 0 < self.consensus_threshold <= 1 or not 0 < self.polarization_threshold <= .5:
            raise ValueError("Invalid endpoint thresholds")

    def digest(self):
        return sha256(yaml.safe_dump(asdict(self), sort_keys=True).encode()).hexdigest()


def load_config(path):
    cfg = FlagConfig(**yaml.safe_load(Path(path).read_text()))
    cfg.validate()
    return cfg
