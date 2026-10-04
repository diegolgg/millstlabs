import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class EnvironmentConfig:
    size: int = 20
    obstacle_fraction: float = 0.10
    founders: int = 8
    population_cap: int = 16
    stations: int = 3
    predators: int = 1
    max_energy: float = 100
    founder_energy: float = 60
    newborn_energy: float = 30
    metabolism: float = 0.5
    movement_cost: float = 0.25
    watching_cost: float = 0.1
    intake: float = 8
    initial_stock: float = 30
    stock_cap: float = 60
    replenishment_rates: list[int] = field(default_factory=lambda: [1, 2, 3])
    rate_change_probability: float = 1 / 128
    detection_radius: int = 6
    attack_cooldown: int = 8
    alert_death_probability: float = 0.15
    unalert_death_probability: float = 0.45
    maturity: int = 128
    birth_energy: float = 85
    birth_cost: float = 40
    birth_cooldown: int = 128
    social_denominator: float = 16
    # Resource guard: identities are never recycled within an ecological trial.
    max_individuals: int = 100_000

    def validate(self):
        assert self.size >= 12 and 0 <= self.obstacle_fraction < 0.4
        assert 1 <= self.founders <= self.population_cap < self.max_individuals
        assert self.stations >= 1 and self.predators == 1, "v1 supports one predator"
        assert self.social_denominator > 0
        assert 0 < self.newborn_energy <= self.birth_cost <= self.max_energy
        assert 0 < self.founder_energy <= self.max_energy
        assert 0 < self.birth_energy <= self.max_energy
        assert self.birth_cooldown > 0 and self.maturity > 0
        assert self.replenishment_rates and min(self.replenishment_rates) >= 0


@dataclass
class TrainingConfig:
    backend: str = "smollm"
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    revision: str = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
    device: str = "auto"
    max_tokens: int = 256
    decisions: int = 2_000_000
    warmstart_transitions: int = 50_000
    rollout_ticks: int = 128
    social_window: int = 256
    epochs: int = 2
    clip: float = 0.2
    gamma: float = 0.999
    gae_lambda: float = 0.95
    adapter_lr: float = 1e-4
    controller_lr: float = 3e-4
    entropy: float = 0.01
    value_coefficient: float = 0.5
    sequence_length: int = 32
    max_grad_norm: float = 0.5
    mutation_rms: float = 0.01
    mutation_watch_std: float = 0.05
    mutation_kl: float = 0.01
    checkpoints: list[int] = field(default_factory=lambda: [250_000, 500_000, 1_000_000, 2_000_000])
    evaluation_maps: int = 8
    final_evaluation_maps: int = 32
    evaluation_horizon: int = 2048
    evaluation_policies: int = 8
    evaluation_seed: int = 1_000_000
    max_wall_seconds: float = 0
    cpu_threads: int = 4
    newborn_evaluation_every: int = 16
    newborn_evaluation_maps: int = 2
    newborn_evaluation_horizon: int = 128


@dataclass
class ExperimentConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # More granular preferences can be added without code changes.
    profiles: dict = field(default_factory=lambda: {
        "individual": {"social_preference": 0.0, "vigilance_bias": 0.0},
        "prosocial": {"social_preference": 0.5, "vigilance_bias": 0.0},
        "competitive": {"social_preference": -0.5, "vigilance_bias": 0.0},
        "vigilance": {"social_preference": 0.0, "vigilance_bias": 1.0},
    })
    seeds: list[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])
    methods: list[str] = field(default_factory=lambda: ["iteration", "r_adult", "r_initial"])

    def validate(self):
        self.environment.validate()
        t = self.training
        assert t.backend in {"tiny", "smollm"}
        assert t.decisions > 0 and t.warmstart_transitions >= 0
        assert 0 < t.sequence_length <= t.rollout_ticks
        assert t.social_window % t.rollout_ticks == 0
        assert t.max_tokens <= 256 and t.max_tokens >= 64
        assert t.epochs > 0 and t.evaluation_horizon > 0
        assert t.evaluation_policies <= self.environment.population_cap
        assert set(self.methods) <= {"iteration", "r_adult", "r_initial"}
        assert 0 <= t.mutation_kl and 0 < t.gamma <= 1
        assert self.profiles and self.seeds

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def load_config(path=None):
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    raw = raw or {}
    cfg = ExperimentConfig(
        environment=EnvironmentConfig(**raw.pop("environment", {})),
        training=TrainingConfig(**raw.pop("training", {})), **raw,
    )
    cfg.validate()
    return cfg
