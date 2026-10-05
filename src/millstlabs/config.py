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
    message_symbols: int = 1  # Includes silence. One preserves the original seven actions.
    message_radius: int = 6
    message_capacity: int = 4
    message_cost: float = 0.02
    predator_temperature: float | None = None  # None preserves the legacy movement kernel.

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
        assert 1 <= self.message_symbols <= 8 and self.message_radius > 0
        assert self.message_capacity > 0 and self.message_cost >= 0
        assert self.predator_temperature is None or self.predator_temperature >= 0


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
    separate_critic: bool = True
    value_scale: float = 100.0
    critic_lr: float = 1e-3
    temperature_start: float = 1.0
    temperature_end: float = 1.0
    temperature_decay_decisions: int = 0
    evaluation_temperature: float = 1.0
    deliver_messages: bool = True
    development_maps: int = 0
    development_seed: int = 3_000_000
    evaluation_ablations: bool = False
    predator_evaluation_temperatures: list[float] = field(default_factory=list)
    controller_architecture: str = "legacy"
    corpus_mode: str = "off"
    corpus_slots: int = 4
    corpus_food_ttl: int = 16
    corpus_tool_cost: float = 0.05
    intrinsic_critic: bool = False
    novelty_beta_start: float = 0.0
    novelty_beta_end: float = 0.0
    novelty_decay_decisions: int = 32768
    predator_curriculum: list[dict] = field(default_factory=list)
    corpus_interface: str = "tools"  # Legacy interface remains checkpoint-compatible.
    note_style: str = "grounded"
    note_interval: int = 16
    note_max_tokens: int = 48
    note_credit: float = 0.1
    note_write_penalty: float = 0.01
    actor_grounding: bool = False
    feed_retention: float = 0.0
    action_policy: str = "controller"
    note_memory: str = "map_facts"

    @property
    def split_controller(self):
        return self.backend == "structured" or self.controller_architecture == "split"


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
        assert t.backend in {"tiny", "smollm", "structured"}
        assert t.decisions > 0 and t.warmstart_transitions >= 0
        assert 0 < t.sequence_length <= t.rollout_ticks
        assert t.social_window % t.rollout_ticks == 0
        assert 64 <= t.max_tokens <= 1024
        assert t.epochs > 0 and t.evaluation_horizon > 0
        assert t.evaluation_policies <= self.environment.population_cap
        assert set(self.methods) <= {"iteration", "r_adult", "r_initial"}
        assert 0 <= t.mutation_kl and 0 < t.gamma <= 1
        assert self.profiles and self.seeds
        assert t.value_scale > 0 and t.critic_lr > 0
        assert min(t.temperature_start, t.temperature_end, t.evaluation_temperature) > 0
        assert t.temperature_decay_decisions >= 0 and t.development_maps >= 0
        assert t.controller_architecture in {"legacy", "split"}
        assert t.controller_architecture != "split" or t.backend == "smollm"
        assert self.environment.message_symbols == 1 or t.split_controller
        assert all(x >= 0 for x in t.predator_evaluation_temperatures)
        assert t.corpus_mode in {"off", "private", "shared"}
        assert t.corpus_slots > 0 and t.corpus_food_ttl > 0 and t.corpus_tool_cost >= 0
        assert t.corpus_mode == "off" or (t.split_controller and self.environment.message_symbols == 1)
        assert not t.intrinsic_critic or t.split_controller
        assert min(t.novelty_beta_start, t.novelty_beta_end) >= 0 and t.novelty_decay_decisions > 0
        assert not (t.novelty_beta_start or t.novelty_beta_end) or t.intrinsic_critic
        assert all(s["decisions"] >= 0 and s["temperature"] >= 0 for s in t.predator_curriculum)
        assert [s["decisions"] for s in t.predator_curriculum] == sorted({s["decisions"] for s in t.predator_curriculum})
        assert t.corpus_interface in {"tools", "notes"}
        assert t.note_style in {"grounded", "prose"}
        assert t.note_interval > 0 and 1 <= t.note_max_tokens <= 128
        assert min(t.note_credit, t.note_write_penalty, t.feed_retention) >= 0
        assert not t.actor_grounding or t.split_controller
        if t.corpus_interface == "notes":
            assert t.corpus_mode != "off" and t.split_controller
            assert t.note_style != "prose" or t.backend == "smollm"
        assert t.action_policy in {"controller", "lm_token"}
        assert t.note_memory in {"map_facts", "tactics"}
        if t.action_policy == "lm_token":
            assert t.backend == "smollm" and t.controller_architecture == "split" and t.separate_critic
            assert t.warmstart_transitions == 0 and t.feed_retention == 0
            assert not t.actor_grounding and not t.intrinsic_critic
            assert t.corpus_interface == "notes" and self.environment.message_symbols == 1
            assert t.max_tokens >= 512
            assert t.corpus_tool_cost == 0 and t.note_write_penalty == 0
        if t.note_memory == "tactics":
            assert t.corpus_interface == "notes" and t.corpus_tool_cost == 0

    def digest(self):
        # Default extensions do not invalidate already-running v1 checkpoints.
        data = asdict(self)
        extensions = {
            "environment": ["message_symbols", "message_radius", "message_capacity", "message_cost", "predator_temperature"],
            "training": ["separate_critic", "value_scale", "critic_lr", "temperature_start", "temperature_end",
                         "temperature_decay_decisions", "evaluation_temperature", "deliver_messages",
                         "development_maps", "development_seed", "evaluation_ablations", "predator_evaluation_temperatures",
                         "controller_architecture", "corpus_mode", "corpus_slots", "corpus_food_ttl", "corpus_tool_cost",
                         "intrinsic_critic", "novelty_beta_start", "novelty_beta_end", "novelty_decay_decisions", "predator_curriculum",
                         "corpus_interface", "note_style", "note_interval", "note_max_tokens", "note_credit",
                         "note_write_penalty", "actor_grounding", "feed_retention", "action_policy", "note_memory"],
        }
        for section, names in extensions.items():
            defaults = EnvironmentConfig() if section == "environment" else TrainingConfig()
            for name in names:
                if data[section][name] == getattr(defaults, name):
                    del data[section][name]
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_config(path=None):
    raw = yaml.safe_load(Path(path).read_text()) if path else {}
    raw = raw or {}
    cfg = ExperimentConfig(
        environment=EnvironmentConfig(**raw.pop("environment", {})),
        training=TrainingConfig(**raw.pop("training", {})), **raw,
    )
    cfg.validate()
    return cfg
