"""Pregnancy state and automatic reproduction, independent of learner weights."""
from dataclasses import dataclass

from gymnasium import spaces


class SurvivalNoteSpace(spaces.Text):
    """Bounded private Unicode text from the frozen language-model writer."""
    def __init__(self):
        super().__init__(4096, min_length=0)

    def contains(self, value):
        return isinstance(value, str) and len(value) <= self.max_length


SURVIVAL_PROMPT = (
    "Write a short survival note for your newborn, at most 64 tokens. "
    "Summarize lessons about food, energy, predators, watching and other prey. "
    "Use only the supplied private observations, action outcomes and inherited note. "
    "Advice may be uncertain; do not invent hidden information."
)


@dataclass
class Pregnancy:
    conceived_tick: int
    completion_tick: int
    survival_note: str | None = None

    def remaining(self, tick):
        return max(0, self.completion_tick - tick)


def resolve_reproduction(env, note_writer=None):
    """Complete existing pregnancies before conceiving; newborns cannot act here.

    Writers receive only the mother's bounded private history and inherited note.
    Neural writing is supplied by the trainer, never stored in simulator state.
    """
    parents = list(env.rng.permutation(list(env.prey)))
    for i in parents:
        p = env.prey[i]
        pregnancy = p.pregnancy
        if pregnancy is None or pregnancy.remaining(env.tick):
            continue
        if pregnancy.survival_note is None:
            note, tokens = note_writer(tuple(p.history), p.survival_note) if note_writer else ("", 0)
            if not isinstance(note, str) or not 0 <= tokens <= 64:
                raise ValueError("Survival writer must return text and at most 64 tokens")
            pregnancy.survival_note = note
            env.events.append({"type": "gestation_complete", "id": i, "tick": env.tick,
                               "conception_tick": pregnancy.conceived_tick,
                               "gestation_completion_tick": pregnancy.completion_tick,
                               "survival_note": note, "note_tokens": tokens})
        occupied = {q.pos for q in env.prey.values()} | {env.predator}
        free = [c for c in env.neighbors(p.pos) if c not in occupied]
        if not free or len(env.prey) >= env.cfg.population_cap:
            continue
        child = env._spawn(free[int(env.rng.integers(len(free)))], p)
        child.survival_note = pregnancy.survival_note
        p.cooldown = env.cfg.birth_cooldown
        p.pregnancy = None
        env.total_births += 1
        env.birth_wait_ticks += env.tick - pregnancy.completion_tick
        env.events.append({"type": "birth", "id": child.id, "parent": i,
                           "lineage": child.lineage, "generation": child.generation,
                           "tick": env.tick, "birth_tick": env.tick,
                           "conception_tick": pregnancy.conceived_tick,
                           "gestation_completion_tick": pregnancy.completion_tick,
                           "birth_wait_ticks": env.tick - pregnancy.completion_tick,
                           "survival_note": child.survival_note})
    for i in env.rng.permutation(parents):
        p = env.prey[i]
        if (p.pregnancy is not None or p.age < env.cfg.maturity or p.cooldown
                or p.energy < env.cfg.birth_energy or len(env.prey) >= env.cfg.population_cap):
            continue
        p.energy -= env.cfg.birth_cost
        p.pregnancy = Pregnancy(env.tick, env.tick + env.cfg.gestation_ticks)
        env.total_conceptions += 1
        env.events.append({"type": "conception", "id": i, "tick": env.tick,
                           "conception_tick": env.tick,
                           "gestation_completion_tick": p.pregnancy.completion_tick})


def resolve_immediate_reproduction(env):
    """Original birth-time eligibility, energy charge and adjacent-cell requirement."""
    for i in env.rng.permutation(list(env.prey)):
        p = env.prey[i]
        if p.age < env.cfg.maturity or p.cooldown or p.energy < env.cfg.birth_energy:
            continue
        occupied = {q.pos for q in env.prey.values()} | {env.predator}
        free = [c for c in env.neighbors(p.pos) if c not in occupied]
        if not free or len(env.prey) >= env.cfg.population_cap:
            continue
        p.energy -= env.cfg.birth_cost
        p.cooldown = env.cfg.birth_cooldown
        child = env._spawn(free[int(env.rng.integers(len(free)))], p)
        env.total_births += 1
        env.events.append({"type": "birth", "id": child.id, "parent": p.id,
                           "lineage": child.lineage, "generation": child.generation, "tick": env.tick})
