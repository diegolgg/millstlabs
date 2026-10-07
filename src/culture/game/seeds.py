"""Seed discipline: `seed = base + game_index`, shared seed sets per experiment, content-hashed so pairing is verifiable."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

# Generation seed sets are spaced far apart so selfplay/crossplay/verification/anchor sets never overlap.
GENERATION_STRIDE = 1_000_003
# Within a generation the purposes are disjoint: `feedback` deals are the only ones whose traces an LLM ever sees,
# `verify` deals are used by receivers to check payloads, `eval` deals are held out (selection, metrics, credit).
PURPOSE_OFFSETS = {"eval": 0, "verify": 200_000, "ladder": 400_000, "probe": 600_000, "reeval": 800_000,
                   "feedback": 900_000}
MAX_GAMES_PER_PURPOSE = 100_000  # keeps purpose blocks (and generations) disjoint


@dataclass(frozen=True)
class SeedSet:
    base: int
    n: int

    @property
    def seeds(self) -> list[int]:
        return [self.base + i for i in range(self.n)]

    def digest(self) -> str:
        return hashlib.sha256(f"{self.base}:{self.n}".encode()).hexdigest()[:12]

    def head(self, n: int) -> "SeedSet":
        return SeedSet(self.base, min(n, self.n))


def generation_seeds(experiment_seed: int, generation: int, n: int, purpose: str = "eval") -> SeedSet:
    """The shared deal set for one generation. Depends only on (experiment seed, generation, purpose), never on the
    condition or the population seed, so every condition of an experiment plays the same deals."""
    if n > MAX_GAMES_PER_PURPOSE:
        raise ValueError(f"at most {MAX_GAMES_PER_PURPOSE} games per purpose per generation")
    base = experiment_seed * 7_919_993 + generation * GENERATION_STRIDE + PURPOSE_OFFSETS[purpose]
    return SeedSet(base % (2**31 - 2**24), n)


BOT_SEED_SALT = "culture/bot-seed/v1"


def bot_seed(game_seed: int, seat: int) -> int:
    """Seed handed to a bot's `reset` (its only legitimate randomness source). A one-way hash of (salt, game seed,
    seat), so a bot cannot recover the deal seed from it (the old `game_seed * 16 + seat + 1` was trivially
    invertible, and the deck is a deterministic function of the deal seed). Deterministic across runs and processes.
    Seat -1 seeds the module-level `random` state that `play_game` resets before each game."""
    h = hashlib.sha256(f"{BOT_SEED_SALT}:{int(game_seed)}:{int(seat)}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)
