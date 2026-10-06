"""N seeded games of one artifact with itself."""

from __future__ import annotations

from ..bots.runner import BotSpec, GameResult
from .pool import Evaluator


def selfplay(ev: Evaluator, spec: BotSpec, seeds: list[int]) -> list[GameResult]:
    return ev.run([((spec,) * ev.params.players, seeds)])[0]


def seat_rates(results: list[GameResult], key: str) -> dict[str, float]:
    """Illegal / timeout / slow rates over the moves made by seats holding `key`."""
    moves = illegal = timeouts = slow = 0
    for r in results:
        for k, m, il, to, sl in zip(r.seats, r.moves, r.illegal, r.timeouts, r.slow):
            if k == key:
                moves += m
                illegal += il
                timeouts += to
                slow += sl
    d = max(moves, 1)
    return {"illegal_rate": illegal / d, "timeout_rate": timeouts / d, "slow_rate": slow / d, "moves": moves}


def first_error(results: list[GameResult], key: str) -> str | None:
    for r in results:
        for k, e in zip(r.seats, r.errors):
            if k == key and e:
                return e
    return None
