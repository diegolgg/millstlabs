"""Game protocol. The organization layer only ever talks to this interface (nothing in org/ imports game/hanabi)."""

from __future__ import annotations

from typing import Any, Protocol


class Match(Protocol):
    """One game in progress. Created by `Game.new_game(seed)`; one object per game so the RNG never leaks across games."""

    seed: int
    turn: int

    @property
    def current_player(self) -> int: ...

    def is_terminal(self) -> bool: ...

    def score(self) -> int: ...

    def observation(self, player: int) -> dict[str, Any]: ...

    def legal_moves(self, player: int) -> list[dict[str, Any]]: ...

    def is_legal(self, move: dict[str, Any]) -> bool: ...

    def step(self, move: dict[str, Any]) -> None: ...


class Game(Protocol):
    name: str
    num_players: int

    def new_game(self, seed: int) -> Match: ...

    def describe(self) -> dict[str, Any]: ...

    def fallback_move(self, legal_moves: list[dict[str, Any]]) -> dict[str, Any]: ...

    def digest(self) -> str: ...


class IllegalMove(ValueError):
    pass
