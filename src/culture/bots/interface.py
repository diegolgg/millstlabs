"""Bot protocol (spec section 4). LLM-written bot.py files must define `class Bot` with this shape."""

from __future__ import annotations

from typing import Any, Protocol

BOT_INTERFACE_DOC = '''
class Bot:
    name: str = "my_bot"
    def reset(self, game: dict, my_id: int, seed: int) -> None: ...
    def act(self, obs: dict) -> dict:  # must return an element of obs["legal_moves"]
        ...
'''.strip()

ALLOWED_IMPORTS = ("math", "random", "itertools", "collections", "functools", "dataclasses", "typing")


class Bot(Protocol):
    name: str

    def reset(self, game: dict[str, Any], my_id: int, seed: int) -> None: ...

    def act(self, obs: dict[str, Any]) -> dict[str, Any]: ...
