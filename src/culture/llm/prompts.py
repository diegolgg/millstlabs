"""Versioned prompt templates (files in prompts/; their hashes go in the run manifest).

Prompts describe the task and the interface only. They never describe credit, selection or allocation rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from string import Template

from ..bots.interface import ALLOWED_IMPORTS, BOT_INTERFACE_DOC

PROMPT_DIR = Path(__file__).parent / "prompts"
TEMPLATES = ("system", "author", "revise", "teach", "ingest", "merge", "repair")

OBSERVATION_EXAMPLE = {
    "game": {"num_players": 2, "hand_size": 5, "colors": ["R", "Y", "G", "W", "B"], "ranks": [1, 2, 3, 4, 5],
             "max_information_tokens": 8, "max_life_tokens": 3},
    "me": 0, "current_player": 0, "turn": 12, "deck_size": 28, "info_tokens": 6, "life_tokens": 3,
    "fireworks": {"R": 1, "Y": 0, "G": 2, "W": 0, "B": 0},
    "discards": [{"color": "R", "rank": 1}],
    "hands": {"1": [{"color": "G", "rank": 3, "hints": {"color": None, "rank": 3},
                     "possible": {"colors": ["R", "G", "B"], "ranks": [3]}}]},
    "my_hand": [{"color": None, "rank": None, "hints": {"color": "R", "rank": None},
                 "possible": {"colors": ["R"], "ranks": [2, 3, 4, 5]}}],
    "last_moves": [{"player": 1, "type": "REVEAL_COLOR", "target": 0, "color": "R", "touched": [0]}],
    "legal_moves": [{"type": "PLAY", "card_index": 0}, {"type": "DISCARD", "card_index": 0},
                    {"type": "REVEAL_COLOR", "target": 1, "color": "G"}, {"type": "REVEAL_RANK", "target": 1, "rank": 3}],
}


def load(name: str) -> Template:
    return Template((PROMPT_DIR / f"{name}.md").read_text())


def render(name: str, **kw) -> str:
    return load(name).safe_substitute(**kw)


def prompt_hashes() -> dict[str, str]:
    return {n: hashlib.sha256((PROMPT_DIR / f"{n}.md").read_bytes()).hexdigest()[:16] for n in TEMPLATES}


def system_prompt(game_desc: dict, limits, variant_note: str = "") -> str:
    return render(
        "system",
        variant_note=variant_note,
        colors=" ".join(game_desc["colors"]),
        max_rank=max(game_desc["ranks"]),
        hand_size=game_desc["hand_size"],
        max_info=game_desc.get("max_information_tokens", 8),
        max_life=game_desc.get("max_life_tokens", 3),
        bot_interface=BOT_INTERFACE_DOC,
        observation_example=json.dumps(OBSERVATION_EXAMPLE, indent=1),
        allowed_imports=", ".join(ALLOWED_IMPORTS),
        soft_ms=int(limits.soft_timeout_ms),
        hard_s=limits.hard_timeout_s,
        max_lines=limits.max_lines,
    )
