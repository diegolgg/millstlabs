"""Anchor bots: random, simple (HLE SimpleAgent port), Canaan Piers / IGGI / Flawed.

Two forms of the Canaan agents exist:
- `canaan_piers`, `canaan_iggi`, `canaan_flawed`: the vendored, unmodified files behind `CanaanBot` (ground truth).
- `piers`, `iggi`, `flawed`: `rulebot.py` with Canaan's rule lists, a sandbox-valid artifact that reproduces the vendored
  agents move for move. These are the anchors used in evaluation (faster, and they can be handed around as artifacts).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_DIR = Path(__file__).parent
CONFIG_RE = re.compile(r"# === CONFIG ===\n(.*?)# === END CONFIG ===\n", re.S)

PIERS_RULES = [
    ["hail_mary", {}],
    ["play_safe_card", {}],
    ["play_probably_safe", {"threshold": 0.6, "require_extra_lives": True}],
    ["tell_anyone_useful_card", {}],
    ["tell_dispensable", {"min_information_tokens": 3}],
    ["osawa_discard", {}],
    ["discard_oldest_first", {}],
    ["tell_randomly", {}],
    ["discard_randomly", {}],
]
IGGI_RULES = [
    ["play_if_certain", {}],
    ["play_safe_card", {}],
    ["tell_playable_card_outer", {}],
    ["osawa_discard", {}],
    ["discard_oldest_first", {}],
    ["legal_random", {}],
]
FLAWED_RULES = [
    ["play_safe_card", {}],
    ["play_probably_safe", {"threshold": 0.25}],
    ["tell_randomly", {}],
    ["osawa_discard", {}],
    ["discard_oldest_first", {}],
    ["discard_randomly", {}],
]
RULE_LISTS = {"piers": PIERS_RULES, "iggi": IGGI_RULES, "flawed": FLAWED_RULES}

# What each rule does, in prose; used to write conventions.md for rule-list artifacts.
RULE_DOCS = {
    "hail_mary": "When the deck is empty and we have more than one life, play the card most likely to be playable.",
    "play_safe_card": "Play a card if every identity still possible for it is playable now.",
    "play_if_certain": "Play a card whose color and rank were both hinted and that is playable now.",
    "play_probably_safe": "Play the card most likely to be playable if that probability is at least {threshold}"
    "{require_extra_lives_text}.",
    "tell_anyone_useful_card": "If a partner holds a playable card, hint its rank (or its color if rank is known).",
    "tell_playable_card_outer": "Same as hinting a useful card: rank first, then color.",
    "tell_dispensable": "With fewer than {min_information_tokens} hint tokens, hint partner cards that are already useless.",
    "osawa_discard": "Discard a card known to be useless (dead color, already played, or unplayable forever).",
    "discard_oldest_first": "Discard the oldest card (slot 0) when hint tokens are not full.",
    "discard_probably_useless": "Discard the card most likely to be useless if that probability is at least {threshold}.",
    "tell_randomly": "Give the next player a random color or rank hint about a random card.",
    "tell_unknown": "Hint the next player the first color or rank they do not yet know.",
    "discard_randomly": "Discard a random card when hint tokens are not full.",
    "legal_random": "Make a random legal move.",
}


def rulebot_source(rules: list, noise: int = 0) -> str:
    """bot.py text for a rule list (the template with its CONFIG block replaced). `noise != 0` (the null stub) adds
    `NOISE = <k>` and seeds the bot's RNG with `seed + NOISE`: the same policy with a different random stream, so the
    expected score is unchanged and only the realized games differ. noise = 0 leaves the template text untouched."""
    template = (_DIR / "rulebot.py").read_text()
    block = "# === CONFIG ===\nRULES = " + repr(rules) + "\n# === END CONFIG ===\n"
    src = CONFIG_RE.sub(lambda _: block, template, count=1)
    if noise:
        src = src.replace("COPIES = [3, 2, 2, 2, 1]\n", f"COPIES = [3, 2, 2, 2, 1]\nNOISE = {int(noise)}\n", 1)
        src = src.replace("self.rng = random.Random(seed)", "self.rng = random.Random(seed + NOISE)", 1)
    return src


def parse_rules(code: str) -> list | None:
    """Recover the rule list from a rule-list bot.py, or None if the CONFIG block is absent or malformed."""
    m = CONFIG_RE.search(code)
    if not m:
        return None
    body = m.group(1).strip()
    if not body.startswith("RULES = "):
        return None
    try:
        rules = ast.literal_eval(body[len("RULES = "):])
    except (ValueError, SyntaxError):
        return None
    return rules if isinstance(rules, list) else None


def conventions_for(rules: list, title: str = "Conventions") -> str:
    lines = [f"# {title}", "", "Rules are tried in order; the first that applies decides the move.", ""]
    for i, (name, params) in enumerate(rules, 1):
        p = dict(params)
        p.setdefault("threshold", 0.95)
        p.setdefault("min_information_tokens", 8)
        p["require_extra_lives_text"] = " and we have more than one life" if p.get("require_extra_lives") else ""
        lines.append(f"{i}. **{name}**: " + RULE_DOCS.get(name, name).format(**p))
    return "\n".join(lines) + "\n"


def anchor_source(name: str) -> str:
    if name in RULE_LISTS:
        return rulebot_source(RULE_LISTS[name])
    if name in ("random", "simple"):
        return (_DIR / f"{name}.py").read_text()
    raise KeyError(f"no source anchor named {name!r}")


def anchor_conventions(name: str) -> str:
    if name in RULE_LISTS:
        return conventions_for(RULE_LISTS[name], title=f"{name.capitalize()} conventions (Canaan et al. rule list)")
    if name == "random":
        return "# Random\n\nMake a uniformly random legal move.\n"
    if name == "simple":
        return "# Simple\n\nPlay any hinted card; else color-hint a partner's playable card; else discard oldest.\n"
    raise KeyError(name)


SOURCE_ANCHORS = ("random", "simple", "piers", "iggi", "flawed")
CANAAN_ANCHORS = ("canaan_piers", "canaan_iggi", "canaan_flawed")
