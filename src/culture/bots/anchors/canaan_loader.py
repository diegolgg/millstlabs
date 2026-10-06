"""Load the vendored, unmodified Canaan et al. agents and drive them either from raw HLE dicts or from our observation JSON.

The vendored files import `rl_env`, `pyhanabi`, `rulebased_agent`, `ruleset` as top-level modules. We alias the first two
to the HLE package and load the rest from `canaan/` under private names. Canaan's rules draw from the module-level
`random`; we swap `ruleset.random` for the acting bot's own `random.Random(seed)` before every call, so each bot has an
independent seeded stream and games are deterministic.

`CanaanBot` implements our Bot protocol: it rebuilds the HLE-format dict (relative offsets, 0-based ranks, plus a
`pyhanabi` stand-in exposing `card_knowledge()` plausibility) from our JSON. If the adapter dropped or leaked information,
trajectories would diverge from raw HLE (test_adapter.py) and Flawed would stop scoring about 0.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from typing import Any

from hanabi_learning_environment import pyhanabi, rl_env

_DIR = Path(__file__).parent / "canaan"
_MODULES: dict[str, Any] = {}
AGENT_CLASSES = {"piers": ("piers_agent", "PiersAgent"), "iggi": ("iggi_agent", "IGGIAgent"), "flawed": ("flawed_agent", "FlawedAgent")}


def _load(name: str):
    if name in _MODULES:
        return _MODULES[name]
    sys.modules.setdefault("rl_env", rl_env)
    sys.modules.setdefault("pyhanabi", pyhanabi)
    for dep in ("ruleset", "rulebased_agent"):
        if dep not in _MODULES:
            spec = importlib.util.spec_from_file_location(dep, _DIR / f"{dep}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[dep] = mod  # the agent files do `from ruleset import Ruleset`
            spec.loader.exec_module(mod)
            _MODULES[dep] = mod
    if name not in _MODULES:
        spec = importlib.util.spec_from_file_location(f"canaan_{name}", _DIR / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULES[name] = mod
    return _MODULES[name]


def make_agent(kind: str):
    """Instantiate the unmodified Canaan agent class (`piers`, `iggi`, `flawed`)."""
    mod_name, cls_name = AGENT_CLASSES[kind]
    return getattr(_load(mod_name), cls_name)({"information_tokens": 8})


def ruleset_module():
    _load("ruleset")
    return _MODULES["ruleset"]


def act_with_rng(agent, hle_obs: dict, rng: random.Random):
    rs = ruleset_module()
    rs.random = rng
    try:
        return agent.act(hle_obs)
    finally:
        rs.random = random


# ----------------------------------------------------------------------- our JSON -> HLE dict
class _Knowledge:
    __slots__ = ("_cp", "_rp")

    def __init__(self, cp: set[int], rp: set[int]):
        self._cp, self._rp = cp, rp

    def color_plausible(self, i: int) -> bool:
        return i in self._cp

    def rank_plausible(self, i: int) -> bool:
        return i in self._rp


class _ObservationShim:
    __slots__ = ("_k",)

    def __init__(self, knowledge):
        self._k = knowledge

    def card_knowledge(self):
        return self._k


def json_to_hle(obs: dict[str, Any]) -> dict[str, Any]:
    g = obs["game"]
    n, me, colors = g["num_players"], obs["me"], g["colors"]
    cidx = {c: i for i, c in enumerate(colors)}
    observed, knowledge, shim = [], [], []
    for off in range(n):
        p = (me + off) % n
        cards = obs["my_hand"] if off == 0 else obs["hands"][str(p)]
        hand, kn, sk = [], [], []
        for c in cards:
            hand.append({"color": None, "rank": -1} if off == 0 else {"color": c["color"], "rank": c["rank"] - 1})
            h = c["hints"]
            kn.append({"color": h["color"], "rank": None if h["rank"] is None else h["rank"] - 1})
            sk.append(_Knowledge({cidx[x] for x in c["possible"]["colors"]}, {r - 1 for r in c["possible"]["ranks"]}))
        observed.append(hand)
        knowledge.append(kn)
        shim.append(sk)
    legal = []
    for m in obs["legal_moves"]:
        t = m["type"]
        if t in ("PLAY", "DISCARD"):
            legal.append({"action_type": t, "card_index": m["card_index"]})
        elif t == "REVEAL_COLOR":
            legal.append({"action_type": t, "color": m["color"], "target_offset": (m["target"] - me) % n})
        else:
            legal.append({"action_type": t, "rank": m["rank"] - 1, "target_offset": (m["target"] - me) % n})
    return {
        "current_player": obs["current_player"],
        "current_player_offset": (obs["current_player"] - me) % n,
        "life_tokens": obs["life_tokens"],
        "information_tokens": obs["info_tokens"],
        "num_players": n,
        "deck_size": obs["deck_size"],
        "fireworks": dict(obs["fireworks"]),
        "legal_moves": legal,
        "observed_hands": observed,
        "discard_pile": [{"color": d["color"], "rank": d["rank"] - 1} for d in obs["discards"]],
        "card_knowledge": knowledge,
        "pyhanabi": _ObservationShim(shim),
    }


def hle_action_to_json(action: dict[str, Any], me: int, n: int) -> dict[str, Any]:
    t = action["action_type"]
    if t in ("PLAY", "DISCARD"):
        return {"type": t, "card_index": int(action["card_index"])}
    target = (me + int(action["target_offset"])) % n
    if t == "REVEAL_COLOR":
        return {"type": t, "target": target, "color": action["color"]}
    return {"type": t, "target": target, "rank": int(action["rank"]) + 1}


class CanaanBot:
    """Our Bot protocol around an unmodified Canaan agent."""

    def __init__(self, kind: str):
        self.kind = kind
        self.name = f"canaan_{kind}"
        self._agent = None
        self._rng = random.Random(0)
        self._me = 0
        self._n = 2

    def reset(self, game: dict, my_id: int, seed: int) -> None:
        self._agent = make_agent(self.kind)
        self._rng = random.Random(seed)
        self._me, self._n = my_id, game["num_players"]

    def act(self, obs: dict) -> dict:
        action = act_with_rng(self._agent, json_to_hle(obs), self._rng)
        return hle_action_to_json(action, self._me, self._n)
