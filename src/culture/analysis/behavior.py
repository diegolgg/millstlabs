"""Behavioural fingerprints of artifacts on a fixed probe set of observations (spec section 9, diversity).

The probe set is a deterministic sample of observations from games between anchor bots. Each artifact is reset with a
fixed seed and asked to act on every probe observation; its answers give (a) categorical features for IF-PCA and (b)
pairwise action-disagreement distances."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np

from ..bots.runner import BotSpec, SandboxLimits, bot_factory, play_game
from ..game.hanabi import HanabiParams, move_key

CATEGORIES = ("PLAY", "DISCARD", "REVEAL_COLOR", "REVEAL_RANK")


def probe_set(n: int = 200, seed: int = 0, params: HanabiParams | None = None,
              pairs=(("piers", "iggi"), ("iggi", "piers"), ("piers", "piers"), ("simple", "piers"))) -> list[dict]:
    params = params or HanabiParams()
    rng = random.Random(seed)
    obs, g = [], 0
    while len(obs) < n:
        a, b = pairs[g % len(pairs)]
        r = play_game(params, [BotSpec.anchor(a), BotSpec.anchor(b)], 900_000 + seed * 1000 + g, trace=True)
        turns = [t["obs"] for t in r.trace if len(t["obs"]["legal_moves"]) > 1]
        obs += rng.sample(turns, min(len(turns), 4))
        g += 1
    return obs[:n]


def actions(code_or_anchor: str, probes: list[dict], seed: int = 12345, limits: SandboxLimits | None = None) -> list:
    """The move each probe observation gets from a freshly reset bot (None if it errors or is illegal)."""
    limits = limits or SandboxLimits()
    spec = BotSpec.anchor(code_or_anchor) if not code_or_anchor.lstrip().startswith(("import", "class", '"""', "#", "from")) \
        else BotSpec("probe:" + hashlib.sha256(code_or_anchor.encode()).hexdigest()[:16], code_or_anchor)
    factory = bot_factory(spec, limits)
    out = []
    for o in probes:
        try:
            bot = factory()
            bot.reset(o["game"], o["me"], seed)
            m = bot.act(json.loads(json.dumps(o)))
            k = move_key(m)
            out.append(k if k in {move_key(x) for x in o["legal_moves"]} else None)
        except Exception:  # noqa: BLE001 - a broken bot simply has no answer on this probe
            out.append(None)
    return out


def features(action_lists: list[list]) -> np.ndarray:
    """One-hot action category per probe (n_artifacts x 4 * n_probes)."""
    rows = []
    for acts in action_lists:
        v = np.zeros((len(acts), len(CATEGORIES)))
        for i, a in enumerate(acts):
            if a is not None:
                v[i, CATEGORIES.index(a[0])] = 1.0
        rows.append(v.ravel())
    return np.array(rows)


def disagreement(action_lists: list[list]) -> np.ndarray:
    """Pairwise fraction of probes on which two artifacts choose different moves."""
    n = len(action_lists)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = action_lists[i], action_lists[j]
            D[i, j] = D[j, i] = float(np.mean([x != y for x, y in zip(a, b)]))
    return D
