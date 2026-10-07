"""Fix round 1, step 7: the bot seed is a one-way hash of (salt, game seed, seat), still deterministic."""

import os
import subprocess
import sys
from pathlib import Path

from culture.bots.runner import BotSpec, play_game
from culture.game.hanabi import HanabiParams
from culture.game.seeds import bot_seed

ROOT = Path(__file__).resolve().parents[1]


def test_hashed_seed_is_deterministic_across_processes():
    here = [bot_seed(s, p) for s in (0, 1, 12345, 2**30) for p in (-1, 0, 1)]
    code = ("from culture.game.seeds import bot_seed; "
            "print([bot_seed(s, p) for s in (0, 1, 12345, 2**30) for p in (-1, 0, 1)])")
    for hs in ("0", "1", "random"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT,
                             env=dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT / "src"))).stdout
        assert out.strip() == str(here)


def test_seed_does_not_reveal_the_deal_seed():
    seeds = [bot_seed(s, 0) for s in range(200)]
    assert all(seeds[i] != s * 16 + 1 for i, s in enumerate(range(200)))  # not the old linear formula
    diffs = {b - a for a, b in zip(seeds, seeds[1:])}
    assert len(diffs) > 150  # no constant stride to invert
    assert bot_seed(5, 0) != bot_seed(5, 1) and all(0 <= x < 2**31 - 1 for x in seeds)


def test_games_stay_reproducible():
    from culture.bots.anchors import anchor_source

    s = BotSpec("piers_seed", anchor_source("piers"))
    assert [play_game(HanabiParams(), [s, s], k).digest for k in range(5)] == \
           [play_game(HanabiParams(), [s, s], k).digest for k in range(5)]
