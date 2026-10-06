"""Engine layer: determinism, seeding discipline, observation hygiene, legality."""

import json
import multiprocessing as mp

import pytest

from culture.bots.runner import BotSpec, play_game
from culture.game.hanabi import HanabiGame, HanabiParams, IllegalMove
from culture.game.seeds import SeedSet, bot_seed, generation_seeds

from _raw_helpers import digest_for

P = HanabiParams()


def test_same_seed_byte_identical_action_logs_across_runs():
    a = [play_game(P, [BotSpec.anchor("random")] * 2, s, record=True) for s in range(20)]
    b = [play_game(P, [BotSpec.anchor("random")] * 2, s, record=True) for s in range(20)]
    assert [json.dumps(x.actions) for x in a] == [json.dumps(x.actions) for x in b]
    assert len({x.digest for x in a}) > 15  # different seeds give different games


def test_byte_identical_across_processes():
    local = [digest_for(("random", s)) for s in range(12)] + [digest_for(("piers", s)) for s in range(6)]
    ctx = mp.get_context("spawn")
    with ctx.Pool(2) as pool:
        remote = pool.map(digest_for, [("random", s) for s in range(12)] + [("piers", s) for s in range(6)])
    assert local == remote


def test_one_env_per_game_order_independent():
    fwd = [play_game(P, [BotSpec.anchor("piers")] * 2, s).digest for s in range(10)]
    rev = [play_game(P, [BotSpec.anchor("piers")] * 2, s).digest for s in reversed(range(10))]
    assert fwd == list(reversed(rev))


def test_seed_sets():
    s = SeedSet(100, 5)
    assert s.seeds == [100, 101, 102, 103, 104]
    a, b = generation_seeds(7, 3, 50), generation_seeds(7, 3, 50)
    assert a == b and a.digest() == b.digest()
    assert set(generation_seeds(7, 3, 200).seeds).isdisjoint(generation_seeds(7, 4, 200).seeds)
    assert set(generation_seeds(7, 3, 200, "eval").seeds).isdisjoint(generation_seeds(7, 3, 200, "verify").seeds)
    assert bot_seed(5, 0) != bot_seed(5, 1)


def test_observation_hides_own_hand_and_is_json():
    m = HanabiGame(P).new_game(3)
    obs = m.observation(0)
    json.dumps(obs)
    assert all(c["color"] is None and c["rank"] is None for c in obs["my_hand"])
    assert all(c["color"] in "RYGWB" and 1 <= c["rank"] <= 5 for c in obs["hands"]["1"])
    assert obs["legal_moves"] and m.observation(1)["legal_moves"] == []
    assert "pyhanabi" not in obs and "vectorized" not in obs
    assert obs["game"]["ranks"] == [1, 2, 3, 4, 5]


def test_hint_targets_absolute_and_last_moves():
    m = HanabiGame(P).new_game(5)
    hint = next(x for x in m.legal_moves() if x["type"] == "REVEAL_RANK")
    m.step(hint)
    o1 = m.observation(1)
    lm = o1["last_moves"][0]
    assert lm["player"] == 0 and lm["target"] == 1 and lm["rank"] == hint["rank"]
    touched = lm["touched"]
    assert touched and all(o1["my_hand"][i]["hints"]["rank"] == hint["rank"] for i in touched)
    # negative information shows up in `possible`
    untouched = [i for i in range(5) if i not in touched]
    assert all(hint["rank"] not in o1["my_hand"][i]["possible"]["ranks"] for i in untouched)


def test_illegal_moves_rejected_before_engine():
    m = HanabiGame(P).new_game(1)
    with pytest.raises(IllegalMove):
        m.step({"type": "DISCARD", "card_index": 0})  # 8 tokens: discard is illegal
    with pytest.raises(IllegalMove):
        m.step({"type": "REVEAL_COLOR", "target": 0, "color": "R"})  # cannot hint yourself
    with pytest.raises(IllegalMove):
        m.step({"type": "FLY"})
    assert m.turn == 0 and not m.is_terminal()


def test_deal_order_tracking():
    r = play_game(P, [BotSpec.anchor("piers")] * 2, 11, record=True)
    assert len(r.deck) >= 10 and len(r.deck) <= 50
    plays = [a for a in r.hl_actions if a["type"] in (0, 1)]
    assert len({a["target"] for a in plays}) == len(plays)  # each card leaves a hand at most once
