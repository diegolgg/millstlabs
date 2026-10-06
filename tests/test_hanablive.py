"""hanab.live export round trip (spec section 3): export a game, reload it with an independent rules
implementation, same final score (and same number of turns)."""

import json

import pytest

from culture.bots.anchors import anchor_source
from culture.bots.runner import BotSpec, play_game
from culture.game import hanablive
from culture.game.hanabi import HanabiParams

P = HanabiParams()


@pytest.mark.parametrize("pair", [("piers", "piers"), ("iggi", "piers"), ("flawed", "flawed"), ("random", "simple")])
def test_export_reload_same_final_score(tmp_path, pair):
    for seed in range(25):
        r = play_game(P, [BotSpec.anchor(pair[0]), BotSpec.anchor(pair[1])], seed, record=True)
        game = hanablive.export(r.deck, r.hl_actions, players=["Alice", "Bob"])
        path = tmp_path / f"g{seed}.json"
        path.write_text(json.dumps(game))
        again = hanablive.replay(json.loads(path.read_text()))
        assert again["score"] == r.score and again["turns"] == r.turns, (pair, seed)


def test_export_format():
    r = play_game(P, [BotSpec("x", anchor_source("piers"))] * 2, 3, record=True)
    g = hanablive.export(r.deck, r.hl_actions)
    assert g["options"]["variant"] == "No Variant" and g["players"] == ["Alice", "Bob"]
    assert len(g["deck"]) == 50 and all(set(c) == {"suitIndex", "rank"} for c in g["deck"])
    counts = {}
    for c in g["deck"]:
        counts[(c["suitIndex"], c["rank"])] = counts.get((c["suitIndex"], c["rank"]), 0) + 1
    assert all(counts[(s, 1)] == 3 and counts[(s, 5)] == 1 for s in range(5))
    assert all(a["type"] in (0, 1, 2, 3) for a in g["actions"])
    assert all(0 <= a["target"] < 50 for a in g["actions"] if a["type"] in (0, 1))
    assert all(a["target"] in (0, 1) for a in g["actions"] if a["type"] in (2, 3))


def test_reloader_rejects_corrupted_replays():
    r = play_game(P, [BotSpec.anchor("piers")] * 2, 5, record=True)
    g = hanablive.export(r.deck, r.hl_actions)
    clue = next(i for i, a in enumerate(g["actions"]) if a["type"] == 3)
    g["actions"][clue]["target"] = 1 - g["actions"][clue]["target"]  # clue to self (the acting player)
    with pytest.raises(ValueError):
        hanablive.replay(g)
