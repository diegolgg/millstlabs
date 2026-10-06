"""Module-level helpers so spawned worker processes can import them."""

from culture.bots.runner import BotSpec, play_game
from culture.game.hanabi import HanabiParams
from culture.game.hle_raw import play_raw


def raw_and_adapter(args):
    kind, seed = args
    score_raw, log_raw = play_raw([kind, kind], seed)
    r = play_game(HanabiParams(), [BotSpec.anchor("canaan_" + kind)] * 2, seed, record=True)
    return seed, score_raw, log_raw == r.actions, r.score


def digest_for(args):
    name, seed = args
    return play_game(HanabiParams(), [BotSpec.anchor(name)] * 2, seed).digest
