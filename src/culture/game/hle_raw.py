"""Reference driver: unmodified Canaan agents on raw HLE `rl_env.HanabiEnv` dicts, no adapter in the loop.

Used only by the adapter ground-truth tests: the same agents driven through our adapter must produce identical action logs.
"""

from __future__ import annotations

import random

from hanabi_learning_environment import rl_env

from ..bots.anchors.canaan_loader import act_with_rng, hle_action_to_json, make_agent
from .hanabi import HanabiParams
from .seeds import bot_seed


def play_raw(kinds: list[str], seed: int, params: HanabiParams | None = None) -> tuple[int, list]:
    params = params or HanabiParams()
    env = rl_env.HanabiEnv(config=params.hle_config(seed))
    obs = env.reset()
    agents = [make_agent(k) for k in kinds]
    rngs = [random.Random(bot_seed(seed, p)) for p in range(params.players)]
    log = []
    done = env.state.is_terminal()
    while not done:
        cur = obs["current_player"]
        action = act_with_rng(agents[cur], obs["player_observations"][cur], rngs[cur])
        log.append([cur, hle_action_to_json(action, cur, params.players)])
        obs, _, done, _ = env.step(action)
    return env.state.score(), log
