"""Uniform-random legal move. A valid artifact bot.py (scores about 0: it loses its three lives)."""

import random


class Bot:
    name = "random"

    def reset(self, game, my_id, seed):
        self.rng = random.Random(seed)

    def act(self, obs):
        return self.rng.choice(obs["legal_moves"])
