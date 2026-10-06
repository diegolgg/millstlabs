"""Port of HLE's SimpleAgent to the observation JSON. A valid artifact bot.py."""


class Bot:
    name = "simple"

    def reset(self, game, my_id, seed):
        self.me = my_id
        self.n = game["num_players"]
        self.max_info = game.get("max_information_tokens", 8)

    def act(self, obs):
        for i, card in enumerate(obs["my_hand"]):
            if card["hints"]["color"] is not None or card["hints"]["rank"] is not None:
                return {"type": "PLAY", "card_index": i}
        fw = obs["fireworks"]
        if obs["info_tokens"] > 0:
            for off in range(1, self.n):
                p = (self.me + off) % self.n
                for card in obs["hands"][str(p)]:
                    if card["rank"] - 1 == fw[card["color"]] and card["hints"]["color"] is None:
                        return {"type": "REVEAL_COLOR", "target": p, "color": card["color"]}
        if obs["info_tokens"] < self.max_info:
            return {"type": "DISCARD", "card_index": 0}
        return {"type": "PLAY", "card_index": 0}
