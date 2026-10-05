"""Bounded, observation-only interfaces shared by heuristics and neural policies."""
from collections import deque

import numpy as np

from .env import DELTAS, FEED, REST, WATCH, distance


def observation_text(obs):
    s = obs["self"]
    terrain = "".join("? .#Fpxa".replace(" ", "")[int(v) + 1] for v in obs["local"].flat)
    food = ";".join(",".join(str(int(v)) for v in row) for row in obs["stations"])
    threats = ";".join(f"{x},{y}" for x, y in obs["threats"] if x >= 0) or "none"
    peers = obs["companions"]
    peers = peers[peers[:, 0] >= 0]
    # Local grid carries all positions; energy of nearest/weakest peer and watch
    # count retain the probe-relevant information within a 256-token budget.
    weakest = min(peers, key=lambda p: p[2]) if len(peers) else [-1, -1, -1, 0]
    radio = ""
    if "messages" in obs:
        packets = ";".join(f"{x},{y}:{int(symbol)}" for x, y, symbol in obs["messages"] if symbol > 0) or "none"
        radio = f"radio_xy_symbol={packets} "
    return (radio + f"energy={s[0]:.1f} age={int(s[1])} xy={int(s[2])},{int(s[3])} "
            f"cooldown={int(s[4])} fertile={int(s[5])} tick={int(s[6])} "
            f"alert={int(s[7])} outcome={int(s[8])} size={int(s[9])} "
            f"grid={terrain} food_xy_stock_seen={food} predator={threats} "
            f"peers={len(peers)} watchers={int(peers[:, 3].sum()) if len(peers) else 0} "
            f"weakest={int(weakest[0])},{int(weakest[1])},{int(weakest[2])} "
            f"legal={''.join(str(int(v)) for v in obs['action_mask'][:7])}")


def vector_observation(obs):
    # Cheap offline backend only; no privileged state is introduced.
    s = obs["self"].copy()
    s /= np.asarray([100, 2048, 20, 20, 128, 1, 2048, 1, 2, 20])
    stations = obs["stations"] / np.asarray([20, 20, 60, 2048])
    companions = obs["companions"] / np.asarray([20, 20, 100, 1])
    return np.concatenate([s, obs["local"].flatten() / 5, stations.flatten(),
                           obs["threats"].flatten() / 20, companions.flatten(), obs["action_mask"]]).astype(np.float32)


class Heuristic:
    """Tracks only observed walls. BFS treats unexplored cells as traversable."""
    def __init__(self, seed=0, vigilance=True):
        self.rng = np.random.default_rng(seed)
        self.walls = set()
        self.vigilance = vigilance

    def act(self, obs):
        s = obs["self"]
        pos = tuple(int(v) for v in s[2:4])
        size = int(s[9])
        mask = obs["action_mask"]
        for iy in range(5):
            for ix in range(5):
                if obs["local"][iy, ix] == 1:
                    self.walls.add((pos[0] + ix - 2, pos[1] + iy - 2))
        threats = [tuple(row) for row in obs["threats"] if row[0] >= 0]
        if threats:
            near = min(distance(pos, t) for t in threats)
            if near <= 2 and self.vigilance and not s[7]:
                return WATCH
            if near <= 3 and s[0] > 12:
                options = [(min(distance((pos[0] + dx, pos[1] + dy), t) for t in threats), a)
                           for a, (dx, dy) in enumerate(DELTAS) if mask[a]]
                if options:
                    best = max(d for d, a in options)
                    return int(self.rng.choice([a for d, a in options if d == best]))
        if mask[FEED] and s[0] < 99:
            return FEED
        if s[0] >= 99:
            return REST
        # Search for a feeding cell using remembered obstacles. Never inspect the env.
        stations = [tuple(int(v) for v in row[:2]) for row in obs["stations"]]
        queue = deque([(pos, None)])
        seen = {pos}
        while queue:
            cell, first = queue.popleft()
            if first is not None and any(distance(cell, f) <= 1 for f in stations):
                return first
            for a in self.rng.permutation(4):
                dx, dy = DELTAS[a]
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt not in seen and nxt not in self.walls and 0 <= nxt[0] < size and 0 <= nxt[1] < size:
                    if first is None and not mask[a]:
                        continue
                    seen.add(nxt)
                    queue.append((nxt, int(a) if first is None else first))
        return REST
