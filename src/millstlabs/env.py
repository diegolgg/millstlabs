"""Simultaneous ecological dynamics; no policy weights or learner state live here."""
import copy
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .config import EnvironmentConfig
from .reproduction import Pregnancy, SurvivalNoteSpace, resolve_immediate_reproduction, resolve_reproduction

NORTH, SOUTH, EAST, WEST, FEED, WATCH, REST = range(7)
DELTAS = [(0, -1), (0, 1), (1, 0), (-1, 0)]
ACTION_NAMES = ["north", "south", "east", "west", "feed", "watch", "rest"]


def diffusion_probabilities(distances, temperature):
    """Conditional-on-moving Boltzmann kernel over legal neighboring cells.

    Distances are shortest-path steps to the current target. Zero means greedy,
    ties uniform; infinity approaches a uniform random walk. Move rate stays 0.9.
    """
    costs = np.asarray(distances, dtype=float)
    costs -= costs.min()
    weights = (costs == 0).astype(float) if temperature == 0 else np.exp(-costs / temperature)
    return weights / weights.sum()


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class Prey:
    id: str
    pos: tuple
    energy: float
    age: int
    cooldown: int = 0
    parent: str | None = None
    lineage: str = ""
    generation: int = 0
    alert_until: int = -1
    supplies: dict = field(default_factory=dict)
    alarms: list = field(default_factory=list)
    outcome: int = 0
    born_tick: int = 0
    last_action: int = REST
    messages: list = field(default_factory=list)
    pregnancy: Pregnancy | None = None
    survival_note: str = ""
    history: deque = field(default_factory=lambda: deque(maxlen=128))


class PopulationEnv(ParallelEnv):
    """PettingZoo Parallel API with unique birth identities and dictionary observations.

    Observation is the only policy input. Privileged metrics are exposed in info.
    Dead agents receive one terminal observation, never another action request.
    """

    metadata = {"name": "millst_population_v0", "render_modes": ["ansi"], "is_parallelizable": True}

    def __init__(self, config=None, reproduction=True, social_preference=0.0, render_mode=None, deliver_messages=True):
        self.cfg = config or EnvironmentConfig()
        self.cfg.validate()
        self.reproduction = reproduction
        self.social_preference = social_preference
        self.render_mode = render_mode
        self.deliver_messages = deliver_messages
        self.possible_agents = [f"prey_{i}" for i in range(self.cfg.max_individuals)]
        self.agents = []
        self._action_space = spaces.Discrete(7 * self.cfg.message_symbols)
        n = self.cfg.size
        self._observation_space = spaces.Dict({
            "self": spaces.Box(-1, np.inf, (12,), np.float32),
            "survival_note": SurvivalNoteSpace(),
            "local": spaces.Box(-1, 5, (5, 5), np.int8),
            "stations": spaces.Box(-1, np.inf, (self.cfg.stations, 4), np.float32),
            "threats": spaces.Box(-1, n, (self.cfg.predators, 2), np.int16),
            "companions": spaces.Box(-1, np.inf, (24, 4), np.float32),
            "action_mask": spaces.MultiBinary(7 * self.cfg.message_symbols),
        })
        if self.cfg.reproduction_mode == "immediate":
            self._observation_space.spaces["self"] = spaces.Box(-1, np.inf, (10,), np.float32)
            del self._observation_space.spaces["survival_note"]
        if self.cfg.message_symbols > 1:
            self._observation_space.spaces["messages"] = spaces.Box(-1, max(n, self.cfg.message_symbols),
                                                                   (self.cfg.message_capacity, 3), np.int16)

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def legal(self, p):
        return 0 <= p[0] < self.cfg.size and 0 <= p[1] < self.cfg.size and p not in self.obstacles

    def neighbors(self, p):
        return [(p[0] + dx, p[1] + dy) for dx, dy in DELTAS if self.legal((p[0] + dx, p[1] + dy))]

    def visible(self, a, b):
        # Bresenham line of sight: walls are visible, cells behind walls are not.
        x, y = a
        dx, dy = abs(b[0] - x), abs(b[1] - y)
        sx, sy = (1 if x < b[0] else -1), (1 if y < b[1] else -1)
        err = dx - dy
        while (x, y) != b:
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            if (x, y) in self.obstacles and (x, y) != b:
                return False
        return True

    def _connected(self, free):
        seen = {next(iter(free))}
        todo = list(seen)
        while todo:
            x, y = todo.pop()
            for dx, dy in DELTAS:
                p = (x + dx, y + dy)
                if p in free and p not in seen:
                    seen.add(p)
                    todo.append(p)
        return len(seen) == len(free)

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.next_id = 0
        self.prey = {}
        self.events = []
        self.attack_ready = 0
        self.total_births = 0
        self.total_conceptions = self.pregnancy_losses = self.birth_wait_ticks = 0
        self.total_matured = 0
        self.total_consumption = 0.0
        self.messages_sent = self.messages_delivered = 0
        self.predator_entropy_sum = self.predator_kernel_ticks = 0
        self.deaths = {"starvation": 0, "predation": 0}
        n = self.cfg.size
        cells = [(x, y) for x in range(n) for y in range(n)]
        target = round(len(cells) * self.cfg.obstacle_fraction)
        self.obstacles = set()
        free = set(cells)
        for idx in self.rng.permutation(len(cells)):
            if len(self.obstacles) >= target:
                break
            candidate = cells[idx]
            free.remove(candidate)
            if self._connected(free):
                self.obstacles.add(candidate)
            else:
                free.add(candidate)
        free = sorted(free)
        self.station_positions = [free[int(self.rng.integers(len(free)))]]
        for _ in range(self.cfg.stations - 1):
            # Maximin placement, random tie breaking, connected by construction.
            scores = [min(distance(p, s) for s in self.station_positions) for p in free]
            candidates = np.flatnonzero(np.asarray(scores) == max(scores))
            self.station_positions.append(free[int(self.rng.choice(candidates))])
        self.stock = np.full(self.cfg.stations, self.cfg.initial_stock, dtype=float)
        self.rates = self.rng.choice(self.cfg.replenishment_rates, size=self.cfg.stations)
        # Choose a predator location that permits all founders near food.
        found = False
        for idx in self.rng.permutation(len(free)):
            predator = free[idx]
            starts = [p for p in free if distance(p, predator) >= 8
                      and min(distance(p, s) for s in self.station_positions) <= 5]
            if len(starts) >= self.cfg.founders:
                found = True
                break
        if not found:
            raise ValueError("map cannot place founders with the required predator separation")
        self.predator = predator
        self.patrol = self.station_positions[int(self.rng.integers(self.cfg.stations))]
        for idx in self.rng.choice(len(starts), self.cfg.founders, replace=False):
            self._spawn(starts[idx])
        self.agents = list(self.prey)
        self._refresh_memories()
        return {i: self.observe(i) for i in self.agents}, {i: {} for i in self.agents}

    def _spawn(self, pos, parent=None):
        if self.next_id >= self.cfg.max_individuals:
            raise RuntimeError("Identity guard exhausted; increase max_individuals and resume")
        i = f"prey_{self.next_id}"
        self.next_id += 1
        p = Prey(i, pos, self.cfg.newborn_energy if parent else self.cfg.founder_energy,
                 0 if parent else self.cfg.maturity,
                 parent=parent.id if parent else None, lineage=parent.lineage if parent else i,
                 generation=parent.generation + 1 if parent else 0, born_tick=self.tick)
        self.prey[i] = p
        return p

    def _refresh_memories(self):
        for p in self.prey.values():
            for j, s in enumerate(self.station_positions):
                if max(abs(p.pos[k] - s[k]) for k in (0, 1)) <= 2 and self.visible(p.pos, s):
                    p.supplies[j] = (float(self.stock[j]), self.tick)

    def observe(self, i):
        p = self.prey[i]
        local = np.full((5, 5), -1, np.int8)
        occupancy = {q.pos: q for q in self.prey.values() if q.id != i}
        for iy, dy in enumerate(range(-2, 3)):
            for ix, dx in enumerate(range(-2, 3)):
                cell = (p.pos[0] + dx, p.pos[1] + dy)
                if not (0 <= cell[0] < self.cfg.size and 0 <= cell[1] < self.cfg.size):
                    local[iy, ix] = 1
                elif self.visible(p.pos, cell):
                    code = 1 if cell in self.obstacles else 0
                    if cell in self.station_positions:
                        code = 2
                    if cell in occupancy:
                        code = 5 if occupancy[cell].alert_until >= self.tick else 3
                    if cell == self.predator:
                        code = 4
                    local[iy, ix] = code
        threats = set(tuple(a) for a in p.alarms)
        ordinary = max(abs(p.pos[k] - self.predator[k]) for k in (0, 1)) <= 2
        if ordinary and self.visible(p.pos, self.predator):
            # Direct current sight supersedes an alarm emitted before predator movement.
            threats = {self.predator}
        threat_array = np.full((self.cfg.predators, 2), -1, np.int16)
        for j, cell in enumerate(sorted(threats)[:self.cfg.predators]):
            threat_array[j] = cell
        station_obs = [[*s, *p.supplies.get(j, (-1, -1))] for j, s in enumerate(self.station_positions)]
        companions = np.full((24, 4), -1, np.float32)
        nearby = [q for q in self.prey.values() if q.id != i
                  and max(abs(q.pos[k] - p.pos[k]) for k in (0, 1)) <= 2
                  and self.visible(p.pos, q.pos)]
        for row, q in zip(companions, sorted(nearby, key=lambda q: (distance(p.pos, q.pos), q.pos))):
            row[:] = [*q.pos, q.energy, q.last_action == WATCH]
        mask = np.ones(7, np.int8)
        for k, (dx, dy) in enumerate(DELTAS):
            # Only visible terrain constrains masks; occupancy can change simultaneously.
            mask[k] = self.legal((p.pos[0] + dx, p.pos[1] + dy))
        mask[FEED] = any(distance(p.pos, s) <= 1 for s in self.station_positions)
        result = {"self": np.asarray([p.energy, p.age, *p.pos, p.cooldown, self.reproduction,
                                    self.tick, p.alert_until >= self.tick, p.outcome, self.cfg.size,
                                    p.pregnancy is not None,
                                    p.pregnancy.remaining(self.tick) if p.pregnancy else 0], np.float32),
                "survival_note": p.survival_note,
                "local": local, "stations": np.asarray(station_obs, np.float32),
                "threats": threat_array, "companions": companions, "action_mask": np.tile(mask, self.cfg.message_symbols)}
        if self.cfg.reproduction_mode == "immediate":
            result["self"] = result["self"][:10]
            del result["survival_note"]
        if self.cfg.message_symbols > 1:
            messages = np.full((self.cfg.message_capacity, 3), -1, np.int16)
            messages[:, 2] = 0
            for row, packet in zip(messages, p.messages):
                row[:] = packet
            result["messages"] = messages
        return result

    def _moves(self, actions):
        occupied = {p.pos: i for i, p in self.prey.items()}
        requests = defaultdict(list)
        for i, a in actions.items():
            p = self.prey[i]
            p.outcome = 0
            if a < 4:
                dx, dy = DELTAS[a]
                dest = (p.pos[0] + dx, p.pos[1] + dy)
                if self.legal(dest):
                    requests[dest].append(i)
        winners = {str(self.rng.choice(ids)): dest for dest, ids in sorted(requests.items())}
        # Remove chains ending at a stationary occupant; allow swaps and closed cycles.
        changed = True
        while changed:
            blocked = [i for i, dest in winners.items() if dest in occupied and occupied[dest] not in winners]
            changed = bool(blocked)
            for i in blocked:
                del winners[i]
        for i, dest in winners.items():
            self.prey[i].pos = dest
            self.prey[i].outcome = 1

    def _feed(self, actions):
        groups = defaultdict(list)
        for i, a in actions.items():
            if a == FEED:
                eligible = [j for j, s in enumerate(self.station_positions) if distance(self.prey[i].pos, s) <= 1]
                if eligible:
                    # In overlapping catchments choose most stocked, random ties.
                    best = max(self.stock[j] for j in eligible)
                    j = int(self.rng.choice([j for j in eligible if self.stock[j] == best]))
                    groups[j].append(i)
        for j, ids in groups.items():
            needs = {i: min(self.cfg.intake, self.cfg.max_energy - self.prey[i].energy) for i in ids}
            remaining = float(self.stock[j])
            received = dict.fromkeys(ids, 0.0)
            active = list(ids)
            while active and remaining > 1e-10:
                share = remaining / len(active)
                for i in active:
                    amount = min(share, needs[i] - received[i])
                    received[i] += amount
                    remaining -= amount
                active = [i for i in active if needs[i] - received[i] > 1e-10]
            self.stock[j] = max(0, remaining)
            for i, amount in received.items():
                self.prey[i].energy += amount
                self.prey[i].outcome = 2 if amount else 0
                self.total_consumption += amount

    def _kill(self, i, cause):
        p = self.prey.pop(i)
        self.deaths[cause] += 1
        if p.pregnancy is not None:
            self.pregnancy_losses += 1
            self.events.append({"type": "pregnancy_loss", "id": i, "cause": cause,
                                "tick": self.tick, "conception_tick": p.pregnancy.conceived_tick,
                                "gestation_completion_tick": p.pregnancy.completion_tick})
        self.events.append({"type": "death", "id": i, "cause": cause, "age": p.age,
                            "lineage": p.lineage, "generation": p.generation, "tick": self.tick})

    def _predator(self):
        visible = [p for p in self.prey.values() if distance(p.pos, self.predator) <= self.cfg.detection_radius
                   and self.visible(self.predator, p.pos)]
        if visible:
            nearest = min(distance(p.pos, self.predator) for p in visible)
            targets = [p.pos for p in visible if distance(p.pos, self.predator) == nearest]
            target = targets[int(self.rng.integers(len(targets)))]
        else:
            if self.predator == self.patrol:
                self.patrol = self.station_positions[int(self.rng.integers(self.cfg.stations))]
            target = self.patrol
        legal = self.neighbors(self.predator)
        draw = self.rng.random()
        if legal and draw < 0.9:
            temperature = self.cfg.predator_temperature
            if temperature is not None or draw < 0.7:
                # Shortest path around obstacles, rather than getting stuck on walls.
                distances = {target: 0}
                todo = deque([target])
                while todo:
                    cell = todo.popleft()
                    for nxt in self.neighbors(cell):
                        if nxt not in distances:
                            distances[nxt] = distances[cell] + 1
                            todo.append(nxt)
                if temperature is not None:
                    probs = diffusion_probabilities([distances[c] for c in legal], temperature)
                    self.predator_entropy_sum += float(-(probs * np.log(probs.clip(1e-20))).sum())
                    self.predator_kernel_ticks += 1
                else:
                    best = min(distances[c] for c in legal)
                    legal = [c for c in legal if distances[c] == best]
            if temperature is None:
                self.predator = legal[int(self.rng.integers(len(legal)))]
            else:
                self.predator = legal[int(self.rng.choice(len(legal), p=probs))]
        adjacent = [i for i, p in self.prey.items() if distance(p.pos, self.predator) <= 1]
        if self.tick >= self.attack_ready and adjacent:
            i = str(self.rng.choice(adjacent))
            alert = self.prey[i].alert_until >= self.tick
            prob = self.cfg.alert_death_probability if alert else self.cfg.unalert_death_probability
            self.attack_ready = self.tick + self.cfg.attack_cooldown
            died = self.rng.random() < prob
            self.events.append({"type": "attack", "id": i, "alert": alert, "died": died, "tick": self.tick})
            if died:
                self._kill(i, "predation")

    def step(self, actions, note_writer=None, private_observations=None):
        if set(actions) != set(self.agents):
            raise ValueError("Supply exactly one action for every current living prey")
        if any(not self._action_space.contains(a) for a in actions.values()):
            raise ValueError("Action is outside this environment's discrete action space")
        if not self.agents:
            return {}, {}, {}, {}, {}
        acting = list(self.agents)
        terminal_observations = {i: self.observe(i) for i in acting}
        self.events = []
        symbols = {i: int(a) // 7 for i, a in actions.items()}
        actions = {i: int(a) % 7 for i, a in actions.items()}
        inboxes = {i: [] for i in acting}
        if self.cfg.message_symbols > 1:
            for i, symbol in symbols.items():
                if not symbol:
                    continue
                sender = self.prey[i]
                recipients = [j for j in acting if j != i and distance(sender.pos, self.prey[j].pos) <= self.cfg.message_radius]
                self.messages_sent += 1
                if self.deliver_messages:
                    for j in recipients:
                        inboxes[j].append([*sender.pos, symbol])
                self.events.append({"type": "message", "id": i, "symbol": symbol, "tick": self.tick})
        self._moves(actions)
        for p in self.prey.values():
            p.alarms = []
        for i, a in actions.items():
            if a == WATCH:
                p = self.prey[i]
                p.alert_until = self.tick + 1
                if distance(p.pos, self.predator) <= self.cfg.detection_radius and self.visible(p.pos, self.predator):
                    for q in self.prey.values():
                        if distance(q.pos, p.pos) <= self.cfg.detection_radius:
                            q.alert_until = self.tick + 1
                            q.alarms = [self.predator]
                    self.events.append({"type": "alarm", "id": i, "tick": self.tick})
        self._feed(actions)
        for i in acting:
            p = self.prey[i]
            p.last_action = actions[i]
            p.energy -= (self.cfg.metabolism + self.cfg.movement_cost * (actions[i] < 4)
                         + self.cfg.watching_cost * (actions[i] == WATCH)
                         + self.cfg.message_cost * (symbols[i] > 0))
            p.age += 1
            p.cooldown = max(0, p.cooldown - 1)
            if p.parent and p.age == self.cfg.maturity:
                self.total_matured += 1
                self.events.append({"type": "maturation", "id": i, "tick": self.tick})
            if p.energy <= 0:
                self._kill(i, "starvation")
        self._predator()
        if self.reproduction and self.cfg.reproduction_mode == "immediate":
            resolve_immediate_reproduction(self)
        elif self.reproduction:
            from .observations import observation_text
            for i in acting:
                if i in self.prey:
                    p = self.prey[i]
                    observed = (private_observations or terminal_observations)[i]
                    # The inherited note is supplied separately; never recursively log it.
                    observed = {**observed, "survival_note": ""}
                    food = observed["stations"][:, 2].tolist()
                    threats = observed["threats"].tolist()
                    p.history.append(f"tick={self.tick} action={ACTION_NAMES[actions[i]]} "
                                     f"outcome={p.outcome} energy_after={p.energy:.1f} "
                                     f"food={food} threats={threats} {observation_text(observed)}")
            resolve_reproduction(self, note_writer)
        for j in range(self.cfg.stations):
            if self.rng.random() < self.cfg.rate_change_probability:
                self.rates[j] = self.rng.choice(self.cfg.replenishment_rates)
        self.stock = np.minimum(self.cfg.stock_cap, self.stock + self.rates)
        self.tick += 1
        self.agents = list(self.prey)
        if self.cfg.message_symbols > 1:
            for i, p in self.prey.items():
                # Positions/range at send time, delivered after the simultaneous turn.
                origin = terminal_observations[i]["self"][2:4] if i in terminal_observations else p.pos
                p.messages = sorted(inboxes.get(i, []), key=lambda m: (distance(m[:2], origin), m))[:self.cfg.message_capacity]
                self.messages_delivered += len(p.messages)
        self._refresh_memories()
        keys = list(dict.fromkeys(acting + self.agents))
        observations = {i: self.observe(i) if i in self.prey else terminal_observations[i] for i in keys}
        terms = {i: i not in self.prey for i in keys}
        infos, rewards = {}, {}
        for i in keys:
            personal = float(i in self.prey)
            social = (len(self.prey) - personal) / self.cfg.social_denominator
            infos[i] = {"personal_reward": personal, "social_reward": social,
                        "events": self.events, "tick": self.tick}
            rewards[i] = personal + self.social_preference * social if i in acting else 0.0
        return observations, rewards, terms, dict.fromkeys(keys, False), infos

    def metrics(self):
        lineages = [p.lineage for p in self.prey.values()]
        counts = [lineages.count(i) for i in set(lineages)]
        extra = {}
        if self.cfg.message_symbols > 1:
            extra.update(messages_sent=self.messages_sent, messages_delivered=self.messages_delivered)
        if self.cfg.predator_temperature is not None:
            extra.update(predator_temperature=self.cfg.predator_temperature,
                         predator_move_entropy=self.predator_entropy_sum / max(1, self.predator_kernel_ticks))
        return {**extra, "tick": self.tick, "population": len(self.prey), "births": self.total_births,
                "conceptions": self.total_conceptions, "pregnancy_losses": self.pregnancy_losses,
                "pregnant": sum(p.pregnancy is not None for p in self.prey.values()),
                "birth_wait_ticks": self.birth_wait_ticks,
                "matured": self.total_matured, "consumption": self.total_consumption, **self.deaths,
                "at_cap": len(self.prey) == self.cfg.population_cap,
                "lineages": len(counts), "lineage_entropy": -sum(c / len(lineages) * math.log(c / len(lineages))
                                                                             for c in counts),
                "generation_max": max((p.generation for p in self.prey.values()), default=0),
                "ages": [p.age for p in self.prey.values()]}

    def fork(self):
        """Exact simulator + RNG copy for coupled counterfactuals (until paths diverge)."""
        return copy.deepcopy(self)

    def render(self):
        grid = [["." for _ in range(self.cfg.size)] for _ in range(self.cfg.size)]
        for x, y in self.obstacles:
            grid[y][x] = "#"
        for x, y in self.station_positions:
            grid[y][x] = "F"
        for p in self.prey.values():
            x, y = p.pos
            grid[y][x] = "a" if p.alert_until >= self.tick else "p"
        x, y = self.predator
        grid[y][x] = "X"
        return "\n".join("".join(row) for row in grid)

    def close(self):
        pass


def parallel_env(**kwargs):
    return PopulationEnv(**kwargs)
