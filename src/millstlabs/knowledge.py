"""Explicit, evidence-checked corpus tools. The audit union is never a retrieval source."""
import copy
from dataclasses import dataclass

import numpy as np

NONE, PUT_TERRAIN, PUT_FOOD, GET_TERRAIN, GET_FOOD = range(5)
TOOL_NAMES = ["none", "put_terrain", "put_food", "get_terrain", "get_food"]


@dataclass(frozen=True)
class Fact:
    kind: str
    x: int
    y: int
    value: float
    observed_tick: int
    source: str

    @property
    def key(self):
        return self.kind, self.x, self.y


class KnowledgeCorpus:
    """One map namespace. Dead authors' deposits persist until that map ends.

    Policy tools can deposit only registered first-hand evidence. Private and
    shared modes have identical tool contracts and per-call record limits.
    """

    def __init__(self, mode, slots=4, food_ttl=16):
        if mode not in {"private", "shared"}:
            raise ValueError("Corpus mode must be private or shared")
        self.mode, self.slots, self.food_ttl = mode, slots, food_ttl
        self.evidence, self.private, self.shared = {}, {}, {}
        self.submitted = {}
        self.audit_union = set()
        self.initial_union = set()
        self.imports, self.confirmed = set(), set()
        self.responses, self.status = {}, {}
        self.events = []
        self.counts = dict(deposit_calls=0, retrieve_calls=0, accepted_records=0, rejected_records=0,
                           returned_records=0, peer_records=0, empty_retrievals=0)

    @staticmethod
    def observed_facts(agent, obs):
        s = obs["self"]
        x, y, tick, size = int(s[2]), int(s[3]), int(s[6]), int(s[9])
        facts = []
        for iy, row in enumerate(obs["local"]):
            for ix, code in enumerate(row):
                px, py = x + ix - 2, y + iy - 2
                if code >= 0 and 0 <= px < size and 0 <= py < size:
                    facts.append(Fact("terrain", px, py, float(code == 1), tick, agent))
        for fx, fy, stock, seen in obs["stations"]:
            if seen == tick and stock >= 0:
                facts.append(Fact("food", int(fx), int(fy), float(stock), tick, agent))
        return facts

    def observe(self, observations, initial=False):
        """Evidence comes only from living agents' permitted observations."""
        for agent, obs in observations.items():
            own = self.evidence.setdefault(agent, {})
            for fact in self.observed_facts(agent, obs):
                if (agent, fact.key) in self.imports and fact.kind == "terrain":
                    self.confirmed.add((agent, fact.key))
                own[fact.key] = fact
                self.audit_union.add(fact.key)
        if initial:
            self.initial_union = set(self.audit_union)

    def _store(self, agent):
        return self.shared if self.mode == "shared" else self.private.setdefault(agent, {})

    def deposit(self, agent, facts, tick):
        """Callable tool API; returns accepted count. Fabrications are rejected."""
        if len(facts) > self.slots:
            raise ValueError("Deposit exceeds per-call record limit")
        self.counts["deposit_calls"] += 1
        accepted = 0
        for fact in facts:
            verified = self.evidence.get(agent, {}).get(fact.key)
            valid = fact.source == agent and verified == fact and fact.observed_tick <= tick
            if not valid:
                self.counts["rejected_records"] += 1
                self.events.append({"type": "deposit_rejected", "agent": agent, "fact": vars(fact), "tick": tick})
                continue
            store = self._store(agent)
            previous = store.get(fact.key)
            if previous is None or (fact.kind == "food" and fact.observed_tick >= previous.observed_tick):
                store[fact.key] = fact
            self.submitted.setdefault(agent, {})[fact.key] = fact
            accepted += 1
            self.counts["accepted_records"] += 1
            self.events.append({"type": "deposit", "agent": agent, "fact": vars(fact), "tick": tick})
        return accepted

    def retrieve(self, agent, kind, position, tick, goal=None):
        """Callable tool API, constrained by map scope, provenance, TTL and budget."""
        if kind not in {"terrain", "food"}:
            raise ValueError("Unknown fact kind")
        self.counts["retrieve_calls"] += 1
        rows = [f for f in self._store(agent).values() if f.kind == kind and
                (kind == "terrain" or 0 <= tick - f.observed_tick <= self.food_ttl)]
        rows = [f for f in rows if self.evidence.get(agent, {}).get(f.key) is None or
                self.evidence[agent][f.key].observed_tick < tick-1]
        # Same deterministic spatial query for both arms; walls before known floor.
        goal = position if goal is None else goal
        rows.sort(key=lambda f: (0 if kind == "food" else -f.value,
                                 abs(f.x-position[0])+abs(f.y-position[1])+abs(f.x-goal[0])+abs(f.y-goal[1]),
                                 abs(f.x-position[0])+abs(f.y-position[1]), -f.observed_tick, f.x, f.y))
        rows = rows[:self.slots]
        self.counts["returned_records"] += len(rows)
        self.counts["empty_retrievals"] += not rows
        peer = 0
        for fact in rows:
            if fact.source != agent:
                peer += 1
                self.imports.add((agent, fact.key))
        self.counts["peer_records"] += peer
        self.events.append({"type": "retrieve", "agent": agent, "kind": kind, "tick": tick,
                            "facts": [vars(f) for f in rows]})
        return rows

    def candidates(self, agent, kind, tick):
        # Availability/masks depend on own submissions, never on unread peer data.
        store = self.submitted.get(agent, {})
        facts = [f for f in self.evidence.get(agent, {}).values() if f.kind == kind and
                 (kind == "terrain" or tick - f.observed_tick <= self.food_ttl)]
        # Avoid rewarding/wasting budget on identical repeated writes.
        facts = [f for f in facts if f.key not in store or (kind == "food" and store[f.key].observed_tick < f.observed_tick)]
        facts.sort(key=lambda f: (-f.observed_tick, -f.value if kind == "terrain" else 0, f.x, f.y))
        return facts[:self.slots]

    def execute(self, actions, observations, tick):
        """All reads precede all writes. Results are consumed on the next observation."""
        self.responses = {i: [] for i in actions}
        self.status = {i: [int(a)//7, 0, 0, 0] for i, a in actions.items()}
        # Compute write payloads before any write, removing identity-order advantage.
        payloads = {i: self.candidates(i, "terrain" if int(a)//7 == PUT_TERRAIN else "food", tick)
                    for i, a in actions.items() if int(a)//7 in {PUT_TERRAIN, PUT_FOOD}}
        for i, a in actions.items():
            tool = int(a)//7
            if tool in {GET_TERRAIN, GET_FOOD}:
                position = observations[i]["self"][2:4]
                goal = min(observations[i]["stations"][:, :2], key=lambda s: abs(s[0]-position[0])+abs(s[1]-position[1]))
                rows = self.retrieve(i, "terrain" if tool == GET_TERRAIN else "food", position, tick+1, goal)
                self.responses[i] = rows
                self.status[i][2:] = [len(rows), sum(f.source != i for f in rows)]
        for i, rows in payloads.items():
            self.status[i][1] = self.deposit(i, rows, tick)
        return {i: int(a) % 7 for i, a in actions.items()}

    def augment(self, observations):
        augmented = copy.deepcopy(observations)
        for i, obs in augmented.items():
            tick = int(obs["self"][6])
            rows = np.full((self.slots, 7), -1, np.float32)
            for row, fact in zip(rows, self.responses.get(i, [])):
                row[:] = [fact.kind == "food", fact.x, fact.y, fact.value,
                          tick-fact.observed_tick, fact.source != i, 1]
            obs["corpus"] = rows
            obs["corpus_status"] = np.asarray(self.status.get(i, [0, 0, 0, 0]), np.float32)
            legal = [True, bool(self.candidates(i, "terrain", tick)), bool(self.candidates(i, "food", tick)), True, True]
            obs["action_mask"] = np.concatenate([obs["action_mask"][:7] * ok for ok in legal]).astype(np.int8)
        return augmented

    def metrics(self):
        deposited_union = set(self.shared)
        for store in self.private.values():
            deposited_union.update(store)
        return {**self.counts, "unique_observed_facts": len(self.audit_union),
                "initial_observed_facts": len(self.initial_union),
                "new_observed_facts": len(self.audit_union-self.initial_union),
                "unique_deposited_facts": len(deposited_union), "distinct_peer_imports": len(self.imports),
                "peer_terrain_reconfirmations": len(self.confirmed)}


def make_corpus(training, observations, mode=None):
    mode = training.corpus_mode if mode is None else mode
    if mode == "off":
        return None, observations
    if training.corpus_interface == "notes":
        from .notes import NoteCorpus
        corpus = NoteCorpus(mode, training)
    else:
        corpus = KnowledgeCorpus(mode, training.corpus_slots, training.corpus_food_ttl)
    corpus.observe(observations, initial=True)
    return corpus, corpus.augment(observations)


class CountNovelty:
    """Trainer-side lineage counts, never policy input. Birth cannot reset counts."""
    def __init__(self):
        self.counts = {}

    def reward(self, lineage, observation):
        s = observation["self"]
        # Coarse, time/identity-free state; counts persist over extinction recovery.
        key = (lineage, int(s[2])//3, int(s[3])//3, bool(s[0] < 40), bool((observation["threats"][:, 0] >= 0).any()))
        n = self.counts.get(key, 0)
        self.counts[key] = n+1
        return 1 / np.sqrt(1+n)


def novelty_beta(training, decisions):
    fraction = min(1., decisions/training.novelty_decay_decisions)
    return training.novelty_beta_start + fraction*(training.novelty_beta_end-training.novelty_beta_start)


def execute_tools(corpus, actions, observations, env, training):
    if corpus is None:
        return actions
    physical = corpus.execute(actions, observations, env.tick)
    for i, action in actions.items():
        if int(action)//7:
            env.prey[i].energy -= training.corpus_tool_cost
    return physical


def next_observations(corpus, observations, living):
    if corpus is None:
        return observations
    corpus.observe({i: observations[i] for i in living})
    return corpus.augment(observations)


def teacher_tool(corpus, agent, observation, offset=0):
    """An explicit observed-only tool-use demonstration, common to all arms."""
    tick = int(observation["self"][6])
    phase = (tick+offset) % 4
    if phase == 0:
        if corpus.candidates(agent, "food", tick):
            return PUT_FOOD
        if corpus.candidates(agent, "terrain", tick):
            return PUT_TERRAIN
    if phase == 1:
        return GET_FOOD if tick % 8 < 4 else GET_TERRAIN
    return NONE
