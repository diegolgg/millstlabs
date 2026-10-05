"""Observation-grounded notes: automatic context, optional publication, seven moves.

Prose is an author's interpretation, never certified as true. Evidence is attached
by the service from permitted observations; no access to simulator hidden state.
"""
import copy
from dataclasses import asdict, dataclass

import numpy as np

from .knowledge import Fact, KnowledgeCorpus


def evidence_text(facts):
    return "; ".join(
        f"food ({f.x},{f.y}) stock {f.value:g} at tick {f.observed_tick}" if f.kind == "food"
        else f"{'wall' if f.value else 'passable'} ({f.x},{f.y})"
        for f in facts)


@dataclass
class Note:
    id: int
    author: str
    tick: int
    text: str
    facts: list[Fact]
    style: str


class NoteCorpus(KnowledgeCorpus):
    def __init__(self, mode, training):
        super().__init__(mode, training.corpus_slots, training.corpus_food_ttl)
        self.training = training
        self.notes = []
        self.last_write = {}
        self.transferred = set()
        self.pending = set()
        self.pending_rewards = {}
        self.living = set()
        self.serial = 0
        self.counts.update(notes_written=0, note_opportunities=0, publication_reward=0.,
                           new_peer_facts=0, note_contexts=0, prose_generated_tokens=0)

    def payload(self, agent, obs):
        tick = int(obs["self"][6])
        if tick - self.last_write.get(agent, -self.training.note_interval) < self.training.note_interval:
            return []
        food = self.candidates(agent, "food", tick)
        walls = [f for f in self.candidates(agent, "terrain", tick) if f.value == 1]
        # Fresh food and remote obstacles offer decisions that peers cannot infer
        # from their local view. Do not pay for enumerating every floor tile.
        return food[:1] or walls[:self.slots]

    def observe(self, observations, initial=False):
        self.living = set(observations)
        super().observe(observations, initial)

    def _fresh(self, fact, tick):
        return fact.kind == "terrain" or 0 <= tick-fact.observed_tick <= self.food_ttl

    def context(self, agent, obs):
        tick, pos = int(obs["self"][6]), obs["self"][2:4]
        own = self.evidence.get(agent, {})
        visible = {f.key for f in self.observed_facts(agent, obs)}
        available = []
        for note in self.notes:
            if self.mode == "private" and note.author != agent:
                continue
            # A prose sentence may mention any attached fact. Expire the whole
            # note rather than leaving stale claims in its text.
            if not all(self._fresh(f, tick) for f in note.facts):
                continue
            facts = [f for f in note.facts if f.key not in visible and
                     (f.kind != "food" or f.key not in own or own[f.key].observed_tick < f.observed_tick)]
            if not facts:
                continue
            goal = min(obs["stations"][:, :2], key=lambda p: np.abs(p-pos).sum())
            score = min(abs(f.x-pos[0])+abs(f.y-pos[1])+abs(f.x-goal[0])+abs(f.y-goal[1]) for f in facts)
            priority = 0 if obs["self"][0] < 80 and any(f.kind == "food" for f in facts) else 1
            available.append((priority, score, -note.tick, note.id, note, facts))
        available.sort(key=lambda row: row[:4])
        # One short note per decision. Stable spatial ranking, no LLM search call.
        return [(row[4], row[5][:self.slots]) for row in available[:1]]

    def augment(self, observations):
        augmented = copy.deepcopy(observations)
        for agent, obs in augmented.items():
            tick = int(obs["self"][6])
            selected = self.context(agent, obs) if agent in self.living else []
            rows = np.full((self.slots, 7), -1, np.float32)
            facts = [f for _, fs in selected for f in fs]
            for row, fact in zip(rows, facts):
                row[:] = [fact.kind == "food", fact.x, fact.y, fact.value,
                          tick-fact.observed_tick, fact.source != agent, 1]
                if fact.source != agent:
                    self.imports.add((agent, fact.key))
            self.counts["returned_records"] += len(facts)
            self.counts["peer_records"] += sum(f.source != agent for f in facts)
            self.counts["note_contexts"] += bool(selected)
            obs["corpus"] = rows  # Critic and demonstration evidence; actor reads the text.
            obs["corpus_status"] = np.zeros(4, np.float32)
            obs["notes"] = [{"id": n.id, "author": n.author, "age": tick-n.tick,
                             "text": n.text, "evidence": evidence_text(fs)} for n, fs in selected]
            obs["note_food_ttl"] = self.food_ttl
            obs["note_candidate"] = [asdict(f) for f in self.payload(agent, obs)] if agent in self.living else []
            obs["note_opportunity"] = bool(obs["note_candidate"])
            obs["action_mask"] = obs["action_mask"][:7].copy()
            if agent in self.living:
                self._credit_delivery(agent, selected)
        return augmented

    def _credit_delivery(self, receiver, selected):
        for note, facts in selected:
            if note.id not in self.pending or receiver == note.author:
                continue
            credited = 0
            for fact in facts:
                own = self.evidence.get(receiver, {}).get(fact.key)
                if own is not None and (fact.kind == "terrain" or own.observed_tick >= fact.observed_tick):
                    continue
                version = fact.observed_tick//self.food_ttl if fact.kind == "food" else 0
                key = (receiver, fact.key, version)
                if key not in self.transferred:
                    self.transferred.add(key)
                    credited += 1
            if credited:
                self.pending_rewards[note.author] += self.training.note_credit
                self.counts["publication_reward"] += self.training.note_credit
                self.counts["new_peer_facts"] += credited
                self.events.append({"type": "note_credit", "note": note.id, "author": note.author,
                                    "receiver": receiver, "facts": credited,
                                    "reward": self.training.note_credit, "tick": note.tick+1})

    def publish(self, decisions, observations, bank, policy_ids=None, texts=None):
        """Publish after all actors choose. Reward each novel peer fact once/map.

        Credit is for new information delivered in the receiver's next context,
        not note volume or claimed success. It is deliberately a shaping proxy.
        """
        rewards = dict.fromkeys(decisions, 0.)
        self.pending = set()
        self.pending_rewards = rewards
        for agent, write in decisions.items():
            if write < 0:
                continue
            self.counts["note_opportunities"] += 1
            if not write:
                continue
            obs = observations[agent]
            tick = int(obs["self"][6])
            facts = [Fact(**f) for f in obs["note_candidate"]]
            if not facts:
                raise ValueError("Publication without observed evidence")
            text = evidence_text(facts)
            if texts is not None and agent in texts:
                text = texts[agent]
            elif self.training.note_style == "prose":
                text, generated = bank.describe_note((policy_ids or {}).get(agent, agent), text)
                self.counts["prose_generated_tokens"] += generated
            accepted = self.deposit(agent, facts, tick)
            if accepted != len(facts):
                raise ValueError("Publication evidence changed before commit")
            note = Note(self.serial, agent, tick, text[:480], facts, self.training.note_style)
            self.serial += 1
            self.notes.append(note)
            self.notes = self.notes[-256:]
            self.last_write[agent] = tick
            self.counts["notes_written"] += 1
            rewards[agent] -= self.training.note_write_penalty
            self.pending.add(note.id)
            self.events.append({"type": "note", **asdict(note)})
        self.counts["publication_reward"] += sum(rewards.values())
        return rewards

    def execute(self, actions, observations, tick):
        if any(not 0 <= int(a) < 7 for a in actions.values()):
            raise ValueError("Notes use seven physical actions; tool action indices are invalid")
        return actions


def publish_notes(corpus, writes, observations, bank, env, training, policy_ids=None, texts=None):
    if not isinstance(corpus, NoteCorpus):
        return {}
    rewards = corpus.publish(writes, observations, bank, policy_ids, texts)
    for agent, write in writes.items():
        if write == 1:
            env.prey[agent].energy -= training.corpus_tool_cost
    return rewards
