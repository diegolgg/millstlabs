"""Persistent, tentative tactics grounded in an author's own action outcomes."""
import copy
import json
from dataclasses import dataclass, field

from .env import ACTION_NAMES, FEED
from .notes import NoteCorpus


@dataclass
class Tactic:
    id: int
    author: str
    owner: str
    tick: int
    world: int
    text: str
    situation: tuple
    experience: list[dict]
    style: str
    facts: list = field(default_factory=list)


def situation(obs):
    return (bool(obs["self"][0] < 50), bool(obs["action_mask"][FEED]),
            bool((obs["threats"][:, 0] >= 0).any()))


def signature(row):
    return (int(row["energy_before"]//20), *row["situation"], row["action"],
            round(row["energy_change"], 1) if row["energy_change"] is not None else None,
            row["survived"], row["gave_birth"], tuple(row.get("food_distance", [])))


def food_distance(obs):
    """Distance to known station coordinates, not hidden stock or a privileged path."""
    x, y = obs["self"][2:4]
    distances = [abs(float(row[0])-x)+abs(float(row[1])-y) for row in obs["stations"] if row[0] >= 0]
    return int(min(distances)) if distances else None


class TacticCorpus(NoteCorpus):
    """Only text and outcome provenance persist; map coordinates never do.

    Private controls retain the author's lineage's notes. Shared controls expose
    other lineages too. A tactic is an unverified claim, not a certified rule.
    """
    def __init__(self, mode, training):
        super().__init__(mode, training)
        self.experiences = {}
        self.owners = {}
        self.world = 0
        self.block_access = False
        self.counts.update(tactics_written=0, novel_outcome_deliveries=0)

    def payload(self, agent, obs):
        return []

    def context(self, agent, obs):
        if self.block_access:
            return []
        owner = self.owners.get(agent, agent)
        target = situation(obs)
        candidates = [n for n in self.notes if self.mode == "shared" or n.owner == owner]
        candidates.sort(key=lambda n: (-sum(a == b for a, b in zip(n.situation, target)), -n.id))
        return [(n, []) for n in candidates[:1]]

    def augment(self, observations):
        result = super().augment(observations)
        for agent, obs in result.items():
            selected = self.context(agent, obs) if agent in self.living else []
            self.counts["returned_records"] += len(selected)
            self.counts["peer_records"] += sum(n.owner != self.owners.get(agent, agent) for n, _ in selected)
            obs["notes"] = [{"id": n.id, "author": n.author, "age": max(0, int(obs["self"][6])-n.tick) if n.world == self.world else None,
                             "scope": "tentative tactic", "origin_world": n.world, "text": n.text,
                             "evidence": json.dumps(n.experience[-3:], separators=(",", ":"))} for n, _ in selected]
            obs["recent_experience"] = copy.deepcopy(self.experiences.get(agent, [])[-4:])
            tick = int(obs["self"][6])
            obs["note_opportunity"] = bool(agent in self.living and self.experiences.get(agent) and
                tick-self.last_write.get(agent, -self.training.note_interval) >= self.training.note_interval)
        return result

    def record(self, actions, before, after, env):
        self.owners.update({i: p.lineage for i, p in env.prey.items()})
        for agent, action in actions.items():
            alive = agent in env.agents
            energy = float(after[agent]["self"][0]) if alive else None
            prior = float(before[agent]["self"][0])
            row = {"action": ACTION_NAMES[action], "energy_before": round(prior, 2),
                   "energy_change": round(energy-prior, 2) if alive else None,
                   "food_distance": [food_distance(before[agent]), food_distance(after[agent]) if alive else None],
                   "survived": alive, "situation": situation(before[agent]),
                   "gave_birth": any(e["type"] == "birth" and e["parent"] == agent for e in env.events)}
            history = self.experiences.setdefault(agent, [])
            history.append(row)
            self.experiences[agent] = history[-6:]
            self.transferred.add((self.owners.get(agent, agent), signature(row)))

    def publish(self, decisions, observations, bank, policy_ids=None, texts=None):
        self.pending = set()
        rewards = dict.fromkeys(decisions, 0.)
        self.pending_rewards = rewards
        for agent, write in decisions.items():
            if write < 0:
                continue
            self.counts["note_opportunities"] += 1
            if not write:
                continue
            obs = observations[agent]
            if not obs.get("note_opportunity"):
                raise ValueError("A tactic requires the author's own experience and an available publication slot")
            experience = copy.deepcopy(self.experiences[agent][-3:])
            if texts is not None and agent in texts:
                text = texts[agent]
            elif self.training.note_style == "prose":
                text, count = bank.describe_tactic((policy_ids or {}).get(agent, agent), experience)
                self.counts["prose_generated_tokens"] += count
            else:
                text = "Observed action outcomes; generalization untested: "+json.dumps(experience, separators=(",", ":"))
            tick = int(obs["self"][6])
            note = Tactic(self.serial, agent, self.owners.get(agent, agent), tick, self.world,
                          text[:480], tuple(experience[-1]["situation"]), experience, self.training.note_style)
            self.serial += 1
            self.notes.append(note)
            self.notes = self.notes[-256:]
            self.last_write[agent] = tick
            self.pending.add(note.id)
            self.counts["notes_written"] += 1
            self.counts["tactics_written"] += 1
            self.events.append({"type": "tactic", "id": note.id, "author": agent, "owner": note.owner,
                                "world": self.world, "tick": tick, "text": note.text, "experience": experience})
        return rewards

    def _credit_delivery(self, receiver, selected):
        owner = self.owners.get(receiver, receiver)
        for note, _ in selected:
            if note.id not in self.pending or note.owner == owner:
                continue
            novel = 0
            for row in note.experience:
                key = (owner, signature(row))
                if key not in self.transferred:
                    self.transferred.add(key)
                    novel += 1
            if novel:
                reward = self.training.note_credit
                self.pending_rewards[note.author] += reward
                self.counts["publication_reward"] += reward
                self.counts["novel_outcome_deliveries"] += novel
                self.events.append({"type": "tactic_credit", "id": note.id, "author": note.author,
                                    "receiver": receiver, "novel_outcomes": novel, "reward": reward})

    def restore_memory(self, previous, owners, evaluation=False):
        self.notes = copy.deepcopy(previous.notes)
        self.serial = previous.serial
        self.world = previous.world+1
        self.owners = dict(owners)
        if not evaluation:
            self.transferred = copy.deepcopy(previous.transferred)
        # No old sensory facts, recent trajectories, live IDs or pending rewards.


def record_experience(corpus, actions, before, after, env):
    if isinstance(corpus, TacticCorpus):
        corpus.record(actions, before, after, env)
