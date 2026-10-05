"""Verification uses private crop pixels, never the hidden country label."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ColorFact:
    source: int
    rgb: tuple[int, int, int]
    witness: tuple[int, int]  # Crop-local coordinate, not position on the hidden flag.

    @property
    def key(self):
        return self.source, self.rgb

    def text(self):
        color = "#" + bytes(self.rgb).hex()
        return f"Verified sensor fact: agent {self.source}'s private crop contains RGB {color}."


class VerifiedCorpus:
    def __init__(self, crops, mode, limit=2):
        if mode not in {"off", "private", "shared"}:
            raise ValueError("Invalid corpus mode")
        self.mode, self.limit = mode, limit
        self.evidence = []
        for i, crop in enumerate(crops):
            facts = {}
            for y, row in enumerate(crop):
                for x, pixel in enumerate(row):
                    rgb = tuple(int(v) for v in pixel)
                    facts.setdefault(rgb, ColorFact(i, rgb, (x, y)))
            self.evidence.append(list(facts.values()))
        self.deposits = {}
        self.seen = [set() for _ in crops]
        self.colors_known = [{f.rgb for f in rows} for rows in self.evidence]
        self.events = []
        self.counts = dict(deposit_calls=0, retrieve_calls=0, rejected_deposits=0,
                           retrieved_records=0, peer_records=0, novel_color_imports=0)

    def deposit(self, agent, fact):
        self.counts["deposit_calls"] += 1
        if fact not in self.evidence[agent] or fact.source != agent:
            self.counts["rejected_deposits"] += 1
            return False
        self.deposits[fact.key] = fact
        self.events.append({"kind": "deposit", "agent": agent, "fact": asdict(fact)})
        return True

    def retrieve(self, agent):
        self.counts["retrieve_calls"] += 1
        rows = [fact for key, fact in self.deposits.items() if key not in self.seen[agent]
                and (self.mode == "shared" or fact.source == agent)]
        rows = rows[:self.limit]  # Deposit order; no hidden-country relevance ranking.
        for fact in rows:
            self.seen[agent].add(fact.key)
            self.counts["peer_records"] += fact.source != agent
            self.counts["novel_color_imports"] += fact.rgb not in self.colors_known[agent]
            self.colors_known[agent].add(fact.rgb)
        self.counts["retrieved_records"] += len(rows)
        self.events.append({"kind": "retrieve", "agent": agent, "facts": [asdict(f) for f in rows]})
        return rows

    def mask(self, agent, slots):
        # NONE, one PUT per locally available color slot, GET. Never inspect peer deposits.
        available = [f.key not in self.deposits for f in self.evidence[agent]]
        return [True] + available + [False]*(slots-len(available)) + [True]

    def execute(self, actions, slots):
        if self.mode == "off":
            return {i: [] for i in actions}
        for i, a in actions.items():
            if not 0 <= a < slots+2 or not self.mask(i, slots)[a]:
                raise ValueError("Invalid corpus action")
        # Synchronous probes: all reads precede all writes, independent of agent iteration order.
        responses = {i: self.retrieve(i) if a == slots+1 else [] for i, a in actions.items()}
        for i, a in actions.items():
            if 1 <= a <= slots:
                self.deposit(i, self.evidence[i][a-1])
        return responses

    def metrics(self):
        return {**self.counts, "deposited_facts": len(self.deposits),
                "distinct_deposited_colors": len({f.rgb for f in self.deposits.values()})}
