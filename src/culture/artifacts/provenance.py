"""Provenance DAG: nodes are artifacts; edges are `parents` and teaching sources. Used by credit and selection
policies and by the graph analyses (analysis/graphs.py)."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


class Provenance:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self.nodes: dict[str, dict[str, Any]] = {}
        self.parents: dict[str, list[str]] = {}
        self.children: dict[str, list[str]] = {}
        self.teaching: dict[str, list[dict[str, Any]]] = {}  # child -> [{message, sender, source}]
        if self.path and self.path.exists():
            with open(self.path) as f:
                for line in f:
                    if line.strip():
                        self._apply(json.loads(line))

    def _apply(self, ev: dict[str, Any]) -> None:
        aid = ev["id"]
        if aid not in self.nodes:
            self.nodes[aid] = {k: ev[k] for k in ("author", "group", "generation", "origin") if k in ev}
            self.parents[aid] = []
            self.children.setdefault(aid, [])
        for p in ev.get("parents", []):
            if p not in self.parents[aid] and p != aid:
                self.parents[aid].append(p)
                self.children.setdefault(p, []).append(aid)
        for t in ev.get("teaching", []):
            self.teaching.setdefault(aid, []).append(t)

    def add(self, aid: str, author: str, group: str, generation: int, parents: list[str],
            teaching: list[dict[str, Any]] | None = None, origin: str = "revise") -> None:
        ev = {"id": aid, "author": author, "group": group, "generation": generation, "parents": list(parents),
              "teaching": list(teaching or []), "origin": origin}
        self._apply(ev)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(ev, sort_keys=True) + "\n")

    def ancestors(self, aid: str, max_depth: int | None = None) -> dict[str, int]:
        """Ancestor id -> distance (1 = parent)."""
        out: dict[str, int] = {}
        q = deque([(aid, 0)])
        while q:
            x, d = q.popleft()
            if max_depth is not None and d >= max_depth:
                continue
            for p in self.parents.get(x, []):
                if p not in out:
                    out[p] = d + 1
                    q.append((p, d + 1))
        return out

    def clade(self, aid: str) -> set[str]:
        """The artifact and all its descendants."""
        out, q = {aid}, deque([aid])
        while q:
            for c in self.children.get(q.popleft(), []):
                if c not in out:
                    out.add(c)
                    q.append(c)
        return out

    def depth(self, aid: str) -> int:
        anc = self.ancestors(aid)
        return max(anc.values()) if anc else 0

    def teachers_of(self, aid: str, max_depth: int | None = None) -> dict[str, int]:
        """Sender agent -> distance, over the artifact's own teaching sources and those of its ancestors."""
        out: dict[str, int] = {}
        for node, dist in [(aid, 0)] + sorted(self.ancestors(aid, max_depth).items(), key=lambda kv: kv[1]):
            for t in self.teaching.get(node, []):
                out.setdefault(t["sender"], dist)
        return out

    def edges(self) -> list[tuple[str, str, str]]:
        """(parent, child, kind) with kind 'parent' or 'teaching'."""
        out = [(p, c, "parent") for c, ps in self.parents.items() for p in ps]
        out += [(t["source"], c, "teaching") for c, ts in self.teaching.items() for t in ts if t.get("source")]
        return out
