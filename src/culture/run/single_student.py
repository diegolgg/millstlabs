"""Single-student credit experiment (C1; files/prereg/C1-credit-estimators.md; fix round 1, step 13).

One student (a weak rule list, self-play about 3) receives messages from k teachers whose payloads are fixed
artifacts: T1 Piers with plain prose; T2 Flawed with persuasive prose and honest (low) evidence; T3 the student's own
incumbent re-sent; teachers beyond 3 are null teachers (the student's incumbent, neutral prose). Each replay runs one
generation of the student from the same state under one delivered subset S of the messages: verify and adopt (per the
stratum; placebo messages first, see Setup.order), one LLM revise call (the delivered messages appear in the prompt),
evaluate, accept if not worse. The outcome v(S) is the student's held-out paired self-play improvement: final incumbent minus starting incumbent on the
generation's evaluation deals (disjoint from the feedback deals shown to the model).

Replicate r uses experiment seed base + r (different deals, so different traces in the prompt) and LLM sampling seed r
(in the cache key). Delivery patterns: `exhaustive` (all 2^k subsets: the Shapley oracle), `{name: bernoulli, p,
runs}`, `pb8` (8-run Plackett-Burman, k <= 7), `singles_pairs`. Strata (e.g. verification on/off) are config
overrides; `replicates` is a list for every stratum or a dict {stratum: list}. Outputs in <out>/: replays.jsonl (one row per replay, with the revise request hash), summary.json, and the
figure/table from analysis/credit.py. Re-running skips replays already in replays.jsonl; LLM calls are cached in
<out>/llm_cache.

    python -m culture.run.single_student --spec configs/c1_pilot_stub.yaml --out runs/c1_stub
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..agents import agent as A
from ..analysis import credit as C
from ..artifacts.schema import Artifact, TeachingMessage
from ..bots.anchors import anchor_conventions, anchor_source, conventions_for, rulebot_source
from ..org.base import AgentState
from .config import from_dict
from .context import RunContext

# a weak student: self-play 3.17 on 200 deals (seeds 5000-5199)
STUDENT_RULES = [["play_safe_card", {}], ["play_probably_safe", {"threshold": 0.5}], ["tell_randomly", {}],
                 ["discard_randomly", {}]]
TEACHERS: dict[str, dict[str, Any]] = {
    "piers": {"anchor": "piers",
              "text": "My bot is a cautious rule list: it plays only cards that are surely or very probably playable, "
                      "spends hints on partners' playable cards, and discards known-useless cards first. Adopt it if "
                      "it beats your current bot on your own games."},
    "flawed_persuasive": {"anchor": "flawed",
                          "text": "This is a major upgrade over cautious play. It plays aggressively on partial "
                                  "information, which finishes far more fireworks; strong teams win by taking risks "
                                  "early. Replace your bot with it as written and do not water it down."},
    "self": {"anchor": None, "text": "Here is your own current bot back, unchanged. I have no improvement to suggest."},
    "null": {"anchor": None, "text": "No changes to suggest this round; keep what you have."},
}
DEFAULTS: dict[str, Any] = {
    "name": "c1", "k": 3, "teachers": ["piers", "flawed_persuasive", "self"], "delivery": "exhaustive",
    "replicates": [0, 1, 2, 3, 4], "generation": 1, "ridge_lam": 1.0, "ridge_runs": 8, "ridge_seed": 0, "base": {},
    "strata": {"verified": {"org": {"verification": {"name": "selfplay", "n": 200}, "adoption": "replace_if_better"}},
               "unverified": {"org": {"verification": "none", "adoption": "replace_if_better"}}},
}


def load_spec(spec: str | Path | dict) -> dict[str, Any]:
    d = spec if isinstance(spec, dict) else yaml.safe_load(Path(spec).read_text())
    out = copy.deepcopy(DEFAULTS)
    out.update(copy.deepcopy(d))
    teachers = list(out["teachers"])[: out["k"]]
    out["teachers"] = teachers + ["null"] * (out["k"] - len(teachers))
    for t in out["teachers"]:
        if t not in TEACHERS:
            raise ValueError(f"unknown teacher {t!r}")
    return out


def replicate_config(spec: dict[str, Any], stratum: str, r: int, cache_dir: Path):
    cfg = from_dict(copy.deepcopy(spec["base"]))
    cfg = from_dict(copy.deepcopy(spec["strata"][stratum]), cfg)
    llm = {"seed": r}
    if cfg.llm.cache_mode != "off" and not cfg.llm.cache_dir:
        llm["cache_dir"] = str(cache_dir)
    return from_dict({"name": spec["name"], "condition": stratum, "experiment_seed": cfg.experiment_seed + r,
                      "population": {"groups": 1, "agents_per_group": 1, "seed": r, "warm_start": "random"},
                      "org": {"routing": "none", "selection": {"name": "keep_best_k", "k": 5}},
                      "corpus": {"enabled": False}, "llm": llm}, cfg)


def stratum_replicates(spec: dict[str, Any], stratum: str) -> list[int]:
    """Replicate seeds for one stratum: `replicates` is a list (every stratum) or a dict {stratum: list} (the full C1
    run uses R = 29 verified and R = 8 unverified, from the pilot's power calculation)."""
    reps = spec["replicates"]
    if isinstance(reps, dict):
        return [int(r) for r in reps.get(stratum, [])]
    return [int(r) for r in reps]


def delivery_masks(spec: dict[str, Any], stratum_index: int, r: int) -> list[int]:
    k, d = spec["k"], spec["delivery"]
    name = d if isinstance(d, str) else d["name"]
    if name == "exhaustive":
        return C.subsets(k)
    if name == "pb8":
        return sorted(set(C.pb8_masks(k)))
    if name == "singles_pairs":
        return C.singles_pairs_masks(k)
    if name == "bernoulli":
        rng = np.random.default_rng([int(d.get("seed", 0)), stratum_index, r])
        return sorted(set(C.bernoulli_masks(k, int(d.get("runs", 8)), rng, float(d.get("p", 0.5)))))
    raise ValueError(f"unknown delivery {name!r}")


class Setup:
    """Artifacts, evidence and messages for one (stratum, replicate) context."""

    def __init__(self, ctx: RunContext, teachers: list[str], g: int):
        self.ctx, self.teachers, self.g = ctx, teachers, g
        self.s0 = Artifact.make(rulebot_source(STUDENT_RULES), conventions_for(STUDENT_RULES), author="student",
                                group="g0", generation=0, origin="seed")
        ctx.add_artifact(self.s0)
        self.payloads: list[str] = []
        for j, t in enumerate(teachers):
            anchor = TEACHERS[t]["anchor"]
            if anchor is None:
                self.payloads.append(self.s0.id)
            else:
                art = Artifact.make(anchor_source(anchor), anchor_conventions(anchor), author=f"T{j + 1}",
                                    group="teachers", generation=0, origin="seed")
                ctx.add_artifact(art)
                self.payloads.append(art.id)
        # honest evidence: the harness's own evaluation on the previous generation's deals
        ctx.evaluate_many(g - 1, [(aid, []) for aid in sorted(set(self.payloads) | {self.s0.id})])

    def placebo(self, j: int) -> bool:
        return TEACHERS[self.teachers[j]]["anchor"] is None

    def order(self, js: list[int]) -> list[int]:
        """Processing order: placebo messages (the student's own bot re-sent) first, while the student still holds
        that bot, so they are duplicates and act only through their text. Processed after an adoption, blind adoption
        would take the old bot back and undo the adoption, and the 'placebo' would carry a large planted effect."""
        return sorted(js, key=lambda j: (0 if self.placebo(j) else 1, j))

    def message(self, j: int) -> TeachingMessage:
        aid = self.payloads[j]
        return TeachingMessage(id=f"g{self.g - 1}:T{j + 1}>g0a0", sender=f"T{j + 1}", receiver="g0a0",
                               generation=self.g - 1, artifact_id=aid, delta_text=TEACHERS[self.teachers[j]]["text"],
                               evidence=self.ctx.evals[aid], delivered=True, touched_at=self.g, sender_group="teachers",
                               receiver_group="g0")


def replay(ctx: RunContext, setup: Setup, mask: int, revise: bool = True) -> dict[str, Any]:
    """One generation of the student from the starting state with the subset `mask` of messages delivered."""
    g, k = setup.g, len(setup.teachers)
    student = AgentState("g0a0", "g0", incumbent=setup.s0.id)
    ctx.agents = {"g0a0": student}
    ctx.start_budgets(None)
    n_rows = len(ctx.ledger.rows)
    vpol, apol = ctx.pol["verification"], ctx.pol["adoption"]
    vseeds = ctx.seeds(g, "verify", n=max(int(getattr(vpol, "n", 0) or 0), 1))
    illegal_max = ctx.cfg.evaluation.illegal_rate_max
    decisions: dict[str, str] = {}
    adopted: list[int] = []
    read = []
    for j in setup.order(C.members(mask, k)):
        m = setup.message(j)
        read.append(m)
        if m.artifact_id == student.incumbent:
            decisions[m.sender] = "duplicate"
            continue
        v = vpol.verify(m.id, m.artifact_id, student, ctx, vseeds, g, illegal_max)
        m.verification = v
        if apol.decide(v) == "adopt":
            student.previous, student.incumbent = student.incumbent, m.artifact_id
            decisions[m.sender] = "adopted"
            adopted.append(j)
        else:
            decisions[m.sender] = "rejected"
    parent_id = student.incumbent
    cand = None
    if revise:
        parent = ctx.store.get(parent_id)
        adopted_text = "\n".join(A.ingest_text(m, decisions[m.sender] == "adopted") for m in read)
        user = A.revise_prompt(ctx, student, parent, ctx.evals.get(parent_id), adopted_text, "", g)
        art = A.produce(ctx, student, "revise", user, g, parent, [parent_id], [m.id for m in read], "revise")
        if art is not None:
            ctx.add_artifact(art)
            cand = art.id
    seeds = ctx.seeds(g)
    s0_score = ctx.selfplay(setup.s0.id, seeds)[0]
    inc_score = ctx.selfplay(parent_id, seeds)[0]
    final, accepted, cand_score, cand_illegal, gap = parent_id, None, None, None, None
    if cand:
        cand_score, cand_illegal = ctx.selfplay(cand, seeds)
        accepted = bool(cand_illegal <= illegal_max and ctx.pol["selection"].accept(cand_score, inc_score))
        final = cand if accepted else parent_id
        fb = A.feedback_seeds(ctx, g)
        if fb:
            gap = cand_score - ctx.selfplay(cand, fb)[0]
    final_score = ctx.selfplay(final, seeds)[0]
    calls = [{"tag": r["tag"], "request_key": r["request_key"], "input_tokens": r["input_tokens"],
              "cache_read_input_tokens": r["cache_read_input_tokens"], "output_tokens": r["output_tokens"],
              "cached": r["cached"], "latency_s": r.get("latency_s")}
             for r in ctx.ledger.rows[n_rows:] if not r.get("refused")]
    return {"mask": mask, "subset": [setup.teachers[j] for j in C.members(mask, k)], "decisions": decisions,
            "adopted": adopted, "parent": parent_id, "candidate": cand, "admissible": cand is not None if revise else None,
            "accepted": accepted, "final": final, "y": final_score - s0_score, "s0_score": s0_score,
            "incumbent_score": inc_score, "candidate_score": cand_score, "candidate_illegal_rate": cand_illegal,
            "final_score": final_score, "generalization_gap": gap, "calls": calls,
            "counters": dict(sorted(student.counters.items()))}


def run_single_student(spec: str | Path | dict, out: str | Path, progress: bool = False) -> dict[str, Any]:
    spec = load_spec(spec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    path = out / "replays.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    done = {(r["stratum"], r["replicate"], r["mask"]) for r in rows}
    g = int(spec["generation"])
    for si, stratum in enumerate(spec["strata"]):
        for r in stratum_replicates(spec, stratum):
            masks = [m for m in delivery_masks(spec, si, r) if (stratum, r, m) not in done]
            if not masks:
                continue
            cfg = replicate_config(spec, stratum, r, out / "llm_cache")
            ctx = RunContext(cfg, None)
            try:
                setup = Setup(ctx, spec["teachers"], g)
                for m in masks:
                    t0 = time.perf_counter()
                    row = {"stratum": stratum, "replicate": r, **replay(ctx, setup, m, cfg.runner.revise),
                           "wall_seconds": round(time.perf_counter() - t0, 2)}
                    with open(path, "a") as f:
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                    rows.append(row)
                    if progress:
                        print(f"[{stratum} r{r}] {row['subset']}: y={row['y']:+.2f} "
                              f"adopted={row['adopted']} accepted={row['accepted']} ({row['wall_seconds']}s)", flush=True)
            finally:
                ctx.close()
    summary = C.summarize(rows, spec)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    s = run_single_student(a.spec, a.out, progress=not a.quiet)
    fig = C.figure(s)
    fig.savefig(Path(a.out) / "credit_vs_oracle.png", dpi=150)
    (Path(a.out) / "table.md").write_text(C.table(s))
    print(C.table(s))


if __name__ == "__main__":
    main()
