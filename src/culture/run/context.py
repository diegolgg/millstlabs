"""Run state and services shared by the generation loop: engine judge, batched evaluation, budget, logs, checkpoint
state. Everything that must survive a crash is either in `state_dict()` (bounded) or in an append-only JSONL log whose
byte offset is checkpointed (resume truncates to it)."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts.provenance import Provenance
from ..artifacts.schema import Artifact, Evaluation, TeachingMessage
from ..artifacts.store import ArtifactStore, Corpus, EvidenceRegistry, TouchLog
from ..bots.runner import BotSpec, SandboxLimits, complexity
from ..evaluate import stats
from ..evaluate.crossplay import crossplay, pair_jobs
from ..evaluate.pool import Evaluator
from ..evaluate.selfplay import first_error, seat_rates, selfplay
from ..game.hanabi import HanabiParams
from ..game.seeds import generation_seeds
from ..llm.backend import CostLedger, Request, Response, Usage, approx_tokens, cost_usd
from ..llm.cache import CachedBackend, CallCache
from ..llm.stub_backend import StubBackend, StubConfig
from ..org import registry
from ..org.base import AgentState
from ..org.population import Population
from .config import ExperimentConfig

LOGS = ("generations", "artifacts", "messages", "ledger", "touch", "provenance")


def make_backend(cfg: ExperimentConfig):
    """The inner LLM backend named by the config (the cache and ledger wrap it)."""
    if cfg.llm.backend == "openai_compat":
        from ..llm.openai_compat import OpenAICompatBackend, OpenAICompatConfig

        c = cfg.llm
        return OpenAICompatBackend(OpenAICompatConfig(base_url=c.base_url, seed=c.seed, temperature=c.temperature,
                                                      request_timeout_s=c.request_timeout_s,
                                                      extra_body=dict(c.extra_body)))
    return StubBackend(StubConfig(**cfg.llm.stub))


class RunContext:
    def __init__(self, cfg: ExperimentConfig, out: Path | None):
        self.cfg = cfg
        self.out = Path(out) if out else None
        self.run_id = f"{cfg.name}/{cfg.condition}/p{cfg.population.seed}"
        self.seed_prefix = f"{cfg.name}/p{cfg.population.seed}"  # condition-free: warm starts are paired
        self.base_params = HanabiParams(**asdict(cfg.game))
        self.game_params = self.base_params
        self.variant_note = ""
        self.limits = SandboxLimits(**asdict(cfg.sandbox))
        self.evaluator = Evaluator(self.game_params, self.limits, workers=cfg.evaluation.workers)
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
        self.ledger = CostLedger(self._p("ledger"))
        cache = None
        if cfg.llm.cache_mode != "off":
            cache_dir = Path(cfg.llm.cache_dir) if cfg.llm.cache_dir else (self.out / "llm_cache" if self.out else None)
            cache = CallCache(cache_dir) if cache_dir else None
        self.backend = CachedBackend(make_backend(cfg), cache, cfg.llm.cache_mode, self.ledger)
        self.store = ArtifactStore(self.out / "artifacts" if self.out else None)
        self.provenance = Provenance(self._p("provenance"))
        self.touch = TouchLog(self._p("touch"))
        self.registry = EvidenceRegistry()
        self.pop = Population(cfg.population.groups, cfg.population.agents_per_group)
        self.agents = {a: AgentState(a, self.pop.group_of(a)) for a in self.pop.agents}
        self.corpora = {g: Corpus(g, cfg.corpus.capacity) for g in self.pop.groups}
        self.pol = {f: registry.make(f, getattr(cfg.org, f)) for f in registry.REGISTRIES}
        from ..org.migration import NoMigration, RandomMigration
        from ..org.population import Islands

        if isinstance(self.pol["topology"], Islands) and isinstance(self.pol["migration"], NoMigration):
            t = self.pol["topology"]  # islands(G, migration_rate, interval) carries its own migration rule
            self.pol["migration"] = RandomMigration(t.migration_rate, t.interval)
        self.outbox: list[TeachingMessage] = []
        self.group_budget: dict[str, float] = {g: float("inf") for g in self.pop.groups}
        self.evals: dict[str, Evaluation] = {}
        self.frozen: list[list] = []  # [generation, label, artifact id] snapshots for the ladder
        self.budget_bonus: dict[str, float] = {}
        self.generation = -1  # last completed generation
        self._specs: dict[str, BotSpec] = {}
        self.timing: dict[str, float] = {}
        # ids already in artifacts.jsonl (read after the runner truncated it): store files can outlive a crash, so
        # whether to log is decided from the log, never from the store
        self._logged: set[str] = set()
        if self.out and self._p("artifacts").exists():
            with open(self._p("artifacts")) as f:
                self._logged = {json.loads(line)["id"] for line in f if line.strip()}

    # ------------------------------------------------------------------ paths and logs
    def _p(self, name: str) -> Path | None:
        return self.out / f"{name}.jsonl" if self.out else None

    def log(self, name: str, row: dict[str, Any]) -> None:
        if self.out:
            with open(self._p(name), "a") as f:
                f.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")

    def offsets(self) -> dict[str, int]:
        return {n: (self._p(n).stat().st_size if self.out and self._p(n).exists() else 0) for n in LOGS}

    def truncate(self, offsets: dict[str, int]) -> None:
        for n, off in offsets.items():
            p = self._p(n)
            if p and p.exists() and p.stat().st_size > off:
                with open(p, "r+b") as f:
                    f.truncate(off)

    # ------------------------------------------------------------------ randomness
    def rng(self, generation: int, purpose: str) -> random.Random:
        return random.Random(f"{self.cfg.population.seed}/{generation}/{purpose}")

    def seeds(self, generation: int, purpose: str = "eval", n: int | None = None) -> list[int]:
        n = n if n is not None else self.cfg.evaluation.selfplay_games
        return generation_seeds(self.cfg.experiment_seed, generation, n, purpose).seeds

    # ------------------------------------------------------------------ artifacts
    def add_artifact(self, art: Artifact, teaching: list[dict] | None = None) -> Artifact:
        if art.id not in self.store:
            self.store.put(art)
        if art.id not in self._logged:
            self._logged.add(art.id)
            self.log("artifacts", art.meta())
        self.provenance.add(art.id, art.author, art.group, art.generation, art.parents, teaching, art.origin)
        return art

    def spec(self, aid: str) -> BotSpec:
        if aid.startswith("anchor:"):
            return BotSpec(aid)
        if aid not in self._specs:
            if len(self._specs) > 2048:
                self._specs.clear()
            self._specs[aid] = BotSpec(aid, self.store.get(aid).code)
        return self._specs[aid]

    def set_game_params(self, params: HanabiParams, note: str = "") -> None:
        self.game_params = params
        self.variant_note = note
        self.evaluator.set_params(params)

    # ------------------------------------------------------------------ budget
    # The budget is a per-generation pool per group (spec section 7: a call debits the agent's share of its group's
    # budget; when the group's budget runs out, remaining agents skip revise/teach). Reserving the estimated cost of a
    # call before sending — and refusing if it would overspend — is what stops a group allocated 1.2 calls from
    # spending 4. The pool is transient (recomputed at each generation start, so resume re-derives it).
    def start_budgets(self, per_group: dict[str, float] | None) -> None:
        if per_group is None:  # unbudgeted (warm start, probe): bonuses are kept for the next budgeted generation
            self.group_budget = {g: float("inf") for g in self.pop.groups}
        else:
            self.group_budget = {g: float(per_group.get(g, 0.0)) for g in self.pop.groups}
            for aid, bonus in list(self.budget_bonus.items()):  # fold earned credit-share bonus into the group's pool
                g = self.agents[aid].group if aid in self.agents else None
                if g in self.group_budget:
                    self.group_budget[g] += bonus
            self.budget_bonus = {}
        for a in self.agents.values():  # per-agent mirror, for logs and the checkpoint
            a.budget_left = self.group_budget.get(a.group, 0.0)

    def estimate_cost(self, req: Request) -> float:
        """Upper bound on what one call will cost in the budget's unit, reserved before the call is sent."""
        unit = self.cfg.budget.unit
        if unit == "calls":
            return 1.0
        new_tokens = approx_tokens(req.prompt_text()) + req.max_tokens  # input estimate + the full output allowance
        if unit == "tokens":
            return float(new_tokens)
        return float(cost_usd(req.model, Usage(input_tokens=approx_tokens(req.prompt_text()),
                                               output_tokens=req.max_tokens)) or 0.0)

    def reserve(self, agent: AgentState, req: Request) -> tuple[bool, str]:
        """Can the agent's group afford this call? Returns (ok, reason-if-refused). Does not debit (charge does)."""
        est = self.estimate_cost(req)
        left = self.group_budget.get(agent.group, 0.0)
        if est > left:
            return False, f"estimated {self.cfg.budget.unit} cost {est:g} > group budget left {left:g}"
        return True, ""

    def log_refusal(self, agent: AgentState, req: Request, reason: str) -> None:
        agent.bump("refused_for_budget")
        self.ledger.record_refusal(req, reason, self.group_budget.get(agent.group, 0.0))

    def charge(self, agent: AgentState, resp: Response, extra: float = 0.0) -> None:
        self.spend(agent.group, self._amount(resp) + extra)
        agent.budget_left = self.group_budget.get(agent.group, 0.0)

    def _amount(self, resp: Response) -> float:
        unit = self.cfg.budget.unit
        return 1.0 if unit == "calls" else float(resp.usage.total) if unit == "tokens" else float(resp.cost_usd or 0.0)

    def spend(self, group: str, amount: float) -> None:
        """Debit the group's pool directly (teaching cost, which is not an LLM call)."""
        self.group_budget[group] = self.group_budget.get(group, 0.0) - amount

    # ------------------------------------------------------------------ judge (engine-time measurements for org/)
    def selfplay(self, aid: str, seeds: list[int]) -> tuple[float, float]:
        res = selfplay(self.evaluator, self.spec(aid), seeds)
        return float(np.mean([r.score for r in res])), seat_rates(res, aid)["illegal_rate"]

    def crossplay(self, a: str, b: str, seeds: list[int]) -> float:
        return float(np.mean([r.score for r in crossplay(self.evaluator, self.spec(a), self.spec(b), seeds)]))

    def anchors(self, aid: str, seeds: list[int]) -> dict[str, float]:
        return {n: self.crossplay(aid, f"anchor:{n}", seeds) for n in sorted(self.cfg.evaluation.anchors)}

    def anchor_mean(self, aid: str, seeds: list[int]) -> float:
        a = self.anchors(aid, seeds)
        return float(np.mean(list(a.values()))) if a else 0.0

    # ------------------------------------------------------------------ evaluation
    def evaluate_many(self, generation: int, items: list[tuple[str, list[str]]]) -> dict[str, Evaluation]:
        """Evaluate artifacts on this generation's shared seeds: self-play, cross-play with the listed partners,
        anchors. All games are dispatched in one batch; results are memoized for the generation."""
        e = self.cfg.evaluation
        seeds = self.seeds(generation)
        cs, an = seeds[: e.crossplay_games], seeds[: e.anchor_games]
        jobs = []
        for aid, partners in items:
            s = self.spec(aid)
            jobs.append(((s, s), seeds))
            for p in partners:
                if p != aid:
                    jobs += pair_jobs(s, self.spec(p), cs)
            for name in e.anchors:
                jobs += pair_jobs(s, BotSpec.anchor(name), an)
        t0 = time.perf_counter()
        self.evaluator.run(jobs)
        dt = time.perf_counter() - t0
        out = {}
        for aid, partners in items:
            res = selfplay(self.evaluator, self.spec(aid), seeds)
            sc = [r.score for r in res]
            art = self.store.get(aid)
            ev = Evaluation(
                artifact_id=aid, generation=generation, seed_base=seeds[0], n_games=len(seeds), selfplay_scores=sc,
                selfplay_iqm=stats.iqm(sc), selfplay_mean=float(np.mean(sc)),
                selfplay_ci=stats.iqm_bootstrap_ci(sc, n_boot=500, seed=generation),
                crossplay={p: self.crossplay(aid, p, cs) for p in sorted(partners) if p != aid} if cs else {},
                anchors=self.anchors(aid, an) if an else {},
                illegal_rate=seat_rates(res, aid)["illegal_rate"], lines=len(art.code.splitlines()),
                complexity=complexity(art.code), wall_seconds=round(dt / max(len(items), 1), 4),
                error=first_error(res, aid),
            )
            self.registry.issue(ev)
            self.evals[aid] = ev
            out[aid] = ev
        return out

    # ------------------------------------------------------------------ checkpoint state
    def state_dict(self) -> dict[str, Any]:
        live = {a.incumbent for a in self.agents.values() if a.incumbent}
        live |= {m.artifact_id for m in self.outbox}
        return {
            "generation": self.generation,
            "agents": {k: asdict(v) for k, v in self.agents.items()},
            "population": self.pop.state_dict(),
            "corpora": {g: c.state_dict() for g, c in self.corpora.items()},
            "registry": self.registry.state_dict(),
            "policies": {k: p.state_dict() for k, p in self.pol.items()},
            "outbox": [m.to_dict() | {"evidence_full": m.evidence.to_dict() if m.evidence else None} for m in self.outbox],
            "evals": {k: v.to_dict() for k, v in self.evals.items() if k in live},
            "frozen": self.frozen,
            "budget_bonus": self.budget_bonus,
            "backend": self.backend.state_dict(),
            "game_params": asdict(self.game_params),
            "variant_note": self.variant_note,
        }

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.generation = s["generation"]
        self.agents = {k: AgentState(**v) for k, v in s["agents"].items()}
        self.pop.load_state_dict(s["population"])
        self.corpora = {g: Corpus.from_state(c) for g, c in s["corpora"].items()}
        self.registry.load_state_dict(s["registry"])
        for k, st in s["policies"].items():
            self.pol[k].load_state_dict(st)
        self.outbox = []
        for d in s["outbox"]:
            m = TeachingMessage.from_dict(d)
            m.evidence = Evaluation.from_dict(d["evidence_full"]) if d.get("evidence_full") else None
            self.outbox.append(m)
        self.evals = {k: Evaluation.from_dict(v) for k, v in s["evals"].items()}
        self.frozen = s["frozen"]
        self.budget_bonus = s["budget_bonus"]
        self.backend.load_state_dict(s["backend"])
        self.set_game_params(HanabiParams(**s["game_params"]), s.get("variant_note", ""))

    def close(self) -> None:
        self.evaluator.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))
