"""One generation for the whole population. This is the only module that orders the steps:

    allocate -> receive -> verify/adopt -> revise -> evaluate/accept -> credit -> deposit -> teach -> migrate -> metrics

Generation 0 is the warm start (author or seed from an anchor, evaluate, deposit, teach). Policies are injected via
`ctx.pol`; nothing here knows which policy is configured.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np

from ..agents import agent as A
from ..artifacts.schema import Artifact, TeachingMessage
from ..artifacts.store import EvidenceRejected
from ..bots.anchors import anchor_conventions, anchor_source
from ..evaluate import stats
from ..evaluate.crossplay import crossplay_matrix
from ..evaluate.selfplay import selfplay
from ..game.hanabi import HanabiParams
from ..org.credit import Offer, Window
from .context import RunContext


def maybe_kill(ctx: RunContext, g: int, step: str) -> None:
    d = ctx.cfg.debug
    if d.kill_at_generation is not None and d.kill_at_generation == g and d.kill_after_step == step:
        os._exit(17)  # simulated crash: no cleanup, partial logs left behind on purpose


def _apply_environment(ctx: RunContext, g: int) -> None:
    base = {k: getattr(ctx.base_params, k) for k in ctx.base_params.__dataclass_fields__}
    params = ctx.pol["environment"].params_for(g, base)
    note = params.pop("_note", "")
    p = HanabiParams(**params)
    if p != ctx.game_params:
        ctx.set_game_params(p, note)


def _group_stats(ctx: RunContext) -> dict[str, dict[str, Any]]:
    out = {}
    for gname, members in ctx.pop.members.items():
        scores = [ctx.agents[a].score for a in members] or [0.0]
        incs = {ctx.agents[a].incumbent for a in members}
        best = max(members, key=lambda a: (ctx.agents[a].score, a)) if members else None
        ev = ctx.evals.get(ctx.agents[best].incumbent) if best else None
        out[gname] = {"credit": sum(ctx.agents[a].credit for a in members), "score": max(scores),
                      "mean": float(np.mean(scores)), "size": len(members), "incumbents": incs,
                      "scores": list(ev.selfplay_scores) if ev else []}
    for gname, s in out.items():  # number of groups holding exactly the same artifact set (cloning detection)
        s["clones"] = sum(1 for o in out.values() if o["incumbents"] == s["incumbents"])
    for s in out.values():
        s.pop("incumbents")
    return out


def _evaluate_population(ctx: RunContext, g: int, extra: dict[str, list[str]]) -> None:
    items: dict[str, list[str]] = {}
    for gname, members in ctx.pop.members.items():
        partners = sorted({ctx.agents[a].incumbent for a in members if ctx.agents[a].incumbent})
        for a in members:
            for aid in [ctx.agents[a].incumbent] + extra.get(a, []):
                if aid:
                    items.setdefault(aid, partners)
    ctx.evaluate_many(g, sorted(items.items()))


def _refresh_scores(ctx: RunContext) -> None:
    for ag in ctx.agents.values():
        ev = ctx.evals[ag.incumbent]
        ag.score = ev.selfplay_mean
        ag.anchor_score = float(np.mean(list(ev.anchors.values()))) if ev.anchors else 0.0


def _deposit(ctx: RunContext) -> None:
    if not ctx.cfg.corpus.enabled:
        return
    holders = {a.id: a.incumbent for a in ctx.agents.values()}
    for ag in sorted(ctx.agents.values(), key=lambda x: x.id):
        try:
            ctx.corpora[ag.group].deposit(ctx.store.get(ag.incumbent), ctx.evals[ag.incumbent], ag.id, ctx.registry,
                                          holders)
        except EvidenceRejected:
            ag.bump("deposit_rejected")


def _teach(ctx: RunContext, g: int) -> list[TeachingMessage]:
    routes = ctx.pol["routing"].route(ctx.agents, ctx.pop, ctx.pol["topology"], ctx.rng(g, "routing"))
    by_sender: dict[str, list[str]] = {}
    for s, r in routes:
        by_sender.setdefault(s, []).append(r)
    cost = ctx.pol["teaching_cost"].cost()
    sent = []
    for s in sorted(by_sender):
        sender = ctx.agents[s]
        receivers = sorted(set(by_sender[s]))
        art = ctx.store.get(sender.incumbent)
        ev = ctx.evals.get(sender.incumbent)
        if not ctx.registry.check(ev):
            continue
        text = A.teach(ctx, sender, receivers, art, ev, g)
        if text is None:
            continue
        sender.budget_left -= cost
        sender.bump("teach_events")
        for r in receivers:
            m = TeachingMessage(id=f"g{g}:{s}>{r}", sender=s, receiver=r, generation=g, artifact_id=art.id,
                                delta_text=text, evidence=ev, sender_group=sender.group,
                                receiver_group=ctx.agents[r].group)
            sent.append(m)
    ctx.outbox = sent
    return sent


# ------------------------------------------------------------------------------------------------ generation 0
def warm_start(ctx: RunContext) -> dict[str, Any]:
    g = 0
    t0 = time.perf_counter()
    _apply_environment(ctx, g)
    ctx.start_budgets(None)
    pcfg = ctx.cfg.population
    for aid in sorted(ctx.agents):
        ag = ctx.agents[aid]
        anchor = pcfg.warm_start_overrides.get(aid) or (pcfg.warm_start if pcfg.warm_start != "author" else None)
        art = None
        if anchor is None:
            art = A.produce(ctx, ag, "author", A.author_prompt(ag, g), g, None, [], [], "author")
            if art is None:
                anchor = "random"  # authoring failed twice: start from the random bot (counted in failed_revisions)
        if art is None:
            art = Artifact.make(anchor_source(anchor), anchor_conventions(anchor), author=f"anchor:{anchor}",
                                group=ag.group, generation=0, origin="seed")
        ctx.add_artifact(art)
        ag.incumbent = art.id
    maybe_kill(ctx, g, "revise")
    _evaluate_population(ctx, g, {})
    _refresh_scores(ctx)
    for ag in ctx.agents.values():
        ctx.pol["selection"].record(ag.group, ag.incumbent, ag.score, None, g)
    _deposit(ctx)
    sent = _teach(ctx, g)
    return _finish(ctx, g, t0, {"teaching": {"sent": len(sent)}}, candidates={},
                   starts={a: ag.incumbent for a, ag in ctx.agents.items()})


# ------------------------------------------------------------------------------------------------ generation g >= 1
def run_generation(ctx: RunContext, g: int) -> dict[str, Any]:
    if g == 0:
        return warm_start(ctx)
    cfg, P = ctx.cfg, ctx.pol
    t0 = time.perf_counter()
    _apply_environment(ctx, g)
    illegal_max = cfg.evaluation.illegal_rate_max
    vpol = P["verification"]
    vseeds = ctx.seeds(g, "verify", n=max(int(getattr(vpol, "n", 0) or 0), 1))

    # allocate
    alloc = None
    if cfg.budget.per_group_per_generation is not None:
        total = cfg.budget.per_group_per_generation * len(ctx.pop.groups)
        alloc = P["allocation"].allocate(ctx.pop.groups, _group_stats(ctx), total)
    ctx.start_budgets(alloc)

    # receive
    inbox = sorted(ctx.outbox, key=lambda m: m.id)
    ctx.outbox = []
    for m, flag in zip(inbox, P["delivery"].deliver(len(inbox), ctx.rng(g, "delivery"))):
        m.delivered = bool(flag)
    starts = {a: ag.incumbent for a, ag in ctx.agents.items()}
    maybe_kill(ctx, g, "receive")

    # verify and adopt
    offers: dict[str, list[Offer]] = {a: [] for a in ctx.agents}
    read: dict[str, list[TeachingMessage]] = {a: [] for a in ctx.agents}
    adopted: dict[str, set[str]] = {a: set() for a in ctx.agents}
    counts = {"delivered": 0, "undelivered": 0, "verified": 0, "passed": 0, "adopted": 0, "merged": 0, "rejected": 0,
              "reverted": 0, "rechecked": 0}
    for m in inbox:
        if m.receiver not in ctx.agents:
            continue
        rcv = ctx.agents[m.receiver]
        offer = Offer(m.id, m.sender, m.delivered, False, m.artifact_id)
        offers[rcv.id].append(offer)
        if not m.delivered:
            counts["undelivered"] += 1
            m.decision = "undelivered"
            ctx.log("messages", m.to_dict())
            continue
        counts["delivered"] += 1
        m.touched_at = g
        ctx.touch.add(rcv.id, m.id, "message", g, used=True)
        read[rcv.id].append(m)
        if m.artifact_id == rcv.incumbent:
            m.decision = "duplicate"
            ctx.log("messages", m.to_dict())
            continue
        v = vpol.verify(m.id, m.artifact_id, rcv, ctx, vseeds, g, illegal_max)
        if v is not None and v.n_games:
            counts["verified"] += 1
            counts["passed"] += int(v.passed)
        decision = P["adoption"].decide(v)
        if decision == "adopt":
            rcv.previous, rcv.incumbent = rcv.incumbent, m.artifact_id
            if vpol.adopt_first:
                prior = rcv.pending_check if rcv.pending_check and rcv.pending_check["since"] == g else None
                rcv.pending_check = {"since": g, "previous": prior["previous"] if prior else rcv.previous,
                                     "message": m.id}
            m.decision = "adopted"
        elif decision == "merge":
            mine, theirs = ctx.store.get(rcv.incumbent), ctx.store.get(m.artifact_id)
            art = A.merge(ctx, rcv, mine, theirs, ctx.evals.get(mine.id), m.evidence, m.sender, m.id, g)
            if art is not None:
                ctx.add_artifact(art, [{"message": m.id, "sender": m.sender, "source": m.artifact_id, "adopted": True}])
                rcv.previous, rcv.incumbent = rcv.incumbent, art.id
                m.decision = "merged"
            else:
                m.decision = "merge_failed"
        else:
            m.decision = "rejected"
        if v is not None:
            v.decision = m.decision
        m.verification = v
        if m.decision in ("adopted", "merged"):
            offer.adopted = True
            adopted[rcv.id].add(m.id)
            rcv.bump("adoptions")
        counts[m.decision if m.decision in counts else "rejected"] += 1
        ctx.log("messages", m.to_dict())
    maybe_kill(ctx, g, "verify")

    # revise
    candidates: dict[str, tuple[str, str]] = {}
    if cfg.runner.revise:
        rng = ctx.rng(g, "selection")
        for a in sorted(ctx.agents):
            ag = ctx.agents[a]
            if not P["selection"].wants_revision(ag, rng):
                continue
            parent_id = P["selection"].choose_parent(ag, rng) or ag.incumbent
            parent = ctx.store.get(parent_id)
            adopted_text = "\n".join(A.ingest_text(m, m.id in adopted[a]) for m in read[a])
            corpus_text = ""
            if cfg.corpus.enabled:
                entries = ctx.corpora[ag.group].retrieve(a, g, cfg.corpus.retrieve_k, ctx.touch, exclude={parent_id},
                                                         policy=cfg.corpus.retrieve_policy)
                corpus_text = "\n".join(f"- {e.artifact_id} from {e.depositor} (generation {e.generation}): {e.summary}"
                                        for e in entries)
            user = A.revise_prompt(ctx, ag, parent, ctx.evals.get(parent_id), adopted_text, corpus_text, g)
            art = A.produce(ctx, ag, "revise", user, g, parent, [parent_id], [m.id for m in read[a]], "revise")
            if art is not None:
                teaching = [{"message": m.id, "sender": m.sender, "source": m.artifact_id,
                             "adopted": m.id in adopted[a]} for m in read[a]]
                ctx.add_artifact(art, teaching)
                candidates[a] = (art.id, parent_id)
    maybe_kill(ctx, g, "revise")

    # evaluate and accept
    extra = {a: [x for x in (starts[a], candidates.get(a, (None,))[0],
                             (ctx.agents[a].pending_check or {}).get("previous")) if x] for a in ctx.agents}
    _evaluate_population(ctx, g, extra)
    accepted = {}
    for a, (cid, parent_id) in sorted(candidates.items()):
        ag = ctx.agents[a]
        c_ev, i_ev = ctx.evals[cid], ctx.evals[ag.incumbent]
        ok = c_ev.illegal_rate <= illegal_max and P["selection"].accept(c_ev.selfplay_mean, i_ev.selfplay_mean)
        P["selection"].record(ag.group, cid, c_ev.selfplay_mean, parent_id, g)
        if ok:
            ag.incumbent = cid
            ag.bump("accepted_revisions")
        accepted[a] = ok
    for ag in ctx.agents.values():
        P["selection"].record(ag.group, ag.incumbent, ctx.evals[ag.incumbent].selfplay_mean, None, g)
        pc = ag.pending_check
        if pc and pc["since"] == g and hasattr(vpol, "recheck"):
            res = vpol.recheck(ag.incumbent, pc["previous"], ctx, vseeds, illegal_max)
            counts["rechecked"] += 1
            if not res["keep"]:
                ag.incumbent = pc["previous"]
                counts["reverted"] += 1
                ag.bump("reverts")
                ctx.log("messages", {"id": pc["message"], "decision": "reverted", "generation": g, **res})
            ag.pending_check = None
    _evaluate_population(ctx, g, {})  # no-op for already-evaluated incumbents (memoized), covers reverts
    _refresh_scores(ctx)
    maybe_kill(ctx, g, "evaluate")

    # credit
    seeds_a = ctx.seeds(g)[: cfg.evaluation.anchor_games]
    windows = []
    for a in sorted(ctx.agents):
        if not offers[a]:
            continue
        before = ctx.anchor_mean(starts[a], seeds_a) if seeds_a else ctx.evals[starts[a]].selfplay_mean
        after = ctx.agents[a].anchor_score if seeds_a else ctx.agents[a].score
        windows.append(Window(a, g, before, after, offers[a], self_revised=bool(accepted.get(a))))
    increments, credit_notes = P["credit"].assign(windows, ctx.provenance)
    alpha = P["teaching_cost"].credit_share()
    for s, v in sorted(increments.items()):
        if s in ctx.agents:
            ctx.agents[s].credit += v
            if alpha > 0 and v > 0:
                ctx.budget_bonus[s] = ctx.budget_bonus.get(s, 0.0) + alpha * v

    # deposit, teach, migrate
    _deposit(ctx)
    sent = _teach(ctx, g)
    moves = P["migration"].migrate(ctx.pop, ctx.agents, g, ctx.rng(g, "migration"))
    for a, _, dst in moves:
        ctx.agents[a].group = dst
    extra_rec = {"teaching": counts | {"sent": len(sent)}, "credit_increments": increments, "allocation": alloc,
                 "parents": {a: p for a, (_, p) in sorted(candidates.items())},
                 "credit_windows": credit_notes, "migrations": moves,
                 "accepted": accepted, "windows": [{"student": w.student, "before": w.before, "after": w.after,
                                                    "offers": [o.__dict__ for o in w.offers]} for w in windows]}
    return _finish(ctx, g, t0, extra_rec, candidates, starts)


# ------------------------------------------------------------------------------------------------ metrics and logs
def _token_vectors(codes: dict[str, str]) -> dict[str, dict[str, int]]:
    import re

    out = {}
    for k, c in codes.items():
        v: dict[str, int] = {}
        for t in re.findall(r"[A-Za-z_]+|\d+\.?\d*", c):
            v[t] = v.get(t, 0) + 1
        out[k] = v
    return out


def code_clusters(codes: dict[str, str], threshold: float = 0.98) -> int:
    """Number of artifact clusters by cosine similarity of token-count vectors (a code-embedding stand-in)."""
    vecs = _token_vectors(codes)
    keys = sorted(vecs)
    reps: list[str] = []
    for k in keys:
        v = vecs[k]
        nv = sum(x * x for x in v.values()) ** 0.5
        joined = False
        for r in reps:
            w = vecs[r]
            nw = sum(x * x for x in w.values()) ** 0.5
            if sum(v[t] * w.get(t, 0) for t in v) / (nv * nw + 1e-12) >= threshold:
                joined = True
                break
        if not joined:
            reps.append(k)
    return len(reps)


def _hc_record(ctx: RunContext, g: int, starts: dict[str, str]) -> dict[str, Any]:
    seeds = ctx.seeds(g)
    z = []
    for a, ag in sorted(ctx.agents.items()):
        if not starts.get(a) or g == 0:
            continue
        after = np.array([r.score for r in selfplay(ctx.evaluator, ctx.spec(ag.incumbent), seeds)], float)
        before = np.array([r.score for r in selfplay(ctx.evaluator, ctx.spec(starts[a]), seeds)], float)
        d = after - before
        sd = d.std(ddof=1) if d.size > 1 else 0.0
        z.append(float(np.sign(d.mean())) * 8.0 if sd == 0 else float(d.mean() / (sd / np.sqrt(d.size))))
    if not z:
        return {"z": [], "hc": None}
    hc, k = stats.higher_criticism(stats.z_to_p(z))
    return {"z": [round(x, 3) for x in z], "hc": round(hc, 4), "hc_k": k}


def _finish(ctx: RunContext, g: int, t0: float, extra: dict[str, Any], candidates: dict, starts: dict) -> dict[str, Any]:
    cfg = ctx.cfg
    groups = {}
    best_ids = {}
    for gname, members in ctx.pop.members.items():
        if not members:
            continue
        best = max(members, key=lambda a: (ctx.agents[a].score, a))
        best_ids[gname] = ctx.agents[best].incumbent
        sc = [ctx.agents[a].score for a in members]
        groups[gname] = {"best": max(sc), "mean": float(np.mean(sc)), "best_artifact": best_ids[gname],
                         "credit": sum(ctx.agents[a].credit for a in members)}
    between = None
    if cfg.evaluation.between_group_games and len(best_ids) > 1:
        names = sorted(best_ids)
        m = crossplay_matrix(ctx.evaluator, [ctx.spec(best_ids[n]) for n in names],
                             ctx.seeds(g, "eval", cfg.evaluation.between_group_games))
        off = m[~np.eye(len(names), dtype=bool)]
        between = {"groups": names, "matrix": np.round(m, 3).tolist(), "offdiag_mean": float(off.mean()),
                   "diag_mean": float(np.diag(m).mean())}
    within = [float(np.mean(list(ctx.evals[ag.incumbent].crossplay.values())))
              for ag in ctx.agents.values() if ctx.evals[ag.incumbent].crossplay]
    ladder = None
    if cfg.evaluation.ladder_every and g % cfg.evaluation.ladder_every == 0:
        for gname, aid in sorted(best_ids.items()):
            ctx.frozen.append([g, gname, aid])
        pop_best = max(ctx.agents.values(), key=lambda x: (x.score, x.id)).incumbent
        lseeds = ctx.seeds(g, "ladder", cfg.evaluation.ladder_games)
        ladder = {"current": pop_best, "vs_frozen": {f"{fg}:{fn}": round(ctx.crossplay(pop_best, fid, lseeds), 3)
                                                      for fg, fn, fid in ctx.frozen if fg < g},
                  "anchors": {k: round(v, 3) for k, v in ctx.anchors(pop_best, lseeds).items()}}
    incs = {ag.incumbent for ag in ctx.agents.values()}
    led = ctx.ledger.totals()
    by_tag = ctx.ledger.totals("tag")
    rec = {
        "generation": g,
        "game_params": {k: getattr(ctx.game_params, k) for k in ("hand_size", "max_information_tokens", "colors")},
        "agents": {a: {"group": ag.group, "incumbent": ag.incumbent, "score": round(ag.score, 4),
                       "anchor_score": round(ag.anchor_score, 4), "credit": round(ag.credit, 4),
                       "candidate": candidates.get(a, (None,))[0],
                       "parent": candidates[a][1] if a in candidates else None,
                       "candidate_score": (round(ctx.evals[candidates[a][0]].selfplay_mean, 4) if a in candidates else None),
                       "illegal_rate": ctx.evals[ag.incumbent].illegal_rate, "lines": ctx.evals[ag.incumbent].lines,
                       "start": starts.get(a)}
                   for a, ag in sorted(ctx.agents.items())},
        "groups": groups,
        "population_best": max(ag.score for ag in ctx.agents.values()),
        "population_mean": float(np.mean([ag.score for ag in ctx.agents.values()])),
        "between_group": between,
        "within_group_crossplay": float(np.mean(within)) if within else None,
        "diversity": {"distinct_incumbents": len(incs),
                      "code_clusters": code_clusters({i: ctx.store.get(i).code for i in incs})},
        "hc": _hc_record(ctx, g, starts),
        "ladder": ladder,
        "cost": {"tokens": led["tokens"], "spend_usd": led["spend_usd"], "new_spend_usd": led["new_spend_usd"],
                 "calls": led["calls"], "by_tag": {k: {"tokens": v["tokens"], "calls": v["calls"],
                                                       "spend_usd": v["spend_usd"]} for k, v in sorted(by_tag.items())}},
        "counters": {a: dict(sorted(ag.counters.items())) for a, ag in sorted(ctx.agents.items())},
        **extra,
        "volatile": {"wall_seconds": round(time.perf_counter() - t0, 3), "games_played": ctx.evaluator.games_played},
    }
    ctx.log("generations", rec)
    ctx.touch.flush()
    ctx.registry.prune(g)
    keep = {ag.incumbent for ag in ctx.agents.values()} | {m.artifact_id for m in ctx.outbox}
    keep |= {ag.previous for ag in ctx.agents.values() if ag.pending_check} - {None}
    ctx.evals = {k: v for k, v in ctx.evals.items() if k in keep}
    ctx.evaluator.clear_memo()
    ctx.generation = g
    return rec
