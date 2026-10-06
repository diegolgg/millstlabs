"""One agent's LLM-facing actions: author, revise (with one repair call), teach, merge.

The ordering of steps within a generation lives in run/generation.py only; this module provides the pieces.
Every call goes through the cached backend (record/replay + ledger) and debits the agent's budget.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from ..artifacts.schema import Artifact, Evaluation, TeachingMessage
from ..bots.runner import BotSpec, check_source, play_game
from ..llm import prompts
from ..llm.backend import Request, Response
from ..llm.parsing import ParseError, parse_artifact, tag
from ..org.base import AgentState

if TYPE_CHECKING:
    from ..run.context import RunContext


def system_blocks(ctx: "RunContext") -> list[dict]:
    text = prompts.system_prompt(ctx.game_params.describe(), ctx.limits, ctx.variant_note)
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def call(ctx: "RunContext", agent: AgentState, tag_: str, user: str, generation: int, attempt: int = 0) -> Response | None:
    """One budgeted LLM call. Returns None when the agent's group cannot afford the call (the estimated cost is
    reserved before sending, so a group never overspends its per-generation budget); the refusal is logged."""
    cfg = ctx.cfg.llm
    req = Request(system=system_blocks(ctx), messages=[{"role": "user", "content": user}], model=cfg.model,
                  tag=tag_, seed_tag=f"{ctx.seed_prefix}/{agent.id}/g{generation}/{tag_}/{attempt}",
                  max_tokens=cfg.max_tokens, effort=cfg.effort,
                  meta={"run": ctx.run_id, "group": agent.group, "agent": agent.id, "generation": generation})
    ok, reason = ctx.reserve(agent, req)
    if not ok:
        agent.bump("skipped_for_budget")
        ctx.log_refusal(agent, req, reason)
        return None
    resp = ctx.backend.complete(req)
    ctx.charge(agent, resp)
    agent.bump(f"calls_{tag_}")
    return resp


def _admissible(ctx: "RunContext", code: str) -> str | None:
    problems = check_source(code, ctx.limits.max_lines)
    if problems:
        return "; ".join(problems[:3])
    # a cheap smoke game: catches import-time and reset/act crashes before the artifact is evaluated
    key = "smoke:" + hashlib.sha256(code.encode()).hexdigest()[:16]
    res = play_game(ctx.game_params, [BotSpec(key, code)] * 2, 0, ctx.limits)
    if res.errors[0] and res.illegal[0] == res.moves[0]:
        return res.errors[0]
    return None


def produce(ctx: "RunContext", agent: AgentState, tag_: str, user: str, generation: int, parent: Artifact | None,
            parents: list[str], teaching_sources: list[str], origin: str) -> Artifact | None:
    """LLM call -> parse -> sandbox check -> (one repair call) -> Artifact, or None for a failed revision."""
    resp = call(ctx, agent, tag_, user, generation)
    if resp is None:
        return None
    cur_code = parent.code if parent else None
    cur_conv = parent.conventions if parent else None
    for attempt in range(ctx.cfg.llm.repair_attempts + 1):
        try:
            code, conv, mode = parse_artifact(resp.text, cur_code, cur_conv)
            err = _admissible(ctx, code)
        except ParseError as e:
            code, conv, mode, err = (cur_code or ""), (cur_conv or ""), "full", f"ParseError: {e}"
        if err is None:
            art = Artifact.make(code, conv, author=agent.id, group=agent.group, generation=generation,
                                parents=list(dict.fromkeys(parents)), teaching_sources=list(teaching_sources),
                                origin=origin, edit_mode=mode)
            if attempt:
                agent.bump("repaired")
            return art
        if attempt >= ctx.cfg.llm.repair_attempts:
            break
        agent.bump("repair_calls")
        resp = call(ctx, agent, "repair", prompts.render("repair", error=err, code=code), generation, attempt + 1)
        if resp is None:
            break
    agent.bump("failed_revisions")
    return None


def author_prompt(agent: AgentState, generation: int) -> str:
    return prompts.render("author", generation=generation, agent=agent.id)


def failure_text(ctx: "RunContext", art: Artifact, ev: Evaluation | None) -> str:
    """Compact traces of the k lowest-scoring self-play games (replayed deterministically with tracing on)."""
    if ev is None or not ev.selfplay_scores:
        return "(no evaluation yet)"
    k = ctx.cfg.evaluation.failure_traces
    order = sorted(range(len(ev.selfplay_scores)), key=lambda i: (ev.selfplay_scores[i], i))[:k]
    spec = ctx.spec(art.id)
    out = []
    for i in order:
        seed = ev.seed_base + i
        r = play_game(ctx.game_params, [spec, spec], seed, ctx.limits, trace=True)
        lines = [f"Game seed {seed}: score {r.score}, {r.turns} turns."]
        shown = 0
        for j, t in enumerate(r.trace):
            o, m = t["obs"], t["move"]
            mv = m["type"] + (f" slot {m['card_index']}" if "card_index" in m else
                              f" -> P{m['target']} {m.get('color', m.get('rank'))}")
            lines.append(f"t{o['turn']} P{o['me']} {mv}{'' if t['ok'] else ' (fallback: illegal/timeout)'}")
            nxt = r.trace[j + 1]["obs"]["life_tokens"] if j + 1 < len(r.trace) else o["life_tokens"]
            if m["type"] == "PLAY" and nxt < o["life_tokens"] and shown < 1:
                lines.append("  misfire; observation before it: " + json.dumps(_trim_obs(o)))
                shown += 1
        out.append("\n".join(lines[:80]))
    return "\n\n".join(out)


def _trim_obs(o: dict) -> dict:
    d = {k: o[k] for k in ("turn", "info_tokens", "life_tokens", "fireworks", "deck_size")}
    d["my_hand"] = [{"hints": c["hints"], "possible": c["possible"]} for c in o["my_hand"]]
    d["hands"] = o["hands"]
    return d


def revise_prompt(ctx: "RunContext", agent: AgentState, parent: Artifact, ev: Evaluation | None, adopted_text: str,
                  corpus_text: str, generation: int) -> str:
    return prompts.render("revise", generation=generation, agent=agent.id, conventions=parent.conventions.strip(),
                          code=parent.code.rstrip(), evaluation=ev.summary() if ev else "(not evaluated yet)",
                          failures=failure_text(ctx, parent, ev), adopted=adopted_text or "(none)",
                          corpus=corpus_text or "(none)")


def ingest_text(msg: TeachingMessage, adopted: bool) -> str:
    ev = msg.evidence.summary() if msg.evidence else "(none)"
    v = msg.verification
    vt = (f"Your verification: payload {v.payload_score:.2f} vs your incumbent {v.incumbent_score:.2f} on "
          f"{v.n_games} games ({'passed' if v.passed else 'failed'}).") if v and v.n_games else ""
    return prompts.render("ingest", message_id=msg.id, sender=msg.sender,
                          adopted_note=" (adopted: it is now your current bot)" if adopted else "",
                          delta_text=msg.delta_text.strip(), evidence=ev, verification=vt)


def teach(ctx: "RunContext", sender: AgentState, receivers: list[str], art: Artifact, ev: Evaluation | None,
          generation: int) -> str | None:
    user = prompts.render("teach", generation=generation, agent=sender.id, receivers=", ".join(receivers),
                          conventions=art.conventions.strip() + "\n\n```python\n" + art.code.rstrip() + "\n```",
                          evaluation=ev.summary() if ev else "(none)")
    resp = call(ctx, sender, "teach", user, generation)
    if resp is None:
        return None
    return tag(resp.text, "delta") or resp.text.strip()[:2000]


def merge(ctx: "RunContext", agent: AgentState, mine: Artifact, theirs: Artifact, ev_a: Evaluation | None,
          ev_b: Evaluation | None, sender: str, message_id: str, generation: int) -> Artifact | None:
    user = prompts.render("merge", generation=generation, agent=agent.id, sender=sender,
                          eval_a=ev_a.summary() if ev_a else "(none)", eval_b=ev_b.summary() if ev_b else "(none)",
                          code_a=mine.code.rstrip(), code_b=theirs.code.rstrip(), conventions_a=mine.conventions,
                          conventions_b=theirs.conventions)
    return produce(ctx, agent, "merge", user, generation, mine, [mine.id, theirs.id], [message_id], "merge")
