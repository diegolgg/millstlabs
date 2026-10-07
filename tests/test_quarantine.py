"""Overnight 2, step 2: the `quarantine_unverified` policy (files/prereg/C2-quarantine.md).

With quarantine on, a received message reaches the revision prompt only if it passed verification; a rejected,
duplicate or unverified message is withheld entirely (prose, evidence, verification line). With it off, every read
message is shown, as before."""

import json

import pytest

from culture.agents import agent as A
from culture.run import single_student as SS
from culture.run.config import ConfigError, ExperimentConfig, from_dict
from culture.run.context import RunContext
from culture.run.runner import run_config

SMALL = {"selfplay_games": 30, "crossplay_games": 0, "anchor_games": 0, "between_group_games": 0,
         "ladder_every": 0, "workers": 0}
FLAWED_TEXT = SS.TEACHERS["flawed_persuasive"]["text"]
PIERS_TEXT = SS.TEACHERS["piers"]["text"]
SELF_TEXT = SS.TEACHERS["self"]["text"]


def _spy(monkeypatch):
    prompts = []
    orig = A.produce

    def spy(ctx, agent, tag_, user, *a, **kw):
        if tag_ == "revise":
            prompts.append(user)
        return orig(ctx, agent, tag_, user, *a, **kw)

    monkeypatch.setattr(A, "produce", spy)
    return prompts


def _replay(tmp_path, monkeypatch, stratum, mask):
    spec = SS.load_spec({"name": "q_test", "k": 3, "replicates": [0], "strata": [stratum],
                         "base": {"experiment_seed": 7, "evaluation": SMALL}})
    spec["strata"][stratum]["org"]["verification"]["n"] = 30
    cfg = SS.replicate_config(spec, stratum, 0, tmp_path / "cache")
    ctx = RunContext(cfg, None)
    try:
        setup = SS.Setup(ctx, spec["teachers"], 1)
        prompts = _spy(monkeypatch)
        row = SS.replay(ctx, setup, mask)
        msgs = {j: setup.message(j) for j in range(3)}
        evidence = {j: ctx.evals[setup.payloads[j]].summary() for j in range(3)}
    finally:
        ctx.close()
    assert len(prompts) == 1
    return prompts[0], row, msgs, evidence


def test_rejected_message_is_withheld_from_the_revise_prompt_only_under_quarantine(tmp_path, monkeypatch):
    flawed = 1 << 1
    on, row_on, msgs, ev = _replay(tmp_path / "on", monkeypatch, "verified_quarantine", flawed)
    off, row_off, _, _ = _replay(tmp_path / "off", monkeypatch, "verified", flawed)
    assert row_on["decisions"] == row_off["decisions"] == {"T2": "rejected"}
    # quarantine on: none of the rejected message's text (prose, id, evidence, verification line) is in the prompt
    for piece in (FLAWED_TEXT, msgs[1].id, ev[1], "Your verification:"):
        assert piece not in on
    assert row_on["withheld"] == ["T2"]
    # quarantine off: the same message is shown
    for piece in (FLAWED_TEXT, msgs[1].id, ev[1], "Your verification:"):
        assert piece in off
    assert row_off["withheld"] == []


def test_passed_message_is_shown_and_duplicate_is_withheld(tmp_path, monkeypatch):
    on, row, msgs, _ = _replay(tmp_path / "all", monkeypatch, "verified_quarantine", 0b111)
    assert row["decisions"] == {"T1": "adopted", "T2": "rejected", "T3": "duplicate"}
    assert PIERS_TEXT in on and msgs[0].id in on  # Piers passed verification
    assert FLAWED_TEXT not in on and SELF_TEXT not in on  # rejected and duplicate (never verified)
    assert sorted(row["withheld"]) == ["T2", "T3"]
    off, row_off, _, _ = _replay(tmp_path / "all_off", monkeypatch, "verified", 0b111)
    assert PIERS_TEXT in off and FLAWED_TEXT in off and SELF_TEXT in off


def test_received_text_helper():
    from culture.artifacts.schema import TeachingMessage, Verification

    def msg(i, passed, n=10):
        v = None if n is None else Verification(f"m{i}", "r", 0, "SelfPlay", n, 1.0, 0.0, None, {}, 0.0, passed)
        return TeachingMessage(f"m{i}", f"s{i}", "r", 0, "a", f"text-{i}", None, verification=v)

    ms = [msg(0, True), msg(1, False), msg(2, True, n=None), msg(3, True, n=0)]
    text, withheld = A.received_text(ms, set(), quarantine=True)
    assert "text-0" in text and withheld == ["m1", "m2", "m3"]
    text, withheld = A.received_text(ms, set(), quarantine=False)
    assert all(f"text-{i}" in text for i in range(4)) and withheld == []
    assert text == "\n".join(A.ingest_text(m, False) for m in ms)  # unchanged behaviour when off


def test_generation_loop_withholds_rejected_sabotage(tmp_path, monkeypatch):
    base = {"name": "q_gen", "population": {"groups": 1, "agents_per_group": 3, "seed": 2, "warm_start": "piers"},
            "evaluation": {"selfplay_games": 12, "crossplay_games": 0, "anchor_games": 0, "between_group_games": 0,
                           "ladder_every": 0, "workers": 0},
            "org": {"routing": "broadcast_group", "verification": {"name": "selfplay", "n": 12}, "credit": "none"},
            "corpus": {"enabled": False}, "llm": {"stub": {"mode": "null"}}, "runner": {"generations": 2},
            "sabotage": {"epsilon": 1.0}}
    text = ExperimentConfig().sabotage.text
    seen = {}
    for flag in (False, True):
        prompts = _spy(monkeypatch)
        cfg = json.loads(json.dumps(base))
        cfg["org"]["quarantine_unverified"] = flag
        run_config(cfg, tmp_path / f"q{int(flag)}")
        monkeypatch.undo()
        seen[flag] = prompts
        msgs = [json.loads(x) for x in (tmp_path / f"q{int(flag)}" / "messages.jsonl").read_text().splitlines()]
        assert any(m["decision"] == "rejected" and m["sabotaged"] for m in msgs)
    received = [p for p in seen[False] if "### Message" in p]
    assert received and all(text in p for p in received)
    assert seen[True] and not any(text in p or "### Message" in p for p in seen[True])
    recs = [json.loads(x) for x in (tmp_path / "q1" / "generations.jsonl").read_text().splitlines()]
    assert any(c.get("withheld", 0) > 0 for r in recs for c in (r.get("revise_context") or {}).values())


def test_config_flag_default_off_and_digest_stable():
    off = from_dict({})
    assert off.org.quarantine_unverified is False
    d = off.to_dict()
    assert "quarantine_unverified" in d["org"]
    on = from_dict({"org": {"quarantine_unverified": True}})
    assert on.digest() != off.digest()
    # the digest of a default config does not depend on the new flag's presence
    import hashlib

    d.pop("debug"), d["runner"].pop("generations"), d["runner"].pop("wall_clock_budget_s")
    d["evaluation"].pop("workers"), d["org"].pop("quarantine_unverified")
    assert off.digest() == hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    with pytest.raises(ConfigError):
        from_dict({"org": {"quarantine_unverified": "yes"}})


def test_c2_summary_on_a_synthetic_game():
    from culture.analysis.quarantine import c2_summary

    teachers = ["piers", "flawed_persuasive", "self"]
    rows = []
    for r in range(6):
        for m in range(8):
            piers, flawed, me = m & 1, m >> 1 & 1, m >> 2 & 1
            # verified: Piers +10, Flawed's text -3 (never adopted), the placebo +-1 by seed
            rows.append({"stratum": "verified", "replicate": r, "mask": m, "adopted": [0] if piers else [],
                         "y": 5 + 10 * piers - (3 + 0.1 * r) * flawed + (-1) ** r * me})
            # quarantine: only the passed message (Piers) reaches the prompt
            rows.append({"stratum": "verified_quarantine", "replicate": r, "mask": m, "adopted": [0] if piers else [],
                         "y": 5 + (10 + 0.2 * r) * piers})
    s = c2_summary(rows, teachers)
    assert s["seeds"] == list(range(6)) and s["collapse_max_abs_over_seeds"] == 0.0
    assert [p["primary"] for p in s["per_seed"].values()] == pytest.approx([3 + 0.1 * r for r in range(6)])
    assert [p["secondary"] for p in s["per_seed"].values()] == pytest.approx([0.2 * r for r in range(6)])
    assert s["tau_flawed_quarantine"]["iqm"] == pytest.approx(0.0)
    assert s["decision_rules"]["removes_influence"]["met_on_these_seeds"]
    assert s["decision_rules"]["removes_influence"]["verdict"] == "not judged (pilot)"  # one model block
    assert s["primary"]["R_for_80pct_power"] is not None and s["primary"]["R_capped"] <= 29
