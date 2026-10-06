"""LLM layer: cache modes, ledger, simulated prompt caching, stub determinism, parsing."""

import json

import pytest

from culture.bots.anchors import PIERS_RULES, parse_rules, rulebot_source
from culture.bots.runner import BotSpec, check_source, play_game
from culture.game.hanabi import HanabiParams
from culture.llm import prompts
from culture.llm.backend import CostLedger, Request, cost_usd, Usage
from culture.llm.cache import CacheMiss, CachedBackend, CallCache, request_key
from culture.llm.parsing import ParseError, apply_search_replace, parse_artifact
from culture.llm.stub_backend import StubBackend, StubConfig


def req(tag="revise", seed_tag="run/a0/g1", user=None, system="frozen prefix " * 50, run="r1"):
    user = user if user is not None else "```python\n" + rulebot_source(PIERS_RULES) + "```"
    return Request(system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                   messages=[{"role": "user", "content": user}], model="claude-haiku-4-5", tag=tag,
                   seed_tag=seed_tag, meta={"run": run, "group": "g0", "agent": "a0", "generation": 1})


def test_replay_reproduces_byte_identical_responses(tmp_path):
    cache = CallCache(tmp_path / "cache")
    rec = CachedBackend(StubBackend(), cache, mode="record")
    a = rec.complete(req())
    rep = CachedBackend(StubBackend(), cache, mode="replay_strict")
    b = rep.complete(req())
    assert a.text == b.text and b.cached is False  # same run id: replay of its own call, counted as spend
    other = CachedBackend(StubBackend(), cache, mode="replay")
    c = other.complete(req(run="r2"))  # same key (meta is not hashed), different run: a true cache hit
    assert c.text == a.text and c.cached is True
    assert other.ledger.rows[-1]["new_spend_usd"] == 0.0 and other.ledger.rows[-1]["spend_usd"] > 0


def test_replay_strict_raises_on_miss(tmp_path):
    rep = CachedBackend(StubBackend(), CallCache(tmp_path / "c"), mode="replay_strict")
    with pytest.raises(CacheMiss):
        rep.complete(req())


def test_record_mode_calls_and_stores(tmp_path):
    cache = CallCache(tmp_path / "c")
    be = CachedBackend(StubBackend(), cache, mode="record")
    be.complete(req())
    be.complete(req(seed_tag="run/a1/g1"))
    assert len(cache) == 2 and be.misses == 2


def test_ledger_totals_equal_sum_of_calls(tmp_path):
    ledger = CostLedger(tmp_path / "ledger.jsonl")
    be = CachedBackend(StubBackend(), None, mode="off", ledger=ledger)
    resps = [be.complete(req(seed_tag=f"run/a{i}/g1", tag=t)) for i, t in enumerate(["author", "revise", "teach"] * 3)]
    tot = ledger.totals()
    assert tot["calls"] == 9
    assert tot["input_tokens"] == sum(r.usage.input_tokens for r in resps)
    assert tot["output_tokens"] == sum(r.usage.output_tokens for r in resps)
    assert tot["tokens"] == sum(r.usage.total for r in resps)
    assert abs(tot["spend_usd"] - sum(r.cost_usd for r in resps)) < 1e-12
    by_tag = ledger.totals("tag")
    assert set(by_tag) == {"author", "revise", "teach"} and sum(v["calls"] for v in by_tag.values()) == 9
    assert CostLedger(tmp_path / "ledger.jsonl").totals() == tot  # persisted


def test_prompt_cache_hits_on_repeated_frozen_prefix():
    be = StubBackend()
    first = be.complete(req(seed_tag="x/1"))
    second = be.complete(req(seed_tag="x/2"))
    assert first.usage.cache_creation_input_tokens > 0 and first.usage.cache_read_input_tokens == 0
    assert second.usage.cache_read_input_tokens == first.usage.cache_creation_input_tokens
    assert second.cost_usd < first.cost_usd
    other = be.complete(req(seed_tag="x/3", system="a different prefix"))
    assert other.usage.cache_read_input_tokens == 0


def test_cost_table():
    u = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd("claude-haiku-4-5", u) == 6.0
    assert cost_usd("claude-haiku-4-5", u, batch=True) == 3.0
    assert cost_usd("unknown-model", u) is None


def test_stub_is_deterministic_and_seed_tag_sensitive():
    a = StubBackend().complete(req(tag="author", user="write a bot"))
    b = StubBackend().complete(req(tag="author", user="write a bot"))
    assert a.text == b.text
    texts = {StubBackend().complete(req(tag="author", user="write a bot", seed_tag=f"r/a{i}")).text for i in range(8)}
    assert len(texts) > 4


def test_stub_artifacts_parse_compile_and_play():
    be = StubBackend(StubConfig(invalid_rate=0.0, diff_rate=0.5))
    code = rulebot_source(PIERS_RULES)
    seen_modes = set()
    for i in range(12):
        r = be.complete(req(seed_tag=f"r/a{i}/g2", user="Current bot\n```python\n" + code + "```"))
        new_code, conv, mode = parse_artifact(r.text, code, "old conventions")
        seen_modes.add(mode)
        assert check_source(new_code) == [] and parse_rules(new_code) is not None and conv
        res = play_game(HanabiParams(), [BotSpec(f"k{i}", new_code)] * 2, 0)
        assert res.illegal == [0, 0]
    assert seen_modes == {"full", "diff"}


def test_stub_invalid_then_repair():
    be = StubBackend(StubConfig(invalid_rate=1.0))
    r = be.complete(req(tag="author", user="write"))
    code, _, _ = parse_artifact(r.text, None, None)
    assert check_source(code)  # broken on purpose
    fixed = be.complete(req(tag="repair", user=prompts.render("repair", error="SyntaxError", code=code)))
    code2, _, _ = parse_artifact(fixed.text, code, "")
    assert check_source(code2) == []


def test_parse_errors_and_search_replace():
    with pytest.raises(ParseError):
        parse_artifact("no tags here", "x", "y")
    with pytest.raises(ParseError):
        apply_search_replace("abc", "<<<<<<< SEARCH\nzzz\n=======\nq\n>>>>>>> REPLACE")
    assert apply_search_replace("a = 1\nb = 2\n", "<<<<<<< SEARCH\nb = 2\n=======\nb = 3\n>>>>>>> REPLACE") == "a = 1\nb = 3\n"
    _, conv, _ = parse_artifact("<bot>\n```python\nx\n```\n</bot><conventions_delta>more</conventions_delta>", None, "base")
    assert conv.startswith("base") and conv.strip().endswith("more")


def test_request_key_ignores_meta_but_not_seed_tag():
    a, b = req(run="r1"), req(run="r9")
    assert request_key("stub", a) == request_key("stub", b)
    assert request_key("stub", req(seed_tag="other")) != request_key("stub", a)


def test_prompt_templates_hash_and_never_mention_credit_rules():
    hashes = prompts.prompt_hashes()
    assert set(hashes) == set(prompts.TEMPLATES)
    for name in prompts.TEMPLATES:
        text = (prompts.PROMPT_DIR / f"{name}.md").read_text().lower()
        for word in ("credit", "allocation", "selection", "budget share"):
            assert word not in text, (name, word)
    sp = prompts.system_prompt(HanabiParams().describe(), __import__("culture.bots.runner", fromlist=["x"]).SandboxLimits())
    assert "legal_moves" in sp and "${" not in sp
    json.loads(json.dumps(prompts.OBSERVATION_EXAMPLE))
