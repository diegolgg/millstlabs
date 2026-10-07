"""Hosted OpenAI-compatible backend: spend guard, failure semantics, key hygiene, concurrency and the local path's
byte-identity, all against a fake HTTP server (no network, no paid call)."""

import importlib.util
import json
import multiprocessing as mp
import random
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

from culture.bots.anchors import RULE_LISTS, anchor_source, conventions_for
from culture.llm.backend import CostLedger, Request
from culture.llm.cache import CachedBackend, CallCache, request_key
from culture.llm.openai_compat import BackendError, OpenAICompatBackend, OpenAICompatConfig
from culture.llm.spend import BudgetExhausted, SpendGuard, UncertainOutcome, estimate_usd, usage_cost_usd
from culture.run.config import ConfigError, from_dict

ROOT = Path(__file__).resolve().parents[1]
KEY_ENV = "CULTURE_TEST_HOSTED_KEY"
SECRET = "sk-test-0123456789abcdef-NEVER-WRITTEN"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
P_IN, P_OUT = 0.5, 2.0
PROMPT_TOKENS, COMPLETION_TOKENS = 1234, 567
COST = usage_cost_usd(PROMPT_TOKENS, COMPLETION_TOKENS, P_IN, P_OUT)
BOT = anchor_source("iggi")
REPLY = ("Kept the rule order.\n<bot>\n```python\n" + BOT + "```\n</bot>\n<conventions>\n"
         + conventions_for(RULE_LISTS["iggi"]) + "</conventions>\n")


class _Handler(BaseHTTPRequestHandler):
    """Replies from `server.script` (a list of HTTP statuses, consumed in order; 200 when empty). A 200 echoes the last
    user message when `server.echo` is set, else `server.reply`; it carries an id, a fingerprint and fixed usage."""

    def do_POST(self):  # noqa: N802
        srv = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with srv.lock:
            srv.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
            status = srv.script.pop(0) if srv.script else 200
            n = len(srv.requests)
        if srv.guard_path:  # what the spend database holds while this request is in flight
            db = sqlite3.connect(srv.guard_path)
            srv.seen_states.append(db.execute("SELECT state, reserved FROM calls WHERE state='pending'").fetchall())
            db.close()
        if srv.jitter:
            time.sleep(random.random() * srv.jitter)
        if status != 200:
            data = json.dumps({"error": {"message": f"scripted {status}"}}).encode()
        else:
            text = body["messages"][-1]["content"] if srv.echo else srv.reply
            out = {"id": f"req-{n}", "object": "chat.completion", "model": body["model"], "system_fingerprint": "fp-1",
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS,
                             "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS}}
            data = json.dumps(out).encode() if status == 200 else b""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if status == 429:
            self.send_header("Retry-After", "0")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests, srv.script, srv.reply, srv.echo, srv.jitter = [], [], REPLY, False, 0.0
    srv.lock, srv.guard_path, srv.seen_states = threading.Lock(), None, []
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(OpenAICompatBackend, "backoff_base_s", 0.001)
    monkeypatch.setattr(OpenAICompatBackend, "backoff_cap_s", 0.01)


def _url(srv):
    return f"http://127.0.0.1:{srv.server_port}/v1"


def _hosted(srv, tmp_path, monkeypatch, **kw):
    monkeypatch.setenv(KEY_ENV, SECRET)
    cfg = dict(base_url=_url(srv), api_key_env=KEY_ENV, price_in_per_mtok=P_IN, price_out_per_mtok=P_OUT,
               max_usd=1.0, spend_file=str(tmp_path / "spend.sqlite"))
    cfg.update(kw)
    return OpenAICompatBackend(OpenAICompatConfig(**cfg))


def _req(text="hello", run="r1", model=MODEL):
    return Request(system=[{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}],
                   messages=[{"role": "user", "content": text}], model=model, tag="revise",
                   seed_tag="t/g1/revise/0", max_tokens=4096, meta={"run": run})


def _est(req):
    return estimate_usd(req.prompt_text(), req.max_tokens, P_IN, P_OUT)


# ---------------------------------------------------------------- reserve, settle, cap
def test_reserves_before_send_and_settles_after(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch)
    server.guard_path = str(tmp_path / "spend.sqlite")
    resp = be.complete(_req())
    assert server.seen_states == [[("pending", pytest.approx(_est(_req())))]]  # reserved while in flight
    assert resp.cost_usd == pytest.approx(COST) and resp.provider_request_id == "req-1"
    assert resp.system_fingerprint == "fp-1"
    t = be.guard.totals()
    assert t["spent"] == pytest.approx(COST) and t["reserved"] == 0 and t["uncertain"] == 0 and t["calls"] == 1
    assert _est(_req()) >= COST  # the estimate bounds the real cost for this request


def test_cache_read_tokens_are_charged_at_the_full_input_rate(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch)
    from culture.llm.backend import Usage

    assert be.price(Usage(input_tokens=1000, cache_read_input_tokens=234, output_tokens=567)) == pytest.approx(COST)


def test_cap_refuses_and_sends_nothing(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch, max_usd=_est(_req()) / 2)
    with pytest.raises(BudgetExhausted, match="No request was sent"):
        be.complete(_req())
    assert server.requests == [] and be.guard.totals()["committed"] == 0


def test_missing_key_sends_nothing(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch)
    monkeypatch.delenv(KEY_ENV)
    with pytest.raises(BackendError, match=KEY_ENV):
        be.complete(_req())
    assert server.requests == [] and be.guard.totals()["committed"] == 0


def test_cap_is_cumulative_across_guard_instances(tmp_path):
    p = tmp_path / "s.sqlite"
    g1 = SpendGuard(p, 1.0)
    g1.reserve("a", 0.5)
    g1.settle("a", 0.75)  # actual above the estimate is recorded as is
    g2 = SpendGuard(p, 1.0)
    assert g2.totals()["spent"] == 0.75 and g2.spent_or_reserved() == 0.75
    with pytest.raises(BudgetExhausted):
        g2.reserve("b", 0.5)
    g2.reserve("b", 0.25)
    with pytest.raises(BudgetExhausted):
        SpendGuard(p, 1.0).reserve("c", 0.001)
    assert SpendGuard(p, 2.0).totals()["committed"] == 1.0


def _reserve_in_process(args):
    path, key, start_at = args
    time.sleep(max(0.0, start_at - time.time()))
    try:
        SpendGuard(path, 1.25).reserve(key, 0.125)  # exact binary fractions: the cap admits exactly 10
        return True
    except BudgetExhausted:
        return False


def test_cap_is_cumulative_across_processes(tmp_path):
    path = str(tmp_path / "s.sqlite")
    SpendGuard(path, 1.25)
    start = time.time() + 4.0
    with mp.get_context("spawn").Pool(20) as pool:
        ok = pool.map(_reserve_in_process, [(path, f"k{i}", start) for i in range(20)])
    assert sum(ok) == 10
    t = SpendGuard(path, 1.25).totals()
    assert t["reserved"] == 1.25 and t["committed"] == 1.25


# ---------------------------------------------------------------- failure semantics
def test_401_releases_without_retry(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch)
    server.script = [401]
    with pytest.raises(BackendError, match="HTTP 401"):
        be.complete(_req())
    assert len(server.requests) == 1
    t = be.guard.totals()
    assert t["committed"] == 0 and t["released_calls"] == 1
    be.complete(_req())  # a released key can be sent again
    assert len(server.requests) == 2 and be.guard.totals()["calls"] == 1


@pytest.mark.parametrize("status", [400, 403, 404])
def test_other_rejections_release(server, tmp_path, monkeypatch, status):
    be = _hosted(server, tmp_path, monkeypatch)
    server.script = [status]
    with pytest.raises(BackendError, match=f"HTTP {status}"):
        be.complete(_req())
    assert len(server.requests) == 1 and be.guard.totals()["committed"] == 0


def test_429_retries_unbilled_then_succeeds(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch)
    server.script = [429, 429]
    resp = be.complete(_req())
    assert len(server.requests) == 3 and resp.cost_usd == pytest.approx(COST)
    t = be.guard.totals()
    assert t["spent"] == pytest.approx(COST) and t["calls"] == 1 and t["released_calls"] == 2
    assert t["committed"] == pytest.approx(COST)


def test_429_gives_up_after_the_retry_limit(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch, max_rate_limit_retries=2)
    server.script = [429] * 5
    with pytest.raises(BackendError, match="gave up after 2 retries"):
        be.complete(_req())
    assert len(server.requests) == 3 and be.guard.totals()["committed"] == 0


def test_connection_refused_is_unbilled(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV, SECRET)
    be = OpenAICompatBackend(OpenAICompatConfig(base_url="http://127.0.0.1:9/v1", api_key_env=KEY_ENV,
                                                price_in_per_mtok=P_IN, price_out_per_mtok=P_OUT, max_usd=1.0,
                                                spend_file=str(tmp_path / "s.sqlite"), max_rate_limit_retries=1,
                                                request_timeout_s=2))
    with pytest.raises(BackendError, match="unbilled"):
        be.complete(_req())
    t = be.guard.totals()
    assert t["committed"] == 0 and t["released_calls"] == 2


def test_500_is_uncertain_and_retry_needs_the_flag(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch)
    server.script = [500]
    with pytest.raises(BackendError, match="retry_uncertain"):
        be.complete(_req())
    assert len(server.requests) == 1
    est = _est(_req())
    assert be.guard.totals()["uncertain"] == pytest.approx(est)
    with pytest.raises(UncertainOutcome, match="retry_uncertain"):  # identical request: refused, nothing sent
        be.complete(_req())
    assert len(server.requests) == 1
    be.complete(_req("a different request"))  # other requests are unaffected
    assert len(server.requests) == 2
    retry = _hosted(server, tmp_path, monkeypatch, retry_uncertain=True)
    resp = retry.complete(_req())
    assert len(server.requests) == 3 and resp.cost_usd == pytest.approx(COST)
    t = retry.guard.totals()
    assert t["uncertain"] == pytest.approx(est) and t["uncertain_calls"] == 1  # the old reservation is retained
    assert t["spent"] == pytest.approx(2 * COST) and t["committed"] == pytest.approx(2 * COST + est)


def test_malformed_body_is_uncertain(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch)
    server.script = [299]  # a 2xx with an empty body
    with pytest.raises(BackendError, match="malformed"):
        be.complete(_req())
    assert be.guard.totals()["uncertain_calls"] == 1


def test_runs_sharing_a_spend_file_do_not_block_each_other(server, tmp_path, monkeypatch):
    be = _hosted(server, tmp_path, monkeypatch)
    server.script = [500]
    with pytest.raises(BackendError):
        be.complete(_req(run="run-a"))
    be.complete(_req(run="run-b"))  # same request in another run: a different guard key
    assert len(server.requests) == 2


# ---------------------------------------------------------------- key hygiene
def test_key_is_sent_but_never_written(server, tmp_path, monkeypatch):
    from culture.run.runner import run_config

    monkeypatch.setenv(KEY_ENV, SECRET)
    cfg = from_dict({"name": "hosted_fake", "population": {"groups": 1, "agents_per_group": 1, "warm_start": "piers"},
                     "evaluation": {"selfplay_games": 4, "crossplay_games": 0, "anchor_games": 2,
                                    "between_group_games": 0, "ladder_every": 0, "workers": 0},
                     "llm": {"backend": "openai_compat", "model": MODEL, "base_url": _url(server), "max_tokens": 4096,
                             "api_key_env": KEY_ENV, "price_in_per_mtok": P_IN, "price_out_per_mtok": P_OUT,
                             "max_usd": 1.0},
                     "budget": {"unit": "usd", "per_group_per_generation": 0.5},
                     "runner": {"generations": 2}})
    out = tmp_path / "r"
    recs = run_config(cfg, out)
    assert recs[-1]["agents"]["g0a0"]["candidate"] is not None
    assert server.requests and all(r["auth"] == f"Bearer {SECRET}" for r in server.requests)
    files = [p for p in out.rglob("*") if p.is_file()]
    assert (out / "spend.sqlite") in files and any("llm_cache" in str(p) for p in files)
    for p in files:
        assert SECRET.encode() not in p.read_bytes(), p
    assert SECRET not in cfg.digest() and SECRET not in json.dumps(cfg.to_dict())
    man = json.loads((out / "manifest.json").read_text())
    prov = man["llm_provider"]
    assert prov["api_key_env"] == KEY_ENV and prov["hosted"] and prov["base_url"] == _url(server)
    assert prov["model"] == MODEL and prov["price_in_per_mtok"] == P_IN and prov["max_usd"] == 1.0
    led = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines()]
    rev = [r for r in led if r["tag"] == "revise"]
    assert rev and all(r["spend_usd"] == pytest.approx(COST) and r["provider_request_id"].startswith("req-")
                       and r["system_fingerprint"] == "fp-1" for r in rev)
    assert CostLedger(out / "ledger.jsonl").totals()["real_spend_usd"] == pytest.approx(COST * len(rev))
    assert SpendGuard(out / "spend.sqlite", 1.0).totals()["spent"] == pytest.approx(COST * len(rev))


# ---------------------------------------------------------------- concurrency
def test_concurrent_batch_keeps_request_order(server, tmp_path, monkeypatch):
    server.echo, server.jitter = True, 0.05
    be = _hosted(server, tmp_path, monkeypatch, concurrency=4)
    reqs = [_req(f"message {i}") for i in range(12)]
    assert [r.text for r in be.complete_batch(reqs)] == [f"message {i}" for i in range(12)]
    assert be.guard.totals()["calls"] == 12
    # through the cache layer: misses run on the pool, ledger rows land in request order, a repeat is sent once
    ledger = CostLedger(tmp_path / "ledger.jsonl")
    cb = CachedBackend(be, CallCache(tmp_path / "cache"), "replay", ledger)
    reqs2 = [_req(f"second {i}") for i in range(6)] + [_req("second 0")]
    n0 = len(server.requests)
    assert [r.text for r in cb.complete_batch(reqs2)] == [f"second {i}" for i in range(6)] + ["second 0"]
    assert len(server.requests) - n0 == 6
    assert [r["request_key"] for r in ledger.rows] == [cb._key(r) for r in reqs2]


def test_concurrency_needs_a_hosted_provider():
    with pytest.raises(ConfigError, match="concurrency"):
        from_dict({"llm": {"backend": "openai_compat", "model": "m", "concurrency": 4}})


# ---------------------------------------------------------------- the local path is unchanged
# Recorded before the hosted backend existed (commit 2cb0596) by computing the cache key of this exact request.
LOCAL_KEY_BEFORE = "73135187cb56e9681d0497929f4e7d9aa9640a278e199291468a4679d221247a"


def test_local_config_is_byte_identical(server, tmp_path):
    cfg = from_dict({"llm": {"backend": "openai_compat", "model": "mlx-community/Qwen3.6-35B-A3B-4bit", "seed": 3,
                             "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}})
    from culture.run.context import make_backend

    be = make_backend(cfg, tmp_path)
    req = Request(system=[{"type": "text", "text": "SYS"}], messages=[{"role": "user", "content": "hi"}],
                  model=cfg.llm.model, tag="revise", seed_tag="x/g1", max_tokens=100)
    assert request_key(be.name, req, be.key_extra()) == LOCAL_KEY_BEFORE
    assert cfg.digest() == "5d9996d6ac8358ae"  # also recorded at 2cb0596
    assert be.body(req) == {"model": "mlx-community/Qwen3.6-35B-A3B-4bit", "max_tokens": 100, "temperature": 0.0,
                            "seed": 3, "stream": False, "chat_template_kwargs": {"enable_thinking": False},
                            "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]}
    assert be.guard is None
    local = OpenAICompatBackend(OpenAICompatConfig(base_url=_url(server), seed=7))
    cache = CallCache(tmp_path / "c")
    resp = CachedBackend(local, cache, "replay").complete(_req(model="mlx-community/Qwen3.6-35B-A3B-4bit"))
    assert server.requests[-1]["auth"] == "Bearer local" and resp.cost_usd == 0.0
    assert resp.provider_request_id is None and resp.system_fingerprint is None
    entry = json.loads(next((tmp_path / "c").glob("*/*.json")).read_text())
    assert set(entry["response"]) == {"text", "usage", "cost_usd", "model", "backend", "cached", "request_key",
                                      "batch"}
    assert not list(tmp_path.rglob("*.sqlite"))


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("llm,match", [
    ({"api_key_env": KEY_ENV}, "price and a cap"),
    ({"api_key_env": KEY_ENV, "price_in_per_mtok": 0.1, "price_out_per_mtok": 0.2}, "price and a cap"),
    ({"api_key_env": KEY_ENV, "price_in_per_mtok": 0.1, "max_usd": 1.0}, "price and a cap"),
    ({"api_key_env": KEY_ENV, "price_in_per_mtok": -1.0, "price_out_per_mtok": 0.2, "max_usd": 1.0}, "non-negative"),
    ({"max_usd": -1.0}, "max_usd"),
    ({"max_rate_limit_retries": -1}, "max_rate_limit_retries"),
    ({"model": "claude-haiku-4-5"}, "claude"),
])
def test_validation_errors(llm, match):
    with pytest.raises(ConfigError, match=match):
        from_dict({"llm": {"backend": "openai_compat", "model": MODEL, **llm}})


def test_api_key_env_needs_openai_compat():
    with pytest.raises(ConfigError, match="openai_compat"):
        from_dict({"llm": {"backend": "stub", "api_key_env": KEY_ENV, "price_in_per_mtok": 0.1,
                           "price_out_per_mtok": 0.2, "max_usd": 1.0}})


def test_operational_fields_do_not_change_the_digest():
    base = {"backend": "openai_compat", "model": MODEL, "api_key_env": KEY_ENV, "price_in_per_mtok": 0.1,
            "price_out_per_mtok": 0.2, "max_usd": 1.0}
    a = from_dict({"llm": base})
    b = from_dict({"llm": {**base, "max_usd": 9.0, "spend_file": "/x/s.sqlite", "concurrency": 8,
                           "retry_uncertain": True, "max_rate_limit_retries": 2}})
    c = from_dict({"llm": {**base, "price_out_per_mtok": 0.3}})
    assert a.digest() == b.digest() != c.digest()


def test_hosted_backend_needs_a_spend_file(monkeypatch):
    from culture.run.context import make_backend

    cfg = from_dict({"llm": {"backend": "openai_compat", "model": MODEL, "api_key_env": KEY_ENV,
                             "price_in_per_mtok": 0.1, "price_out_per_mtok": 0.2, "max_usd": 1.0}})
    with pytest.raises(BackendError, match="spend file"):
        make_backend(cfg, None)


# ---------------------------------------------------------------- experiments share one spend file
def test_experiment_shares_one_spend_file(server, tmp_path, monkeypatch):
    from culture.run.experiment import run_experiment

    monkeypatch.setenv(KEY_ENV, SECRET)  # inherited by the spawned workers
    spec = {"base": {"name": "hosted_exp", "population": {"groups": 1, "agents_per_group": 1, "warm_start": "piers"},
                     "evaluation": {"selfplay_games": 4, "crossplay_games": 0, "anchor_games": 2,
                                    "between_group_games": 0, "ladder_every": 0, "workers": 0},
                     "corpus": {"enabled": False},
                     "llm": {"backend": "openai_compat", "model": MODEL, "base_url": _url(server), "max_tokens": 4096,
                             "api_key_env": KEY_ENV, "price_in_per_mtok": P_IN, "price_out_per_mtok": P_OUT,
                             "max_usd": 1.0},
                     "runner": {"generations": 2}},
            "conditions": {"solo": {}}, "population_seeds": [0, 1]}
    sp = tmp_path / "spec.yaml"
    sp.write_text(yaml.safe_dump(spec))
    out = tmp_path / "out"
    run_experiment(sp, out, processes=2)
    shared = out / "spend.sqlite"
    assert shared.exists() and not list((out / "solo").rglob("spend.sqlite"))
    for ps in (0, 1):
        saved = json.loads((out / "solo" / f"p{ps}" / "config.json").read_text())
        assert saved["config"]["llm"]["spend_file"] == str(shared.resolve())
    assert SpendGuard(shared, 1.0).totals()["calls"] == len(server.requests) >= 2


# ---------------------------------------------------------------- scripts/hosted_check.py
def _hosted_check():
    spec = importlib.util.spec_from_file_location("hosted_check", ROOT / "scripts" / "hosted_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hosted_check_script(server, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(KEY_ENV, SECRET)
    server.echo = True
    args = ["--base-url", _url(server), "--model", MODEL, "--api-key-env", KEY_ENV, "--price-in", str(P_IN),
            "--price-out", str(P_OUT), "--spend-file", str(tmp_path / "s.sqlite")]
    hc = _hosted_check()
    assert hc.main(args) == 2 and hc.main(args + ["--max-usd", "0"]) == 2
    assert server.requests == []  # refused without a positive cap
    assert hc.main(args + ["--max-usd", "0.05", "--n", "2"]) == 0
    out = capsys.readouterr().out
    assert len(server.requests) == 3 and all(r["body"]["max_tokens"] == 32 for r in server.requests)
    assert "identical texts = True" in out and "req-1" in out and "fp-1" in out and SECRET not in out
    assert SpendGuard(tmp_path / "s.sqlite", 0.05).totals()["calls"] == 3
