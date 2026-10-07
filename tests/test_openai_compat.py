"""Fix round 1, step 9: OpenAI-compatible backend for a local MLX server, tested against a fake HTTP server (no
network). A live check against the real server is opt-in: CULTURE_LIVE=1 (see scripts/live_backend_check.py)."""

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from culture.bots.anchors import anchor_source, conventions_for, RULE_LISTS
from culture.llm.backend import Request
from culture.llm.cache import CachedBackend, CallCache, request_key
from culture.llm.openai_compat import BackendError, OpenAICompatBackend, OpenAICompatConfig
from culture.run.config import ConfigError, from_dict

BOT = anchor_source("iggi")
REPLY = ("Kept the rule order.\n<bot>\n```python\n" + BOT + "```\n</bot>\n<conventions>\n"
         + conventions_for(RULE_LISTS["iggi"]) + "</conventions>\n")


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append({"path": self.path, "body": body})
        out = {"id": "x", "object": "chat.completion", "model": body["model"],
               "choices": [{"index": 0, "message": {"role": "assistant", "content": self.server.reply},
                            "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 1234, "completion_tokens": 567, "total_tokens": 1801}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    srv.requests, srv.reply = [], REPLY
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()


def _req(text="hello", **kw):
    return Request(system=[{"type": "text", "text": "SYSTEM", "cache_control": {"type": "ephemeral"}}],
                   messages=[{"role": "user", "content": text}], model="mlx-community/Qwen3.6-35B-A3B-4bit",
                   tag="revise", seed_tag="t/g1/revise/0", max_tokens=4096, **kw)


def test_request_body_and_usage(fake_server):
    be = OpenAICompatBackend(OpenAICompatConfig(base_url=f"http://127.0.0.1:{fake_server.server_port}/v1", seed=7,
                                                temperature=0.0))
    resp = be.complete(_req())
    sent = fake_server.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    b = sent["body"]
    assert b["seed"] == 7 and b["temperature"] == 0.0 and b["max_tokens"] == 4096 and b["stream"] is False
    assert b["model"] == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert b["messages"] == [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "hello"}]
    assert "options" not in b and "think" not in b  # thinking is disabled at server start (chat-template-args)
    assert resp.usage.input_tokens == 1234 and resp.usage.output_tokens == 567 and resp.cost_usd == 0.0
    assert resp.backend == "openai_compat" and "class Bot" in resp.text


def test_extra_body_is_merged(fake_server):
    be = OpenAICompatBackend(OpenAICompatConfig(base_url=f"http://127.0.0.1:{fake_server.server_port}/v1",
                                                extra_body={"chat_template_kwargs": {"enable_thinking": False}}))
    be.complete(_req())
    assert fake_server.requests[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_seed_is_part_of_the_cache_key(fake_server, tmp_path):
    url = f"http://127.0.0.1:{fake_server.server_port}/v1"
    a, b = OpenAICompatBackend(OpenAICompatConfig(base_url=url, seed=1)), OpenAICompatBackend(OpenAICompatConfig(base_url=url, seed=2))
    assert request_key(a.name, _req(), a.key_extra()) != request_key(b.name, _req(), b.key_extra())
    cache = CallCache(tmp_path / "c")
    CachedBackend(a, cache, "replay").complete(_req())
    CachedBackend(a, cache, "replay").complete(_req())  # hit: same seed
    assert len(fake_server.requests) == 1
    CachedBackend(b, cache, "replay").complete(_req())  # miss: different seed, never served seed 1's answer
    assert len(fake_server.requests) == 2 and fake_server.requests[-1]["body"]["seed"] == 2


def test_server_down_is_a_clear_error():
    be = OpenAICompatBackend(OpenAICompatConfig(base_url="http://127.0.0.1:9/v1", request_timeout_s=2))
    with pytest.raises(BackendError):
        be.complete(_req())


def test_config_validation():
    ok = from_dict({"llm": {"backend": "openai_compat", "model": "mlx-community/Qwen3.6-35B-A3B-4bit"}})
    assert ok.llm.base_url == "http://127.0.0.1:8080/v1" and ok.llm.request_timeout_s == 1200.0
    with pytest.raises(ConfigError):
        from_dict({"llm": {"backend": "openai_compat"}})  # a claude-* model name on a local server
    with pytest.raises(ConfigError):
        from_dict({"llm": {"backend": "anthropic"}})


def test_full_agent_step_through_fake_server(fake_server, tmp_path):
    """One revise through the real agent step (prompt, call, parse, sandbox admission) with the fake server."""
    from culture.run.runner import run_config

    cfg = from_dict({"name": "oai_fake", "population": {"groups": 1, "agents_per_group": 1, "warm_start": "piers"},
                     "evaluation": {"selfplay_games": 4, "crossplay_games": 0, "anchor_games": 2,
                                    "between_group_games": 0, "ladder_every": 0, "workers": 0},
                     "llm": {"backend": "openai_compat", "model": "mlx-community/Qwen3.6-35B-A3B-4bit",
                             "base_url": f"http://127.0.0.1:{fake_server.server_port}/v1", "max_tokens": 4096},
                     "runner": {"generations": 2}})
    recs = run_config(cfg, tmp_path / "r")
    assert recs[-1]["agents"]["g0a0"]["candidate"] is not None  # the revision was parsed and admitted
    led = [json.loads(x) for x in (tmp_path / "r" / "ledger.jsonl").read_text().splitlines()]
    assert [r["backend"] for r in led if r["tag"] == "revise"] == ["openai_compat"]
    assert all(r["latency_s"] >= 0 for r in led if r["tag"] == "revise")


@pytest.mark.skipif(os.environ.get("CULTURE_LIVE") != "1", reason="live MLX server check is opt-in (CULTURE_LIVE=1)")
def test_live_mlx_server():
    be = OpenAICompatBackend(OpenAICompatConfig())
    r1 = be.complete(_req("Reply with the single word: ok"))
    r2 = be.complete(_req("Reply with the single word: ok"))
    assert r1.text == r2.text and r1.usage.output_tokens > 0
