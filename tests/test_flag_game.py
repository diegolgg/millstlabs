import base64
import copy
import io
import json
from dataclasses import asdict, replace

import httpx
import numpy as np
import pytest
import torch
from PIL import Image

from millstlabs.flags.config import FlagConfig
from millstlabs.flags.controller import Controllers
from millstlabs.flags.corpus import VerifiedCorpus
from millstlabs.flags.engine import feature_width, run_episode
from millstlabs.flags.provider import BudgetExhausted, MockProvider, OpenAIProvider, parse_report, user_text
from millstlabs.flags.runner import deploy, plan
from millstlabs.flags.task import (
    catalog_images,
    crops_for,
    endpoint,
    generate_trial,
    image_url,
    prepare_trials,
)


def config():
    return FlagConfig(backend="mock", train_trials=1, evaluation_trials=2, interaction_rounds=1)


def distinct_crops():
    return [np.full((4, 6, 3), rgb, np.uint8) for rgb in [(255, 0, 0), (0, 0, 255)]]


def test_verified_corpus_rejects_forgery_and_requires_explicit_tools():
    crops = distinct_crops()
    corpus = VerifiedCorpus(crops, "shared")
    assert corpus.retrieve(1) == []
    red = corpus.evidence[0][0]
    assert not corpus.deposit(1, red)
    assert not corpus.deposit(0, replace(red, rgb=(0, 255, 0)))
    assert corpus.deposit(0, red)
    assert corpus.retrieve(1) == [red]
    assert corpus.metrics()["peer_records"] == corpus.metrics()["novel_color_imports"] == 1
    assert corpus.retrieve(1) == []
    assert not VerifiedCorpus(crops, "shared").deposits
    private = VerifiedCorpus(crops, "private")
    private.deposit(0, red)
    assert private.retrieve(1) == []


def test_simultaneous_reads_and_masks_do_not_leak_unread_peer_facts():
    corpus = VerifiedCorpus(distinct_crops(), "shared")
    before = corpus.mask(1, 24)
    result = corpus.execute({0: 1, 1: 25}, 24)
    assert result[1] == []
    assert corpus.mask(1, 24) == before
    assert corpus.execute({1: 25}, 24)[1]
    assert not corpus.mask(0, 24)[1]


def test_fixed_trial_assignments_and_no_hidden_metadata_in_image():
    c = config()
    _, images = catalog_images(c.catalog, c)
    first = generate_trial(11, c, list(images))
    assert first == generate_trial(11, c, list(images))
    assert len(first.contacts) == c.population*c.interaction_rounds
    crop = crops_for(first, images, c)[0]
    assert crop.shape == (4, 6, 3)
    png = image_url(crop, c.render_scale)
    im = Image.open(io.BytesIO(base64.b64decode(png.split(",")[1])))
    assert im.size == (150, 100) and not im.info
    text = user_text(list(images), [], 3)
    assert "private crop" in text and "country" in text and "reason" in text
    assert "position" not in text and "target" not in text


def test_manifest_rejects_evidence_leakage_across_training_and_evaluation(tmp_path):
    c = config()
    _, images = catalog_images(c.catalog, c)
    trials = prepare_trials(c, images)
    data = {k: [asdict(t) for t in v] for k, v in trials.items()}
    data["evaluation"][0] = {**data["train"][0], "seed": 123456}
    path = tmp_path/"trials.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="evidence assignments overlap"):
        prepare_trials(replace(c, trial_manifest=str(path)), images)


def test_paper_metrics_and_invalid_outputs():
    c = config()
    initial = ["France"]*4+["Peru"]*4
    correct = endpoint(initial, ["France"]*7+["Peru"], "France", c)
    assert correct["initial_accuracy"] == .5 and correct["terminal_truth_mass"] == .875
    assert correct["social_uplift"] == .375 and correct["endpoint"] == "correct_consensus"
    assert endpoint(initial, ["Peru"]*8, "France", c)["endpoint"] == "wrong_consensus"
    assert endpoint(initial, initial, "France", c)["endpoint"] == "polarization"
    assert endpoint(initial, ["__invalid__"]*8, "France", c)["endpoint"] == "fragmentation"
    assert not parse_report('{"country":"secret answer"}', ["France"], 1)["valid"]
    assert not parse_report('{"country":"France","extra":true}', ["France"], 1)["valid"]


def transport_fixture(seen):
    def handle(request):
        data = json.loads(request.content)
        seen.append(data)
        assert request.url.host == "api.openai.com"
        assert data["temperature"] == .2 and data["top_p"] == 1
        assert data["max_completion_tokens"] == 200
        assert data["messages"][1]["content"][1]["image_url"]["detail"] == "high"
        return httpx.Response(200, json={"model": data["model"], "system_fingerprint": "test-fingerprint",
            "choices": [{"message": {"content": '{"country":"France"}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    return httpx.MockTransport(handle)


def test_api_contract_cache_and_cumulative_cost_guard(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    seen = []
    c = replace(config(), backend="openai")
    path = tmp_path/"api.sqlite3"
    p = OpenAIProvider(c, path, 1, transport_fixture(seen))
    first = p.complete(distinct_crops()[0], [], ["France"], 1, 11)
    assert first["valid"] and len(seen) == 1
    spent = p.spent
    p.close()
    p = OpenAIProvider(c, path, spent+.000001, transport_fixture(seen))
    assert p.complete(distinct_crops()[0], [], ["France"], 1, 11) == first
    assert len(seen) == 1 and p.spent == spent
    with pytest.raises(BudgetExhausted):
        p.complete(distinct_crops()[0], [], ["France"], 1, 12)
    assert len(seen) == 1
    p.close()


def test_missing_key_and_uncertain_request_do_not_trigger_automatic_retries(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = []
    c = replace(config(), backend="openai")
    path = tmp_path/"api.sqlite3"
    p = OpenAIProvider(c, path, 1, transport_fixture(seen))
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        p.complete(distinct_crops()[0], [], ["France"], 1, 1)
    assert not seen and p.spent == 0
    p.close()
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    def broken(request):
        raise httpx.ReadTimeout("simulated uncertain response")
    p = OpenAIProvider(c, path, 1, httpx.MockTransport(broken))
    with pytest.raises(httpx.ReadTimeout):
        p.complete(distinct_crops()[0], [], ["France"], 1, 1)
    reserved = p.spent
    assert reserved > 0
    p.close()
    p = OpenAIProvider(c, path, 1, transport_fixture(seen))
    with pytest.raises(RuntimeError, match="uncertain outcome"):
        p.complete(distinct_crops()[0], [], ["France"], 1, 1)
    assert not seen
    p.close()
    p = OpenAIProvider(c, path, 1, transport_fixture(seen), retry_uncertain=True)
    assert p.complete(distinct_crops()[0], [], ["France"], 1, 1)["valid"]
    assert p.spent > reserved
    p.close()


def test_episode_keeps_memory_budget_and_frozen_evaluation_weights():
    c = config()
    _, images = catalog_images(c.catalog, c)
    trial = generate_trial(11, c, list(images))
    bank = Controllers(c, feature_width(c, images), 26)
    before = copy.deepcopy(bank.state())
    out, rows, correct = run_episode(c, trial, images, MockProvider(images), bank, "shared")
    assert all(len(call["transcript"]) <= 8 for call in out["calls"])
    assert out["queries"] == 24 and sum(map(len, rows)) == 16
    for i, model in enumerate(bank.models):
        for key, tensor in model.state_dict().items():
            assert torch.equal(tensor, before["models"][i][key])
    metrics = bank.learn(rows, correct)
    assert metrics and bank.updates > 0
    assert any(not torch.equal(v, before["models"][i][k]) for i, m in enumerate(bank.models) for k, v in m.state_dict().items())
    assert bank.models[0].actor[0].weight.data_ptr() != bank.models[1].actor[0].weight.data_ptr()


def test_resume_and_paired_controls_are_deterministic(tmp_path):
    c = config()
    root = tmp_path/"continued"
    partial = deploy(c, root, 0, max_jobs=1)
    assert not partial["finished"]
    resumed = deploy(c, root, 0)
    direct = deploy(c, tmp_path/"direct", 0)
    assert resumed["finished"] and resumed["comparisons"] == direct["comparisons"]
    a = torch.load(root/"latest.pt", map_location="cpu", weights_only=False)
    b = torch.load(tmp_path/"direct/latest.pt", map_location="cpu", weights_only=False)
    for arm in a["learners"]:
        for left, right in zip(a["learners"][arm]["models"], b["learners"][arm]["models"]):
            assert all(torch.equal(left[k], right[k]) for k in left)
    assert resumed["comparisons"]["shared_rl_minus_private_rl"]["paired_trials"] == 2
    for row in a["results"]:
        if row["arm"].startswith("private") or row["arm"] == "shared_rl_private_access":
            assert row["peer_records"] == 0
    groups = {}
    for row in a["results"]:
        if row["phase"] == "evaluation":
            groups.setdefault(row["trial_seed"], set()).add(row["trial_hash"])
    assert all(len(values) == 1 for values in groups.values())


def test_frozen_asset_changes_are_rejected(tmp_path):
    c = config()
    root = tmp_path/"run"
    deploy(c, root, 0, max_jobs=1)
    with pytest.raises(ValueError, match="Frozen"):
        deploy(replace(c, interaction_rounds=2), root, 0)
    p = plan(FlagConfig())
    assert p["api_call_upper_bound_total"] == 81984
    assert not p["exact_paper_reproduction"] and p["reproduction_gaps"]


def test_failure_after_update_rolls_back_uncommitted_controller(tmp_path, monkeypatch):
    from millstlabs.flags import runner
    c = config()
    root = tmp_path/"interrupted"
    original = runner.atomic_json
    failed = False
    def fail_one_trial(path, value):
        nonlocal failed
        if path.parent.name == "train" and not failed:
            failed = True
            raise OSError("simulated disk failure after PPO")
        return original(path, value)
    monkeypatch.setattr(runner, "atomic_json", fail_one_trial)
    with pytest.raises(OSError, match="simulated disk failure"):
        deploy(c, root, 0)
    state = torch.load(root/"latest.pt", map_location="cpu", weights_only=False)
    assert state["next_job"] == 0 and state["learners"]["private_rl"]["updates"] == 0
    resumed = deploy(c, root, 0)
    direct = deploy(c, tmp_path/"uninterrupted", 0)
    assert resumed["comparisons"] == direct["comparisons"]
