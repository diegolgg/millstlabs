from dataclasses import replace

import numpy as np
import pytest
import torch

from millstlabs.config import EnvironmentConfig, ExperimentConfig, TrainingConfig
from millstlabs.env import PopulationEnv
from millstlabs.knowledge import make_corpus, next_observations
from millstlabs.note_probe import cue_pair
from millstlabs.notes import NoteCorpus
from millstlabs.observations import Heuristic, observation_text
from millstlabs.policy import PolicyBank
from millstlabs.ppo import Transition, update
from millstlabs.trainer import Trainer, warmstart


def config():
    return ExperimentConfig(environment=replace(EnvironmentConfig(), max_individuals=1000, predator_temperature=1),
        training=TrainingConfig(backend="structured", device="cpu", cpu_threads=1,
            corpus_interface="notes", corpus_mode="shared", actor_grounding=True,
            decisions=128, warmstart_transitions=32, rollout_ticks=8, social_window=16,
            sequence_length=4, checkpoints=[0, 128], evaluation_maps=1, final_evaluation_maps=1,
            evaluation_horizon=8, newborn_evaluation_every=0, feed_retention=.1),
        profiles={"prosocial": {"social_preference": .5, "vigilance_bias": 0}},
        methods=["r_adult"], seeds=[11])


def fixture(mode="shared", style="grounded"):
    env = PopulationEnv(replace(EnvironmentConfig(), obstacle_fraction=0), reproduction=False)
    obs, _ = env.reset(seed=11)
    author, receiver = env.agents[:2]
    env.prey[author].pos = (1, 1)
    env.prey[receiver].pos = (15, 15)
    obs = {i: env.observe(i) for i in [author, receiver]}
    obs[author]["local"][1, 1] = 1
    obs[author]["stations"][:] = -1
    obs[receiver]["stations"][:] = -1
    training = replace(config().training, corpus_mode=mode, note_style=style)
    corpus, augmented = make_corpus(training, obs)
    return corpus, obs, augmented, author, receiver


def test_seven_actions_no_implicit_publication_and_real_delivery_credit():
    c, obs, augmented, author, receiver = fixture()
    assert not augmented[receiver]["notes"]
    assert len(augmented[author]["action_mask"]) == 7
    c.publish({author: 0, receiver: -1}, augmented, None)
    assert not c.notes
    rewards = c.publish({author: 1, receiver: -1}, augmented, None)
    assert rewards[author] == -c.training.note_write_penalty
    # A publication is unavailable to already-collected actions.
    assert not augmented[receiver]["notes"]
    for o in obs.values():
        o["self"][6] += 1
    next_obs = next_observations(c, obs, [author, receiver])
    assert next_obs[receiver]["notes"]
    assert rewards[author] == pytest.approx(c.training.note_credit-c.training.note_write_penalty)
    previous = rewards.copy()
    c.augment(obs)
    assert rewards == previous  # Repeated reads cannot farm reward.
    assert "tools=" not in observation_text(next_obs[receiver])
    with pytest.raises(ValueError):
        c.execute({author: 7}, obs, 0)


def test_no_credit_for_private_corpus_or_dead_receivers():
    for mode, living in [("private", True), ("shared", False)]:
        c, obs, augmented, author, receiver = fixture(mode)
        rewards = c.publish({author: 1, receiver: -1}, augmented, None)
        next_observations(c, obs, [author, receiver] if living else [author])
        assert rewards[author] == -c.training.note_write_penalty
        if mode == "private":
            assert not c.context(receiver, obs[receiver])


def test_untrusted_prose_cannot_fabricate_evidence_and_expires():
    c, obs, _, author, receiver = fixture(style="prose")
    obs[author]["stations"][0] = [1, 1, 30, 0]
    c.observe(obs)
    augmented = c.augment(obs)

    class Writer:
        def describe_note(self, policy, evidence):
            assert "stock 30" in evidence
            return "I think there are 999 units at (19,19).", 12

    c.publish({author: 1}, augmented, Writer())
    note = c.notes[0]
    assert note.text.startswith("I think")  # Interpretation is stored, not certified.
    assert all(f.x == 1 and f.y == 1 and f.value == 30 for f in note.facts)
    obs[receiver]["self"][6] = c.food_ttl+1
    assert not c.context(receiver, obs[receiver])  # Neither stale evidence nor its prose leaks.
    clean = NoteCorpus("shared", c.training)
    assert not clean.notes and not clean.transferred  # Map reset cannot reuse coordinates.


def test_teacher_changes_direction_only_when_corpus_cue_changes():
    _, _, augmented, author, _ = fixture()
    # Restore the three actual station rows through the controlled test fixture.
    a, b = cue_pair(augmented[author])
    for key in ["self", "local", "stations", "threats", "companions", "action_mask"]:
        np.testing.assert_array_equal(a[key], b[key])
    assert Heuristic(0).act(a) == 3
    assert Heuristic(0).act(b) == 2


def test_publication_reward_learns_gate_without_changing_action_space():
    cfg = config()
    _, _, obs, author, _ = fixture()
    bank = PolicyBank(cfg.training, obs[author])
    bank.add("agent")
    h = bank.zero_hidden()
    out, _ = bank.sequence("agent", [obs[author]], h)
    before = float(out[0][-1].probs.detach())
    action = int(out[0][0].probs.argmax())
    row = Transition(obs[author], h, action, float(out[0][0].log_prob(torch.tensor(action)).detach()),
                     0., 0., 0., 0., True, True,
                     publication=1, publication_log_probability=float(out[0][-1].log_prob(torch.tensor(1.)).detach()),
                     publication_reward=1.)
    update(bank, "agent", [row]*4, .5)
    after, _ = bank.sequence("agent", [obs[author]], h)
    assert after[0][0].probs.shape[-1] == 7
    assert float(after[0][-1].probs.detach()) > before


def test_notes_resume_replays_exactly_and_evaluates_withheld_access(tmp_path):
    cfg = config()
    cfg.validate()
    warm = tmp_path/"warm.pt"
    warmstart(cfg, 11, warm)
    a = Trainer(cfg, "prosocial", "r_adult", 11, tmp_path/"a", warm)
    for _ in range(3):
        a.step()
    a.checkpoint()
    b = Trainer(cfg, "prosocial", "r_adult", 11, tmp_path/"b", warm, tmp_path/"a/latest.pt")
    for _ in range(6):
        a.step()
        b.step()
    assert a.corpus.notes == b.corpus.notes
    assert a.corpus.metrics() == b.corpus.metrics()
    for i in a.bank.controllers:
        for key, value in a.bank.state(i)["controller"].items():
            assert torch.equal(value, b.bank.state(i)["controller"][key])
    a.cfg.training.checkpoints = [128]  # Initial assay deliberately omitted in this unit test.
    assert a.run()["finished"]
    assert (tmp_path/"a/note_probe.jsonl").exists()
    assert (tmp_path/"a/evaluation_private_corpus.jsonl").exists()
    assert (tmp_path/"a/corpus-checkpoint-128.json").exists()


def test_direct_actor_grounding_does_not_expose_corpus_numbers():
    _, _, obs, author, _ = fixture()
    a, b = cue_pair(obs[author])
    np.testing.assert_array_equal(PolicyBank.physical_features(a), PolicyBank.physical_features(b))
    assert observation_text(a) != observation_text(b)


def test_http_notes_interface_validates_whole_batch_and_does_not_mutate_on_get():
    from fastapi.testclient import TestClient

    from millstlabs.api import app, corpora
    client = TestClient(app)
    result = client.post("/environments", json={"seed": 11, "reproduction": False,
        "corpus_mode": "shared", "corpus_interface": "notes", "note_style": "prose"}).json()
    session = result["id"]
    observations = result["observations"]
    authors = [a for a, o in observations.items() if o["note_opportunity"]]
    assert authors
    assert all(len(o["action_mask"]) == 7 for o in observations.values())
    body = {"expected_tick": 0, "actions": dict.fromkeys(result["agents"], 6),
            "notes": {authors[0]: "An observation to share.", "missing_agent": "bad"}}
    assert client.post(f"/environments/{session}/step", json=body).status_code == 422
    assert not corpora[session][0].notes
    del body["notes"]["missing_agent"]
    response = client.post(f"/environments/{session}/step", json=body)
    assert response.status_code == 200
    before = corpora[session][0].metrics()
    assert client.get(f"/environments/{session}").status_code == 200
    assert corpora[session][0].metrics() == before
    assert client.delete(f"/environments/{session}").status_code == 204
