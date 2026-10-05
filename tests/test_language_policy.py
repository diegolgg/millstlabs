"""Direct language-token actions, zero examples, energy-free persistent memory."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from millstlabs.config import load_config
from millstlabs.env import ACTION_NAMES, REST, PopulationEnv
from millstlabs.evaluation import evaluate
from millstlabs.knowledge import make_corpus, next_observations
from millstlabs.language_policy import bounded_prompt
from millstlabs.notes import publish_notes
from millstlabs.policy import PolicyBank
from millstlabs.ppo import Transition, update
from millstlabs.tactics import record_experience
from millstlabs.trainer import Trainer, warmstart


class Tokenizer:
    pad_token_id = 0
    vocabulary = {word: i+1 for i, word in enumerate([*ACTION_NAMES, "no", "yes"])}

    def encode(self, text, **kwargs):
        return [self.vocabulary[text]] if text in self.vocabulary else list(range(1, 17))

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(m["content"] for m in messages)+"\nAssistant:"

    def __call__(self, texts, **kwargs):
        ids = torch.arange(1, 17).repeat(len(texts), 1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


@pytest.fixture
def cfg(monkeypatch):
    architecture = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *a, **k: LlamaModel(architecture))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: Tokenizer())
    config = load_config("configs/autonomous-llm.yaml")
    config.training = replace(config.training, note_style="grounded", decisions=32, rollout_ticks=4,
        social_window=8, checkpoints=[0, 32], evaluation_maps=1, final_evaluation_maps=1, evaluation_horizon=2)
    return config


def test_actual_lm_logits_choose_actions_and_ppo_updates_only_private_lora(cfg):
    env = PopulationEnv(cfg.environment, reproduction=False)
    raw, _ = env.reset(seed=11)
    _, observations = make_corpus(cfg.training, raw)
    obs = next(iter(observations.values()))
    bank = PolicyBank(cfg.training, obs)
    bank.add("a")
    bank.add("b", bank.state("a"))
    neighbor = bank.state("b")
    frozen = {k: p.clone() for k, p in bank.base.named_parameters() if "lora_" not in k}
    with torch.no_grad():
        features = bank.encode("a", [obs])
        logits = features @ bank.base.get_input_embeddings().weight[bank.action_token_ids].T
        expected = logits.masked_fill(~torch.tensor(obs["action_mask"]).bool(), -1e9).softmax(-1)
    out, _ = bank.sequence("a", [obs], bank.zero_hidden())
    assert torch.allclose(out[0][0].probs, expected)
    assert not hasattr(bank.controllers["a"], "action")
    with torch.no_grad():
        bank.controllers["a"].personal_value.weight.fill_(10)
    other, _ = bank.sequence("a", [obs], bank.zero_hidden())
    assert torch.equal(out[0][0].probs, other[0][0].probs)
    prior = bank.state("a")
    h = bank.zero_hidden()
    rows = []
    for reward in [0., 1., 0., 2.]:
        a, lp, vp, vs, nh = bank.act("a", obs, h)
        rows.append(Transition(obs, h, a, lp, vp, vs, reward, .25, False, False))
        h = nh
    update(bank, "a", rows, .5)
    assert any(not torch.equal(v, bank.state("a")["adapter"][k]) for k, v in prior["adapter"].items())
    assert all(torch.equal(v, dict(bank.base.named_parameters())[k]) for k, v in frozen.items())
    for section in neighbor:
        assert all(torch.equal(v, bank.state("b")[section][k]) for k, v in neighbor[section].items())


def test_publication_uses_lm_yes_no_logits(cfg):
    env = PopulationEnv(cfg.environment, reproduction=False)
    raw, _ = env.reset(seed=11)
    _, obs = make_corpus(cfg.training, raw)
    o = next(iter(obs.values()))
    o["note_opportunity"] = True
    bank = PolicyBank(cfg.training, o)
    bank.add("a")
    features = bank.encode("a", [o], purpose="publication")
    scores = features @ bank.base.get_input_embeddings().weight[bank.publication_token_ids].T
    out, _ = bank.sequence("a", [o], bank.zero_hidden())
    assert torch.allclose(out[0][-1].probs, scores.softmax(-1)[:, 1])


def experienced_corpus(cfg):
    env = PopulationEnv(cfg.environment, reproduction=False)
    raw, _ = env.reset(seed=11)
    corpus, before = make_corpus(cfg.training, raw)
    actions = dict.fromkeys(env.agents, REST)
    after, *_ = env.step(actions)
    record_experience(corpus, actions, before, after, env)
    obs = next_observations(corpus, after, env.agents)
    return env, corpus, obs


def test_knowledge_reads_and_writes_change_no_energy_or_physical_outcome(cfg):
    env, corpus, obs = experienced_corpus(cfg)
    control = copy.deepcopy(env)
    author = env.agents[0]
    before = {i: p.energy for i, p in env.prey.items()}
    publish_notes(corpus, {author: 1}, obs, None, env, cfg.training)
    corpus.augment(obs)
    assert before == {i: p.energy for i, p in env.prey.items()}
    assert corpus.notes[0].experience[0]["action"] == "rest"
    actions = dict.fromkeys(env.agents, REST)
    result, *_ = env.step(actions)
    expected, *_ = control.step(actions)
    assert env.events == control.events
    for i in result:
        np.testing.assert_array_equal(result[i]["self"], expected[i]["self"])


def test_tactics_survive_map_reset_with_provenance_and_lineage_privacy(cfg):
    env, old, obs = experienced_corpus(cfg)
    author, peer = env.agents[:2]
    old.publish({author: 1}, obs, None)
    fresh, _ = env.reset(seed=101)
    new, fresh = make_corpus(cfg.training, fresh, "private")
    new.restore_memory(old, {author: author, peer: peer})
    augmented = new.augment(fresh)
    assert augmented[author]["notes"] and not augmented[peer]["notes"]
    assert augmented[author]["notes"][0]["age"] is None
    assert not new.experiences and not new.pending
    assert not new.notes[0].facts
    new.notes[0].text = "changed"
    assert old.notes[0].text != "changed"


def test_zero_demo_initialization_recovery_and_exact_resume(cfg, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Direct LLM must never instantiate a demonstration teacher")
    monkeypatch.setattr("millstlabs.trainer.Heuristic", forbidden)
    monkeypatch.setattr("millstlabs.evaluation.Heuristic", forbidden)
    path = tmp_path/"initial.pt"
    initial = warmstart(cfg, 11, path)
    assert initial["transitions"] == initial["resources"]["gradient_updates"] == 0
    trainer = Trainer(cfg, "prosocial", "r_adult", 11, tmp_path/"a", path)
    trainer.step()
    author = trainer.env.agents[0]
    trainer.corpus.publish({author: 1}, trainer.obs, None)
    text = trainer.corpus.notes[-1].text
    for p in trainer.env.prey.values():
        p.energy = -100
    trainer.step()
    assert trainer.trial == 1
    assert text in [n.text for n in trainer.corpus.notes]
    trainer.checkpoint()
    trainer.step()
    expected = {i: trainer.bank.state(i) for i in trainer.env.agents}
    restored = Trainer(cfg, "prosocial", "r_adult", 11, tmp_path/"b", path, tmp_path/"a"/"latest.pt")
    restored.step()
    assert restored.decisions == trainer.decisions
    assert restored.env.metrics() == trainer.env.metrics()
    for i, state in expected.items():
        for section in state:
            assert all(torch.equal(v, restored.bank.state(i)[section][k]) for k, v in state[section].items())


def test_evaluation_copies_memory_and_never_leaks_it_back(cfg):
    env, corpus, obs = experienced_corpus(cfg)
    author = env.agents[0]
    corpus.publish({author: 1}, obs, None)
    bank = PolicyBank(cfg.training, obs[author])
    bank.add(author)
    original = copy.deepcopy(corpus.__dict__)
    result = evaluate(bank, [author], cfg.environment, cfg.training, 0, initial_corpus=corpus,
                      policy_owners={author: author}, without_tactics=True)
    assert result["tactic_access"] == "none"
    assert result["episodes"][0]["note_contexts"] == 0
    assert corpus.notes == original["notes"]
    assert corpus.counts == original["counts"]
    assert corpus.experiences == original["experiences"]


def test_prompt_budget_preserves_action_request_and_fails_if_sensors_do_not_fit(cfg):
    class CharacterTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            return list(text)

    _, _, obs = experienced_corpus(cfg)
    obs = next(iter(obs.values()))
    obs["recent_experience"] = [{"padding": "x"*8000}]
    obs["notes"] = [{"text": "y"*8000, "evidence": "z"*8000}]
    prompt = bounded_prompt(CharacterTokenizer(), obs, 3000)
    assert len(prompt) <= 3000
    assert "Choose your next action" in prompt and "energy=" in prompt
    assert "Your recent action outcomes: []" in prompt
    assert "Retrieved note" not in prompt
    with pytest.raises(ValueError, match="Essential game prompt"):
        bounded_prompt(CharacterTokenizer(), obs, 5)


@pytest.mark.parametrize("field,value", [("warmstart_transitions", 1), ("feed_retention", .1),
    ("corpus_tool_cost", .01), ("note_write_penalty", .01)])
def test_direct_mode_rejects_demonstrations_and_knowledge_costs(cfg, field, value):
    setattr(cfg.training, field, value)
    with pytest.raises(AssertionError):
        cfg.validate()
