"""Exercise the real Transformers/PEFT stack with a small local Llama, no downloads."""
import copy

import pytest
import torch
from transformers import LlamaConfig, LlamaModel

from millstlabs.config import TrainingConfig
from millstlabs.env import PopulationEnv
from millstlabs.policy import PolicyBank
from millstlabs.ppo import Transition, update


class LocalTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        ids = torch.arange(1, 17).repeat(len(text), 1)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_real_peft_targets_gradients_isolation_and_removal(monkeypatch):
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *a, **k: LlamaModel(config))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: LocalTokenizer())
    env = PopulationEnv()
    obs, _ = env.reset(seed=11)
    obs = next(iter(obs.values()))
    bank = PolicyBank(TrainingConfig(backend="smollm", device="cpu", cpu_threads=1), obs)
    bank.add("agent_one")
    bank.add("agent_two", bank.state("agent_one"))
    assert len(bank._adapter_parameters("agent_one")) == 16  # four layers x Q/V x A/B
    assert all("layers.0." not in k and "layers.1." not in k for k in bank.state("agent_one")["adapter"])
    shared = {k: v.clone() for k, v in bank.base.named_parameters() if "lora_" not in k}
    neighbor = copy.deepcopy(bank.state("agent_two"))
    prior = copy.deepcopy(bank.state("agent_one"))
    h = bank.zero_hidden()
    a, lp, vp, vs, _ = bank.act("agent_one", obs, h)
    row = Transition(obs, h, a, lp, vp, vs, 1, 0.25, False, False)
    update(bank, "agent_one", [row] * 2, 0.5)
    assert any(not torch.equal(v, bank.state("agent_one")["adapter"][k]) for k, v in prior["adapter"].items())
    for section in neighbor:
        assert all(torch.equal(v, bank.state("agent_two")[section][k]) for k, v in neighbor[section].items())
    assert all(torch.equal(v, dict(bank.base.named_parameters())[k]) for k, v in shared.items())
    bank.remove("agent_one")
    assert "agent_one" not in bank.base.peft_config
    bank.act("agent_two", obs, h)


@pytest.mark.parametrize("separate", [True, False])
def test_llm_split_actor_learns_and_separate_critic_cannot_change_lora(monkeypatch, separate):
    from dataclasses import replace

    from millstlabs.config import EnvironmentConfig
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2)
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda *a, **k: LlamaModel(config))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: LocalTokenizer())
    env = PopulationEnv(replace(EnvironmentConfig(), message_symbols=4))
    obs, _ = env.reset(seed=11)
    obs = next(iter(obs.values()))
    bank = PolicyBank(TrainingConfig(backend="smollm", controller_architecture="split", separate_critic=separate,
                                     device="cpu", cpu_threads=1), obs)
    bank.add("agent_one")
    controller = bank.controllers["agent_one"]
    with torch.no_grad():
        controller.personal_value.weight.fill_(.1)
    out,_ = bank.sequence("agent_one", [obs], bank.zero_hidden())
    out[0][1].square().sum().backward()
    if separate:
        assert all(p.grad is None for p in bank._adapter_parameters("agent_one"))
        assert all(p.grad is None for p in controller.actor_parameters())
    else:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in bank._adapter_parameters("agent_one"))
    bank.optimizers["agent_one"].zero_grad(set_to_none=True)
    prior = bank.state("agent_one")
    h=bank.zero_hidden()
    rows=[]
    for _ in range(4):
        a,lp,vp,vs,nh = bank.act("agent_one",obs,h,temperature=.8)
        rows.append(Transition(obs,h,a,lp,vp,vs,1,.25,False,False,.8))
        h=nh
    result=update(bank,"agent_one",rows,.5)
    assert result['critic_grad_norm'] is not None
    assert any(not torch.equal(v,bank.state("agent_one")["adapter"][k]) for k,v in prior['adapter'].items())
