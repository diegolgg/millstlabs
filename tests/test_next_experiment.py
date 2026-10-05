import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from pettingzoo.test import parallel_api_test

from millstlabs.config import EnvironmentConfig, ExperimentConfig, TrainingConfig, load_config
from millstlabs.env import PopulationEnv, diffusion_probabilities
from millstlabs.next_experiment import promotion_gate
from millstlabs.policy import PolicyBank
from millstlabs.ppo import Transition, update
from millstlabs.trainer import Trainer, seed_all, warmstart


def config(symbols=1):
    return ExperimentConfig(environment=replace(EnvironmentConfig(), max_individuals=1000, message_symbols=symbols,
                                                predator_temperature=1.0),
        training=TrainingConfig(backend='structured', device='cpu', cpu_threads=1, warmstart_transitions=32,
            decisions=128, rollout_ticks=8, sequence_length=4, social_window=16, checkpoints=[0,128],
            evaluation_maps=1, final_evaluation_maps=1, evaluation_horizon=8, development_maps=1,
            evaluation_ablations=True, predator_evaluation_temperatures=[0.,4.],
            temperature_start=1., temperature_end=.7, temperature_decay_decisions=128))


def test_existing_experiment_digest_unchanged():
    assert load_config('configs/overnight-cpu.yaml').digest() == '6893651dc9309c073293ba3582e8e10ec2d1780766e2854cec60804918d05cb8'


def test_predator_diffusion_limits_and_entropy():
    np.testing.assert_allclose(diffusion_probabilities([1,1,3,3],0), [.5,.5,0,0])
    probs = [diffusion_probabilities([1,1,3,3],t) for t in [.25,1.,4.,1e8]]
    entropies = [-sum(p*np.log(p.clip(1e-30))) for p in probs]
    assert all(a < b for a,b in zip(entropies,entropies[1:]))
    np.testing.assert_allclose(probs[-1], [.25]*4, atol=1e-7)


def test_messages_delayed_bounded_and_muted_dynamics_match():
    cfg = replace(config(4).environment, message_radius=2, message_capacity=1)
    a = PopulationEnv(cfg, reproduction=False)
    a.reset(seed=11)
    a.obstacles = set()
    a.predator = (19,19)
    for j,p in enumerate(a.prey.values()):
        p.pos = (j,0)
    before = a.observe('prey_1')
    assert not before['messages'][:,2].any()
    b = a.fork()
    b.deliver_messages = False
    actions = dict.fromkeys(a.agents, 6)
    actions['prey_0'] = 7*2+6
    energy = a.prey['prey_0'].energy
    oa,*_ = a.step(actions)
    ob,*_ = b.step(actions)
    assert a.prey['prey_0'].energy == pytest.approx(energy-cfg.metabolism-cfg.message_cost)
    np.testing.assert_array_equal(oa['prey_1']['messages'][0], [0,0,2])
    assert not oa['prey_3']['messages'][:,2].any()
    assert not ob['prey_1']['messages'][:,2].any()
    assert a.render() == b.render()
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert [p.energy for p in a.prey.values()] == [p.energy for p in b.prey.values()]
    a.step(dict.fromkeys(a.agents,6))
    assert not a.observe('prey_1')['messages'][:,2].any()
    parallel_api_test(PopulationEnv(cfg), num_cycles=15)


def test_critic_cannot_update_actor_and_temperature_replays_logprob():
    cfg = config(4)
    obs,_ = PopulationEnv(cfg.environment).reset(seed=11)
    o = next(iter(obs.values()))
    bank = PolicyBank(cfg.training, o)
    bank.add('a')
    bank.add('b',bank.state('a'))
    controller = bank.controllers['a']
    # Nonzero critic head makes the test exercise critic representation gradients too.
    with torch.no_grad():
        controller.personal_value.weight.fill_(.1)
    outputs,_ = bank.sequence('a',[o],bank.zero_hidden())
    outputs[0][1].square().sum().backward()
    assert all(p.grad is None for p in controller.actor_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in controller.critic_parameters())
    bank.optimizers['a'].zero_grad(set_to_none=True)
    rows=[]
    h=bank.zero_hidden()
    for temp in [1.,.9,.8,.7]:
        a,lp,vp,vs,next_h=bank.act('a',o,h,temperature=temp)
        rows.append(Transition(o,h,a,lp,vp,vs,1,.5,False,False,temp))
        h=next_h
    cfg.training.temperature_start=3.
    outputs,_=bank.sequence('a',[r.observation for r in rows],rows[0].hidden,[r.temperature for r in rows])
    for r,(dist,*_) in zip(rows,outputs):
        assert float(dist.log_prob(torch.tensor(r.action)).detach()) == pytest.approx(r.log_probability,abs=1e-6)
    other=bank.state('b')
    stats=update(bank,'a',rows,.5)
    assert all(v is None or np.isfinite(v) for v in stats.values())
    for name,tensor in other['controller'].items():
        assert torch.equal(tensor,bank.state('b')['controller'][name])
    source=bank.state('a')
    from millstlabs.evaluation import calibration_histories
    bank.inherit('child',source,calibration_histories(cfg.environment),torch.Generator().manual_seed(3))
    for name,tensor in source['controller'].items():
        if name not in {'action.weight','watch_bias'}:
            assert torch.equal(tensor,bank.state('child')['controller'][name])


@pytest.mark.parametrize('method',['r_adult','r_initial','iteration'])
def test_structured_resume_and_final_assays(tmp_path,method):
    cfg=config(4)
    warm=tmp_path/'warm.pt'
    warmstart(cfg,11,warm)
    a=Trainer(cfg,'prosocial',method,11,tmp_path/'a',warm)
    for _ in range(5):
        a.step()
    a.checkpoint()
    b=Trainer(cfg,'prosocial',method,11,tmp_path/'b',warm,tmp_path/'a/latest.pt')
    for _ in range(5):
        a.step()
        b.step()
    assert a.env.metrics() == b.env.metrics()
    for i in a.bank.controllers:
        for k,v in a.bank.state(i)['controller'].items():
            assert torch.equal(v,b.bank.state(i)['controller'][k])
    a.assess(0)  # Test fixture records an artificial baseline; production run prevents late checkpoint zero.
    assert a.run()['finished']
    for name in ['development_evaluation','evaluation_muted','predator_evaluation']:
        records=[json.loads(x) for x in (tmp_path/'a'/f'{name}.jsonl').read_text().splitlines()]
        assert records[-1]['checkpoint'] == 128
        assert records[-1]['policy_temperature'] == 1.
    muted=json.loads((tmp_path/'a/evaluation_muted.jsonl').read_text().splitlines()[-1])
    assert all(r['messages_delivered']==0 for r in muted['episodes'])
    assert promotion_gate(tmp_path/'a')['ready_to_test_harder_stage'] is False


def test_muted_and_delivered_arms_accept_identical_warm_checkpoint(tmp_path):
    cfg=config(4)
    warm=tmp_path/'warm.pt'
    warmstart(cfg,11,warm)
    a=Trainer(cfg,'individual','r_adult',11,tmp_path/'a',warm)
    cfg=copy.deepcopy(cfg)
    cfg.training.deliver_messages=False
    b=Trainer(cfg,'individual','r_adult',11,tmp_path/'b',warm)
    assert a.warmstart_sha256==b.warmstart_sha256
    assert a.env.deliver_messages and not b.env.deliver_messages


def test_physical_demonstrations_do_not_train_arbitrary_message_head(tmp_path):
    cfg = config(4)
    seed_all(11)
    observations, _ = PopulationEnv(cfg.environment, reproduction=False).reset(seed=11)
    bank = PolicyBank(cfg.training, next(iter(observations.values())))
    bank.add("warm")
    before = bank.state("warm")["controller"]
    result = warmstart(cfg, 11, tmp_path / "warm.pt")
    assert all(torch.equal(value, result["state"]["controller"][name])
               for name, value in before.items() if name.startswith("message."))
