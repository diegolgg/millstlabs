import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from millstlabs.api import app
from millstlabs.config import EnvironmentConfig, ExperimentConfig, TrainingConfig, load_config
from millstlabs.discovery import arms, proposal, report
from millstlabs.env import PopulationEnv
from millstlabs.knowledge import (
    GET_TERRAIN,
    PUT_TERRAIN,
    CountNovelty,
    Fact,
    KnowledgeCorpus,
    novelty_beta,
)
from millstlabs.ppo import Transition, intrinsic_advantages
from millstlabs.trainer import Trainer, warmstart


def config():
    return ExperimentConfig(environment=replace(EnvironmentConfig(), max_individuals=1000, predator_temperature=1),
        training=TrainingConfig(backend='structured',device='cpu',cpu_threads=1,
            corpus_mode='private',intrinsic_critic=True,decisions=128,warmstart_transitions=32,
            rollout_ticks=8,social_window=16,sequence_length=4,checkpoints=[0,128],
            evaluation_maps=1,final_evaluation_maps=1,evaluation_horizon=8,development_maps=0,
            newborn_evaluation_every=0),profiles={'prosocial':{'social_preference':.5,'vigilance_bias':0}},
        methods=['r_adult'],seeds=[11])


def observations():
    env=PopulationEnv(replace(EnvironmentConfig(),obstacle_fraction=0),reproduction=False)
    obs,_=env.reset(seed=11)
    env.prey['prey_0'].pos=(1,1)
    env.prey['prey_1'].pos=(15,15)
    return env,{i:env.observe(i) for i in env.agents}


def test_explicit_deposit_retrieval_and_private_control():
    env,obs=observations()
    shared=KnowledgeCorpus('shared')
    private=KnowledgeCorpus('private')
    for c in [shared,private]:
        c.observe(obs,initial=True)
        assert not c.retrieve('prey_1','terrain',(15,15),0)  # Observation alone never publishes.
        assert not c.shared
        fact=c.evidence['prey_0'][('terrain',1,1)]
        assert c.deposit('prey_0',[fact],0)==1
    assert shared.retrieve('prey_1','terrain',(15,15),0)==[fact]
    assert not private.retrieve('prey_1','terrain',(15,15),0)
    assert shared.metrics()['new_observed_facts']==private.metrics()['new_observed_facts']==0
    assert shared.metrics()['distinct_peer_imports']==1
    # An unread shared store never changes another agent's tool availability.
    np.testing.assert_array_equal(shared.augment(obs)['prey_1']['action_mask'],private.augment(obs)['prey_1']['action_mask'])


def test_fabricated_deposit_rejected_and_food_ttl():
    _,obs=observations()
    c=KnowledgeCorpus('shared',food_ttl=2)
    c.observe(obs,initial=True)
    fact=next(iter(c.evidence['prey_0'].values()))
    assert c.deposit('prey_0',[replace(fact,value=1-fact.value)],0)==0
    assert c.deposit('prey_1',[fact],0)==0
    food=Fact('food',1,1,30.,0,'prey_0')
    local=copy.deepcopy(obs['prey_0'])
    local['stations'][0]=[1,1,30.,0]
    c.observe({'prey_0':local})
    assert c.deposit('prey_0',[food],0)==1
    assert c.retrieve('prey_1','food',(1,1),2)==[food]
    assert not c.retrieve('prey_1','food',(1,1),3)
    fresh=KnowledgeCorpus('shared')
    assert not fresh.retrieve('prey_1','food',(1,1),0)  # No cross-map corpus leakage.


def test_simultaneous_reads_do_not_see_same_tick_writes():
    env,obs=observations()
    c=KnowledgeCorpus('shared')
    c.observe(obs,initial=True)
    actions=dict.fromkeys(env.agents,6)
    actions['prey_0']=7*PUT_TERRAIN+6
    actions['prey_1']=7*GET_TERRAIN+6
    c.execute(actions,obs,0)
    assert not c.responses['prey_1']
    c.execute(dict(actions,prey_0=6),obs,1)
    assert c.responses['prey_1']
    a=c.augment(obs)
    assert a['prey_1']['corpus'][:,6].max()==1
    c.execute(dict.fromkeys(env.agents,6),obs,2)
    assert not c.responses['prey_1']  # Reading once does not create a free streaming subscription.


def test_guidance_decay_counts_and_intrinsic_terminal():
    _,obs=observations()
    tracker=CountNovelty()
    o=obs['prey_0']
    assert tracker.reward('family',o)==1
    assert tracker.reward('family',o)==pytest.approx(1/np.sqrt(2))
    t=TrainingConfig(intrinsic_critic=True,backend='structured',novelty_beta_start=.05,novelty_decay_decisions=100)
    assert novelty_beta(t,0)==.05 and novelty_beta(t,100)==0
    row=Transition({},torch.zeros(1,128),6,0,0,0,0,0,True,False,intrinsic_reward=.05)
    assert intrinsic_advantages([row],t,bootstrap=999)[0]==.05


def test_all_arms_have_identical_architecture_and_task_at_evaluation():
    cfg=load_config('configs/discovery.yaml')
    variants=arms(cfg,full=True)
    assert len(variants)==8 and all(c.training.backend=='smollm' for c in variants.values())
    assert all(c.environment==cfg.environment for c in variants.values())
    assert len({c.training.intrinsic_critic for c in variants.values()})==1
    assert proposal(cfg)['training_decisions']==131072
    assert load_config('configs/next-cooperative.yaml').digest()=='3d958501ecd7aedc2944846635ac1f42ff18b4e77ca19165384f840cdf226470'


def test_resume_preserves_corpus_novelty_and_curriculum(tmp_path):
    cfg=arms(config())['shared_guided']
    w=tmp_path/'warm.pt'
    warmstart(cfg,11,w)
    a=Trainer(cfg,'prosocial','r_adult',11,tmp_path/'a',w)
    a.assess(0)
    for _ in range(5):
        a.step()
    a.checkpoint()
    b=Trainer(cfg,'prosocial','r_adult',11,tmp_path/'b',w,tmp_path/'a/latest.pt')
    for _ in range(5):
        a.step()
        b.step()
    assert a.env.cfg.predator_temperature==b.env.cfg.predator_temperature==1
    assert a.corpus.metrics()==b.corpus.metrics()
    assert a.novelty.counts==b.novelty.counts
    for i in a.bank.controllers:
        for k,v in a.bank.state(i)['controller'].items():
            assert torch.equal(v,b.bank.state(i)['controller'][k])
    assert a.run()['finished']
    assert (tmp_path/'a/evaluation_private_corpus.jsonl').exists()
    assert (tmp_path/'a/corpus-checkpoint-128.json').exists()


def test_common_warmstart_works_for_all_four_arms(tmp_path):
    variants=arms(config())
    w=tmp_path/'warm.pt'
    warmstart(variants['private_plain'],11,w)
    hashes=set()
    for name,cfg in variants.items():
        t=Trainer(cfg,'prosocial','r_adult',11,tmp_path/name/'prosocial-r_adult-s11',w)
        hashes.add(t.warmstart_sha256)
        assert t.run()['finished']
    assert len(hashes)==1
    result=report(config(),tmp_path)
    assert result['complete']==4 and len(result['comparisons'])==2
    assert all(not c['exploratory_poc_pass'] for c in result['comparisons'])
    import json
    evaluated=json.loads((tmp_path/'shared_plain/prosocial-r_adult-s11/evaluation.jsonl').read_text().splitlines()[-1])
    row=evaluated['episodes'][0]
    assert row['discovery_curve'][0]['tick']==0 and row['discovery_curve'][-1]['tick']==8
    assert 0 <= row['mean_new_facts_over_time'] <= row['new_observed_facts']
    assert all(r['change_from_warmstart'] is not None for r in result['runs'])


def test_corpus_benchmark_uses_joint_actions(tmp_path):
    from millstlabs.experiments import benchmark
    result=benchmark(config(),2,tmp_path/'benchmark.json')
    assert result['decisions']==16 and result['decisions_per_second_with_ppo']>0


def test_named_http_tools_deposit_and_retrieve():
    client=TestClient(app)
    response=client.post('/environments',json={'seed':11,'corpus_mode':'shared','reproduction':False})
    data=response.json()
    i=data['id']
    agents=data['agents']
    origin=data['observations'][agents[0]]['self'][2:4]
    recipient=max(agents[1:],key=lambda a: sum(abs(x-y) for x,y in zip(origin,data['observations'][a]['self'][2:4])))
    assert len(data['observations'][agents[0]]['action_mask'])==35
    r=client.post(f'/environments/{i}/step',json={'expected_tick':0,'actions':dict.fromkeys(agents,6),
        'knowledge_actions':{agents[0]:'put_terrain'}})
    assert r.status_code==200
    r=client.post(f'/environments/{i}/step',json={'expected_tick':1,'actions':dict.fromkeys(agents,6),
        'knowledge_actions':{recipient:'get_terrain'}})
    assert r.status_code==200
    assert any(row[5]==1 for row in r.json()['observations'][recipient]['corpus'] if row[6]==1)
    assert client.delete(f'/environments/{i}').status_code==204
