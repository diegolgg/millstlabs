"""Organization policy units on synthetic data (v0 options and the cheap named ones)."""

import random

import pytest

from culture.artifacts.provenance import Provenance
from culture.org import registry
from culture.org.adoption import MergeLLM, Never, ReplaceIfBetter
from culture.org.allocation import Proportional, SoftmaxFloor, Uniform
from culture.org.base import AgentState
from culture.org.credit import Offer, PairedDelta, Window
from culture.org.delivery import Bernoulli, Deterministic
from culture.org.population import Full, Isolated, Population, Ring
from culture.org.routing import BestToAll, BroadcastGroup, NoRouting, RandomK
from culture.org.selection import KeepBestK
from culture.org.verification import NoVerification, SelfPlay, SelfPlayCrossplay
from culture.run.config import ConfigError, PolicySpec, from_dict


class FakeJudge:
    """Scores are attributes of the artifact id: 'x@12' self-plays to 12."""

    def selfplay(self, aid, seeds):
        return float(aid.split("@")[1]), 0.0

    def crossplay(self, a, b, seeds):
        return min(self.selfplay(a, seeds)[0], self.selfplay(b, seeds)[0])

    def anchors(self, aid, seeds):
        return {"piers": self.selfplay(aid, seeds)[0] - 1}


def agents_for(pop, scores=None):
    return {a: AgentState(a, pop.group_of(a), score=(scores or {}).get(a, 0.0)) for a in pop.agents}


def test_routing_respects_topology():
    pop = Population(3, 2)
    ag = agents_for(pop)
    routes = BroadcastGroup().route(ag, pop, Isolated(), random.Random(0))
    assert routes and all(pop.group_of(s) == pop.group_of(r) and s != r for s, r in routes)
    assert len(routes) == 3 * 2
    assert NoRouting().route(ag, pop, Full(), random.Random(0)) == []
    rk = RandomK(k=2).route(ag, pop, Full(), random.Random(1))
    assert len(rk) == 12 and all(s != r for s, r in rk)
    rk_iso = RandomK(k=5).route(ag, pop, Isolated(), random.Random(1))
    assert all(pop.group_of(s) == pop.group_of(r) for s, r in rk_iso)
    ring = Population(5, 1)
    assert Ring().reachable("g0a0", "g4a0", ring) and not Ring().reachable("g0a0", "g2a0", ring)


def test_best_to_all_sends_from_group_best_only():
    pop = Population(2, 3)
    ag = agents_for(pop, {"g0a1": 9.0, "g1a2": 4.0})
    routes = BestToAll().route(ag, pop, Isolated(), random.Random(0))
    assert sorted(routes) == [("g0a1", "g0a0"), ("g0a1", "g0a2"), ("g1a2", "g1a0"), ("g1a2", "g1a1")]


def test_bernoulli_delivery_rate_matches_p():
    rng = random.Random(3)
    for p in (0.1, 0.5, 0.9):
        flags = Bernoulli(p).deliver(10_000, rng)
        assert abs(sum(flags) / 10_000 - p) < 0.015
    assert all(Deterministic().deliver(5, rng))
    with pytest.raises(ValueError):
        Bernoulli(1.5)


def test_verification_and_adoption():
    j = FakeJudge()
    rcv = AgentState("r", "g0", incumbent="mine@10")
    v = SelfPlay(n=5).verify("m1", "theirs@12", rcv, j, list(range(5)), 1, 0.01)
    assert v.passed and v.payload_score == 12 and v.incumbent_score == 10 and v.n_games == 5
    worse = SelfPlay(n=5).verify("m2", "theirs@8", rcv, j, list(range(5)), 1, 0.01)
    assert not worse.passed
    assert ReplaceIfBetter().decide(v) == "adopt" and ReplaceIfBetter().decide(worse) == "reject"
    assert ReplaceIfBetter().decide(None) == "adopt"  # no verification: blind adoption
    assert NoVerification().verify("m", "x@1", rcv, j, [0], 1, 0.01) is None
    assert Never().decide(v) == "reject" and MergeLLM().decide(v) == "merge" and MergeLLM().decide(worse) == "reject"
    sc = SelfPlayCrossplay(n=5).verify("m3", "theirs@12", rcv, j, list(range(5)), 1, 0.01)
    assert sc.passed and sc.crossplay_with_incumbent == 10 and sc.anchors == {"piers": 11}


def test_paired_delta_credit_on_synthetic_windows():
    w1 = Window("s1", 3, before=5.0, after=11.0, offers=[Offer("m1", "t1", True, True), Offer("m2", "t2", True, False)])
    w2 = Window("s2", 3, before=8.0, after=6.0, offers=[Offer("m3", "t1", True, True), Offer("m4", "t3", True, True)],
                self_revised=True)
    w3 = Window("s3", 3, before=1.0, after=9.0, offers=[Offer("m5", "t2", False, False)])
    inc, notes = PairedDelta().assign([w1, w2, w3])
    assert inc == {"t1": 6.0 - 1.0, "t3": -1.0}
    assert notes[1]["split"] and notes[1]["self_revised"] and not notes[0]["split"]


def test_credit_propagates_on_synthetic_dag():
    p = Provenance()
    p.add("T0", "t", "g0", 0, [])
    p.add("S0", "s", "g1", 0, [])
    p.add("S1", "s", "g1", 1, ["T0"], teaching=[{"message": "m", "sender": "t", "source": "T0"}])
    assert p.teachers_of("S1") == {"t": 0} and p.clade("T0") == {"T0", "S1"}


def test_allocation_sums_to_budget_and_respects_floor():
    groups = ["g0", "g1", "g2"]
    stats = {"g0": {"credit": 3.0, "score": 20.0}, "g1": {"credit": 1.0, "score": 5.0}, "g2": {"credit": 0.0, "score": 0.0}}
    for pol in (Uniform(), Proportional(), SoftmaxFloor(T=0.5, eps=0.3)):
        out = pol.allocate(groups, stats, 90.0)
        assert abs(sum(out.values()) - 90.0) < 1e-9
    soft = SoftmaxFloor(T=0.5, eps=0.3).allocate(groups, stats, 90.0)
    assert min(soft.values()) >= 0.3 / 3 * 90.0 - 1e-9
    assert Proportional().allocate(groups, stats, 40.0) == {"g0": 30.0, "g1": 10.0, "g2": 0.0}


def test_keep_best_k_archive_and_acceptance():
    sel = KeepBestK(k=2)
    for i, s in enumerate([3.0, 9.0, 5.0, 1.0]):
        sel.record("g0", f"a{i}", s, None, i)
    assert [e["id"] for e in sel.archive["g0"]] == ["a1", "a2"]
    assert sel.accept(5.0, 5.0) and not sel.accept(4.9, 5.0)
    restored = KeepBestK(k=2)
    restored.load_state_dict(sel.state_dict())
    assert restored.archive == sel.archive


def test_registry_and_config_errors():
    assert isinstance(registry.make("routing", PolicySpec("random_k", {"k": 2})), RandomK)
    with pytest.raises(ConfigError):
        from_dict({"org": {"delivery": {"name": "bernoulli", "p": 3}}})
    with pytest.raises(ConfigError):
        from_dict({"org": {"routing": {"name": "random_k", "bogus": 1}}})
    with pytest.raises(ConfigError):
        from_dict({"llm": {"backend": "anthropic"}})


def test_migration_none_conserves_agents():
    from culture.org.migration import NoMigration

    pop = Population(3, 4)
    before = sorted(pop.agents)
    assert NoMigration().migrate(pop, agents_for(pop), 1, random.Random(0)) == []
    assert sorted(pop.agents) == before
