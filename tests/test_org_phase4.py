"""Phase-4 policies on synthetic data with known answers."""

import random

import numpy as np
import pytest

from culture.artifacts.provenance import Provenance
from culture.org.allocation import NashRelative, SoftmaxFloor, maxent_nash
from culture.org.base import AgentState
from culture.org.credit import DatamodelRegression, LineageDecay, Offer, Window
from culture.org.delivery import Bernoulli
from culture.org.environment import VariantSwitch
from culture.org.migration import BestToNeighbor, RandomMigration
from culture.org.population import Islands, Population
from culture.org.selection import HGMCladeTS, ShinkaWeighted
from culture.org.verification import CriticalSocialLearning


class Judge:
    def selfplay(self, aid, seeds):
        return float(aid.split("@")[1]), 0.0

    def crossplay(self, a, b, seeds):
        return 0.0

    def anchors(self, aid, seeds):
        return {}


def agents_for(pop, scores=None):
    return {a: AgentState(a, pop.group_of(a), score=(scores or {}).get(a, 0.0)) for a in pop.agents}


# ------------------------------------------------------------------ islands and migration
def test_random_migration_conserves_agents_and_group_sizes():
    pop = Population(4, 5)
    before_sizes = {g: len(m) for g, m in pop.members.items()}
    all_agents = sorted(pop.agents)
    rng = random.Random(0)
    moved = 0
    for g in range(1, 41):
        moves = RandomMigration(rate=0.2, interval=5).migrate(pop, agents_for(pop), g, rng)
        if g % 5:
            assert moves == []
        moved += len(moves)
        assert sorted(pop.agents) == all_agents
        assert {g_: len(m) for g_, m in pop.members.items()} == before_sizes
    assert moved > 0
    assert Islands(0.1, 3).reachable("g0a0", "g0a1", pop) is not None


def test_best_to_neighbor_moves_bests_along_the_ring():
    pop = Population(3, 2)
    scores = {"g0a1": 9, "g1a0": 8, "g2a1": 7}
    moves = BestToNeighbor(interval=1).migrate(pop, agents_for(pop, scores), 1, random.Random(0))
    assert ("g0a1", "g0", "g1") in moves and ("g1a0", "g1", "g2") in moves and ("g2a1", "g2", "g0") in moves
    assert sorted(len(m) for m in pop.members.values()) == [2, 2, 2]


# ------------------------------------------------------------------ critical social learning
def test_critical_social_learning_adopts_first_then_reverts_below_theta():
    csl = CriticalSocialLearning(theta=10.0, n=5)
    rcv = AgentState("r", "g0", incumbent="mine@12")
    v = csl.verify("m", "theirs@3", rcv, Judge(), [0], 1, 0.01)
    assert v.passed and v.n_games == 0  # provisional: adopt first, no games spent
    assert csl.recheck("theirs@3", "mine@12", Judge(), list(range(5)), 0.01)["keep"] is False  # below theta: revert
    assert csl.recheck("theirs@14", "mine@12", Judge(), list(range(5)), 0.01)["keep"] is True
    assert csl.recheck("theirs@11", "mine@12", Judge(), list(range(5)), 0.01)["keep"] is False  # worse than before


# ------------------------------------------------------------------ credit
def test_lineage_decay_on_synthetic_dag():
    p = Provenance()
    p.add("A0", "a", "g", 0, [])
    p.add("B1", "b", "g", 1, ["A0"], teaching=[{"message": "m0", "sender": "a", "source": "A0"}])
    w = Window("s", 2, before=0.0, after=8.0, offers=[Offer("m1", "b", True, True, "B1")])
    inc, _ = LineageDecay(gamma=0.5).assign([w], p)
    assert inc == {"b": 8.0, "a": 4.0}  # direct teacher full share, grand-teacher gamma^1


def test_datamodel_regression_recovers_planted_teachers():
    """Five teachers with planted per-message effects; Bernoulli(0.5) delivery; noisy deltas. The ridge fit over
    randomized delivery must recover the effects (and the zero-effect teachers must come out near zero)."""
    effects = {"t0": 3.0, "t1": 0.0, "t2": 1.0, "t3": 0.0, "t4": -2.0}
    rng = np.random.default_rng(0)
    dm = DatamodelRegression(lam=0.1, window=10_000)
    deliver = Bernoulli(0.5)
    r = random.Random(1)
    for g in range(200):
        windows = []
        for s in range(6):
            flags = deliver.deliver(len(effects), r)
            offers = [Offer(f"m{g}{s}{t}", t, bool(f), bool(f)) for t, f in zip(effects, flags)]
            delta = sum(effects[o.sender] for o in offers if o.delivered) + rng.normal(0, 1.0)
            windows.append(Window(f"s{s}", g, 0.0, delta, offers))
        dm.assign(windows)
    for t, e in effects.items():
        assert dm.beta[t] == pytest.approx(e, abs=0.15)
    # paired_delta on the same data would credit by adoption only and cannot separate teachers sent together;
    # the regression's total credit for a zero-effect teacher is near zero
    assert abs(dm.last["t1"]) < 0.15 * 600


def test_datamodel_state_roundtrip():
    dm = DatamodelRegression()
    dm.assign([Window("s", 1, 0.0, 1.0, [Offer("m", "t", True, True)]), Window("s", 1, 0.0, 0.0, [Offer("m2", "t", False, False)])])
    dm2 = DatamodelRegression()
    dm2.load_state_dict(dm.state_dict())
    assert dm2.rows == dm.rows and dm2.beta == dm.beta


# ------------------------------------------------------------------ allocation
def test_softmax_floor_and_nash_relative_sum_and_floor():
    groups = ["g0", "g1", "g2"]
    stats = {"g0": {"score": 20.0, "scores": [20, 19, 21, 20]}, "g1": {"score": 10.0, "scores": [10, 12, 9, 11]},
             "g2": {"score": 5.0, "scores": [5, 4, 6, 5]}}
    for pol in (SoftmaxFloor(T=1.0, eps=0.2), NashRelative(eps=0.2)):
        out = pol.allocate(groups, stats, 30.0)
        assert sum(out.values()) == pytest.approx(30.0)
        assert min(out.values()) >= 0.2 / 3 * 30.0 - 1e-6
    nash = NashRelative(eps=0.2).allocate(groups, stats, 30.0)
    assert nash["g0"] == pytest.approx(0.2 / 3 * 30 + 0.8 * 30, abs=1e-3)  # the dominant group takes the mass


def test_nash_relative_resists_cloning_softmax_does_not():
    base = {"g0": {"score": 18.0, "scores": [18, 17, 19, 18, 18]}, "g1": {"score": 17.0, "scores": [17, 18, 18, 17, 16]},
            "g2": {"score": 16.0, "scores": [16, 16, 17, 15, 16]}}
    cloned = dict(base, g3=dict(base["g1"]))  # duplicate group g1
    n0 = NashRelative(eps=0.0).allocate(sorted(base), base, 1.0)
    n1 = NashRelative(eps=0.0).allocate(sorted(cloned), cloned, 1.0)
    assert n1["g1"] + n1["g3"] == pytest.approx(n0["g1"], abs=1e-3)  # clones split, never gain
    s0 = SoftmaxFloor(T=1.0, eps=0.0).allocate(sorted(base), base, 1.0)
    s1 = SoftmaxFloor(T=1.0, eps=0.0).allocate(sorted(cloned), cloned, 1.0)
    assert s1["g1"] + s1["g3"] > s0["g1"] + 0.05  # softmax rewards the clone attack
    assert maxent_nash(np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])) == pytest.approx([1 / 3] * 3, abs=1e-3)


# ------------------------------------------------------------------ selection
def test_shinka_weighted_matches_formula_and_prefers_unexplored_good_parents():
    sel = ShinkaWeighted(lam=10.0)
    for aid, score in [("a", 10.0), ("b", 15.0), ("c", 20.0), ("d", 20.0)]:
        sel.record("g", aid, score, None, 0)
    for _ in range(3):
        sel.record("g", f"c_child{_}", 1.0, "c", 1)
    w = dict(sel.weights("g"))
    sc = [10, 15, 20, 20, 1, 1, 1]
    med = float(np.median(sc))
    mad = float(np.median(np.abs(np.array(sc) - med)))
    expect_d = 1 / (1 + np.exp(-10 * (20 - med) / mad))
    assert w["d"] == pytest.approx(expect_d) and w["c"] == pytest.approx(expect_d / 4)
    rng = random.Random(0)
    picks = [sel.choose_parent(AgentState("x", "g"), rng) for _ in range(2000)]
    tot = sum(w.values())
    for k in ("a", "b", "c", "d"):  # sampling frequencies follow the weights (children penalize c, the median sets a)
        assert picks.count(k) / 2000 == pytest.approx(w[k] / tot, abs=0.03)
    assert picks.count("d") > picks.count("c")
    assert sel.accept(0.0, 25.0)


def test_hgm_clade_counts_thompson_and_widening():
    h = HGMCladeTS(alpha=0.6)
    # parent_score is the parent re-scored on the child's deals (fix round 1, step 4: labels are paired)
    h.record("g", "root", 10.0, None, 0)
    h.record("g", "good", 12.0, "root", 1, parent_score=10.0)    # success
    h.record("g", "good2", 13.0, "good", 2, parent_score=12.0)   # success in good's clade
    h.record("g", "bad", 5.0, "root", 1, parent_score=10.0)      # failure
    h.record("g", "bad2", 4.0, "bad", 2, parent_score=5.0)       # failure
    assert h.clade_counts("g", "root") == (2, 2)
    # clade counts include the node's own outcome (fix round 1, step 4)
    assert h.clade_counts("g", "good") == (2, 0) and h.clade_counts("g", "bad") == (0, 2)
    rng = random.Random(3)
    picks = [h.choose_parent(AgentState("x", "g"), rng) for _ in range(3000)]
    assert picks.count("good") > picks.count("bad")
    # widening: |T| = 5 nodes; expansion only once steps^0.6 >= 5, i.e. from step 15 on
    wants = [h.wants_revision(AgentState("x", "g"), rng) for _ in range(16)]
    assert wants.index(True) == 14


# ------------------------------------------------------------------ environment
def test_variant_switch_changes_rules_at_generation():
    v = VariantSwitch(at_generation=5, variant="hand4_clues6")
    base = {"players": 2, "colors": 5, "ranks": 5, "hand_size": 5, "max_information_tokens": 8, "max_life_tokens": 3}
    assert v.params_for(4, base) == base
    after = v.params_for(5, base)
    assert after["hand_size"] == 4 and after["max_information_tokens"] == 6 and "_note" in after
    assert VariantSwitch(3, {"colors": 4}).params_for(9, base)["colors"] == 4
    with pytest.raises(ValueError):
        VariantSwitch(3, {"players": 3})


# ------------------------------------------------------------------ end-to-end paths of phase-1 options not used elsewhere
def test_merge_llm_and_costly_teaching_through_the_loop(tmp_path):
    import json

    from culture.run.runner import run_config

    cfg = {
        "name": "merge_path",
        "population": {"groups": 1, "agents_per_group": 2, "warm_start_overrides": {"g0a0": "piers", "g0a1": "random"}},
        "evaluation": {"selfplay_games": 10, "crossplay_games": 0, "anchor_games": 4, "between_group_games": 0,
                       "ladder_every": 0, "workers": 0},
        "org": {"routing": "broadcast_group", "verification": {"name": "selfplay", "n": 10}, "adoption": "merge_llm",
                "teaching_cost": {"name": "costly", "c": 1.0, "credit_share": 0.5}},
        "budget": {"unit": "calls", "per_group_per_generation": 20},
        "corpus": {"enabled": False},
        "runner": {"generations": 2, "revise": False},
    }
    recs = run_config(cfg, tmp_path / "m")
    msgs = [json.loads(x) for x in (tmp_path / "m" / "messages.jsonl").read_text().splitlines()]
    assert any(m.get("decision") == "merged" for m in msgs)  # random receiver merges Piers via one LLM call
    ledger = [json.loads(x) for x in (tmp_path / "m" / "ledger.jsonl").read_text().splitlines()]
    assert any(r["tag"] == "merge" for r in ledger)
    assert recs[1]["teaching"]["merged"] >= 1
