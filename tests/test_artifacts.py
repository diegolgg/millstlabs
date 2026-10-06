"""Artifact schema, store, evidence-checked corpus, touch log, provenance DAG."""

import pytest

from culture.artifacts.provenance import Provenance
from culture.artifacts.schema import Artifact, Evaluation, TeachingMessage, artifact_id
from culture.artifacts.store import ArtifactStore, Corpus, EvidenceRegistry, EvidenceRejected, TouchLog


def ev_for(aid, gen=1, score=10.0):
    return Evaluation(artifact_id=aid, generation=gen, seed_base=0, n_games=2, selfplay_scores=[10, 10],
                      selfplay_iqm=score, selfplay_mean=score, selfplay_ci=(score, score), crossplay={}, anchors={},
                      illegal_rate=0.0, lines=10, complexity=3, wall_seconds=0.1)


def test_artifact_roundtrip_and_id(tmp_path):
    a = Artifact.make("code", "conv", author="a0", group="g0", generation=1, evaluation=ev_for("x"))
    assert a.id == artifact_id("code", "conv")
    store = ArtifactStore(tmp_path)
    store.put(a)
    fresh = ArtifactStore(tmp_path)
    b = fresh.get(a.id)
    assert b.to_dict() == a.to_dict() and a.id in fresh
    with pytest.raises(KeyError):
        fresh.get("nope")


def test_evidence_digest_ignores_wall_time():
    e1, e2 = ev_for("x"), ev_for("x")
    e2.wall_seconds = 99
    assert e1.digest() == e2.digest()
    e2.selfplay_mean = 11
    assert e1.digest() != e2.digest()


def test_corpus_deposit_validates_evidence_and_ownership():
    reg = EvidenceRegistry()
    a = Artifact.make("c", "v", author="a0", group="g0", generation=1)
    good = ev_for(a.id)
    reg.issue(good)
    corpus = Corpus("g0", capacity=2)
    with pytest.raises(EvidenceRejected):
        corpus.deposit(a, ev_for(a.id, score=25.0), "a0", reg, {"a0": a.id})  # forged claim
    with pytest.raises(EvidenceRejected):
        corpus.deposit(a, good, "a1", reg, {"a0": a.id, "a1": "other"})  # not the holder
    other = ev_for("zzz")
    reg.issue(other)
    with pytest.raises(EvidenceRejected):
        corpus.deposit(a, other, "a0", reg, {"a0": a.id})  # evidence about another artifact
    corpus.deposit(a, good, "a0", reg, {"a0": a.id})
    for i in range(3):
        b = Artifact.make(f"c{i}", "v", author="a1", group="g0", generation=1)
        e = ev_for(b.id, score=float(i))
        reg.issue(e)
        corpus.deposit(b, e, "a1", reg, {"a1": b.id})
    assert len(corpus.entries) == 2 and corpus.entries[0].score == 10.0
    touch = TouchLog()
    got = corpus.retrieve("a5", 2, k=1, touch=touch)
    assert got[0].artifact_id == a.id and touch.flush()[0]["receiver"] == "a5"
    assert Corpus.from_state(corpus.state_dict()).entries == corpus.entries
    reg.prune(10)
    assert not reg.check(good)


def test_provenance_queries(tmp_path):
    p = Provenance(tmp_path / "prov.jsonl")
    p.add("A", "t", "g0", 0, [])
    p.add("B", "t", "g0", 1, ["A"])
    p.add("S0", "s", "g1", 0, [])
    p.add("S1", "s", "g1", 1, ["S0", "B"], teaching=[{"message": "m1", "sender": "t", "source": "B"}])
    p.add("S2", "s", "g1", 2, ["S1"])
    assert p.ancestors("S2") == {"S1": 1, "S0": 2, "B": 2, "A": 3}
    assert p.clade("A") == {"A", "B", "S1", "S2"}
    assert p.depth("S2") == 3
    assert p.teachers_of("S2") == {"t": 1}
    assert ("B", "S1", "teaching") in p.edges()
    q = Provenance(tmp_path / "prov.jsonl")
    assert q.parents == p.parents and q.teaching == p.teaching


def test_teaching_message_serializes():
    m = TeachingMessage("m1", "a0", "a1", 3, "x", "because", ev_for("x"))
    d = m.to_dict()
    assert d["evidence"]["artifact_id"] == "x" and "selfplay_scores" not in d["evidence"]
