"""Fix round 1, step 4: HGM success labels compare a candidate with its parent on the same deals, ties are unlabelled,
and clade counts include the node's own outcome."""

from culture.artifacts.schema import Artifact
from culture.bots.anchors import anchor_conventions, anchor_source
from culture.org.selection import HGMCladeTS
from culture.run.config import from_dict
from culture.run.context import RunContext
from culture.run.generation import paired_parent_score
from culture.run.runner import run_config

CFG = {
    "name": "hgm_paired",
    "population": {"groups": 1, "agents_per_group": 3, "seed": 2},
    "evaluation": {"selfplay_games": 10, "crossplay_games": 0, "anchor_games": 4, "between_group_games": 0,
                   "ladder_every": 0, "workers": 0},
    "org": {"selection": {"name": "hgm_clade_ts", "alpha": 1.0}},
    "runner": {"generations": 4},
}


def test_tie_is_unlabelled_and_margin_respected():
    h = HGMCladeTS()
    h.record("g", "p", 10.0, None, 0)
    h.record("g", "same", 10.0, "p", 1, parent_score=10.0)
    assert h.tree["g"]["same"]["ok"] is None and h.clade_counts("g", "same") == (0, 0)
    h.record("g", "unpaired", 99.0, "p", 1)  # no paired parent score: never compared across deals
    assert h.tree["g"]["unpaired"]["ok"] is None
    hm = HGMCladeTS(margin=0.5)
    hm.record("g", "p", 10.0, None, 0)
    hm.record("g", "small", 10.4, "p", 1, parent_score=10.0)
    hm.record("g", "big", 10.6, "p", 1, parent_score=10.0)
    assert hm.tree["g"]["small"]["ok"] is None and hm.tree["g"]["big"]["ok"] is True


def test_behaviourally_identical_candidate_never_flips_with_seed_noise():
    """A candidate that plays exactly like its parent ties on every generation's deals (paired), so it is never
    labelled; the old unpaired rule compared it with the parent's score on an earlier generation's deals."""
    ctx = RunContext(from_dict(CFG), None)
    try:
        parent = Artifact.make(anchor_source("iggi"), anchor_conventions("iggi"), author="t", group="g0", generation=0)
        twin = Artifact.make(anchor_source("iggi") + "# a comment only\n", anchor_conventions("iggi"), author="t",
                             group="g0", generation=1)
        for a in (parent, twin):
            ctx.add_artifact(a)
        labels, unpaired = [], []
        for g in (1, 2, 3, 4):
            h = HGMCladeTS()
            h.record("g0", parent.id, paired_parent_score(ctx, parent.id, g - 1), None, g - 1)
            c = ctx.selfplay(twin.id, ctx.seeds(g))[0]
            h.record("g0", twin.id, c, parent.id, g, parent_score=paired_parent_score(ctx, parent.id, g))
            labels.append(h.tree["g0"][twin.id]["ok"])
            unpaired.append(c > paired_parent_score(ctx, parent.id, g - 1))  # what the old code did
        assert labels == [None] * 4
        # the parent's score moves between generations' deals, so the old rule's label depended on seed noise
        scores = {paired_parent_score(ctx, parent.id, g) for g in range(5)}
        assert len(scores) > 1
    finally:
        ctx.close()


def test_run_labels_match_paired_scores(tmp_path):
    run_config(CFG, tmp_path / "r")
    import json

    recs = [json.loads(x) for x in (tmp_path / "r" / "generations.jsonl").read_text().splitlines()]
    ck = json.loads((tmp_path / "r" / "checkpoint.json").read_text())
    tree = ck["state"]["policies"]["selection"]["tree"]["g0"]
    seen, checked = 0, set()
    for r in recs[1:]:
        for a, v in sorted(r["agents"].items()):
            cid = v["candidate"]
            if not cid or cid in checked or cid not in tree or not tree[cid]["parent"]:
                continue
            checked.add(cid)  # the tree keeps the label from the candidate's first evaluation
            d = v["candidate_score"] - v["parent_score"]  # both means of integer scores over 10 deals: exact
            want = True if d > 0 else False if d < 0 else None
            assert tree[cid]["ok"] == want, (r["generation"], a, d)
            seen += 1
    assert seen >= 1
