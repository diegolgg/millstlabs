"""Fix round 1, step 6: cross-play does not depend on argument order and each pair is played once per seed set."""

from culture.bots.anchors import anchor_source
from culture.bots.runner import BotSpec
from culture.evaluate.crossplay import crossplay, crossplay_matrix, pair_jobs
from culture.evaluate.pool import Evaluator
from culture.game.hanabi import HanabiParams

A = BotSpec("zz_iggi", anchor_source("iggi"))
B = BotSpec("aa_piers", anchor_source("piers"))
SEEDS = list(range(101, 121))


def test_argument_order_does_not_matter_and_pair_is_played_once():
    ev = Evaluator(HanabiParams(), workers=0)
    ab = crossplay(ev, A, B, SEEDS)
    n = ev.games_played
    ba = crossplay(ev, B, A, SEEDS)
    assert ev.games_played == n == len(SEEDS)  # the reversed request is served entirely from the memo
    assert [(r.seed, r.score, r.digest, r.seats) for r in ab] == [(r.seed, r.score, r.digest, r.seats) for r in ba]
    # fresh evaluators agree too (not just the memo)
    ev2 = Evaluator(HanabiParams(), workers=0)
    assert [r.digest for r in crossplay(ev2, B, A, SEEDS)] == [r.digest for r in ab]


def test_seats_alternate_by_seed_parity():
    for r in crossplay(Evaluator(HanabiParams(), workers=0), A, B, SEEDS):
        assert r.seats[0] == (B.key if r.seed % 2 == 0 else A.key)  # lower key in seat 0 on even seeds
    jobs = pair_jobs(A, B, SEEDS)
    assert jobs == pair_jobs(B, A, SEEDS) and sorted(s for _, ss in jobs for s in ss) == SEEDS


def test_matrix_is_symmetric_and_order_free():
    ev = Evaluator(HanabiParams(), workers=0)
    m1 = crossplay_matrix(ev, [A, B], SEEDS[:10])
    m2 = crossplay_matrix(Evaluator(HanabiParams(), workers=0), [B, A], SEEDS[:10])
    assert m1[0, 1] == m1[1, 0] == m2[0, 1] and m1[0, 0] == m2[1, 1]
