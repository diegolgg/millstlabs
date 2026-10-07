"""Fix round 1, step 13: credit estimators on synthetic replays with known answers."""

import itertools

import numpy as np
import pytest

from culture.analysis import credit as C


def game_from(a, b=None, c0=0.0):
    """v(S) = c0 + sum_{j in S} a_j + sum_{j<l in S} b_jl."""
    k = len(a)
    b = b or {}
    return {m: c0 + sum(a[j] for j in C.members(m, k))
            + sum(b.get((j, l), 0.0) for j, l in itertools.combinations(C.members(m, k), 2)) for m in C.subsets(k)}


@pytest.mark.parametrize("k", [2, 3, 5])
def test_shapley_efficiency_on_random_games(k):
    rng = np.random.default_rng(k)
    v = {m: float(rng.normal()) for m in C.subsets(k)}
    v[0] = 0.0
    assert C.shapley(v, k).sum() == pytest.approx(v[C.full(k)] - v[0])


def test_hand_computed_two_player_game():
    v = {0: 0.0, 1: 14.0, 2: -3.0, 3: -3.0}  # Piers then Flawed adopted blindly: Flawed wins
    assert C.shapley(v, 2) == pytest.approx([7.0, -10.0])
    assert C.tau(v, 2) == pytest.approx([7.0, -10.0])  # k = 2: Banzhaf equals Shapley
    assert C.leave_one_out(v, 2) == pytest.approx([0.0, -17.0])


def test_additive_game_all_estimators_agree():
    a = [14.0, -3.0, 0.0, 0.5, 0.0, -1.0, 2.0]
    v = game_from(a)
    for est in (C.shapley, C.tau, C.leave_one_out, C.singles_pairs, C.plackett_burman):
        assert est(v, 7) == pytest.approx(a), est.__name__
    rows = [(m, v[m]) for m in C.subsets(7)]
    assert C.ridge(rows, 7, lam=1e-9) == pytest.approx(a, abs=1e-6)


def test_second_order_game():
    a = [5.0, -2.0, 1.0, 0.0]
    b = {(0, 1): -4.0, (1, 2): 2.0, (0, 3): 1.0}
    v = game_from(a, b)
    expect = np.array(a, float)
    for (j, l), x in b.items():
        expect[j] += x / 2
        expect[l] += x / 2
    assert C.shapley(v, 4) == pytest.approx(expect)
    assert C.singles_pairs(v, 4) == pytest.approx(expect)  # exact for second-order games
    assert not np.allclose(C.leave_one_out(v, 4), expect)  # leave-one-out is first order only


def test_pb8_is_a_balanced_design():
    masks = C.pb8_masks(7)
    X = np.array([[m >> j & 1 for j in range(7)] for m in masks])
    assert len(masks) == 8 and (X.sum(0) == 4).all()
    S = 2 * X - 1
    assert (S.T @ S == 8 * np.eye(7)).all()  # orthogonal columns


def test_adopter_effect_and_equal_split():
    t = np.array([6.0, -2.0, 0.0])
    assert C.adopter_effect(t, [0.5, 1.0, 0.0]) == [12.0, -2.0, None]
    assert C.equal_split(10.0, [0, 2], 3) == pytest.approx([5.0, 0.0, 5.0])
    assert C.equal_split(10.0, [], 3) == pytest.approx([0.0, 0.0, 0.0])


def test_metrics_and_power():
    assert C.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert C.spearman([1, 2, 3], [2, 4, 9]) == pytest.approx(1.0) and C.spearman([1, 1, 1], [1, 2, 3]) is None
    assert C.sign_error(-1.0, -5.0) is False and C.sign_error(1.0, -5.0) is True and C.sign_error(1.0, 0.1) is None
    r1, r2 = C.power_r(1.0), C.power_r(2.0)
    assert 9 <= r1 <= 11 and r2 > r1 and C.power_r(0.0) is None


def test_evaluate_replicate_reports_all_estimators():
    v = game_from([14.0, -3.0, 0.0])
    adopted = {m: [j for j in C.members(m, 3) if j < 2] for m in C.subsets(3)}
    res = C.evaluate_replicate(v, 3, adopted, {}, sabotaged=1)
    assert res["efficiency_gap"] == pytest.approx(0.0)
    for n in ("tau_itt", "leave_one_out", "equal_split", "singles_pairs", "plackett_burman"):
        assert n in res["metrics"]
    assert res["metrics"]["leave_one_out"]["rmse"] == pytest.approx(0.0)
    assert res["adopt_rate"] == [1.0, 1.0, 0.0] and res["adopter_effect"][2] is None
