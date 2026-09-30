import numpy as np
import pytest

pytest.importorskip("tensorflow")  # advgen.adv_generator imports it

from responsibility.adversarial import (  # noqa: E402
    adversary_responsibility,
    cat_candidate_probs,
    cat_collision_scores,
    select,
)
from responsibility.metrics import ResponsibilityConfig  # noqa: E402

SIZE = {"w": 2.0, "l": 4.8}


def _line(y0, vy=0.0, vx=10.0, n=80):
    t = np.arange(1, n + 1) * 0.1
    return np.stack([vx * t, y0 + vy * t], -1)


def test_candidate_probabilities_follow_cat():
    scores = np.log(np.array([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02, 0.01]))
    p = cat_candidate_probs(scores)
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(p[6:], p[6])  # CAT flattens everything after the 7th mode
    assert p[0] > p[1] > p[2]


def test_collision_scores_detect_a_crossing_candidate():
    ego = _line(0.0)
    parallel = _line(8.0)  # next-but-one lane, never touches
    merging = _line(6.0, vy=-1.5)  # drifts into the ego's lane
    score, min_dist = cat_collision_scores(np.stack([parallel, merging]), np.array([0.5, 0.5]), [ego], [1.0],
                                           SIZE, SIZE)
    assert score[0] == 0 and score[1] > 0
    assert min_dist.dtype.kind == "i"  # CAT's integer array
    assert min_dist[1] < min_dist[0]


def test_responsibility_is_high_for_the_candidate_that_closes_in():
    ego = _line(0.0)
    samples = np.stack([_line(6.0) for _ in range(10)])  # the adversary usually keeps its lane
    candidates = np.stack([_line(6.0), _line(6.0, vy=-1.5)])
    beta = adversary_responsibility(samples, candidates, [ego], [1.0], ResponsibilityConfig(d_sat=10.0), 80)
    assert beta[0] == pytest.approx(0.0, abs=1e-6)
    assert beta[1] > 4.0


def test_selection_rules():
    score = np.array([0.0, 0.3, 0.1, 0.0])
    min_dist = np.array([5, 0, 1, 2])
    beta = np.array([0.0, 4.0, 0.8, 0.2])
    assert select("cat", score, min_dist, beta) == (1, "collision")
    assert select("constrained", score, min_dist, beta, threshold=1.0)[0] == 2
    assert select("constrained", score, min_dist, beta, threshold=0.5)[0] == 3  # closest within threshold
    assert select("constrained", score, min_dist, beta + 5, threshold=1.0)[0] == 0  # none qualifies: least beta
    assert select("penalized", score, min_dist, beta, penalty=1.0)[0] == 2  # 0.1*e^-0.8 > 0.3*e^-4
    assert select("penalized", score, min_dist, beta, penalty=100.0)[0] == 1
    assert select("cat", np.zeros(4), min_dist, beta) == (1, "closest")
    with pytest.raises(ValueError):
        select("nope", score, min_dist, beta)


def test_path_headings_follow_the_motion_and_hold_while_standing():
    from responsibility.adversarial import path_headings

    traj = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
    h = path_headings(traj, origin=np.array([0.0, 0.0]), start_heading=0.3)
    np.testing.assert_allclose(h, [0.3, 0.3, np.pi / 4, np.pi / 4, np.pi])
    batch = path_headings(np.stack([traj, traj]), origin=np.zeros(2), start_heading=0.3)
    assert batch.shape == (2, 5)


def _ego_motion_set():
    """Half the ego's alternatives keep 10 m/s in its lane, half brake to a stop within 1 s."""
    keep = _line(0.0)
    t = np.arange(1, 81) * 0.1
    brake = np.stack([np.minimum(10 * t - 5 * t ** 2, 5.0) * (t <= 1) + 5.0 * (t > 1), np.zeros(80)], -1)
    return np.stack([keep] * 5 + [brake] * 5)


def test_avoidability_separates_unavoidable_from_avoidable_adversaries():
    from responsibility.adversarial import ego_avoidability

    t = np.arange(1, 81) * 0.1
    parallel = _line(8.0)  # another lane throughout: nothing to avoid
    head_on = np.stack([60.0 - 12.0 * t, np.zeros(80)], -1)  # sweeps the whole lane: nobody escapes
    lateral = np.clip(3.5 - 3.5 * t, 0.0, None)
    cut_in = np.stack([15.0 + 5.0 * t, lateral], -1)  # slow car cutting in ahead: only braking escapes
    avoid, hits = ego_avoidability(_ego_motion_set(), np.stack([parallel, head_on, cut_in]),
                                   (np.zeros(2), 0.0), (np.array([15.0, 3.5]), 0.0), SIZE, SIZE)
    np.testing.assert_allclose(avoid, [1.0, 0.0, 0.5])
    assert hits.shape == (10, 3)
    assert hits[:5, 2].all() and not hits[5:, 2].any()
