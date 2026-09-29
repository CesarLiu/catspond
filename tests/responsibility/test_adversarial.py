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
