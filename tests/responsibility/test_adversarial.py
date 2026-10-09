import numpy as np
import pytest

pytest.importorskip("tensorflow")  # advgen.adv_generator imports it

from responsibility.adversarial import (  # noqa: E402
    adversary_responsibility,
    cat_candidate_probs,
    cat_collision_scores,
    closest_approach,
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


def test_fair_selection_needs_avoidable_candidates():
    score = np.array([0.0, 0.3, 0.1, 0.05, 0.0])
    min_dist = np.array([5, 0, 1, 1, 2])
    beta = np.array([0.0, 4.0, 0.8, 0.2, 0.1])
    avoid = np.array([1.0, 0.0, 0.1, 0.6, 0.9])
    # constrained would take candidate 2, which the ego can hardly avoid
    assert select("constrained", score, min_dist, beta, threshold=1.0)[0] == 2
    assert select("fair", score, min_dist, beta, threshold=1.0, avoid=avoid, min_avoid=0.3) == (
        3, "avoidable collision within threshold")
    # no avoidable collision: the closest avoidable candidate within the threshold
    assert select("fair", score, min_dist, beta, threshold=1.0, avoid=avoid, min_avoid=0.95)[0] == 0
    # nothing avoidable enough: the most avoidable one within the threshold
    assert select("fair", score, min_dist, beta, threshold=1.0, avoid=avoid * 0.5, min_avoid=0.8) == (
        0, "most avoidable (none avoidable enough)")
    # nothing within the threshold either: the most avoidable of all
    assert select("fair", score, min_dist, beta + 10, threshold=1.0, avoid=avoid, min_avoid=0.3)[0] == 0
    with pytest.raises(ValueError, match="avoidability"):
        select("fair", score, min_dist, beta)


def test_runs_are_named_by_their_selection():
    from types import SimpleNamespace

    from responsibility.adversarial import selection_name

    args = SimpleNamespace(adv_selection="fair", resp_threshold=1.0, resp_penalty=1.0, resp_avoid=0.3)
    assert selection_name(args) == "fair1_0.3"
    assert selection_name(SimpleNamespace(adv_selection="cat")) == "cat"


def test_the_plan_is_applied_at_the_scene_step():
    from types import SimpleNamespace

    from advgen.adv_generator import StepAlignedPlan

    rows = np.zeros((91, 5))
    rows[:, 0] = np.arange(91)  # x = the plan's step
    rows[:, 1] = 100.0  # (0, 0) would read as a row without a logged state
    env = SimpleNamespace(engine=SimpleNamespace(episode_step=0))
    plan = StepAlignedPlan(rows, env)
    assert np.array(plan).shape == (91, 5)  # still the whole plan
    # spawned at reset: the k-th env.step applies row k - 1, as CAT's manager did
    applied = []
    for k in range(1, 6):
        env.engine.episode_step = k
        applied.append(plan.pop(0)[0])
    assert applied == [0, 1, 2, 3, 4]
    # spawned at step 8 (it appears late in the log): its first pop is row 8, not row 0
    env.engine.episode_step = 9
    assert plan.pop(0)[0] == 8 and len(plan) == 91
    env.engine.episode_step = 200  # past the end: the last row
    assert plan.pop(0)[0] == 90
    # history rows without a logged state (zeros) hold the nearest logged one
    rows[:2] = 0.0  # appears at step 2
    rows[5] = 0.0  # and is missing at step 5
    held = np.array(StepAlignedPlan(rows, env))
    assert held[:3, 0].tolist() == [2, 2, 2] and held[5, 0] == 4 and held[5, 2] == 0.0
    assert held[6:, 0].tolist() == list(range(6, 91))
    # once the env has moved to another scenario the plan is over: the traffic manager,
    # which keeps it until after the next reset, must not apply it there
    env = SimpleNamespace(engine=SimpleNamespace(episode_step=0), current_seed=7)
    plan = StepAlignedPlan(rows, env)
    assert len(plan) == 91 and plan
    env.current_seed = 8
    assert len(plan) == 0 and not plan


def test_closest_approach_is_the_smallest_gap_between_the_footprints():
    ego = _line(0.0)
    side = np.sqrt(0.8 ** 2 + 1.0 ** 2)  # radius of each of the 3 circles covering a 4.8 x 2 m car
    start = (np.zeros(2), 0.0)

    def approach(y0, vy=0.0, trajs_av=(ego,), probs_av=(1.0,)):  # a candidate starting at (0, y0)
        adv_start = (np.array([0.0, y0]), float(np.arctan2(vy, 10.0)))
        return closest_approach(_line(y0, vy)[None], list(trajs_av), list(probs_av), adv_start, start, SIZE, SIZE)

    assert approach(3.0)[0][0] == pytest.approx(3.0 - 2 * side, abs=1e-6)  # beside
    assert approach(6.0)[0][0] == pytest.approx(6.0 - 2 * side, abs=1e-6)  # further
    gap, gap_min = approach(-8.0, vy=2.0)  # crossing the ego's lane
    assert gap[0] < 0 and gap_min[0] == gap[0]
    # two ego trajectories from the same start: the P(AV_i)-weighted mean gap, and the smaller one
    drift = _line(0.0, vy=0.2)  # 1.6 m toward the candidate by 8 s
    alone = [approach(3.0, trajs_av=(t,))[0][0] for t in (ego, drift)]
    gap2, gap_min2 = approach(3.0, trajs_av=(ego, drift), probs_av=(0.75, 0.25))
    assert alone[1] < alone[0]
    assert gap2[0] == pytest.approx(0.75 * alone[0] + 0.25 * alone[1])
    assert gap_min2[0] == pytest.approx(alone[1])

def test_near_rule_takes_the_most_likely_near_miss_and_never_a_collision():
    gap = np.array([-0.5, 0.4, 1.1, 0.9, 3.0])
    gap_min = gap.copy()
    score = np.array([0.3, 0.0, 0.0, 0.0, 0.0])
    prob = np.array([0.5, 0.2, 0.05, 0.15, 0.1])
    near = dict(gap=gap, gap_min=gap_min, prob=prob, target_gap=1.0, gap_tol=0.5)
    assert select("near", score, np.zeros(5), None, **near) == (3, "most likely near miss")  # 0.9 m, likelier than 1.1
    assert select("near", score, np.zeros(5), None, **dict(near, gap_tol=0.05))[0] == 3  # closest to 1.0
    collide = dict(near, gap_min=np.full(5, -1.0))
    assert select("near", score, np.zeros(5), None, **collide) == (4, "largest gap (every candidate collides)")
    with pytest.raises(ValueError):
        select("near", score, np.zeros(5), None)
