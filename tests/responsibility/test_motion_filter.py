import numpy as np
import pytest

from responsibility.metrics import ResponsibilityConfig, responsibility_at
from responsibility.motion_filter import (MotionFilter, MotionFilterConfig, drivable, kinematically_feasible,
                                          route_deviation)
from tests.responsibility.conftest import FakeModel, make_scene, track

T = np.arange(1, 81) * 0.1  # the 80 predicted steps after the current one (step 10)


def path(x, y):
    """A trajectory [80, 2] from position functions of the time after step 10."""
    return np.stack([x(T), y(T)], axis=-1)


STRAIGHT = path(lambda t: 10 + 10 * t, lambda t: 0 * t)  # the logged motion of agent 0 below
BRAKING = path(lambda t: 10 + 10 * np.minimum(t, 2.5) - 2 * np.minimum(t, 2.5) ** 2, lambda t: 0 * t)  # 4 m/s^2 to a stop
TURNING = path(lambda t: 10 + 10 * np.minimum(t, 2) + 0 * t, lambda t: np.where(t > 2, 10 * (t - 2), 0))
PARALLEL = path(lambda t: 10 + 10 * t, lambda t: 5 + 0 * t)  # 5 m beside the lane, from the start


def scene():
    # agent 0 drives along the lane (y = 0) at 10 m/s, at x = 10 at step 10
    return make_scene({"0": track((10, 0), (10, 0)), "1": track((18, 0), (10, 0)),
                       "2": track((27, 3.0), (0, 0))})


def test_route_keeps_braking_and_drops_another_route():
    dev = route_deviation(scene(), 0, 10, np.stack([STRAIGHT, BRAKING, TURNING]))
    assert dev[0] == pytest.approx(0, abs=1e-6) and dev[1] == pytest.approx(0, abs=1e-6)
    assert dev[2] > 10  # 10 m/s sideways for up to 6 s


def test_drivable_and_kinematics():
    s = scene()
    assert drivable(s, np.stack([STRAIGHT, PARALLEL]), 3.0).tolist() == [True, False]
    cfg = MotionFilterConfig(kinematics=True)
    jump = STRAIGHT.copy()
    jump[10:] += [20.0, 0.0]  # 20 m in 0.1 s
    hard = path(lambda t: 10 + 10 * np.minimum(t, 0.6) - 8 * np.minimum(t, 0.6) ** 2, lambda t: 0 * t)  # 16 m/s^2
    ok = kinematically_feasible(np.stack([STRAIGHT, BRAKING, jump, hard]), np.array([10.0, 0.0]), 10.0, cfg)
    assert ok.tolist() == [True, True, False, False]


def test_collision_with_a_third_agent_but_not_with_the_measured_one():
    s = scene()
    drift = path(lambda t: 10 + 10 * t, lambda t: 3.0 * np.minimum(t, 1.0))  # into the lane beside, where agent 2 stands
    f = MotionFilter(s, 0, 10, np.stack([STRAIGHT, drift]), 20, [1, 2], MotionFilterConfig(collision=True))
    assert f.keep(1).tolist() == [0]  # toward agent 1: driving through agent 2 is no alternative
    assert f.keep(2).tolist() == [0, 1]  # toward agent 2: getting close to it is what beta_s measures
    assert f.stats(f.keep(1)) == {"kept": 1, "kept_mass": 0.5}


def test_nothing_on_the_route_keeps_the_closest():
    off = [path(lambda t, d=d: 10 + 10 * t, lambda t, d=d: d * np.minimum(t, 1.0)) for d in (6.0, 4.0, 8.0)]
    f = MotionFilter(scene(), 0, 10, np.stack(off), 20, [1], MotionFilterConfig(route_tolerance=2.0))
    assert f.keep(1).tolist() == [1]


def test_weighted_set_reports_the_kept_mass():
    f = MotionFilter(scene(), 0, 10, np.stack([STRAIGHT, TURNING, BRAKING]), 20, [1],
                     MotionFilterConfig(route_tolerance=2.0), weights=np.array([0.5, 0.3, 0.2]))
    assert f.keep(1).tolist() == [0, 2] and f.stats(f.keep(1))["kept_mass"] == pytest.approx(0.7)


def test_another_route_no_longer_counts_as_a_safer_alternative():
    # agent 0 closes in on the slower agent 1 (8 m ahead, 5 m/s slower): it hits it within 2 s;
    # half its alternatives swerve off sideways at 5 m/s instead
    s = make_scene({"0": track((10, 0), (10, 0)), "1": track((18, 0), (5, 0))})
    swerve = path(lambda t: 10 + 10 * t, lambda t: 5 * np.minimum(t, 2.0))
    model = FakeModel(lambda n: np.stack([STRAIGHT, swerve] * (n // 2)))
    plain = responsibility_at(model, s, 0, 10, ResponsibilityConfig(courtesy=False))
    route = responsibility_at(model, s, 0, 10, ResponsibilityConfig(
        courtesy=False, filter=MotionFilterConfig(route_tolerance=2.0)))
    assert plain.safety > 1.0  # swerving away "would have kept" metres more
    assert route.safety == pytest.approx(0.0, abs=1e-5)
    entry = route.per_neighbour["1"]
    assert entry["kept"] == 20 and entry["kept_mass"] == 0.5
