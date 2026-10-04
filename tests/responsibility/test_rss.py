import numpy as np
import pytest

from responsibility.blame import rear_end_rule
from responsibility.rollouts import make_rollout, scene_from_rollout
from responsibility.rss import RssParams, max_speed, min_speed, rollout_rss, rss_blame, rss_weight, \
    safe_lateral, safe_longitudinal
from tests.responsibility.conftest import N_STEPS, make_scene, track

DT = 0.1


def _path(x, y, length=4.8, width=2.0):
    """A track along given positions over the 91 steps, heading and velocity
    following the motion."""
    pos = np.zeros((N_STEPS, 3), dtype=np.float32)
    pos[:, 0], pos[:, 1] = x, y
    vel = np.stack([np.gradient(np.asarray(x, float), DT), np.gradient(np.asarray(y, float), DT)], -1)
    return {"type": "VEHICLE",
            "state": {"position": pos, "heading": np.arctan2(vel[:, 1], vel[:, 0]).astype(np.float32),
                      "velocity": vel.astype(np.float32), "length": np.full(N_STEPS, length, np.float32),
                      "width": np.full(N_STEPS, width, np.float32), "height": np.full(N_STEPS, 1.5, np.float32),
                      "valid": np.ones(N_STEPS, dtype=bool)},
            "metadata": {}}


def _drive(x0, v0, accel):
    """Positions of a car starting at x0 with speed v0, accelerating at
    accel(step) m/s^2 (never reversing)."""
    x, v, out = float(x0), float(v0), []
    for k in range(N_STEPS):
        out.append(x)
        a = accel(k)
        v_next = max(v + a * DT, 0.0)
        x += (v + v_next) / 2 * DT
        v = v_next
    return np.array(out)


def _first_contact(x_rear, x_front, length=4.8):
    return int(np.argmax(x_front - x_rear - length <= 0))


def test_safe_distances():
    # both at 10 m/s: 10 + 1.75 + 13.5^2 / 8 - 10^2 / 16
    assert safe_longitudinal(10.0, 10.0, RssParams()) == pytest.approx(28.28, abs=0.01)
    assert safe_longitudinal(0.0, 20.0, RssParams()) == 0.0  # a fast car ahead of a stopped one
    # no lateral motion: mu + twice (0.1 + 0.2^2 / 1.6)
    assert safe_lateral(0.0, 0.0, RssParams()) == pytest.approx(0.35)
    # a car moving away keeps moving away while it brakes laterally
    assert safe_lateral(-1.0, 0.0, RssParams()) == pytest.approx(0.1)


def test_response_envelopes():
    tau = np.array([0.0, 0.5, 1.0, 3.0, 10.0])
    # 10 m/s, up at 3.5 m/s^2 for 1 s, then down at 4 m/s^2 to a stop
    assert max_speed(10.0, tau, 3.5, 4.0, 1.0) == pytest.approx([10.0, 11.75, 13.5, 5.5, 0.0])
    assert min_speed(10.0, tau, 8.0) == pytest.approx([10.0, 6.0, 2.0, 0.0, 0.0])
    # moving away laterally after the response time: any speed up to 0
    assert max_speed(-1.0, tau, 0.2, 0.8, 1.0) == pytest.approx([-1.0, -0.9, -0.8, 0.0, 0.0])


def test_the_tailgater_is_to_blame_once_the_gap_became_unsafe():
    # same lane; the ego (ahead) at 5 m/s, a car 80 m behind closing at 15 m/s without braking
    scene = make_scene({"0": _path(_drive(0, 5, lambda k: 0), np.zeros(N_STEPS)),
                        "1": _path(_drive(-80, 15, lambda k: 0), np.zeros(N_STEPS))})
    crash = _first_contact(scene.position[1, :, 0], scene.position[0, :, 0])
    rss = rss_blame(scene, 0, 1, crash)
    assert rss.verdict == "other" and rss.case == "lon"
    assert 0 < rss.danger_step < crash  # the gap was safe at first
    assert rss.ego_proper and not rss.other_proper
    assert rss_blame(scene, 1, 0, crash).verdict == "ego"  # the same collision seen from the tailgater
    assert rear_end_rule(scene, 0, 1, crash) == "other"  # the rear-end rule agrees here


def test_a_brake_check_is_the_front_cars_fault():
    # both at 10 m/s, just over the safe distance (28.28 m) apart; at step 20 the
    # car ahead brakes at 20 m/s^2 (beyond brake_max); the ego behind responds
    # properly (at most 3.5 m/s^2 for 1 s, then braking at 4 m/s^2) and still hits it
    front = _drive(28.5 + 4.8, 10, lambda k: -20.0 if k >= 20 else 0.0)
    rear = _drive(0, 10, lambda k: 0.0 if k < 20 else 3.4 if k < 30 else -4.05)
    crash = _first_contact(rear, front)
    assert 30 < crash < 91
    scene = make_scene({"0": _path(rear, np.zeros(N_STEPS)), "1": _path(front, np.zeros(N_STEPS))})
    rss = rss_blame(scene, 0, 1, crash)
    assert rss.case == "lon" and rss.danger_step >= 19
    assert rss.ego_proper and not rss.other_proper and rss.verdict == "other"
    assert rear_end_rule(scene, 0, 1, crash) == "ego"  # the rear-end rule blames the follower regardless


def test_cutting_in_is_the_cutters_fault():
    # side by side in neighbouring lanes (3.5 m apart) at 10 m/s; from step 20 the
    # ego drifts toward the other at 1 m/s^2 lateral, up to 1 m/s
    x = _drive(0, 10, lambda k: 0)
    vy = np.clip((np.arange(N_STEPS) - 20) * DT * 1.0, 0.0, 1.0)
    y = 3.5 - np.concatenate([[0.0], np.cumsum((vy[1:] + vy[:-1]) / 2 * DT)])
    scene = make_scene({"0": _path(x, y), "1": _path(x + 1.0, np.zeros(N_STEPS))})
    crash = int(np.argmax(y - 2.0 <= 0))
    rss = rss_blame(scene, 0, 1, crash)
    assert rss.case == "lat" and rss.danger_step > 20
    assert rss.verdict == "ego" and not rss.ego_proper and rss.other_proper
    assert rear_end_rule(scene, 0, 1, crash) == "n/a"  # side by side: the rear-end rule says nothing
    # both drifting toward each other: both responsible
    both = make_scene({"0": _path(x, 1.75 + (y - 1.75) / 2), "1": _path(x + 1.0, 1.75 - (y - 1.75) / 2 - 1.75)})
    crash_both = int(np.argmax(both.position[0, :, 1] - both.position[1, :, 1] - 2.0 <= 0))
    assert rss_blame(both, 0, 1, crash_both).verdict == "shared"


def test_dangerous_from_the_first_step():
    # the follower is already far too close when the clip starts and closes in at 7 m/s
    scene = make_scene({"0": track((0, 0), (5, 0)), "1": track((-15, 0), (12, 0))})
    rss = rss_blame(scene, 0, 1, 28)
    assert rss.case == "lon+lat/first-step" and rss.danger_step == 0
    assert rss.verdict == "other"


def test_no_verdict_outside_rss_scope():
    oncoming = make_scene({"0": track((0, 0), (10, 0)), "1": track((40, 0), (-10, 0), heading=np.pi)})
    assert rss_blame(oncoming, 0, 1, 27).case == "not-same-direction"
    crossing = make_scene({"0": track((0, 0), (10, 0)), "1": track((20, -20), (0, 10), heading=np.pi / 2)})
    assert rss_blame(crossing, 0, 1, 29).verdict == "n/a"
    valid = np.zeros(91, dtype=bool)
    valid[40:] = True
    apart = make_scene({"0": track((0, 0), (12, 0)), "1": track((15, 0), (5, 0), valid=valid)})
    assert rss_blame(apart, 0, 1, 31).case == "not-seen-together"


def test_a_rollouts_collision_and_the_weight():
    # the logged ego keeps its distance; the policy drives 12 m/s into the car ahead
    scene = make_scene({"0": track((0, 0), (5, 0)), "1": track((15, 0), (5, 0)), "2": track((0, 40), (5, 0))})
    t = np.arange(26) * 0.1 - 1.0  # contact at step 25
    ego = {"position": np.stack([12 * t, np.zeros_like(t)], -1), "heading": np.zeros_like(t)}
    rollout = make_rollout("0", "synthetic", "td3", "none", ego=ego,
                           end={"step": 25, "reason": "crash_vehicle", "crash_vehicle": True, "crash_with": "1"},
                           present={"track_ids": ["1", "2"], "mask": np.ones((2, 26), dtype=bool)})
    rss = rollout_rss(scene_from_rollout(scene, rollout), rollout)
    assert rss.verdict == "ego" and rss_weight(rss) == 1.0
    assert rss_weight(None) == 1.0
    assert rss_weight(rss_blame(make_scene({"0": track((0, 0), (5, 0)), "1": track((-15, 0), (12, 0))}),
                                0, 1, 28)) == 0.0  # rear-ended: no penalty
