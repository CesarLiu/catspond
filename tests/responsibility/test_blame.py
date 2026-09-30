import numpy as np
import pytest

from responsibility.blame import crash_blame, crash_partner, rear_end_rule, rollout_blame, split
from responsibility.rollouts import make_rollout, scene_from_rollout
from tests.responsibility.conftest import FakeModel, make_scene, track


def _cv(xy, vel):
    """Constant-velocity motion sets [n, 80, 2] from ``xy`` at the context step."""
    steps = np.arange(1, 81)[:, None] * 0.1
    traj = np.asarray(xy, dtype=np.float32) + np.asarray(vel, dtype=np.float32) * steps
    return lambda n: np.repeat(traj[None], n, 0)


def _at(scene, agent, step):
    return scene.position[scene.index(agent), step, :2]


def test_the_follower_is_to_blame_for_a_rear_end_collision():
    scene = make_scene({"0": track((0, 0), (5, 0)),  # ego, steady
                        "1": track((-15, 0), (12, 0)),  # follower closing in at 7 m/s
                        "2": track((0, 40), (5, 0))})  # far away
    model = FakeModel(None, samples_by_agent={
        0: _cv(_at(scene, "0", 10), (5, 0)),  # the ego's alternatives: what it did
        1: _cv(_at(scene, "1", 10), (5, 0)),  # the follower's: keep the gap
    })
    assert crash_partner(scene, 0, 28) == 1
    blame = crash_blame(model, scene, 0, crash_step=28)
    assert blame.window == 10 and blame.other_id == "1"
    assert blame.beta_ego == pytest.approx(0.0, abs=1e-5)
    assert blame.beta_other > 5.0
    assert blame.share == pytest.approx(0.0) and blame.verdict == "other"
    assert blame.rule == "other"  # the rear-end rule agrees: the follower is at fault


def test_the_ego_is_to_blame_for_cutting_in():
    scene = make_scene({"0": track((0, 3.5), (10, -1.75)),  # ego moving into the next lane
                        "1": track((8, 0), (5, 0))})  # slower car in that lane
    model = FakeModel(None, samples_by_agent={
        0: _cv(_at(scene, "0", 10), (10, 0)),  # the ego's alternatives: stay in lane
        1: _cv(_at(scene, "1", 10), (5, 0)),  # the other's: what it did
    })
    blame = crash_blame(model, scene, 0, crash_step=26, other=1)
    assert blame.beta_ego > 1.0
    assert blame.beta_other == pytest.approx(0.0, abs=1e-5)
    assert blame.share == pytest.approx(1.0) and blame.verdict == "ego"
    assert blame.rule == "n/a"  # side by side at contact: not a rear-end collision


def test_unpredicted_partners_and_missing_windows():
    scene = make_scene({"0": track((0, 0), (5, 0)), "1": track((-15, 0), (12, 0), kind="PEDESTRIAN")})
    model = FakeModel(_cv((0, 0), (5, 0)), predicted={0})
    blame = crash_blame(model, scene, 0, crash_step=28, other=1)
    assert blame.beta_other is None and blame.share is None and blame.verdict == "ego-only"
    # the partner only appears at the collision: no window with both
    valid = np.zeros(91, dtype=bool)
    valid[28:] = True
    late = make_scene({"0": track((0, 0), (5, 0)), "1": track((-15, 0), (12, 0), valid=valid)})
    assert crash_blame(model, late, 0, crash_step=28, other=1) is None


def test_split():
    assert split(0.0, 0.0) == (0.5, "shared")
    assert split(-1.0, -2.0) == (0.5, "shared")
    assert split(1.0, 1.05) == (pytest.approx(1.0 / 2.05), "shared")  # within the margin
    assert split(2.0, None) == (None, "ego")
    assert split(0.05, None) == (None, "ego-only")


def test_the_rear_end_rule():
    # the ego runs into a slower car ahead: the ego is the follower
    scene = make_scene({"0": track((0, 0), (12, 0)), "1": track((15, 0.5), (5, 0))})
    assert rear_end_rule(scene, 0, 1, 25) == "ego"
    assert rear_end_rule(scene, 1, 0, 25) == "other"
    assert rear_end_rule(scene, 0, 1, 30) == "ego"  # also once the boxes overlap
    # oncoming (heading 180 degrees apart), crossing (90 degrees): not rear-end
    oncoming = make_scene({"0": track((0, 0), (10, 0)), "1": track((40, 0), (-10, 0), heading=np.pi)})
    assert rear_end_rule(oncoming, 0, 1, 27) == "n/a"
    crossing = make_scene({"0": track((0, 0), (10, 0)), "1": track((20, -20), (0, 10), heading=np.pi / 2)})
    assert rear_end_rule(crossing, 0, 1, 29) == "n/a"
    # a side-swipe while travelling the same way: not rear-end either
    swipe = make_scene({"0": track((0, 0), (10, 0)), "1": track((0, 2.2), (10, -0.2))})
    assert rear_end_rule(swipe, 0, 1, 12) == "n/a"
    # never seen together
    valid = np.zeros(91, dtype=bool)
    valid[40:] = True
    apart = make_scene({"0": track((0, 0), (12, 0)), "1": track((15, 0), (5, 0), valid=valid)})
    assert rear_end_rule(apart, 0, 1, 31) == "n/a"


def test_a_rollouts_collision_is_attributed_in_its_rebuilt_scene():
    # the logged ego keeps its distance; the policy drives 12 m/s into the car ahead
    scene = make_scene({"0": track((0, 0), (5, 0)), "1": track((15, 0), (5, 0)), "2": track((0, 40), (5, 0))})
    t = np.arange(26) * 0.1 - 1.0  # contact at step 25
    ego = {"position": np.stack([12 * t, np.zeros_like(t)], -1), "heading": np.zeros_like(t)}
    rollout = make_rollout("0", "synthetic", "td3", "none", ego=ego,
                           end={"step": 25, "reason": "crash_vehicle", "crash_vehicle": True, "crash_with": "1"},
                           present={"track_ids": ["1", "2"], "mask": np.ones((2, 26), dtype=bool)})
    played = scene_from_rollout(scene, rollout)
    model = FakeModel(None, samples_by_agent={
        0: _cv(_at(played, "0", 10), (5, 0)),  # the ego's alternatives: the logged speed
        1: _cv(_at(played, "1", 10), (5, 0)),  # the other: what it did
    })
    blame = rollout_blame(model, played, rollout)
    assert blame.other_id == "1" and blame.window == 10
    assert blame.verdict == "ego" and blame.rule == "ego" and blame.share == pytest.approx(1.0)
    # after the end of the log there is nothing to attribute
    late = dict(rollout, end=dict(rollout["end"], step=95))
    assert rollout_blame(model, played, late) is None
