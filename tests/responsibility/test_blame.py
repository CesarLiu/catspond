import numpy as np
import pytest

from responsibility.blame import crash_blame, crash_partner, split
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
