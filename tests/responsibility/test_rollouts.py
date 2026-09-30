import numpy as np
import pytest

from responsibility.metrics import ResponsibilityConfig, scene_responsibility, window_steps
from responsibility.rollouts import (
    check_rollout,
    last_window_step,
    load_rollout,
    make_rollout,
    not_spawned,
    outcome,
    replay_rollout,
    save_rollout,
    scene_from_rollout,
)
from tests.responsibility.conftest import FakeModel, make_scene, track


def _scene():
    return make_scene({
        "0": track((0, 0), (10, 0)),  # ego
        "1": track((30, 3.5), (8, 0)),  # adversary (second object of interest)
        "2": track((50, -3.5), (0, 0)),  # parked car
        "3": track((-20, 0), (12, 0)),  # follower
        "4": track((5, 8), (0, 0), kind="PEDESTRIAN"),  # standing pedestrian: always spawned
    })


def _ego_track(scene, n, dy=0.0):
    pos = scene.position[scene.sdc, :n, :2].copy()
    pos[:, 1] += dy
    return {"position": pos, "heading": scene.heading[scene.sdc, :n]}


def test_static_vehicles_are_not_spawned():
    scene = _scene()
    assert not_spawned(scene) == [scene.index("2")]
    assert not_spawned(scene, no_static_vehicles=False) == []


def test_replayed_rollout_rebuilds_the_log_without_static_vehicles():
    scene = _scene()
    for with_present in (True, False):
        rebuilt = scene_from_rollout(scene, replay_rollout(scene, with_present=with_present))
        np.testing.assert_array_equal(rebuilt.position, scene.position)
        np.testing.assert_array_equal(rebuilt.velocity, scene.velocity)
        expected = scene.valid.copy()
        expected[scene.index("2")] = False
        np.testing.assert_array_equal(rebuilt.valid, expected)


def test_ego_follows_the_simulation_and_ends_with_the_episode():
    scene = _scene()
    n = 41  # ended at step 40
    rollout = make_rollout("7", scene.scenario_id, "td3", "none", ego=_ego_track(scene, n, dy=1.0),
                           end={"step": n - 1, "reason": "out_of_road", "out_of_road": True})
    rebuilt = scene_from_rollout(scene, rollout)
    sdc = scene.sdc
    np.testing.assert_allclose(rebuilt.position[sdc, :n, 1], scene.position[sdc, :n, 1] + 1.0)
    assert rebuilt.valid[sdc, :n].all() and not rebuilt.valid[sdc, n:].any()
    # velocity from finite differences when the rollout has none
    np.testing.assert_allclose(rebuilt.velocity[sdc, :n - 1], [[10.0, 0.0]] * (n - 1), atol=1e-3)
    # the logged scene is untouched
    assert scene.valid[sdc].all()


def test_adversary_is_replaced_and_nan_steps_are_invalid():
    scene = _scene()
    n = 60
    adv = scene.index("1")
    pos = scene.position[adv, :n, :2].copy()
    pos[20:, 1] -= 2.0  # cuts in
    pos[:3] = np.nan  # not yet spawned
    rollout = make_rollout(
        "7", scene.scenario_id, "td3", "cat", ego=_ego_track(scene, n),
        adversary={"track_id": "1", "position": pos, "heading": scene.heading[adv, :n]},
        end={"step": n - 1, "reason": "crash_vehicle", "crash_vehicle": True, "crash_with": "1"})
    rebuilt = scene_from_rollout(scene, rollout)
    np.testing.assert_allclose(rebuilt.position[adv, 20:n, 1], scene.position[adv, 20:n, 1] - 2.0)
    assert not rebuilt.valid[adv, :3].any() and rebuilt.valid[adv, 3:n].all() and not rebuilt.valid[adv, n:].any()


def test_presence_mask_restricts_the_other_objects():
    scene = _scene()
    n = 50
    follower = scene.index("3")
    mask = np.ones((2, n), dtype=bool)
    mask[0, 10:15] = False  # the follower was not spawned for a while (overlapping the ego)
    rollout = make_rollout("7", scene.scenario_id, "td3", "none", ego=_ego_track(scene, n),
                           end={"step": n - 1, "reason": "arrive_dest", "arrive_dest": True},
                           present={"track_ids": ["3", "1"], "mask": mask})
    rebuilt = scene_from_rollout(scene, rollout)
    assert not rebuilt.valid[follower, 10:15].any()
    assert rebuilt.valid[follower, :10].all() and rebuilt.valid[follower, 15:].all()
    # not listed = never in the simulation, whatever the log says
    assert not rebuilt.valid[scene.index("2")].any() and not rebuilt.valid[scene.index("4")].any()


def test_windows_stop_with_the_episode():
    scene = _scene()
    cfg = ResponsibilityConfig()
    crash = make_rollout("7", "s", "td3", "none", ego=_ego_track(scene, 51),
                         end={"step": 50, "reason": "crash_vehicle", "crash_vehicle": True})
    other = make_rollout("7", "s", "td3", "none", ego=_ego_track(scene, 51),
                         end={"step": 50, "reason": "out_of_road", "out_of_road": True})
    assert last_window_step(crash, cfg.metric_horizon) == 50
    assert last_window_step(other, cfg.metric_horizon) == 30
    rebuilt = scene_from_rollout(scene, other)
    assert window_steps(rebuilt, scene.sdc, cfg, last_window_step(other, cfg.metric_horizon))[-1] == 30
    # the metric only sees steps where the ego existed
    model = FakeModel(lambda n: np.repeat(scene.position[scene.sdc, 31:111, :2][None], n, 0))
    observations = scene_responsibility(model, rebuilt, scene.sdc, cfg, last_window_step(crash, 20))
    assert [o.step for o in observations] == list(range(10, 51, 5))


def test_rollout_layout_is_checked(tmp_path):
    scene = _scene()
    rollout = replay_rollout(scene, end_step=70, reason="max_step")
    save_rollout(rollout, tmp_path / "3.pkl")
    loaded = load_rollout(tmp_path / "3.pkl")
    assert loaded["end"]["step"] == 70 and len(loaded["ego"]["position"]) == 71
    row = outcome(loaded)
    assert row["reason"] == "max_step" and row["crash"] == 0
    bad = dict(rollout, end=dict(rollout["end"], step=69))
    with pytest.raises(ValueError, match="end step"):
        check_rollout(bad)
    with pytest.raises(ValueError, match="end reason"):
        check_rollout(dict(rollout, end=dict(rollout["end"], reason="bored")))
    with pytest.raises(ValueError, match="present mask"):
        check_rollout(dict(rollout, present={"track_ids": ["1"], "mask": np.ones((1, 3), bool)}))


def test_rollouts_longer_than_the_log_are_cut():
    scene = _scene()
    n = 120  # MetaDrive keeps going after the log ends
    ego = {"position": np.concatenate([scene.position[scene.sdc, :, :2]] + [scene.position[scene.sdc, -1:, :2]] * 29),
           "heading": np.zeros(n)}
    rollout = make_rollout("7", "s", "td3", "none", ego=ego, end={"step": n - 1, "reason": "max_step"})
    rebuilt = scene_from_rollout(scene, rollout)
    assert rebuilt.n_steps == scene.n_steps and rebuilt.valid[scene.sdc].all()
