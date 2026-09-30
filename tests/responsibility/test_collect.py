"""collect_rollouts.py's recording against a stand-in for MetaDrive's env."""

from types import SimpleNamespace

import numpy as np
import pytest

from responsibility.rollouts import check_rollout
from scripts.responsibility import collect_rollouts as col


class Obj:
    def __init__(self, x, y, vx=10.0):
        self.position = np.array([x, y])
        self.heading_theta = 0.0
        self.velocity = np.array([vx, 0.0])
        self.top_down_length, self.top_down_width = 4.5, 1.9


class FakeEnv:
    """Ego along x at 10 m/s; car "5" ahead from step 3 on; the adversary "9"
    follows a plan one step late, as CAT's traffic manager applies it."""

    def __init__(self, crash_at=20, plan=None):
        self.crash_at, self.plan = crash_at, plan
        self.t = 0
        self.vehicle = Obj(0.0, 0.0)
        self.objects = {}
        self.engine = SimpleNamespace(traffic_manager=SimpleNamespace(_scenario_id_to_obj_id={}),
                                      get_objects=lambda ids: {i: self.objects[i] for i in ids if i in self.objects})
        n = 91
        tracks = {"0": {"state": {"position": np.stack([np.arange(n), np.zeros(n)], -1), "valid": np.ones(n, bool)}},
                  "9": {"state": {"position": np.stack([np.arange(n), np.full(n, 20.0)], -1),
                                  "valid": np.ones(n, bool)}}}
        self.engine.data_manager = SimpleNamespace(current_scenario={"tracks": tracks,
                                                                     "metadata": {"sdc_id": "0"}})
        self._sync()

    def _sync(self):
        ids = self.engine.traffic_manager._scenario_id_to_obj_id
        ids.clear()
        if self.t >= 3:
            ids["5"] = "5"
            self.objects["5"] = Obj(self.vehicle.position[0] + 4.0, 0.0)
        if self.plan is not None:
            ids["9"] = "9"
            x, y = self.plan[max(self.t - 1, 0), :2]
            self.objects["9"] = Obj(x, y)

    def step(self, action):
        self.t += 1
        self.vehicle.position = self.vehicle.position + np.array([1.0, 0.0])
        self._sync()
        crash = self.t == self.crash_at
        info = {"crash_vehicle": crash, "route_completion": self.t / 100}
        return None, 0.0, crash, info


def test_a_normal_episode_is_recorded_step_by_step():
    env = FakeEnv(crash_at=20)
    recorder = col.Recorder(env)
    info = col.play(env, None, lambda s: [0, 0], recorder)
    rollout = recorder.rollout("401", "abc", "td3", "none", info)
    check_rollout(rollout)
    assert rollout["end"]["step"] == 20 and rollout["end"]["reason"] == "crash_vehicle"
    assert rollout["end"]["crash_with"] == "5"
    np.testing.assert_allclose(rollout["ego"]["position"][:, 0], np.arange(21))
    assert rollout["present"]["track_ids"] == ["5"]
    assert rollout["present"]["mask"][0].tolist() == [False] * 3 + [True] * 18
    assert col.replay_error(env, rollout) == pytest.approx(0.0)


def test_the_adversary_and_its_lag_are_recorded():
    plan = np.zeros((91, 5))
    plan[:, 0] = np.arange(91)
    plan[:, 1] = 20.0 - np.minimum(np.arange(91), 20)  # swerves toward the ego's lane
    env = FakeEnv(crash_at=30, plan=plan)
    recorder = col.Recorder(env, adversary="9")
    info = col.play(env, None, lambda s: [0, 0], recorder, max_steps=25)
    rollout = recorder.rollout("401", "abc", "td3", "cat", info, planned=plan)
    assert rollout["end"]["step"] == 25 and rollout["end"]["reason"] == "max_step"
    assert rollout["adversary"]["track_id"] == "9" and rollout["adversary"]["position"].shape == (26, 2)
    moved, lag0, lag1 = col.adversary_check(env, rollout)
    assert moved > 5.0  # it left its logged track
    assert lag1 == pytest.approx(0.0) and lag0 > 0.5  # the plan applied one step late


def test_adversarial_runs_are_named_like_cat_rltrain():
    assert col.adv_mode_name(SimpleNamespace(adv_selection="cat")) == "cat"
    assert col.adv_mode_name(SimpleNamespace(adv_selection="constrained", resp_threshold=1.0,
                                             resp_penalty=1.0)) == "constrained1"
    assert col.adv_mode_name(SimpleNamespace(adv_selection="penalized", resp_threshold=1.0,
                                             resp_penalty=0.5)) == "penalized0.5"
