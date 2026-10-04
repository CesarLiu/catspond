"""The responsibility-weighted collision penalty of RL training, against a
stand-in for CAT's training env (collisions do not end the episode)."""

import csv
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from responsibility.blame import Blame
from responsibility.blame_reward import BlameWeighting, contact_events, penalty_weight, weighted_reward
from tests.responsibility.conftest import FakeModel, make_scene, track

PENALTY, STEP_REWARD = 1.0, 0.5


def _cv(xy, vel):
    steps = np.arange(1, 81)[:, None] * 0.1
    traj = np.asarray(xy, dtype=np.float32) + np.asarray(vel, dtype=np.float32) * steps
    return lambda n: np.repeat(traj[None], n, 0)


class Obj:
    def __init__(self, xy, vx):
        self.position = np.asarray(xy, dtype=float)
        self.heading_theta = 0.0
        self.velocity = np.array([vx, 0.0])
        self.top_down_length, self.top_down_width = 4.8, 2.0
        self.crash_vehicle = False


class TrainEnv:
    """Replays the scene's logged tracks (the ego included, as a policy that
    drives like the log) and rewards like ScenarioEnv with
    crash_vehicle_done=False: -PENALTY at every step in contact, else the
    driving reward."""

    def __init__(self, scene, seed):
        self.scene, self.current_seed, self.t = scene, seed, 0
        self.objects = {}
        self.engine = SimpleNamespace(
            traffic_manager=SimpleNamespace(_scenario_id_to_obj_id={}),
            get_objects=lambda ids: {i: self.objects[i] for i in ids if i in self.objects},
            data_manager=SimpleNamespace(current_scenario={"metadata": {"scenario_id": scene.scenario_id}}))
        self._sync()

    def _state(self, i):
        return Obj(self.scene.position[i, self.t, :2], self.scene.velocity[i, self.t, 0])

    def _sync(self):
        self.vehicle = self._state(self.scene.sdc)
        ids = self.engine.traffic_manager._scenario_id_to_obj_id
        for i, tid in enumerate(self.scene.track_ids):
            if i != self.scene.sdc:
                ids[tid] = f"obj{tid}"
                self.objects[f"obj{tid}"] = self._state(i)
        ego = self.vehicle.position
        self.vehicle.crash_vehicle = any(
            abs(o.position[0] - ego[0]) < 4.8 and abs(o.position[1] - ego[1]) < 2.0 for o in self.objects.values())

    def step(self, action):
        self.t += 1
        self._sync()
        reward = -PENALTY if self.vehicle.crash_vehicle else STEP_REWARD
        return np.zeros(3), reward, False, {"step_reward": STEP_REWARD, "cost": float(self.vehicle.crash_vehicle)}


def _rear_end(tmp_path):
    """The ego steady at 5 m/s, rear-ended by a follower at 12 m/s (in contact
    from step 25 to 38, the follower driving through it), saved as scene 7."""
    scene = make_scene({"0": track((0, 0), (5, 0)), "1": track((-15, 0), (12, 0)), "2": track((0, 40), (5, 0))})
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    with open(scenes / "7.pkl", "wb") as f:
        pickle.dump(scene.to_description(), f)
    at = lambda agent: scene.position[agent, 10, :2]  # noqa: E731
    model = FakeModel(None, samples_by_agent={0: _cv(at(0), (5, 0)),  # the ego's alternatives: what it did
                                              1: _cv(at(1), (5, 0))})  # the follower's: keep the gap
    return scene, scenes, model


def _episode(weighting, scene, steps=45, adversary="1"):
    env = TrainEnv(scene, seed=7)
    weighting.begin(env, adversary)
    rewards = []
    for _ in range(steps):
        state = np.zeros(3)
        next_state, reward, done, info = env.step(None)
        weighting.add(state, [0.0, 0.0], next_state, reward, float(done), info)
        rewards.append(reward)
    return rewards, weighting.end(total_steps=steps)


def test_a_collision_the_ego_did_not_cause_is_not_penalised(tmp_path):
    scene, scenes, model = _rear_end(tmp_path)
    log = tmp_path / "blame.csv"
    weighting = BlameWeighting(model, scenes, PENALTY, log_path=log, verbose=False)
    raw, transitions = _episode(weighting, scene)
    contact = [i for i, r in enumerate(raw) if r == -PENALTY]
    assert contact == list(range(24, 38))  # states 25..38
    assert len(transitions) == 45
    rewards = [t[3] for t in transitions]
    assert rewards == pytest.approx([STEP_REWARD] * 45)  # w = 0: as if nothing had been hit
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 1
    row = rows[0]
    assert row["crash_step"] == "25" and row["other_id"] == "1" and row["scene"] == "7"
    assert row["verdict"] == "other" and row["rule"] == "other" and float(row["weight"]) == 0.0
    assert row["rss"] == "other"  # RSS, logged alongside: the tailgater did not brake
    assert int(row["penalised_steps"]) == 14 and row["adversary"] == "1"
    assert "mean penalty weight 0.00" in weighting.summary()


def test_doubt_keeps_the_full_penalty(tmp_path):
    scene, scenes, model = _rear_end(tmp_path)
    # the sides differ by less than the margin: too close to call
    weighting = BlameWeighting(model, scenes, PENALTY, margin=100.0, verbose=False)
    raw, transitions = _episode(weighting, scene)
    assert [t[3] for t in transitions] == raw
    # the attribution fails: the penalty stays too, and training goes on
    weighting = BlameWeighting(model, scenes, PENALTY, verbose=False)
    weighting.attribute = lambda *a: (_ for _ in ()).throw(RuntimeError("no model"))
    raw, transitions = _episode(weighting, scene)
    assert [t[3] for t in transitions] == raw and weighting.weights == [1.0]


def test_the_rss_baseline_needs_no_model(tmp_path):
    scene, scenes, _ = _rear_end(tmp_path)
    log = tmp_path / "rss.csv"
    weighting = BlameWeighting(None, scenes, PENALTY, log_path=log, verbose=False, mode="rss")
    raw, transitions = _episode(weighting, scene)
    assert min(raw) == -PENALTY
    assert [t[3] for t in transitions] == pytest.approx([STEP_REWARD] * 45)  # RSS blames the follower: w = 0
    row = next(csv.DictReader(open(log)))
    assert row["rss"] == "other" and row["verdict"] == "unknown" and float(row["weight"]) == 0.0
    with pytest.raises(ValueError):
        BlameWeighting(None, scenes, PENALTY, mode="share")


def test_no_collision_leaves_the_episode_as_it_was(tmp_path):
    scene, scenes, model = _rear_end(tmp_path)
    weighting = BlameWeighting(model, scenes, PENALTY, verbose=False)
    raw, transitions = _episode(weighting, scene, steps=20)  # ends before the contact
    assert [t[3] for t in transitions] == raw == [STEP_REWARD] * 20
    assert weighting.weights == []


def test_contact_events():
    contacts = [False, True, True, False, True, True, True]
    partners = [None, "a", "a", None, "a", "b", "b"]
    penalised = [False, True, True, False, True, False, True]  # one step: out of road took precedence
    events = contact_events(contacts, partners, penalised)
    assert [(e.step, e.other, e.transitions) for e in events] == [(2, "a", [1, 2]), (5, "a", [4]),
                                                                   (6, "b", [6])]


def test_penalty_weight():
    def blame(share, verdict):
        return Blame(30, 10, 1, "1", "VEHICLE", 0.0, 0.0, share, verdict)

    assert penalty_weight(None) == 1.0
    assert penalty_weight(blame(0.2, "other")) == pytest.approx(0.2)
    assert penalty_weight(blame(0.9, "ego")) == pytest.approx(0.9)
    assert penalty_weight(blame(0.5, "shared")) == 1.0
    assert penalty_weight(blame(None, "ego-only")) == 1.0
    assert weighted_reward(-1.0, 0.3, 1.0, 1.0) == -1.0
    assert weighted_reward(-1.0, 0.3, 0.0, 1.0) == pytest.approx(0.3)
    assert weighted_reward(-1.0, 0.3, 0.5, 1.0) == pytest.approx(-0.35)
