from types import SimpleNamespace

import numpy as np
import torch

from responsibility.metrics import ResponsibilityConfig, responsibility_at, scene_responsibility
from responsibility.records import load_record, run_scene, save_record, scene_of, sparse_goals
from scripts.responsibility import visualize_responsibility as vis
from tests.responsibility.conftest import make_scene, track


class GoalModel:
    """Stand-in model with goal distributions: 50 goals around each agent,
    samples are straight lines to sampled goals; courtesy for vehicle 1."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def _dist(self, scene, step, agent, shift=0.0):
        origin = scene.position[agent, step, :2]
        goals = np.stack([np.linspace(5, 40, 50), np.linspace(-5, 5, 50) + shift], -1)
        logits = torch.linspace(-1, 1, 50) * (1 + shift)
        return SimpleNamespace(goals=goals, log_prob=torch.log_softmax(logits, -1),
                               to_global=lambda pts, o=origin: np.asarray(pts) + o)

    def distribution(self, scene, step, agent, excluded=(), avoid_partner=()):
        return self._dist(scene, step, agent)

    def sample(self, dist, n, generator=None):
        idx = torch.multinomial(dist.log_prob.exp(), n, replacement=True, generator=generator)
        ends = dist.to_global(dist.goals[idx.numpy()])
        start = dist.to_global(np.zeros((1, 2)))[0]
        frac = np.linspace(1 / 80, 1, 80)[None, :, None]
        return idx, dist.log_prob[idx], start + (ends[:, None] - start) * frac

    def with_and_without(self, scene, step, b, a):
        if scene.types[b] != "VEHICLE":
            return None
        return self._dist(scene, step, b), self._dist(scene, step, b, shift=0.5)


def _scene():
    return make_scene({
        "0": track((0, 0), (10, 0)),
        "1": track((5, 3.5), (10, 0)),
        "2": track((-30, 40), (0, 0)),  # parked far away: no neighbour in some windows
    })


def test_record_captures_what_the_values_were_computed_from():
    record = {}
    obs = responsibility_at(GoalModel(), _scene(), 0, 10, ResponsibilityConfig(n_safety_samples=3), record=record)
    assert record["samples"].shape == (3, 80, 2) and record["sample_log_prob"].shape == (3,)
    assert set(record["neighbours"]) == {1} and set(record["courtesy"]) == {1}
    assert obs.courtesy > 0


def test_records_keep_the_run_values_and_survive_a_round_trip(tmp_path):
    scene = _scene()
    cfg = ResponsibilityConfig(n_safety_samples=6, window_stride=20)
    plain = scene_responsibility(GoalModel(), scene, 0, cfg)
    observations, record = run_scene(GoalModel(), scene, 0, cfg)
    assert [(o.step, o.safety, o.courtesy) for o in observations] == [(o.step, o.safety, o.courtesy) for o in plain]
    save_record(record, tmp_path / "r.pkl")
    loaded = load_record(tmp_path / "r.pkl")
    assert scene_of(loaded).track_ids == scene.track_ids and loaded["agent_id"] == "0"
    frame = loaded["frames"][0]
    assert frame["samples"].dtype == np.float32 and frame["samples"].shape == (6, 80, 2)
    assert set(frame["courtesy"]) == {"1"} and frame["courtesy"]["1"]["kl"] == frame["observation"]["courtesy"]
    assert abs(frame["goals"]["prob"].sum() - frame["goals"]["mass"]) < 1e-5


def test_sparse_goals_keep_the_requested_mass():
    dist = GoalModel()._dist(_scene(), 10, 0)
    kept = sparse_goals(dist, top_mass=0.5)
    assert kept["mass"] >= 0.5 and len(kept["prob"]) < 50
    assert np.all(np.diff(kept["prob"]) <= 0)


def test_windows_without_neighbours_get_display_samples_off_the_metric_stream():
    scene = make_scene({"0": track((0, 0), (10, 0)), "1": track((-30, 40), (0, 0))})
    cfg = ResponsibilityConfig(n_safety_samples=4, window_stride=30)
    observations, record = run_scene(GoalModel(), scene, 0, cfg)
    frame = record["frames"][0]
    assert not frame["metric_samples"] and frame["samples"].shape[0] > 0
    assert observations[0].per_neighbour == {}


def test_offline_rendering_from_a_record(tmp_path):
    cfg = ResponsibilityConfig(n_safety_samples=6, window_stride=30)
    _, record = run_scene(GoalModel(), _scene(), 0, cfg)
    record["scene_file"] = "7"
    save_record(record, tmp_path / "7.pkl")
    levels = tmp_path / "levels.csv"
    levels.write_text("run,scene,agent_id,step,level,aggressive\n" + "".join(
        f"sdc,7,0,{f['step']},{i % 2},{i % 2}\n" for i, f in enumerate(record["frames"])))
    (gif, mp4), radius = vis.render_record(load_record(tmp_path / "7.pkl"), tmp_path / "video", levels,
                                           ego_heatmap=True)
    assert gif.exists() and radius >= 30.0
    assert len(list((tmp_path / "video" / "frames").glob("t_*.png"))) == len(record["frames"])


def test_live_settings_take_use_ooi_and_a_runs_filters(tmp_path):
    import json
    from dataclasses import asdict

    import pytest

    from responsibility.motion_filter import MotionFilterConfig

    args = SimpleNamespace(run=None, n_samples=40, horizon=20, stride=5, seed=0, motion_set="sampled", use_ooi=True)
    assert vis.config_from(args).use_ooi
    cfg = ResponsibilityConfig(filter=MotionFilterConfig(drivable_edges=True))
    (tmp_path / "config.json").write_text(json.dumps({"responsibility": asdict(cfg)}))
    run = SimpleNamespace(**{**vars(args), "run": str(tmp_path)})
    with pytest.raises(SystemExit, match="without --use-ooi"):
        vis.config_from(run)
    loaded = vis.config_from(SimpleNamespace(**{**vars(run), "use_ooi": False}))
    assert isinstance(loaded.filter, MotionFilterConfig) and loaded.filter.drivable_edges
