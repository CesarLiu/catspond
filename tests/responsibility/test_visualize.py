from types import SimpleNamespace

import numpy as np
import torch

from responsibility.metrics import ResponsibilityConfig, responsibility_at
from scripts.responsibility import visualize_responsibility as vis
from tests.responsibility.conftest import FakeModel, make_scene, track


def _dist(origin, n_goals=50, seed=0):
    rng = np.random.default_rng(seed)
    goals = rng.normal(scale=15.0, size=(n_goals, 2))
    return SimpleNamespace(
        goals=goals,
        log_prob=torch.log_softmax(torch.tensor(rng.normal(size=n_goals), dtype=torch.float), -1),
        to_global=lambda pts: np.asarray(pts) + origin,
        goals_global=goals + origin,
    )


def _scene():
    return make_scene({
        "0": track((0, 0), (10, 0)),
        "1": track((5, 3.5), (10, 0)),
        "2": track((0, 30), (5, 0)),
    })


def test_record_captures_what_the_values_were_computed_from():
    scene = _scene()
    samples = lambda n: np.tile(np.stack([np.arange(80) + 1.0, np.zeros(80)], -1), (n, 1, 1))  # noqa: E731
    model = FakeModel(samples, courtesy_logits={1: ([0.0, 0.0], [1.0, 0.0])})
    record = {}
    obs = responsibility_at(model, scene, 0, 10, ResponsibilityConfig(n_safety_samples=3), record=record)
    assert record["samples"].shape == (3, 80, 2) and record["horizon"] == 20
    assert set(record["neighbours"]) == {1} and set(record["courtesy"]) == {1}
    assert obs.courtesy > 0


def test_a_frame_renders_with_and_without_levels():
    scene = _scene()
    origin = scene.position[0, 10, :2]
    obs = SimpleNamespace(safety=0.4, courtesy=0.2, speed=10.0, per_neighbour={
        "1": {"safety": 0.4, "courtesy": 0.2, "min_gap": 1.0, "pet": 0.5, "ttc": 3.0, "type": "VEHICLE"}})
    samples = np.tile(np.stack([np.arange(80) + 1.0, np.zeros(80)], -1), (6, 1, 1)) + origin
    record = {"distribution": _dist(origin), "samples": samples, "horizon": 20,
              "neighbours": {1: {}}, "courtesy": {1: (_dist(scene.position[1, 10, :2]),
                                                     _dist(scene.position[1, 10, :2], seed=1))}}
    frames = [{"step": 10, "obs": obs, "record": record, "samples": samples},
              {"step": 20, "obs": obs, "record": record, "samples": samples}]
    groups = vis.map_segments(scene)
    for levels in ({}, {10: (0, 0), 20: (2, 1)}):
        image = vis.render(scene, 0, frames[0], 0, frames, groups, 40.0, 20, levels, ego_heatmap=True)
        assert image.ndim == 3 and image.shape[2] == 3 and image.std() > 0
    assert vis.fit_radius(scene, 0, frames, 20) >= 30.0
