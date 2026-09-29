import numpy as np
import pytest
import torch

from responsibility.interaction import InteractionConfig, interacting_neighbours
from responsibility.metrics import (
    ResponsibilityConfig,
    goal_kl,
    responsibility_at,
    safety_responsibility,
    scene_responsibility,
    window_steps,
)
from tests.responsibility.conftest import FakeModel, make_scene, track

H = 20


def _line(y, n=None, speed=1.0):
    """Trajectories along x at height y, 1 m per step ([H, 2] or [n, H, 2])."""
    xy = np.stack([np.arange(1, H + 1) * speed, np.full(H, float(y))], -1)
    return xy if n is None else np.repeat(xy[None], n, 0)


def _safety(actual_y, sample_ys, cfg=None):
    cfg = cfg or ResponsibilityConfig(d_sat=None)
    samples = np.concatenate([_line(y, 1) for y in sample_ys])
    ones = np.ones(H, dtype=bool)
    return safety_responsibility(samples, _line(actual_y), ones, _line(6.0), ones, cfg)


def test_closing_in_more_than_the_alternatives_is_positive():
    # neighbour alongside at y=6; the alternatives keep 6 m
    assert _safety(actual_y=3.0, sample_ys=[0.0] * 10) == pytest.approx(3.0)
    # keeping further away than the alternatives is negative
    assert _safety(actual_y=-2.0, sample_ys=[0.0] * 10) == pytest.approx(-2.0)
    # doing what the alternatives do is neutral
    assert _safety(actual_y=0.0, sample_ys=[0.0] * 10) == pytest.approx(0.0)


def test_safety_is_saturated_at_the_interaction_range():
    cfg = ResponsibilityConfig(d_sat=4.0)
    # alternatives at 6 m and the actual 5 m are both beyond d_sat: no responsibility
    assert _safety(actual_y=1.0, sample_ys=[0.0] * 10, cfg=cfg) == pytest.approx(0.0)


def test_safety_only_compares_where_both_futures_are_logged():
    cfg = ResponsibilityConfig(d_sat=None)
    samples = _line(0.0, 5)
    actual = _line(3.0)
    neighbour_valid = np.ones(H, dtype=bool)
    actual_valid = np.zeros(H, dtype=bool)
    actual_valid[:5] = True
    near_end = _line(6.0)
    near_end[10:, 1] = 0.5  # would be close to everyone later, but a's log ends at step 5
    value = safety_responsibility(samples, actual, actual_valid, near_end, neighbour_valid, cfg)
    assert value == pytest.approx(3.0)


def test_goal_kl():
    p = torch.log_softmax(torch.tensor([0.0, 0.0]), -1)
    q = torch.log(torch.tensor([0.9, 0.1]))
    assert goal_kl(p, p) == 0.0
    expected = 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)
    assert goal_kl(p, q) == pytest.approx(expected, rel=1e-5)


def test_window_steps():
    valid = np.ones(91, dtype=bool)
    valid[40:46] = False
    scene = make_scene({"0": track((0, 0), (10, 0), valid=valid), "1": track((0, 5), (10, 0))})
    steps = window_steps(scene, 0, ResponsibilityConfig(window_stride=5, metric_horizon=20))
    assert steps[0] == 10 and steps[-1] == 70
    assert 40 not in steps and 45 not in steps and 50 in steps


def test_neighbours_are_selected_by_interaction_evidence():
    scene = make_scene({
        "0": track((0, 0), (10, 0)),  # ego along +x
        "1": track((20, -20), (0, 10), heading=np.pi / 2),  # crosses the ego's path ~2 s ahead
        "2": track((0, 30), (10, 0)),  # parallel, 30 m to the side
        "3": track((-45, 0), (10, 0)),  # same lane far behind, never closes in
    })
    found = interacting_neighbours(scene, 0, 10, 20, InteractionConfig())
    assert list(found) == [1]
    assert found[1]["pet"] <= 2.0


def test_responsibility_at_combines_neighbours():
    scene = make_scene({
        "0": track((0, 0), (10, 0)),
        "1": track((5, 3.5), (10, 0)),  # vehicle alongside
        "2": track((5, -3.0), (10, 0), kind="PEDESTRIAN", length=0.8, width=0.8),
    })
    far_away = lambda n: np.tile(np.stack([np.arange(80) + 1.0, np.full(80, -1.0)], -1), (n, 1, 1))  # noqa: E731
    model = FakeModel(far_away, courtesy_logits={1: ([0.0, 0.0], [2.0, 0.0])})
    cfg = ResponsibilityConfig(n_safety_samples=4, d_sat=None)
    obs = responsibility_at(model, scene, 0, 10, cfg)
    assert set(obs.per_neighbour) == {"1", "2"}
    assert obs.per_neighbour["2"]["courtesy"] is None  # pedestrians are not predicted
    assert obs.courtesy == pytest.approx(obs.per_neighbour["1"]["courtesy"]) and obs.courtesy > 0
    assert obs.safety == pytest.approx(max(v["safety"] for v in obs.per_neighbour.values()))
    row = obs.as_row()
    assert row["courtesy_toward"] == "1" and row["n_neighbours"] == 2
    assert model.calls == [(10, 1, 0), (10, 2, 0)]


def test_no_prediction_no_observation():
    scene = make_scene({"0": track((0, 0), (10, 0)), "1": track((5, 3.5), (10, 0))})
    model = FakeModel(lambda n: np.zeros((n, 80, 2)), predicted=[1])
    assert responsibility_at(model, scene, 0, 10, ResponsibilityConfig()) is None


def test_scene_responsibility_covers_the_windows():
    scene = make_scene({"0": track((0, 0), (10, 0)), "1": track((5, 3.5), (10, 0))})
    model = FakeModel(lambda n: np.zeros((n, 80, 2)))
    cfg = ResponsibilityConfig(n_safety_samples=2, window_stride=20, courtesy=False)
    obs = scene_responsibility(model, scene, cfg=cfg)
    assert [o.step for o in obs] == [10, 30, 50, 70]
    assert all(o.per_neighbour["1"]["courtesy"] is None for o in obs)
