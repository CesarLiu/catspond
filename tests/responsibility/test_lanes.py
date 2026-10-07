from types import SimpleNamespace

import numpy as np
import pytest
import torch

from responsibility.lanes import reachable_lanes, route_lanes
from responsibility.metrics import ResponsibilityConfig, responsibility_at
from responsibility.motion_filter import MotionFilterConfig, restrict_to_route
from tests.responsibility.conftest import make_scene, track


def lane(points, entry=(), exit=(), left=()):
    return {"type": "LANE_SURFACE_STREET", "polyline": np.array([[x, y, 0.0] for x, y in points]),
            "entry_lanes": list(entry), "exit_lanes": list(exit),
            "left_neighbor": [{"feature_id": x} for x in left], "right_neighbor": []}


# A -> B -> F straight east along y = 0; A also turns right into C (south from x = 55);
# D runs beside A and B (y = 3.5, same way); E the other way (y = -3.5); G far off
MAP = {
    "A": lane([(0, 0), (50, 0)], exit=["B", "C"], left=["D"]),
    "B": lane([(50, 0), (100, 0)], entry=["A"], exit=["F"], left=["D"]),
    "C": lane([(50, 0), (55, -5), (55, -50)], entry=["A"]),
    "D": lane([(0, 3.5), (100, 3.5)]),
    "E": lane([(100, -3.5), (0, -3.5)]),
    "F": lane([(100, 0), (150, 0)], entry=["B"]),
    "G": lane([(0, 40), (100, 40)]),
}
GOALS = np.array([[80.0, 0], [80, 3.5], [55, -30], [40, 40]])  # on B, on D, on C, on G


def scene(**tracks):
    # agent 0 drives east along A and B at 10 m/s (x = 10 at step 10, x = 90 at the end)
    return make_scene({"0": track((10, 0), (10, 0)), **tracks}, map_features=MAP)


def goal_dist(logits, goals=GOALS):
    lp = torch.log_softmax(torch.tensor(logits, dtype=torch.float), -1)
    return SimpleNamespace(goals_global=goals, log_prob=lp)


def test_route_lanes_take_neighbours_and_what_follows_the_log():
    assert route_lanes(scene(), 0, 10) == {"A", "B", "D", "F"}  # not the turn C, the opposite E or far G
    assert reachable_lanes(scene(), 0, 10) == {"A", "B", "C", "D", "F"}


def test_off_every_lane_means_no_restriction():
    s = make_scene({"0": track((10, 20), (10, 0))}, map_features=MAP)
    assert route_lanes(s, 0, 10) is None


def test_goals_restricted_to_the_route_before_sampling():
    restricted, mass = restrict_to_route(scene(), 0, 10, goal_dist([0.0, 0, 0, 0]),
                                         MotionFilterConfig(lane_route=True))
    p = restricted.log_prob.exp().numpy()
    assert mass == pytest.approx(0.5) and p == pytest.approx([0.5, 0.5, 0, 0])


class GoalModel:
    """Goals with a distribution; samples drive straight to their goal; b's
    goals with and without the queried agent come from ``courtesy``."""

    def __init__(self, logits, courtesy=None):
        self.logits, self.courtesy = logits, courtesy or {}

    def distribution(self, scene, step, agent, excluded=(), avoid_partner=()):
        d = goal_dist(self.logits)
        d.origin = scene.position[agent, step, :2]
        return d

    def sample(self, dist, n, generator=None):
        idx = torch.multinomial(dist.log_prob.exp(), n, replacement=True, generator=generator)
        goals = dist.goals_global[idx.numpy()]
        frac = (np.arange(1, 81) / 80.0)[None, :, None]
        return idx, dist.log_prob[idx], dist.origin + frac * (goals[:, None] - dist.origin)

    def with_and_without(self, scene, step, b, a):
        if b not in self.courtesy:
            return None
        return tuple(goal_dist(x) for x in self.courtesy[b])


def test_safety_samples_only_route_goals():
    s = scene(**{"1": track((18, 0), (5, 0))})  # a slower car 8 m ahead: a neighbour
    record = {}
    obs = responsibility_at(GoalModel([0.0, 0, 0, 0]), s, 0, 10,
                            ResponsibilityConfig(courtesy=False, filter=MotionFilterConfig(lane_route=True)),
                            torch.Generator().manual_seed(0), record=record)
    ends = record["samples"][:, -1]
    assert np.all(np.isin(ends[:, 1], [0.0, 3.5]))  # every sample heads for B or D
    assert obs.per_neighbour["1"]["route_goal_mass"] == pytest.approx(0.5)


def test_courtesy_ignores_goals_the_neighbour_cannot_reach():
    # b (agent 1) on A, 10 m behind agent 0; with agent 0 removed, only the probability of G
    # (unreachable for b) changes, the reachable goals keep their ratio
    s = scene(**{"1": track((0, 0), (10, 0))})
    model = GoalModel([0.0, 0, 0, 0], courtesy={1: ([0.0, -50, 0, -6], [0.0, -50, 0, 0.5])})
    plain = responsibility_at(model, s, 0, 10, ResponsibilityConfig())
    valid = responsibility_at(model, s, 0, 10, ResponsibilityConfig(courtesy_valid_goals=True))
    assert plain.per_neighbour["1"]["courtesy"] > 0.2
    assert valid.per_neighbour["1"]["courtesy"] == pytest.approx(0.0, abs=1e-6)
    assert valid.per_neighbour["1"]["courtesy_goal_mass"] == pytest.approx(1.0, abs=0.01)


def along(points, speed=10.0, start=10.0):
    """A track driving the polyline ``points`` at ``speed``, ``start`` m into it at step 10."""
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.clip(start + speed * (np.arange(91) - 10) * 0.1, 0, cum[-1])
    xy = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=-1)
    t = track((0, 0), (0, 0))
    t["state"]["position"][:, :2] = xy
    d = np.gradient(xy, axis=0)
    t["state"]["heading"] = np.arctan2(d[:, 1], d[:, 0]).astype(np.float32)
    t["state"]["velocity"] = (d / 0.1).astype(np.float32)
    return t


def test_a_car_that_turns_has_the_turn_as_its_route():
    s = make_scene({"0": along([(0, 0), (50, 0), (55, -5), (55, -50)])}, map_features=MAP)
    lanes = route_lanes(s, 0, 10)
    assert {"A", "C"} <= lanes and "B" not in lanes and "F" not in lanes


def test_a_car_leaving_the_lanes_gets_no_restriction():
    # it drives east on A, then off into a parking lot north of the road (no lanes there)
    s = make_scene({"0": along([(0, 0), (30, 0), (30, 25)], speed=5.0)}, map_features=MAP)
    assert route_lanes(s, 0, 10) is None and reachable_lanes(s, 0, 10) is None
