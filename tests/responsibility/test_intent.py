"""Maps without lane topology: same intent (heading, lateral, road edges) and drivable area by road edges."""

import json
from dataclasses import asdict

import numpy as np

from responsibility.edges import crosses_road_edge
from responsibility.intent import crosses_edge, same_heading
from responsibility.motion_filter import MotionFilter, MotionFilterConfig
from scripts.responsibility.compute_responsibility import same_settings
from tests.responsibility.conftest import make_scene, track

N_STEPS, DT = 91, 0.1


def driven(headings, speed=10.0, start=(0.0, 0.0)):
    """Positions [len(headings) + 1, 2] of a car driving ``speed`` m/s along
    the given heading at each step."""
    step = speed * DT * np.stack([np.cos(headings), np.sin(headings)], -1)
    return np.concatenate([[start], start + np.cumsum(step, 0)])


def left_turn_headings(n, start=20, turn=30):
    """Straight along +x, then a left turn over ``turn`` steps, then +y."""
    return np.clip((np.arange(n) - start) / turn, 0.0, 1.0) * np.pi / 2


def logged_scene(positions, headings, edges=None):
    """Agent 0 logged at ``positions`` [81, 2] from step 10 on, plus a
    neighbour far away."""
    t = track((0.0, 0.0), (10.0, 0.0))
    pos, head = t["state"]["position"], t["state"]["heading"]
    pos[10:, :2] = positions
    pos[:10, :2] = positions[0] - np.arange(10, 0, -1)[:, None] * (positions[1] - positions[0])
    head[10:] = np.append(headings, headings[-1])[: N_STEPS - 10]
    head[:10] = headings[0]
    lane = {"100": {"type": "LANE_SURFACE_STREET", "polyline": np.array([[-50.0, 0, 0], [300, 0, 0]])}}
    if edges is not None:
        lane.update({str(200 + i): {"type": "ROAD_EDGE_BOUNDARY", "polyline": e} for i, e in enumerate(edges)})
    return make_scene({"0": t, "1": track((0.0, 200.0), (0.0, 0.0))}, map_features=lane)


STRAIGHT = driven(np.zeros(80))  # [81, 2]: the logged path and the same alternative


def test_heading_keeps_braking_and_lane_change_but_not_a_turn():
    scene = logged_scene(STRAIGHT, np.zeros(80))
    braking = driven(np.zeros(80), speed=1.0)  # still moving forward, slowly
    lane_change = STRAIGHT + np.stack([np.zeros(81), np.clip(np.arange(81) / 30, 0, 1) * 3.5], -1)
    turn = driven(left_turn_headings(80, start=5, turn=30))
    trajs = np.stack([STRAIGHT, braking, lane_change, turn])[:, 1:]
    assert same_heading(scene, 0, 10, trajs, 45.0).tolist() == [True, True, True, False]


def test_heading_is_compared_with_the_path_where_the_alternative_is():
    headings = left_turn_headings(80)
    logged = driven(headings)
    scene = logged_scene(logged, headings)
    # half the speed along the same path: at 8 s still in the middle of the turn
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(logged, axis=0), axis=-1))])
    half = np.stack([np.interp(arc / 2, arc, logged[:, i]) for i in range(2)], -1)
    straight_on = driven(np.zeros(80))
    trajs = np.stack([logged, half, straight_on])[:, 1:]
    # going straight ends nearest the end of the turn, where the path heads +y
    assert same_heading(scene, 0, 10, trajs, 45.0).tolist() == [True, True, False]
    f = MotionFilter(scene, 0, 10, trajs, 20, [1], MotionFilterConfig(intent=True))
    assert f.keep(1).tolist() == [0, 1]


def test_a_waiting_agent_shows_no_intent():
    still = np.zeros((81, 2))
    scene = logged_scene(still, np.zeros(80))
    turn = driven(left_turn_headings(80, start=5))
    assert same_heading(scene, 0, 10, np.stack([turn, STRAIGHT])[:, 1:], 45.0).all()


def test_road_edges_separate_a_parallel_road_and_missing_ones_remove_nothing():
    edge = np.array([[-50.0, 6.0, 0.0], [300.0, 6.0, 0.0]])
    scene = logged_scene(STRAIGHT, np.zeros(80), edges=[edge])
    beside = STRAIGHT + [0.0, 3.5]  # the next lane: same road
    across = STRAIGHT + [0.0, 9.0]  # beyond the edge: another road
    trajs = np.stack([STRAIGHT, beside, across])[:, 1:]
    assert crosses_edge(scene, 0, 10, trajs).tolist() == [False, False, True]
    assert not crosses_edge(logged_scene(STRAIGHT, np.zeros(80)), 0, 10, trajs).any()
    cfg = MotionFilterConfig(intent=True, intent_lateral=None, intent_edges=True)
    assert MotionFilter(scene, 0, 10, trajs, 20, [1], cfg).keep(1).tolist() == [0, 1]


def test_runs_made_before_the_intent_filter_still_resume():
    old_filter = {k: v for k, v in asdict(MotionFilterConfig(route_tolerance=2.0)).items()
                  if not k.startswith("intent") and k not in ("lane_route", "lane_radius")}
    settings = {"responsibility": {"n_safety_samples": 40, "motion_set": "sampled",
                                   "filter": asdict(MotionFilterConfig(route_tolerance=2.0)),
                                   "courtesy_valid_goals": False, "use_ooi": False},
                "agent": "sdc", "scenes": "/x", "model": {"name": "densetnt"}}
    old = {"responsibility": {"n_safety_samples": 40, "filter": old_filter}, "agent": "sdc", "scenes": "/x"}
    assert same_settings(json.loads(json.dumps(old)), settings)
    with_intent = json.loads(json.dumps(settings))
    with_intent["responsibility"]["filter"]["intent"] = True
    assert not same_settings(json.loads(json.dumps(old)), with_intent)


def road(edges=()):
    """A lane centreline along y = 0 and the given road edges."""
    f = {"100": {"type": "LANE_SURFACE_STREET", "polyline": np.array([[-50.0, 0, 0], [300.0, 0, 0]])}}
    f.update({str(200 + i): {"type": "ROAD_EDGE_BOUNDARY", "polyline": np.asarray(e, float)} for i, e in enumerate(edges)})
    return f


def test_a_path_across_a_road_edge_is_not_drivable_unless_the_log_crosses_it_too():
    curb = [[-50.0, 6.0, 0.0], [300.0, 6.0, 0.0]]
    swerve = STRAIGHT + np.stack([np.zeros(81), np.clip(np.arange(81) / 30, 0, 1) * 9.0], -1)
    trajs = np.stack([STRAIGHT, STRAIGHT + [0.0, 3.5], swerve])[:, 1:]
    scene = make_scene({"0": track((0.0, 0.0), (10.0, 0.0)), "1": track((0.0, 200.0), (0.0, 0.0))},
                       map_features=road([curb]))
    assert crosses_road_edge(scene, 0, 10, trajs).tolist() == [False, False, True]
    # the log turns into a driveway across the curb: that edge does not count
    driveway = logged_scene(swerve, np.zeros(80), edges=[np.asarray(curb)])
    assert not crosses_road_edge(driveway, 0, 10, trajs).any()
    # a gap in the curb where the swerve crosses it: nothing is removed
    gap = [[[-50.0, 6.0, 0.0], [10.0, 6.0, 0.0]], [[60.0, 6.0, 0.0], [300.0, 6.0, 0.0]]]
    gappy = make_scene({"0": track((0.0, 0.0), (10.0, 0.0)), "1": track((0.0, 200.0), (0.0, 0.0))},
                       map_features=road(gap))
    assert not crosses_road_edge(gappy, 0, 10, trajs).any()
    # the same as a MotionFilter filter
    cfg = MotionFilterConfig(drivable_edges=True)
    assert MotionFilter(scene, 0, 10, trajs, 20, [1], cfg).keep(1).tolist() == [0, 1]

