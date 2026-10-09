"""Right of way at a synthetic four-way intersection (US, right-hand
traffic): roads along x and y, 3.5 m lanes, the intersection lanes from
-10 m to 10 m. Eastbound runs at y = -1.75, westbound at y = 1.75,
northbound at x = 1.75; westbound can turn left (south) on an arc."""

import numpy as np
import pytest

from responsibility import stl
from responsibility.lanes import LaneGraph
from responsibility.right_of_way import RightOfWayParams, lane_sequence, priority, right_of_way_blame, \
    rollout_right_of_way
from responsibility.rollouts import make_rollout, scene_from_rollout
from tests.responsibility.conftest import N_STEPS, make_scene, track

DT = 0.1
STEPS = np.arange(N_STEPS)


def _line(a, b, n=20):
    return np.linspace(a, b, n)


def _arc(center, radius, a0, a1, n=30):
    t = np.linspace(a0, a1, n)
    return np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)], -1)


def _lane(poly, exits=(), neighbours=()):
    poly = np.asarray(poly, dtype=float)
    return {"type": "LANE_SURFACE_STREET", "polyline": np.concatenate([poly, np.zeros((len(poly), 1))], -1),
            "exit_lanes": list(exits), "left_neighbor": [{"feature_id": n} for n in neighbours]}


def _crossroads(stop_sign_on=None):
    lanes = {
        "E_in": _lane(_line((-80, -1.75), (-10, -1.75)), ["E_thr"]),
        "E_thr": _lane(_line((-10, -1.75), (10, -1.75)), ["E_out"]),
        "E_out": _lane(_line((10, -1.75), (80, -1.75))),
        "W_in": _lane(_line((80, 1.75), (10, 1.75)), ["W_thr", "W_left"]),
        "W_thr": _lane(_line((10, 1.75), (-10, 1.75)), ["W_out"]),
        "W_left": _lane(_arc((10, -10), 11.75, np.pi / 2, np.pi), ["S_out"]),
        "W_out": _lane(_line((-10, 1.75), (-80, 1.75))),
        "S_out": _lane(_line((-1.75, -10), (-1.75, -80))),
        "N_in": _lane(_line((1.75, -80), (1.75, -10)), ["N_thr"]),
        "N_thr": _lane(_line((1.75, -10), (1.75, 10)), ["N_out"]),
        "N_out": _lane(_line((1.75, 10), (1.75, 80))),
    }
    if stop_sign_on is not None:
        lanes["sign"] = {"type": "STOP_SIGN", "position": np.array([6.0, -12.0, 0.0]), "lane": [stop_sign_on]}
    return lanes


def _lights(**states):
    """Traffic lights on intersection lanes, each one state throughout."""
    starts = {"E_thr": (-10, -1.75), "N_thr": (1.75, -10), "W_thr": (10, 1.75), "W_left": (10, 1.75)}
    return {f"tl_{lane}": {"type": "TRAFFIC_LIGHT", "lane": lane, "stop_point": np.array([*starts[lane], 0.0]),
                           "state": {"object_state": [f"LANE_STATE_{state}"] * N_STEPS}}
            for lane, state in states.items()}


def _along(poly, s):
    """A track along a polyline at arc lengths s [91]; heading along the
    polyline, velocity from the motion."""
    poly = np.asarray(poly, dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))])
    x, y = np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])
    seg = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(poly) - 2)
    d = poly[seg + 1] - poly[seg]
    pos = np.zeros((N_STEPS, 3), np.float32)
    pos[:, 0], pos[:, 1] = x, y
    vel = np.stack([np.gradient(x, DT), np.gradient(y, DT)], -1).astype(np.float32)
    return {"type": "VEHICLE",
            "state": {"position": pos, "heading": np.arctan2(d[:, 1], d[:, 0]).astype(np.float32), "velocity": vel,
                      "length": np.full(N_STEPS, 4.8, np.float32), "width": np.full(N_STEPS, 2.0, np.float32),
                      "height": np.full(N_STEPS, 1.5, np.float32), "valid": np.ones(N_STEPS, dtype=bool)},
            "metadata": {}}


EAST = np.array([(-80, -1.75), (80, -1.75)])
NORTH = np.array([(1.75, -80), (1.75, 80)])
WEST_LEFT = np.concatenate([[(80, 1.75)], _arc((10, -10), 11.75, np.pi / 2, np.pi), [(-1.75, -80)]])


def _corners(scene, i, t):
    (l, w), h = scene.shape_at(i, t), float(scene.heading[i, t])
    u, v = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
    c = scene.position[i, t, :2].astype(float)
    return np.array([c + sx * l / 2 * u + sy * w / 2 * v for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


def _first_contact(scene, a=0, b=1):
    """The first step the two footprints overlap (separating axis test)."""
    for t in range(N_STEPS):
        pa, pb = _corners(scene, a, t), _corners(scene, b, t)
        axes = [pa[1] - pa[0], pa[3] - pa[0], pb[1] - pb[0], pb[3] - pb[0]]
        if all(max((pa @ ax).min(), (pb @ ax).min()) < min((pa @ ax).max(), (pb @ ax).max()) for ax in axes):
            return t
    raise AssertionError("no contact")


def _crossing(map_features, lights=None, ego_speed=10.0, other_speed=10.0):
    """The ego eastbound and the other northbound, both at the crossing
    point (1.75, -1.75) at step 50."""
    ego = _along(EAST, 81.75 + ego_speed * DT * (STEPS - 50))
    other = _along(NORTH, 78.25 + other_speed * DT * (STEPS - 50))
    scene = make_scene({"0": ego, "1": other}, map_features=map_features, lights=lights or {})
    return scene, _first_contact(scene)


def test_stl_operators():
    x = np.array([1.0, -2.0, 3.0, 0.5])
    assert stl.always(x)[0] == -2.0 and stl.eventually(x)[0] == 3.0
    assert stl.always(x, 2, 3)[0] == 0.5 and stl.always(x, 0, 0).tolist() == x.tolist()
    assert stl.always(x, 5)[0] == np.inf and stl.eventually(x, 5)[0] == -np.inf  # past the end
    assert stl.implies([-1.0, 2.0], [-3.0, -3.0]).tolist() == [1.0, -2.0]
    assert stl.prev(x, first=7.0).tolist() == [7.0, 1.0, -2.0, 3.0]
    # phi U psi: psi must come, phi until then
    phi, psi = np.array([1.0, 2.0, -1.0, 1.0]), np.array([-1.0, -1.0, 0.5, -1.0])
    assert stl.until(phi, psi)[0] == 0.5  # psi at 2 with phi on [0, 2)
    assert stl.until(phi, -np.ones(4))[0] == -1.0  # psi never: false
    assert stl.weak_until(np.ones(4), -np.ones(4))[0] == 1.0  # phi throughout: true weakly


def test_lane_sequence_relabels_the_branch_taken():
    graph = LaneGraph(_crossroads())
    s = np.arange(55.0, 95.0, 1.0)  # through the turn: W_thr and W_left overlap past the stop line
    track = _along(WEST_LEFT, np.pad(s, (0, N_STEPS - len(s)), mode="edge"))["state"]
    seq = lane_sequence(graph, track["position"][:len(s), :2], track["heading"][:len(s)])
    assert "W_thr" not in seq
    assert seq[0] == "W_in" and seq[-1] == "S_out"
    assert seq.index("W_left") == int(np.argmax(track["position"][:len(s), 0] < 10.0))  # once past the stop line


def test_red_light_runner_is_to_blame():
    scene, crash = _crossing(_crossroads(), _lights(E_thr="GO", N_thr="STOP"))
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("ego", "red-light", "other")
    assert row.rho_yield < 0 and row.yield_gap < 0  # the ego could no longer stop when the other entered
    assert row.rho_avoid_ego >= 0
    # the same collision with the lights the other way round
    scene, crash = _crossing(_crossroads(), _lights(E_thr="STOP", N_thr="GO"))
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("other", "red-light", "ego")
    assert row.as_row() == {"right_of_way": "ego", "right_of_way_case": "red-light", "priority": "other"}


def test_no_priority_without_the_signal_state():
    scene, crash = _crossing(_crossroads(), _lights(E_thr="GO", N_thr="UNKNOWN"))
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("none", "signal-unknown/no-violation", "n/a")
    assert row.rho_yield is None
    # a light near the conflict point whose own lane state was never observed: unknown too
    scene, crash = _crossing(_crossroads(), _lights(E_thr="GO"))
    assert priority(scene, 0, 1, crash).case == "signal-unknown"


def test_left_turn_yields_to_oncoming_traffic():
    # both on green; the ego turns left from the westbound road across the eastbound one
    # (P* where the arc meets y = -1.75: 9.31 m into it, x = 1.63)
    ego = _along(WEST_LEFT, 79.31 + 0.6 * (STEPS - 50))
    other = _along(EAST, 81.63 + 1.2 * (STEPS - 50))
    scene = make_scene({"0": ego, "1": other}, map_features=_crossroads(),
                       lights=_lights(E_thr="GO", W_thr="GO", W_left="GO"))
    crash = _first_contact(scene)
    prio = priority(scene, 0, 1, crash)
    assert prio.holder == 1 and prio.case == "left-turn"
    turner, oncoming = prio.approaches
    assert turner.turn == "left" and turner.control == "go" and oncoming.turn == "straight"
    assert prio.conflict.kind == "crossing"
    np.testing.assert_allclose(prio.conflict.point, [1.63, -1.75], atol=0.2)  # polyline chords
    assert right_of_way_blame(scene, 0, 1, crash).verdict == "ego"
    # under a protected arrow the turner goes first and the oncoming car (on red) yields
    scene = make_scene({"0": ego, "1": other}, map_features=_crossroads(),
                       lights=_lights(E_thr="STOP", W_thr="STOP", W_left="ARROW_GO"))
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("ego", "red-light", "other")


def test_stop_sign_yields_unless_the_other_could_have_stopped():
    # the ego (northbound, stop sign) pulls out in front of the other
    scene = make_scene({"0": _along(NORTH, 78.25 + DT * 10 * (STEPS - 50)),
                        "1": _along(EAST, 81.75 + DT * 10 * (STEPS - 50))}, map_features=_crossroads("N_thr"),
                       lights={})
    crash = _first_contact(scene)
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("other", "stop-sign", "ego")
    # the ego entered with the other 46 m away and stalled across its lane; the other
    # could have stopped but ran into it: the other's fault (avoid), not the ego's (yield)
    s_ego = np.minimum(78.25 - 0.5 * (30 - STEPS), 78.25)
    scene = make_scene({"0": _along(NORTH, s_ego), "1": _along(EAST, 81.75 + 1.0 * (STEPS - 70))},
                       map_features=_crossroads("N_thr"), lights={})
    crash = _first_contact(scene)
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("other", "stop-sign", "other")
    assert row.rho_yield >= 0 and row.yield_gap > 10.0 and row.rho_avoid_other < 0


def test_uncontrolled_intersection():
    # an intersection the map shows no control for: no priority by default (in WOMD such
    # junctions are mostly controlled in reality, by signs or lights the map lacks)
    scene, crash = _crossing(_crossroads())
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("none", "uncontrolled/no-violation", "n/a")
    # with the CVC's order for uncontrolled intersections: simultaneous arrival, so
    # yield to the vehicle on the right (the northbound other, for the eastbound ego)
    cvc = RightOfWayParams(uncontrolled_order=True)
    row = right_of_way_blame(scene, 0, 1, crash, cvc)
    assert (row.priority, row.case, row.verdict) == ("other", "yield-right", "ego")
    assert right_of_way_blame(scene, 1, 0, crash, cvc).as_row() == {  # the same collision from the other side
        "right_of_way": "other", "right_of_way_case": "yield-right", "priority": "ego"}
    # the ego, on the right, entered 2.5 s before a fast car from its left: first in goes first
    ego = _along(NORTH, 78.25 + 0.3 * (STEPS - 50))  # 3 m/s, in the intersection from step 23
    other = _along(np.array([(80, -1.75), (-80, -1.75)]), 78.25 + 1.5 * (STEPS - 50))  # westbound, 15 m/s
    scene = make_scene({"0": ego, "1": other}, map_features=_crossroads(), lights={})
    crash = _first_contact(scene)
    prio = priority(scene, 0, 1, crash, cvc)
    assert prio.case == "first-in" and prio.holder == 0
    assert prio.approaches[1].entered - prio.approaches[0].entered > 10


def _two_lanes():
    return {"R": _lane(_line((-100, -5.25), (200, -5.25), 60), neighbours=["L"]),
            "L": _lane(_line((-100, -1.75), (200, -1.75), 60), neighbours=["R"])}


def test_lane_change_into_a_close_follower():
    # the ego changes from the right lane into the left one 6.5 m ahead of a car closing at 5 m/s
    x_ego = 30 + 1.0 * STEPS
    y_ego = np.clip(-5.25 + 0.2 * (STEPS - 30), -5.25, -1.75)
    pos = np.stack([x_ego, y_ego, np.zeros(N_STEPS)], -1).astype(np.float32)
    ego = _along(EAST, np.zeros(N_STEPS))
    vel = np.stack([np.gradient(x_ego, DT), np.gradient(y_ego, DT)], -1).astype(np.float32)
    ego["state"].update(position=pos, velocity=vel, heading=np.arctan2(vel[:, 1], vel[:, 0]).astype(np.float32))
    other = track((15, -1.75), (15, 0))  # x = 1.5 t
    scene = make_scene({"0": ego, "1": other}, map_features=_two_lanes(), lights={})
    crash = _first_contact(scene)
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("other", "lane-change", "ego")


def test_following_is_left_to_the_rear_end_rule():
    scene = make_scene({"0": track((0, -1.75), (5, 0)), "1": track((-15, -1.75), (12, 0))},
                       map_features=_two_lanes(), lights={})
    row = right_of_way_blame(scene, 0, 1, 28)
    assert (row.verdict, row.case, row.priority) == ("n/a", "following", "none")


def test_a_rollouts_collision():
    # the logged ego waits at its stop sign; the policy drives on into the other's path
    wait = np.minimum(78.25 - 6.0 + 1.0 * (STEPS - 50), 78.25 - 9.0)
    scene = make_scene({"0": _along(NORTH, wait), "1": _along(EAST, 81.75 + 1.0 * (STEPS - 50)),
                        "2": track((0, 60), (0, 0))}, map_features=_crossroads("N_thr"), lights={})
    policy = make_scene({"0": _along(NORTH, 78.25 + 1.0 * (STEPS - 50)), "1": _along(EAST, 81.75 + 1.0 * (STEPS - 50))},
                        map_features=_crossroads("N_thr"), lights={})
    crash = _first_contact(policy)
    ego = {"position": policy.position[0, :crash + 1, :2], "heading": policy.heading[0, :crash + 1]}
    rollout = make_rollout("0", "synthetic", "td3", "none", ego=ego,
                           end={"step": crash, "reason": "crash_vehicle", "crash_vehicle": True, "crash_with": "1"},
                           present={"track_ids": ["1", "2"], "mask": np.ones((2, crash + 1), dtype=bool)})
    row = rollout_right_of_way(scene_from_rollout(scene, rollout), rollout)
    assert (row.priority, row.case, row.verdict) == ("other", "stop-sign", "ego")


def test_parameters_move_the_boundary():
    # the stalled ego, hit by a car that was 46 m from the zone when the ego entered it (10 m/s):
    # with a 3.5 s response time the car could not have stopped (35 + 12.5 m)
    s_ego = np.minimum(78.25 - 0.5 * (30 - STEPS), 78.25)
    scene = make_scene({"0": _along(NORTH, s_ego), "1": _along(EAST, 81.75 + 1.0 * (STEPS - 70))},
                       map_features=_crossroads("N_thr"), lights={})
    crash = _first_contact(scene)
    assert right_of_way_blame(scene, 0, 1, crash).verdict == "other"
    slow = RightOfWayParams(rho=3.5)
    assert right_of_way_blame(scene, 0, 1, crash, slow).verdict == "ego"


def _piecewise(*knots):
    """Arc length over the 91 steps through (step, s) knots, linear between them."""
    t, s = zip(*knots)
    return np.interp(STEPS, t, s)


def test_all_way_stop_goes_by_arrival():
    # both approaches stop-controlled; the ego arrives at step 20 and waits, the other
    # arrives later (step 50) and drives on without stopping
    stops = _crossroads(stop_sign_on="N_thr")
    stops["sign_e"] = {"type": "STOP_SIGN", "position": np.array([-12.0, -6.0, 0.0]), "lane": ["E_thr"]}
    ego = _along(EAST, _piecewise((0, 47.0), (20, 67.0), (40, 67.0), (90, 92.0)))
    other = _along(NORTH, _piecewise((0, 15.0), (90, 105.0)))
    scene = make_scene({"0": ego, "1": other}, map_features=stops, lights={})
    crash = _first_contact(scene)
    prio = priority(scene, 0, 1, crash)
    assert (prio.holder, prio.case) == (0, "all-way-stop/first-in")
    assert prio.approaches[0].arrived < prio.approaches[1].arrived
    assert right_of_way_blame(scene, 0, 1, crash).verdict == "other"
    # both already waiting when the clip begins: who came first is unknown
    ego = _along(EAST, _piecewise((0, 67.0), (40, 67.0), (90, 92.0)))
    other = _along(NORTH, _piecewise((0, 66.0), (45, 66.0), (90, 100.0)))
    scene = make_scene({"0": ego, "1": other}, map_features=stops, lights={})
    assert priority(scene, 0, 1, _first_contact(scene)).case == "all-way-stop/order-unknown"


def test_a_road_that_ends_yields_to_the_one_that_continues():
    # a T junction: the northbound side road ends at the eastbound main road and only turns
    tee = {key: lane for key, lane in _crossroads().items() if not key.startswith(("N_", "W_", "S_"))}
    tee["N_in"] = _lane(_line((1.75, -80), (1.75, -10)), ["N_right"])
    tee["N_right"] = _lane(_arc((10, -10), 8.25, np.pi, np.pi / 2), ["E_out"])
    turn = np.concatenate([_line((1.75, -80), (1.75, -10)), _arc((10, -10), 8.25, np.pi, np.pi / 2)[1:],
                           [(80, -1.75)]])
    # the side-road car turns right onto the main road 12 m ahead of a car doing 12 m/s
    ego = _along(turn, 70.0 + 0.5 * (STEPS - 40))
    other = _along(EAST, 81.75 - 12.0 + 1.2 * (STEPS - 40))
    scene = make_scene({"0": ego, "1": other}, map_features=tee, lights={})
    crash = _first_contact(scene)
    prio = priority(scene, 0, 1, crash)
    assert prio.case == "through-road" and prio.holder == 1
    assert prio.approaches[0].through is False and prio.approaches[1].through is True
    assert right_of_way_blame(scene, 0, 1, crash).verdict == "ego"


def test_driveway_and_oncoming():
    # the ego pulls out of a driveway (no lanes) across the eastbound lane
    plain = {key: lane for key, lane in _crossroads().items() if key.startswith("E_")}
    scene = make_scene({"0": _along(np.array([(1.75, -40), (1.75, 40)]), 38.25 + 0.5 * (STEPS - 50)),
                        "1": _along(EAST, 81.75 + 1.0 * (STEPS - 50))}, map_features=plain, lights={})
    crash = _first_contact(scene)
    row = right_of_way_blame(scene, 0, 1, crash)
    assert (row.priority, row.case, row.verdict) == ("other", "driveway", "ego")
    # two cars passing each other on a two-way road: no conflict of priority
    scene = make_scene({"0": track((0, -1.75), (10, 0)), "1": track((60, 1.75), (-10, 0), heading=np.pi)},
                       map_features=_crossroads(), lights={})
    assert priority(scene, 0, 1, 40).case == "oncoming"


def test_old_crash_files_get_the_rule_verdicts(tmp_path):
    import csv
    import json
    import pickle

    from responsibility.rollouts import save_rollout
    from scripts.responsibility import attribute_rules

    # the stop-sign collision of test_a_rollouts_collision, as files of a finished run
    # whose crashes.csv predates the right-of-way columns
    scene = make_scene({"0": _along(NORTH, np.minimum(72.25 + 1.0 * (STEPS - 50), 69.25)),
                        "1": _along(EAST, 81.75 + 1.0 * (STEPS - 50))}, map_features=_crossroads("N_thr"), lights={})
    (tmp_path / "scenes").mkdir()
    with open(tmp_path / "scenes" / "0.pkl", "wb") as f:
        pickle.dump(scene.to_description(), f)
    policy = make_scene({"0": _along(NORTH, 78.25 + 1.0 * (STEPS - 50)), "1": _along(EAST, 81.75 + 1.0 * (STEPS - 50))},
                        map_features=_crossroads("N_thr"), lights={})
    crash = _first_contact(policy)
    rollout = make_rollout("0", "synthetic", "td3", "none",
                           ego={"position": policy.position[0, :crash + 1, :2], "heading": policy.heading[0, :crash + 1]},
                           end={"step": crash, "reason": "crash_vehicle", "crash_vehicle": True, "crash_with": "1"},
                           present={"track_ids": ["1"], "mask": np.ones((1, crash + 1), dtype=bool)})
    save_rollout(rollout, tmp_path / "rollouts" / "0.pkl")
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"scenes": str(tmp_path / "scenes"),
                                                 "rollouts": str(tmp_path / "rollouts")}))
    old = ["scene", "policy", "adv_mode", "crash_step", "verdict", "rule", "rss", "rss_case"]
    with open(run / "crashes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old)
        writer.writeheader()
        writer.writerow(dict(scene="0", policy="td3", adv_mode="none", crash_step=crash, verdict="ego", rule="n/a",
                             rss="n/a", rss_case=""))
    attribute_rules.main(["--runs", str(run)])
    row = next(csv.DictReader(open(run / "crashes.csv")))
    assert row["verdict"] == "ego" and row["crash_step"] == str(crash)  # the counterfactual columns are kept
    assert (row["right_of_way"], row["right_of_way_case"], row["priority"]) == ("ego", "stop-sign", "other")
    assert row["rss_case"] == "not-same-direction"
