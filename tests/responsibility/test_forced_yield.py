"""Forced yields at the synthetic crossroads of test_right_of_way.py: the
agent (northbound, stop sign) crosses the other's path (eastbound, no
control, so it holds the right of way), both at 10 m/s."""

import csv
import json
import pickle

import numpy as np
import pytest

from responsibility.forced_yield import forced_yield, required_braking
from responsibility.scene import Scene, sidecar_index
from tests.responsibility.conftest import make_scene
from tests.responsibility.test_right_of_way import DT, EAST, NORTH, STEPS, _along, _crossroads


def _pair(agent_at: int, other_at: int, agent_speed: float = 10.0):
    """The agent at P* at step agent_at, the other at step other_at."""
    return make_scene({"0": _along(NORTH, 78.25 + DT * agent_speed * (STEPS - agent_at)),
                       "1": _along(EAST, 81.75 + DT * 10 * (STEPS - other_at))},
                      map_features=_crossroads("N_thr"), lights={})


def test_required_braking():
    # 10 m/s, 30 m away, 1 s response: 100 / (2 * 20) = 2.5 m/s^2
    assert required_braking(30.0, 10.0, 1.0) == pytest.approx(2.5)
    assert np.isinf(required_braking(9.0, 10.0, 1.0))  # within one response time: cannot stop
    assert required_braking(5.0, 0.0, 1.0) == 0.0  # standing before the zone


def test_demand_grows_as_the_agent_takes_the_right_of_way():
    lawful = forced_yield(_pair(60, 40), 0, 1)  # the agent crosses 2 s after the holder
    assert (lawful.case, lawful.holder, lawful.forced) == ("stop-sign", "other", 0.0)
    # the agent goes first, the holder 3 s, 2 s and 1 s behind: the braking it would have
    # needed grows from the assertive band past RSS's 4 m/s^2 to impossible
    judged = [forced_yield(_pair(40, 40 + lag), 0, 1) for lag in (30, 20, 10)]
    assert all(j.kind == "crossing" for j in judged)  # the 3 s lag is judged at the holder's arrival
    forced = [j.forced for j in judged]
    assert 1.0 < forced[0] < 4.0 < forced[1] and np.isinf(forced[2])
    # judged the other way round, the holder's demand on the yielder does not count
    swapped = forced_yield(_pair(40, 60), 1, 0)
    assert swapped.holder == "agent" and swapped.forced is None and swapped.demand is not None


def test_go_first_ground_truth_end_to_end(tmp_path):
    from scripts.responsibility import evaluate_forced_yield, make_go_first, validate_right_of_way

    scenes = tmp_path / "scenes"
    scenes.mkdir()
    # two lawful pairs, the yielder at 5 m/s crossing 1.2 s after the holder: late in the
    # clip (the yielder has time to be re-timed ahead) and early (the holder can be delayed)
    for sid, (agent_at, other_at) in {"late": (82, 70), "early": (50, 38)}.items():
        description = _pair(agent_at, other_at, agent_speed=5.0).to_description()
        description["metadata"]["scenario_id"] = sid
        (scenes / f"{sid}.pkl").write_bytes(pickle.dumps(description))
    validate_right_of_way.main(["--scenes", str(scenes), "--out-dir", str(tmp_path / "val")])
    out = tmp_path / "go_first"
    index, why = make_go_first.main(["--pairs", str(tmp_path / "val" / "pairs.csv"), "--out-dir", str(out),
                                     "--pet", "1", "3", "--accel", "8", "--max-speed", "25"])
    assert {name: set(entries) for name, entries in index.items()} == {
        "log": {"late", "early"}, "retime_pet1": {"late"}, "retime_pet3": {"late"},
        "delay_pet1": {"early"}, "delay_pet3": {"early"}}
    retimed = json.loads(sidecar_index(out / "retime_pet1").read_text())["late"]
    assert (retimed["agent"], retimed["other"], retimed["case"]) == ("0", "1", "stop-sign")
    assert 0 < retimed["accel"] <= 8
    log, new = Scene.load(out / "log" / "late.pkl"), Scene.load(out / "retime_pet1" / "late.pkl")
    np.testing.assert_allclose(new.position[1], log.position[1])  # the holder keeps its log
    np.testing.assert_allclose(new.position[0, :retimed["t0"] + 1], log.position[0, :retimed["t0"] + 1])
    shift = json.loads(sidecar_index(out / "delay_pet1").read_text())["early"]["delay"]
    log, new = Scene.load(out / "log" / "early.pkl"), Scene.load(out / "delay_pet1" / "early.pkl")
    np.testing.assert_allclose(new.position[0], log.position[0])  # the yielder keeps its log
    np.testing.assert_allclose(new.position[1, shift:], log.position[1, :-shift])
    assert new.valid[1].all()  # seen from the start, coming from further back at its first speed
    np.testing.assert_allclose(new.position[1, 0, :2], log.position[1, 0, :2] - [shift * DT * 10, 0], atol=1e-4)
    sets = ["log", "retime_pet3", "retime_pet1", "delay_pet3", "delay_pet1"]
    rows = evaluate_forced_yield.main(["--out-dir", str(tmp_path / "eval"), "--sets",
                                       *[f"{name}={out / name}" for name in sets]])
    assert all(r["kind"] == "crossing" and not r["error"] for r in rows)
    forced = {(r["set"], r["scene"]): float(r["forced"]) for r in rows}
    assert forced["log", "late"] == forced["log", "early"] == 0.0
    # 1 s and 3 s after the yielder clears; at 5 m/s it needs 1.4 s to cross its zone, so the
    # holder is 2.4 s and 4.4 s away when it enters
    for scene, mode in (("late", "retime"), ("early", "delay")):
        assert 1.0 < forced[f"{mode}_pet3", scene] < 3.0 < forced[f"{mode}_pet1", scene]
    with open(tmp_path / "eval" / "pairs.csv") as f:
        assert len(list(csv.DictReader(f))) == 6
