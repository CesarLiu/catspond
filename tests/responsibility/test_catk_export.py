import numpy as np

from responsibility.catk_export import dynamic_map_infos, map_infos, numeric_id, track_infos
from tests.responsibility.conftest import make_scene, track


def _scene():
    valid = np.ones(91, dtype=bool)
    valid[:20] = False  # a pedestrian that appears at step 20
    return make_scene({"0": track((0, 0), (5, 0)), "1": track((15, 0), (5, 0)),
                       "7": track((5, 5), (0, 1), kind="PEDESTRIAN", valid=valid)}, sdc="0", ooi=("0", "1"))


def test_tracks_follow_decode_tracks_from_proto():
    scene = _scene()
    t = track_infos(scene, predict=["1"])
    assert t["states"].shape == (3, 91, 9) and t["states"].dtype == np.float32
    assert t["object_id"].tolist() == [0, 1, 7] and t["object_type"].tolist() == [0, 0, 1]
    # x, y, z, length, width, height, heading, vx, vy
    np.testing.assert_allclose(t["states"][0, 10], [0, 0, 0, 4.8, 2.0, 1.5, 0, 5, 0])
    assert not t["states"][2, :20].any() and t["valid"][2].sum() == 71
    assert t["role"].tolist() == [[True, True, False], [False, True, True], [False, False, False]]
    assert numeric_id("abc") == numeric_id("abc") and 0 <= numeric_id("abc") < 2 ** 31


def test_map_types_stop_signs_and_polygons():
    features = {
        "100": {"type": "LANE_SURFACE_STREET", "polyline": np.array([[0.0, 0, 0], [50, 0, 0], [100, 0, 0]])},
        "102": {"type": "LANE_BIKE_LANE", "polyline": np.array([[0.0, 3, 0], [10, 3, 0]])},
        "103": {"type": "LANE_FREEWAY", "polyline": np.array([[0.0, 9, 0]])},  # one point: left out
        "104": {"type": "ROAD_EDGE_MEDIAN", "polyline": np.array([[0.0, 5], [9, 5]])},  # 2-D: z = 0
        "105": {"type": "ROAD_LINE_SOLID_DOUBLE_YELLOW", "polyline": np.array([[0.0, 1, 0], [9, 1, 0]])},
        "106": {"type": "ROAD_LINE_BROKEN_SINGLE_WHITE", "polyline": np.array([[0.0, 2, 0], [9, 2, 0]])},
        "107": {"type": "CROSSWALK", "polygon": np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, .5, 0]])},
        "101": {"type": "STOP_SIGN", "position": np.array([5.0, 5.0, 0.0]), "lane": ["100", "102"]},
    }
    seen = []

    def to_polyline(corners):
        seen.append(corners)
        return np.concatenate([corners, corners[:1]], axis=0)

    m = map_infos(features, to_polyline)
    lanes = {lane["id"]: lane["type"] for lane in m["lane"]}
    assert lanes == {100: 2, 102: 3}  # the stop sign's street lane becomes 2; the bike lane stays
    assert [(e["id"], e["type"]) for e in m["road_edge"]] == [(104, 5)]
    assert [(e["id"], e["type"]) for e in m["road_line"]] == [(105, 8), (106, 6)]
    assert [(e["id"], e["type"]) for e in m["crosswalk"]] == [(107, 9)]
    assert seen[0].shape == (4, 3)  # four of the polygon's points, as catk takes them
    # rows: x, y, z, type, id, in feature order
    assert m["all_polylines"].shape == (3 + 2 + 2 + 2 + 2 + 5, 5)
    np.testing.assert_allclose(m["all_polylines"][0], [0, 0, 0, 1, 100])  # the type before the stop-sign override
    a, b = m["road_edge"][0]["polyline_index"]
    np.testing.assert_allclose(m["all_polylines"][a:b, 2], 0.0)


def test_lights_list_the_observed_signals_per_step():
    states = {"200": {"type": "TRAFFIC_LIGHT", "lane": "100",
                      "state": {"object_state": ["LANE_STATE_GO"] * 3 + [None] + ["LANE_STATE_STOP"] * 2}}}
    d = dynamic_map_infos(states, 6)
    assert len(d["lane_id"]) == 6
    assert d["lane_id"][0].tolist() == [[100]] and d["state"][0].tolist() == [["LANE_STATE_GO"]]
    assert d["lane_id"][3].shape == (1, 0) and d["state"][3].shape == (1, 0)  # not observed at step 3
    assert d["state"][5].tolist() == [["LANE_STATE_STOP"]]
    scene = _scene()  # the conftest scene's light: go for 50 steps, then stop
    assert dynamic_map_infos(scene.dynamic_map_states, 91)["state"][60].tolist() == [["LANE_STATE_STOP"]]
