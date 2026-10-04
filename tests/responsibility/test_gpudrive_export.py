import numpy as np
import pytest

from responsibility.gpudrive_export import ERR_VAL, objects, roads, scenario, scene_flags
from tests.responsibility.conftest import make_scene, track


def _scene():
    valid = np.ones(91, dtype=bool)
    valid[60:] = False  # leaves at step 60
    return make_scene({"0": track((0, 0), (5, 0)), "1": track((15, 0), (5, 0), valid=valid),
                       "7": track((5, 5), (0, 1), kind="PEDESTRIAN")}, sdc="0", ooi=("0", "1"))


def test_objects_follow_gpudrive_conversion():
    obj = objects(_scene())
    assert [o["type"] for o in obj] == ["vehicle", "vehicle", "pedestrian"] and [o["id"] for o in obj] == [0, 1, 7]
    o = obj[1]
    assert len(o["position"]) == 91 and o["valid"][59] and not o["valid"][60]
    assert o["position"][60] == {"x": ERR_VAL, "y": ERR_VAL, "z": ERR_VAL} and o["heading"][60] == ERR_VAL
    # goal: the last valid position (step 59: 15 + 5 * 4.9 s)
    assert o["goalPosition"]["x"] == pytest.approx(15 + 5 * 4.9)
    assert o["velocity"][10] == {"x": 5.0, "y": 0.0} and (o["length"], o["width"]) == pytest.approx((4.8, 2.0))


def test_roads_and_metadata():
    scene = _scene()
    r = {x["id"]: x for x in roads(scene.map_features)}
    assert r[100]["type"] == "lane" and r[100]["map_element_id"] == 2 and len(r[100]["geometry"]) == 3
    assert r[101]["type"] == "stop_sign" and r[101]["map_element_id"] == 17 and len(r[101]["geometry"]) == 1
    data = scenario(scene, "cat_0.json", [{"track_index": 1, "difficulty": 2}], experts=False)
    assert data["metadata"] == {"sdc_track_index": 0, "objects_of_interest": [0, 1],
                                "tracks_to_predict": [{"track_index": 1, "difficulty": 2}]}
    assert data["tl_states"] == {}  # GPUDrive has no signals
    assert scene_flags(scene) == {"traffic_lights": True, "has_3d": False}


def test_experts_are_vehicles_that_start_overlapping():
    pytest.importorskip("trimesh")
    pytest.importorskip("fcl")
    scene = make_scene({"0": track((0, 0), (5, 0)), "1": track((2, 0), (5, 0)), "2": track((40, 20), (5, 0))})
    data = scenario(scene, "x.json")
    assert [o["mark_as_expert"] for o in data["objects"]] == [True, True, False]
