"""convert_womd_split on a synthetic WOMD shard: a Scenario proto built here,
framed as a tfrecord. Skips unless the WOMD protos load (waymo_open_dataset,
or the catk environment's copy) and MetaDrive imports."""

import csv
import struct

import numpy as np
import pytest

from responsibility.right_of_way import priority
from responsibility.scene import Scene, cat_agent_order, womd_features

convert = pytest.importorskip("scripts.responsibility.convert_womd_split")
try:
    converter = convert.load_converter(str(convert.CATK_PROTOS))
    from waymo_open_dataset.protos import scenario_pb2
except Exception as e:  # noqa: BLE001  (no protos, or no MetaDrive)
    pytest.skip(f"WOMD protos or MetaDrive unavailable: {e}", allow_module_level=True)

VEHICLE, PEDESTRIAN = 1, 2


def _scenario(sid, ooi_types, sdc_in_ooi=True):
    s = scenario_pb2.Scenario()
    s.scenario_id = sid
    s.timestamps_seconds.extend(np.arange(91) * 0.1)
    s.current_time_index = 10
    for i, (kind, y) in enumerate(zip([VEHICLE, *ooi_types], [0.0, 3.5, -20.0])):
        track = s.tracks.add()
        track.id, track.object_type = 100 + i, kind
        for t in range(91):
            state = track.states.add()
            state.center_x, state.center_y, state.center_z = -30 + t * 0.5, y, 0.0
            state.length, state.width, state.height, state.heading = 4.5, 2.0, 1.5, 0.0
            state.velocity_x, state.velocity_y, state.valid = 5.0, 0.0, True
    s.sdc_track_index = 0
    s.objects_of_interest.extend([100, 101] if sdc_in_ooi else [101, 102])
    lane = s.map_features.add()
    lane.id = 7
    lane.lane.type = 2  # surface street
    for x in range(-40, 41, 5):
        p = lane.lane.polyline.add()
        p.x, p.y, p.z = float(x), 0.0, 0.0
    drive = s.map_features.add()
    drive.id = 8
    for x, y in ((0, 10), (5, 10), (5, 15)):
        p = drive.driveway.polygon.add()
        p.x, p.y, p.z = float(x), float(y), 0.0
    for t in range(91):
        light = s.dynamic_map_states.add().lane_states.add()
        light.lane, light.state = 7, 6  # LANE_STATE_GO
        light.stop_point.x, light.stop_point.y = 20.0, 0.0
    return s


def _shard(path, scenarios):
    with open(path, "wb") as f:
        for s in scenarios:
            data = s.SerializeToString()
            f.write(struct.pack("<Q", len(data)) + b"\0" * 4 + data + b"\0" * 4)  # CRCs are not checked


def test_a_shard_is_converted_into_scene_files(tmp_path):
    raw = tmp_path / "validation_interactive"
    raw.mkdir()
    _shard(raw / "validation_interactive.tfrecord-00000-of-00002",
           [_scenario("a1", [VEHICLE, VEHICLE]), _scenario("p1", [PEDESTRIAN, VEHICLE]),
            _scenario("a2", [VEHICLE, VEHICLE], sdc_in_ooi=False)])
    _shard(raw / "validation_interactive.tfrecord-00001-of-00002", [_scenario("b1", [VEHICLE, VEHICLE])])
    out = tmp_path / "out"
    rows = convert.main(["--tfrecords", str(raw), "--out-dir", str(out), "--workers", "2"])
    assert sorted(r["scenario_id"] for r in rows) == ["a1", "a2", "b1"]  # the pedestrian pair is left out
    index = {r["scenario_id"]: r for r in csv.DictReader(open(out / "index.csv"))}
    assert index["a1"]["sdc_in_ooi"] == "1" and index["a2"]["sdc_in_ooi"] == "0" and index["a1"]["in_cat"] == "0"
    assert sorted(p.name for p in (out / "scenes").iterdir()) == ["a1.pkl", "a2.pkl", "b1.pkl"]  # scenes only

    scene = Scene.load(out / "scenes" / "a1.pkl")  # CAT's format: the readers take it as they take CAT's files
    assert scene.scenario_id == "a1" and scene.n_steps == 91 and scene.track_ids[scene.sdc] == "100"
    assert sorted(scene.track_ids[i] for i in scene.objects_of_interest) == ["100", "101"]
    np.testing.assert_allclose(scene.position[1, 10, :2], [-25.0, 3.5])
    assert {f["type"] for f in scene.map_features.values()} == {"LANE_SURFACE_STREET", "DRIVEWAY"}
    assert scene.dynamic_map_states["7"]["state"]["object_state"][0] == "LANE_STATE_GO"
    features = womd_features(scene, 10, cat_agent_order(scene))  # DenseTNT's input leaves the driveway out
    assert set(features["roadgraph_samples/type"][features["roadgraph_samples/id"][:, 0] >= 0, 0]) == {2}
    a, b = scene.objects_of_interest
    assert priority(scene, a, b, 50).case == "same-direction"  # side by side, 3.5 m apart

    # a re-run converts nothing again; --limit converts the first kept scenes of a shard
    assert len(convert.main(["--tfrecords", str(raw), "--out-dir", str(out)])) == 3
    rows = convert.main(["--tfrecords", str(raw), "--out-dir", str(tmp_path / "test"), "--shards", "1", "--limit", "1"])
    assert [r["scenario_id"] for r in rows] == ["a1"]
