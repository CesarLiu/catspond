"""Scenes in catk's cached-WOMD format, so CAT-K's SMART and its
responsibility implementation (the catk repository, src/responsibility)
can measure the same scenes and rollouts as DenseTNT here
(responsibility/SMART_PLAN.md).

catk caches a WOMD scenario in two stages (src/data_preprocess.py):
``decode_*_from_proto`` read the protobuf into plain arrays and lists
(tracks, map polylines, traffic-light states), and ``get_agent_features``,
``get_map_features`` / ``process_dynamic_map`` and ``preprocess_map`` turn
those into SMART's tensors. CAT's scenes are the same protobufs converted to
MetaDrive's format, so this module rebuilds only the first stage, from a
Scene, the way the decoders do:

  tracks   states [N, 91, 9] (x, y, z, length, width, height, heading, vx,
           vy), validity, WOMD type - 1 (vehicle 0, pedestrian 1, cyclist
           2, other 3), role (sdc, object of interest, track to predict)
  map      lanes (freeway 0, surface street 1, bike lane 3; a stop sign's
           lanes become 2 unless they are bike lanes), road edges
           (unknown 3, boundary 4, median 5), road lines (broken 6, solid
           single 7, double 8), crosswalks, speed bumps and driveways (9,
           four points of the polygon as a closed polyline); features with
           fewer than two points are left out, as catk does
  lights   per step, the lanes whose signal was observed and its state
           (MetaDrive's None = not observed at that step)

The second stage is catk's own code, run in catk's environment by
scripts/responsibility/export_catk.py. Scenes rebuilt from rollouts
(rollouts.scene_from_rollout) export the same way, with the simulated
tracks in place of the logged ones.
"""

import zlib
from typing import Callable, Dict, Iterable, List

import numpy as np

from responsibility.scene import Scene

AGENT_TYPES = {"VEHICLE": 0, "PEDESTRIAN": 1, "CYCLIST": 2}  # anything else: 3 (WOMD's TYPE_OTHER - 1)
LANE_TYPES = {"LANE_FREEWAY": 0, "LANE_SURFACE_STREET": 1, "LANE_BIKE_LANE": 3}  # other lanes: 1 (undefined)
ROAD_EDGE_TYPES = {"ROAD_EDGE_BOUNDARY": 4, "ROAD_EDGE_MEDIAN": 5}  # other edges: 3 (unknown)
BROKEN_LINES = ("ROAD_LINE_BROKEN_SINGLE_WHITE", "ROAD_LINE_BROKEN_SINGLE_YELLOW", "ROAD_LINE_BROKEN_DOUBLE_YELLOW")
SOLID_SINGLE_LINES = ("ROAD_LINE_SOLID_SINGLE_WHITE", "ROAD_LINE_SOLID_SINGLE_YELLOW")
POLYGON_FEATURES = ("CROSSWALK", "SPEED_BUMP", "DRIVEWAY")
STOP_SIGN_LANE_TYPE = 2


def numeric_id(track_id) -> int:
    """WOMD ids are integers; anything else gets a stable 31-bit number."""
    text = str(track_id)
    return int(text) if text.lstrip("-").isdigit() else zlib.crc32(text.encode()) & 0x7FFFFFFF


def track_infos(scene: Scene, predict: Iterable[str] = ()) -> Dict[str, np.ndarray]:
    """decode_tracks_from_proto's output for ``scene``; ``predict`` holds the
    track ids of the scenario's tracks to predict (metadata, not in Scene)."""
    predict = {str(t) for t in predict}
    interest = set(scene.objects_of_interest)
    states = np.concatenate([
        scene.position[..., :3],
        scene.length[..., None], scene.width[..., None], scene.height[..., None],
        scene.heading[..., None],
        scene.velocity[..., :2],
    ], axis=-1).astype(np.float32)
    states[~scene.valid] = 0.0  # as WOMD stores invalid states
    role = np.zeros((scene.n_agents, 3), dtype=bool)
    role[scene.sdc, 0] = True
    for i, tid in enumerate(scene.track_ids):
        role[i, 1] = i in interest
        role[i, 2] = tid in predict
    return {
        "object_id": np.array([numeric_id(t) for t in scene.track_ids], dtype=np.int64),
        "object_type": np.array([AGENT_TYPES.get(t, 3) for t in scene.types], dtype=np.uint8),
        "states": states,
        "valid": scene.valid.astype(bool),
        "role": role,
    }


def _rows(points: np.ndarray, kind: int, feature_id: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] == 2:
        points = np.concatenate([points, np.zeros((len(points), 1))], axis=-1)
    return np.concatenate([points[:, :3], np.full((len(points), 1), kind), np.full((len(points), 1), feature_id)],
                          axis=-1)


def map_infos(map_features: Dict, polygon_to_polyline: Callable[[np.ndarray], np.ndarray]) -> Dict:
    """decode_map_features_from_proto's output for a MetaDrive map.
    ``polygon_to_polyline`` is catk's get_polylines_from_polygon."""
    out: Dict[str, List] = {"lane": [], "road_edge": [], "road_line": [], "crosswalk": []}
    polylines, count = [], 0

    def add(key, feature_id, kind, rows):
        nonlocal count
        out[key].append({"id": feature_id, "type": kind, "polyline_index": (count, count + len(rows))})
        polylines.append(rows)
        count += len(rows)

    for key, feature in map_features.items():
        kind, fid = feature.get("type", ""), numeric_id(key)
        if kind.startswith("LANE_") or kind.startswith("ROAD_EDGE_") or kind.startswith("ROAD_LINE_"):
            line = np.asarray(feature.get("polyline", []))
            if line.ndim != 2 or len(line) < 2:
                continue
            if kind.startswith("LANE_"):
                add("lane", fid, LANE_TYPES.get(kind, 1), _rows(line, LANE_TYPES.get(kind, 1), fid))
            elif kind.startswith("ROAD_EDGE_"):
                add("road_edge", fid, ROAD_EDGE_TYPES.get(kind, 3), _rows(line, ROAD_EDGE_TYPES.get(kind, 3), fid))
            else:
                t = 6 if kind in BROKEN_LINES else 7 if kind in SOLID_SINGLE_LINES else 8
                add("road_line", fid, t, _rows(line, t, fid))
        elif kind in POLYGON_FEATURES:
            xyz = np.asarray(feature["polygon"], dtype=np.float64)
            if xyz.shape[-1] == 2:
                xyz = np.concatenate([xyz, np.zeros((len(xyz), 1))], axis=-1)
            corners = xyz[np.linspace(0, xyz.shape[0], 4, endpoint=False, dtype=int)]
            add("crosswalk", fid, 9, _rows(polygon_to_polyline(corners), 9, fid))

    stopped = {numeric_id(lane) for f in map_features.values() if f.get("type") == "STOP_SIGN"
               for lane in f.get("lane", [])}
    for lane in out["lane"]:
        if lane["id"] in stopped and lane["type"] < STOP_SIGN_LANE_TYPE:
            lane["type"] = STOP_SIGN_LANE_TYPE
    out["all_polylines"] = (np.concatenate(polylines, axis=0).astype(np.float32) if polylines
                            else np.zeros((0, 8), dtype=np.float32))
    return out


def dynamic_map_infos(dynamic_map_states: Dict, n_steps: int) -> Dict[str, List[np.ndarray]]:
    """decode_dynamic_map_states_from_proto's output: per step, the observed
    signals' lane ids and states, each as a [1, n] array."""
    lights = [(numeric_id(s["lane"]), s["state"]["object_state"]) for s in dynamic_map_states.values()
              if s.get("type") == "TRAFFIC_LIGHT"]
    out = {"lane_id": [], "state": []}
    for t in range(n_steps):
        seen = [(lane, states[t]) for lane, states in lights if t < len(states) and states[t] is not None]
        out["lane_id"].append(np.array([[lane for lane, _ in seen]], dtype=np.int64).reshape(1, -1))
        out["state"].append(np.array([[str(state) for _, state in seen]], dtype=object).reshape(1, -1).astype(str)
                            if seen else np.zeros((1, 0), dtype="<U1"))
    return out
