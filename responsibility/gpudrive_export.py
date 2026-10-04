"""Scenes in GPUDrive's JSON format, to run CAT's scenes in GPUDrive
(github.com/Emerge-Lab/gpudrive) for a speed and feasibility check.

Mirrors GPUDrive's own WOMD conversion (data_utils/process_waymo_files.py,
``waymo_to_scenario``), field for field, from a Scene and its MetaDrive map:

  objects   every track in scene order: per-step positions {x, y, z},
            headings (wrapped to [-pi, pi)) and velocities, ERR_VAL (-1e4)
            where invalid; validity; width, length and height of the last
            valid state; goalPosition, the last valid position; type
            (vehicle, pedestrian, cyclist, other); the numeric track id;
            mark_as_expert (below)
  roads     every map feature: lanes, road lines and edges as polylines,
            stop signs as their position, crosswalks, speed bumps and
            driveways as polygons; GPUDrive's map_element_id (WOMD's
            roadgraph type codes)
  metadata  sdc_track_index, objects_of_interest (ids), tracks_to_predict

mark_as_expert: GPUDrive replays a vehicle or cyclist from its log, instead
of letting a policy control it, when its first box overlaps another agent's
or a road edge, or its logged path crosses a road edge (the meshes and
tests of the conversion script, with trimesh).

Two differences from GPUDrive's own data, both recorded per scene by
``scene_flags``:
  * GPUDrive drops every WOMD scene with traffic lights, and its simulator
    has no signals. Scenes here keep their place but lose their signal
    states (tl_states is empty), so logged agents still obey lights that a
    controlled agent cannot see.
  * GPUDrive drops scenes with 3D road structures (road edges stacked within
    0.2 m in x-y but more than 0.2 m apart in height, e.g. overpasses).
    They are flagged, not dropped.
"""

from typing import Dict, Iterable, List

import numpy as np

from responsibility.scene import Scene

ERR_VAL = -1e4
OBJECT_TYPES = {"VEHICLE": "vehicle", "PEDESTRIAN": "pedestrian", "CYCLIST": "cyclist"}
MAP_ELEMENT_IDS = {
    "LANE_UNKNOWN": 0, "LANE_FREEWAY": 1, "LANE_SURFACE_STREET": 2, "LANE_BIKE_LANE": 3,
    "ROAD_LINE_UNKNOWN": 5, "ROAD_LINE_BROKEN_SINGLE_WHITE": 6, "ROAD_LINE_SOLID_SINGLE_WHITE": 7,
    "ROAD_LINE_SOLID_DOUBLE_WHITE": 8, "ROAD_LINE_BROKEN_SINGLE_YELLOW": 9, "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 10,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": 11, "ROAD_LINE_SOLID_DOUBLE_YELLOW": 12, "ROAD_LINE_PASSING_DOUBLE_YELLOW": 13,
    "ROAD_EDGE_UNKNOWN": 14, "ROAD_EDGE_BOUNDARY": 15, "ROAD_EDGE_MEDIAN": 16,
    "STOP_SIGN": 17, "CROSSWALK": 18, "SPEED_BUMP": 19, "DRIVEWAY": 20,
}
POLYGONS = ("CROSSWALK", "SPEED_BUMP", "DRIVEWAY")


def _id(key) -> int:
    from responsibility.catk_export import numeric_id

    return numeric_id(key)


def _point(p) -> Dict[str, float]:
    p = np.asarray(p, dtype=float)
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0}


def road_type(kind: str) -> str:
    """GPUDrive's road "type": the WOMD feature it came from."""
    if kind.startswith("LANE_"):
        return "lane"
    if kind.startswith("ROAD_LINE_"):
        return "road_line"
    if kind.startswith("ROAD_EDGE_"):
        return "road_edge"
    return kind.lower()  # stop_sign, crosswalk, speed_bump, driveway


def roads(map_features: Dict) -> List[Dict]:
    out = []
    for key, feature in map_features.items():
        kind = feature.get("type", "")
        if kind == "STOP_SIGN":
            geometry = [_point(feature["position"])]
        elif kind in POLYGONS:
            geometry = [_point(p) for p in feature["polygon"]]
        elif "polyline" in feature:
            geometry = [_point(p) for p in feature["polyline"]]
        else:
            continue
        element = MAP_ELEMENT_IDS.get(kind, MAP_ELEMENT_IDS.get(kind.rsplit("_", 1)[0] + "_UNKNOWN", -1))
        out.append({"geometry": geometry, "type": road_type(kind), "map_element_id": element, "id": _id(key)})
    return out


def objects(scene: Scene) -> List[Dict]:
    out = []
    for i in range(scene.n_agents):
        valid = scene.valid[i]
        last = int(np.flatnonzero(valid)[-1]) if valid.any() else 0
        pos, vel = scene.position[i].astype(float), scene.velocity[i].astype(float)
        head = (scene.heading[i].astype(float) + np.pi) % (2 * np.pi) - np.pi
        out.append({
            "position": [_point(pos[t]) if valid[t] else {"x": ERR_VAL, "y": ERR_VAL, "z": ERR_VAL}
                         for t in range(scene.n_steps)],
            "width": float(scene.width[i, last]), "length": float(scene.length[i, last]),
            "height": float(scene.height[i, last]),
            "heading": [float(head[t]) if valid[t] else ERR_VAL for t in range(scene.n_steps)],
            "velocity": [{"x": float(vel[t, 0]), "y": float(vel[t, 1])} if valid[t] else {"x": ERR_VAL, "y": ERR_VAL}
                         for t in range(scene.n_steps)],
            "valid": [bool(v) for v in valid],
            "goalPosition": _point(pos[last]),
            "type": OBJECT_TYPES.get(scene.types[i], "other"),
            "id": _id(scene.track_ids[i]),
            "mark_as_expert": False,
        })
    return out


def _edge_segments(road_list: List[Dict]) -> List:
    segments = []
    for road in road_list:
        if road["type"] == "road_edge":
            v = [[p["x"], p["y"], p["z"]] for p in road["geometry"]]
            segments += [[v[k], v[k + 1]] for k in range(len(v) - 1)]
    return segments


def mark_experts(object_list: List[Dict], road_list: List[Dict]) -> None:
    """GPUDrive's expert marking (waymo_to_scenario), in place."""
    import trimesh

    def filtered(segments):
        return [s for s in segments if np.linalg.norm(np.subtract(s[1], s[0])) >= 1e-6]

    def mesh(segments, height=2.0, width=0.2):
        boxes = []
        for start, end in np.asarray(segments, dtype=np.float64):
            d = end - start
            box = trimesh.creation.box(extents=[1.0, width, height])
            box.apply_translation([0.5, 0, 0])
            box.apply_scale([np.linalg.norm(d), 1.0, 1.0])
            box.apply_transform(trimesh.transformations.rotation_matrix(np.arctan2(d[1], d[0]), [0, 0, 1]))
            box.apply_translation(start)
            boxes.append(box)
        return trimesh.util.concatenate(boxes)

    edges = filtered(_edge_segments(road_list))
    roads_cm = trimesh.collision.CollisionManager()
    if edges:
        roads_cm.add_object("road_edges", mesh(edges))
    agents_cm, paths_cm = trimesh.collision.CollisionManager(), trimesh.collision.CollisionManager()
    for obj in object_list:
        if obj["type"] not in ("vehicle", "cyclist") or not any(obj["valid"]):
            continue
        first = obj["valid"].index(True)
        p = obj["position"][first]
        box = trimesh.creation.box(extents=[obj["length"], obj["width"], obj["height"]])
        box.apply_transform(trimesh.transformations.rotation_matrix(obj["heading"][first], [0, 0, 1]))
        box.apply_translation([p["x"], p["y"], p["z"]])
        agents_cm.add_object(str(obj["id"]), box)
        v = [[q["x"], q["y"], q["z"]] for q in obj["position"]]
        path = filtered([[v[k], v[k + 1]] for k in range(len(v) - 1) if obj["valid"][k] and obj["valid"][k + 1]])
        if path:
            paths_cm.add_object(str(obj["id"]), mesh(path))
    colliding = set()
    _, pairs = agents_cm.in_collision_internal(return_names=True)
    for a, b in pairs:
        colliding |= {a, b}
    if edges:
        _, pairs = agents_cm.in_collision_other(roads_cm, return_names=True)
        colliding |= {a for a, _ in pairs}
        _, pairs = paths_cm.in_collision_other(roads_cm, return_names=True)
        colliding |= {a for a, _ in pairs}
    for obj in object_list:
        obj["mark_as_expert"] = obj["type"] in ("vehicle", "cyclist") and str(obj["id"]) in colliding


def scene_flags(scene: Scene) -> Dict[str, bool]:
    """What GPUDrive's own conversion would have dropped the scene for."""
    lights = any(s.get("type") == "TRAFFIC_LIGHT" and any(x is not None for x in s["state"]["object_state"])
                 for s in scene.dynamic_map_states.values())
    pts = [np.asarray(f["polyline"], dtype=float) for f in scene.map_features.values()
           if f.get("type", "").startswith("ROAD_EDGE_") and len(f.get("polyline", [])) > 1]
    has_3d = False
    if pts:
        pts = np.concatenate([p if p.shape[1] == 3 else np.pad(p, ((0, 0), (0, 1))) for p in pts])
        for k in range(0, len(pts), 1000):
            d = np.linalg.norm(pts[k:k + 1000, None, :2] - pts[None, :, :2], axis=-1)
            a, b = np.nonzero((d < 0.2) & (d > 0))
            if np.any(np.abs(pts[k + a, 2] - pts[b, 2]) > 0.2):
                has_3d = True
                break
    return {"traffic_lights": lights, "has_3d": has_3d}


def scenario(scene: Scene, name: str, tracks_to_predict: Iterable[Dict] = (), experts: bool = True) -> Dict:
    road_list, object_list = roads(scene.map_features), objects(scene)
    if experts:
        mark_experts(object_list, road_list)
    return {
        "name": name,
        "scenario_id": scene.scenario_id,
        "objects": object_list,
        "roads": road_list,
        "tl_states": {},
        "metadata": {
            "sdc_track_index": int(scene.sdc),
            "objects_of_interest": [_id(scene.track_ids[i]) for i in scene.objects_of_interest],
            "tracks_to_predict": [{"track_index": int(t["track_index"]), "difficulty": int(t.get("difficulty", 0))}
                                  for t in tracks_to_predict],
        },
    }
