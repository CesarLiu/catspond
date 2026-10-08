"""Per-scene static data for the GPUDrive environment (gpucat/env.py), in the
scene's own (raw) coordinates -- GPUDrive's positions are these minus the
world mean, which the env adds back:

  route     the SDC's logged path (valid steps) extended EXTENSION metres
            along its last heading: MetaDrive's navigation reference for the
            SDC, against which progress (the driving reward), route
            completion and the 10 m out-of-route test are measured;
            route_len is the logged part's length (completion's denominator)
  edges     road-edge segments (boundary and median) within NEAR_ROUTE
            of the route: MetaDrive's ScenarioBlock turns every road edge
            (is_road_edge) into 0.2 m wide "sidewalk" boxes named
            BOUNDARY_LINE, and touching one (crash_sidewalk) is out of road
  yellow    segments (also within NEAR_ROUTE) of the yellow lines a vehicle must not cross: solid
            single, solid double and passing double yellow: MetaDrive builds
            every yellow line that is not broken as a continuous one named
            LINE_SOLID_SINGLE_YELLOW (on_yellow_continuous_line)
  sizes     length and width of the agent in each GPUDrive slot
            (gpucat.scenes.agent_order; the last valid state's, as
            GPUDrive's conversion takes them), and its type (1 vehicle,
            2 cyclist, 3 pedestrian, 0 other or empty)
"""

from typing import Dict

import numpy as np

from gpucat.scenes import agent_order
from responsibility.scene import Scene

EXTENSION = 100.0  # m past the log's last point
NEAR_ROUTE = 20.0  # m: segments farther from the route are never reached (out of route beyond 10 m)
MAX_SLOTS = 64  # GPUDrive's kMaxAgentCount
EDGE_TYPES = ("ROAD_EDGE_BOUNDARY", "ROAD_EDGE_MEDIAN")
YELLOW_TYPES = ("ROAD_LINE_SOLID_SINGLE_YELLOW", "ROAD_LINE_SOLID_DOUBLE_YELLOW", "ROAD_LINE_PASSING_DOUBLE_YELLOW")
TYPE_CODES = {"VEHICLE": 1, "CYCLIST": 2, "PEDESTRIAN": 3}


def segments(map_features: Dict, types) -> np.ndarray:
    """[N, 4] (x0, y0, x1, y1) of every polyline segment of the given types."""
    out = []
    for f in map_features.values():
        poly = np.asarray(f.get("polyline", []), dtype=float)
        if f.get("type") in types and poly.ndim == 2 and len(poly) >= 2:
            out.append(np.concatenate([poly[:-1, :2], poly[1:, :2]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 4))


def near(segs: np.ndarray, route: np.ndarray, radius: float) -> np.ndarray:
    """The segments [N, 4] with an end or midpoint within radius of a route
    vertex (route points are under a metre apart, so this keeps every
    segment the vehicle can reach while within 10 m of the route)."""
    if len(segs) == 0:
        return segs
    pts = np.concatenate([segs[:, :2], segs[:, 2:], 0.5 * (segs[:, :2] + segs[:, 2:])])  # [3N, 2]
    dense = np.concatenate([np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a))) + 1))
                            for a, b in zip(route[:-1], route[1:])])
    d = np.full(len(pts), np.inf)
    for k in range(0, len(dense), 512):
        d = np.minimum(d, np.linalg.norm(pts[:, None] - dense[None, k:k + 512], axis=-1).min(1))
    return segs[(d.reshape(3, -1) <= radius).any(0)]


def scene_static(scene: Scene) -> Dict[str, np.ndarray]:
    sdc = scene.sdc
    valid = np.flatnonzero(scene.valid[sdc])
    path = scene.position[sdc, valid, :2].astype(float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-3])  # drop repeats
    path = path[keep]
    route_len = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
    h = float(scene.heading[sdc, valid[-1]])
    route = np.concatenate([path, path[-1:] + EXTENSION * np.array([[np.cos(h), np.sin(h)]])])
    order = agent_order(scene)[:MAX_SLOTS]
    sizes = np.zeros((MAX_SLOTS, 2))
    types = np.zeros(MAX_SLOTS, dtype=np.int64)
    for slot, i in enumerate(order):
        v = np.flatnonzero(scene.valid[i])
        last = int(v[-1]) if len(v) else 0
        sizes[slot] = scene.length[i, last], scene.width[i, last]
        types[slot] = TYPE_CODES.get(scene.types[i], 0)
    return {"route": route, "route_len": route_len,
            "edges": near(segments(scene.map_features, EDGE_TYPES), route, NEAR_ROUTE),
            "yellow": near(segments(scene.map_features, YELLOW_TYPES), route, NEAR_ROUTE), "sizes": sizes, "types": types,
            "n_agents": len(order)}


def pad(arrays, fill=0.0) -> np.ndarray:
    """Stacks arrays of different first lengths into [S, max, ...]."""
    n = max(len(a) for a in arrays)
    out = np.full((len(arrays), max(n, 1)) + arrays[0].shape[1:], fill, dtype=float)
    for i, a in enumerate(arrays):
        out[i, : len(a)] = a
    return out


class StaticBank:
    """build_static.py's .npz on a device, rows by scene stem."""

    def __init__(self, path: str, device: str = "cuda"):
        import torch

        z = np.load(path)
        self.stems = [str(s) for s in z["stems"]]
        self.index = {s: i for i, s in enumerate(self.stems)}
        f = torch.float32
        t = lambda k, dtype=f: torch.as_tensor(z[k], dtype=dtype, device=device)  # noqa: E731
        self.route, self.route_n, self.route_len = t("route"), t("route_n", torch.long), t("route_len")
        self.edges, self.edges_n = t("edges"), t("edges_n", torch.long)
        self.yellow, self.yellow_n = t("yellow"), t("yellow_n", torch.long)
        self.sizes, self.types, self.n_agents = t("sizes"), t("types", torch.long), t("n_agents", torch.long)
