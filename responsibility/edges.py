"""Road edges of a scene's map, and whether a path crosses one.

A map built online by perception has road edges (curbs, medians) and lane
lines but few lane centrelines inside intersections and no lane topology.
The edges still outline the road there, intersection corners included, so
they serve both tests that must work on such a map:

  crosses_road_edge  drivable area (motion_filter's "drivable_edges"): the
                     trajectory, driven from the agent's position, crosses
                     no road edge -- except the edges the agent's own logged
                     path crosses (map noise, or a driveway, which CAT's
                     WOMD v1.1 maps lack).
  segments_cross     the primitive, also used by intent.crosses_edge.

A gap in the edges removes nothing: an incomplete map only makes the tests
looser.
"""

import numpy as np
import torch

from responsibility.scene import Scene

EDGE_TYPES = ("ROAD_EDGE_BOUNDARY", "ROAD_EDGE_MEDIAN")
PROBE_EVERY = 5  # steps (0.5 s) between the points of a path checked for crossings


def edge_segments(scene: Scene, types=EDGE_TYPES) -> np.ndarray:
    """The road edges of the map as segments [E, 2, 2] (none: [0, 2, 2])."""
    out = []
    for f in scene.map_features.values():
        poly = np.asarray(f.get("polyline", []), dtype=float)
        if f.get("type") in types and poly.ndim == 2 and len(poly) >= 2:
            out.append(np.stack([poly[:-1, :2], poly[1:, :2]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2, 2))


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def crossing_matrix(p: torch.Tensor, q: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Whether each segment p->q [Q] crosses each edge [E, 2, 2]: [Q, E].
    Half-open along p->q (from p, not to q), so a path through a point on an
    edge crosses it once, in the segment that leaves that point."""
    c, d = edges[:, 0], edges[:, 1]  # [E, 2]
    r, s = (q - p)[:, None], (d - c)[None]  # [Q, 1, 2], [1, E, 2]
    denom = _cross(r, s)  # [Q, E]
    cp = c[None] - p[:, None]
    t = _cross(cp, s) / denom  # inf or nan where parallel: masked below
    u = _cross(cp, r) / denom
    return (denom.abs() > 1e-12) & (t >= 0) & (t < 1) & (u >= 0) & (u <= 1)


def segments_cross(p: torch.Tensor, q: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Whether each segment p->q [Q] crosses any edge [E, 2, 2]."""
    return crossing_matrix(p, q, edges).any(-1)


def near_edges(edges: np.ndarray, points: np.ndarray, margin: float = 1.0) -> torch.Tensor:
    """The edges [E', 2, 2] whose bounding box overlaps that of the points [..., 2]."""
    e = torch.as_tensor(edges, dtype=torch.float64)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    lo = torch.as_tensor(pts.min(0) - margin)
    hi = torch.as_tensor(pts.max(0) + margin)
    return e[((e.amin(1) <= hi) & (e.amax(1) >= lo)).all(-1)]


def path_segments(start: np.ndarray, trajs: np.ndarray, every: int = PROBE_EVERY) -> np.ndarray:
    """The path of each trajectory [N, T, 2] from ``start`` [2] through every
    ``every``-th point and its last: segments [N, K, 2, 2]."""
    idx = sorted(set(range(every - 1, trajs.shape[1], every)) | {trajs.shape[1] - 1})
    pts = np.concatenate([np.broadcast_to(start, (len(trajs), 1, 2)), trajs[:, idx]], axis=1)
    return np.stack([pts[:, :-1], pts[:, 1:]], axis=2)


def crosses_road_edge(scene: Scene, agent: int, step: int, trajs: np.ndarray,
                      edges=None) -> np.ndarray:
    """Whether each trajectory [N, T, 2] (scene frame, from step + 1), driven
    from the agent's position at ``step``, crosses a road edge the agent's
    own logged path does not cross [N] (all False on a map without edges)."""
    edges = edge_segments(scene) if edges is None else edges
    if len(edges) == 0 or len(trajs) == 0:
        return np.zeros(len(trajs), dtype=bool)
    start = scene.position[agent, step, :2]
    segs = path_segments(start, trajs)  # [N, K, 2, 2]
    e = near_edges(edges, segs)
    if len(e) == 0:
        return np.zeros(len(trajs), dtype=bool)
    fut = slice(step + 1, scene.n_steps)
    logged = scene.position[agent, fut, :2][scene.valid[agent, fut]]
    if len(logged):
        own = path_segments(start, logged[None], every=1)[0]
        crossed = crossing_matrix(torch.as_tensor(own[:, 0]), torch.as_tensor(own[:, 1]), e).any(0)
        e = e[~crossed]
        if len(e) == 0:
            return np.zeros(len(trajs), dtype=bool)
    flat = torch.as_tensor(segs.reshape(-1, 2, 2))
    hit = torch.cat([segments_cross(c[:, 0], c[:, 1], e) for c in flat.split(4096)])
    return hit.view(segs.shape[:2]).any(-1).numpy()
