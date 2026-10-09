"""Same intent without lane topology (``--intent``): which alternatives of
an agent's motion set execute the manoeuvre its log shows, judged from the
trajectories themselves and, where the map has them, road edges.

lanes.route_lanes tells intent through WOMD's lane graph (entry and exit
lanes, side neighbours). A map built online by perception has lane lines
and road edges but almost no topology in intersections, so that test has
nothing to work with there. This one needs no lanes at all:

  heading  map-free: at the last step the log covers (T*, at most 8 s
           ahead), the alternative's direction of travel lies within
           ``max_heading`` (deg) of the logged path's direction where the
           alternative is (at the nearest point of the path, extended 100 m
           along its last heading). Comparing with the path where it is, not
           with the log at the same time, keeps a braking alternative that
           is still in the middle of the logged turn. Not applied when the
           logged path is shorter than MIN_INTENT_PATH: an agent that waits
           shows no intent.
  lateral  map-free: the alternative stays within ``max_lateral`` (m) of
           the logged path (extended as above). With the heading test this
           is the manoeuvre: going straight when the log turns left ends
           tens of metres off its path, a left turn into another lane of
           the same road a few metres.
  edges    from the map, if it has road edges: every second up to T*, the
           segment from the alternative's position to the nearest point of
           the logged path crosses no road edge (ROAD_EDGE_BOUNDARY,
           ROAD_EDGE_MEDIAN). That separates a ramp, a frontage road or the
           far side of a median. A missing edge removes nothing, so an
           incomplete map only makes the filter looser, never wrong.

same_heading and crosses_edge return masks [N] over the trajectories;
motion_filter.MotionFilter combines them with its route deviation as the
"intent" filter.
"""

from typing import Optional

import numpy as np
import torch

from responsibility.edges import edge_segments, near_edges, segments_cross
from responsibility.geometry import lateral_deviation, logged_route
from responsibility.scene import Scene

MIN_INTENT_PATH = 5.0  # m of logged path below which the heading shows no intent
CHECK_EVERY = 10  # steps (1 s) between the edge checks
MIN_MOVE = 0.1  # m per step below which a displacement's direction is noise (1 m/s)


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def logged_span(scene: Scene, agent: int, step: int, n: int):
    """(T*, path length): the offset (1-based, in steps after ``step``) of
    the last logged position within ``n`` steps, or 0 if there is none, and
    the length (m) of the logged path up to it."""
    fut = slice(step + 1, step + 1 + n)
    valid = scene.valid[agent, fut]
    if not valid.any():
        return 0, 0.0
    pts = np.concatenate([scene.position[agent, step, :2][None], scene.position[agent, fut, :2][valid]])
    return int(np.flatnonzero(valid).max()) + 1, float(np.linalg.norm(np.diff(pts, axis=0), axis=-1).sum())


def travel_heading(trajs: np.ndarray, heading0: float) -> np.ndarray:
    """Each trajectory's direction of travel at its last step [N]: that of
    its last displacement longer than MIN_MOVE, or ``heading0`` if it never
    moves."""
    d = np.diff(trajs, axis=1)
    moving = np.linalg.norm(d, axis=-1) > MIN_MOVE  # [N, T-1]
    last = np.where(moving.any(1), d.shape[1] - 1 - np.argmax(moving[:, ::-1], axis=1), -1)
    out = np.full(len(trajs), heading0, dtype=float)
    ok = last >= 0
    picked = d[np.flatnonzero(ok), last[ok]]
    out[ok] = np.arctan2(picked[:, 1], picked[:, 0])
    return out


def path_heading(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """The direction [...] of the polyline [P, 2] at its nearest point to
    each point [..., 2] (that of the nearest segment)."""
    start, seg = polyline[:-1], polyline[1:] - polyline[:-1]
    rel = points[..., None, :] - start
    t = ((rel * seg).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
    dist = torch.linalg.norm(rel - t[..., None] * seg, dim=-1)  # [..., S]
    nearest = seg[dist.argmin(-1)]
    return torch.atan2(nearest[..., 1], nearest[..., 0])


def same_heading(scene: Scene, agent: int, step: int, trajs: np.ndarray, max_heading: float) -> np.ndarray:
    """Whether each trajectory [N] heads, at T*, within ``max_heading`` (deg)
    of the logged path where it is (all True when the log shows no intent)."""
    t_star, length = logged_span(scene, agent, step, trajs.shape[1])
    if t_star == 0 or length < MIN_INTENT_PATH:
        return np.ones(len(trajs), dtype=bool)
    heading0 = float(scene.heading[agent, step])
    start = np.broadcast_to(scene.position[agent, step, :2], (len(trajs), 1, 2))
    alt = travel_heading(np.concatenate([start, trajs[:, :t_star]], axis=1), heading0)
    fut = slice(step + 1, step + 1 + t_star)
    path = logged_route(torch.as_tensor(scene.position[agent, step, :2], dtype=torch.float64),
                        torch.as_tensor(scene.position[agent, fut, :2], dtype=torch.float64),
                        torch.as_tensor(scene.valid[agent, fut]),
                        torch.tensor(float(scene.heading[agent, step + t_star]), dtype=torch.float64))
    path = path[torch.cat([torch.ones(1, dtype=torch.bool), torch.linalg.norm(path[1:] - path[:-1], dim=-1) > 1e-6])]
    if len(path) < 2:
        return np.ones(len(trajs), dtype=bool)
    local = path_heading(torch.as_tensor(trajs[:, t_star - 1], dtype=torch.float64), path).numpy()
    return np.abs(_wrap(alt - local)) <= np.deg2rad(max_heading)


def distance_to_logged_path(scene: Scene, agent: int, step: int, points: np.ndarray) -> np.ndarray:
    """Distance [N] (m) of each point [N, 2] from the agent's logged path
    from ``step`` on, extended 100 m along its last logged heading (for b's
    own drive mode in beta_c: metrics.same_mode_support)."""
    fut = slice(step + 1, scene.n_steps)
    valid = scene.valid[agent, fut]
    last = step + 1 + int(np.flatnonzero(valid).max()) if valid.any() else step
    route = logged_route(torch.as_tensor(scene.position[agent, step, :2], dtype=torch.float64),
                         torch.as_tensor(scene.position[agent, fut, :2], dtype=torch.float64), torch.as_tensor(valid),
                         torch.tensor(float(scene.heading[agent, last]), dtype=torch.float64))
    return lateral_deviation(torch.as_tensor(np.asarray(points, dtype=float), dtype=torch.float64), route).numpy()


def nearest_on_polyline(points: torch.Tensor, polyline: torch.Tensor) -> torch.Tensor:
    """The nearest point [..., 2] of the polyline [P, 2] to each point [..., 2]."""
    if len(polyline) == 1:
        return polyline[0].expand_as(points)
    start, seg = polyline[:-1], polyline[1:] - polyline[:-1]
    rel = points[..., None, :] - start
    t = ((rel * seg).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
    closest = start + t[..., None] * seg  # [..., S, 2]
    i = torch.linalg.norm(points[..., None, :] - closest, dim=-1).argmin(-1)
    return torch.gather(closest, -2, i[..., None, None].expand(*i.shape, 1, 2)).squeeze(-2)


def crosses_edge(scene: Scene, agent: int, step: int, trajs: np.ndarray,
                 edges: Optional[np.ndarray] = None) -> np.ndarray:
    """Whether each trajectory [N] is, at some check up to T*, across a road
    edge from the logged path (all False on a map without edges)."""
    edges = edge_segments(scene) if edges is None else edges
    t_star, _ = logged_span(scene, agent, step, trajs.shape[1])
    if len(edges) == 0 or t_star == 0:
        return np.zeros(len(trajs), dtype=bool)
    checks = sorted(set(range(CHECK_EVERY - 1, t_star, CHECK_EVERY)) | {t_star - 1})
    probe = torch.as_tensor(trajs[:, checks], dtype=torch.float64)  # [N, K, 2]
    fut = slice(step + 1, step + 1 + t_star)
    # the logged path without its extension: a point ahead of the log is
    # joined to the log's end along the road, not to a straight line past it
    path = torch.as_tensor(np.concatenate([scene.position[agent, step, :2][None],
                                           scene.position[agent, fut, :2][scene.valid[agent, fut]]]),
                           dtype=torch.float64)
    target = nearest_on_polyline(probe, path)
    e = near_edges(edges, np.concatenate([probe.reshape(-1, 2).numpy(), path.numpy()]))
    if len(e) == 0:
        return np.zeros(len(trajs), dtype=bool)
    p, q = probe.reshape(-1, 2), target.reshape(-1, 2)
    hit = torch.cat([segments_cross(pc, qc, e) for pc, qc in zip(p.split(4096), q.split(4096))])
    return hit.view(probe.shape[:2]).any(-1).numpy()
