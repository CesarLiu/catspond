"""Batched geometry for the GPUDrive environment's own checks (gpucat/env.py):
progress and distance along a route, whether a vehicle's footprint touches
a line, and whether two footprints overlap -- with full-size boxes, where
GPUDrive's own collision boxes are scaled by 0.7 (gpucat/PLAN.md).

All functions take a leading batch dimension W (one world each).
"""

import torch


def route_coords(p: torch.Tensor, route: torch.Tensor, route_n: torch.Tensor):
    """(lateral [W], s [W]): the distance from points p [W, 2] to the route
    polylines route [W, R, 2] (route_n [W] valid points) and the arc length
    at the closest point -- MetaDrive's local_coordinates on the reference
    trajectory (lateral as a distance, unsigned)."""
    a, b = route[:, :-1], route[:, 1:]  # [W, R-1, 2]
    seg = b - a
    seg_len = seg.norm(dim=-1)
    t = ((p[:, None] - a) * seg).sum(-1) / (seg_len ** 2).clamp(min=1e-9)
    t = t.clamp(0.0, 1.0)
    closest = a + t[..., None] * seg
    d = (p[:, None] - closest).norm(dim=-1)  # [W, R-1]
    valid = torch.arange(seg.shape[1], device=p.device)[None] < (route_n[:, None] - 1)
    d = torch.where(valid, d, torch.full_like(d, float("inf")))
    lateral, k = d.min(-1)
    cum = torch.cat([torch.zeros_like(seg_len[:, :1]), torch.cumsum(seg_len * valid, -1)], -1)  # [W, R]
    w = torch.arange(p.shape[0], device=p.device)
    s = cum[w, k] + t[w, k] * seg_len[w, k]
    return lateral, s


def points_along(route: torch.Tensor, route_n: torch.Tensor, s: torch.Tensor, ahead: torch.Tensor) -> torch.Tensor:
    """[W, K, 2]: the points of the route polylines route [W, R, 2] (route_n
    [W] valid points) at arc lengths s [W] + ahead [K] (m), clamped to the
    route's ends -- the navigation checkpoints the SDC is told about."""
    seg = route[:, 1:] - route[:, :-1]
    valid = torch.arange(seg.shape[1], device=route.device)[None] < (route_n[:, None] - 1)
    seg_len = seg.norm(dim=-1) * valid
    cum = torch.cat([torch.zeros_like(seg_len[:, :1]), torch.cumsum(seg_len, -1)], -1)  # [W, R], flat past the end
    total = cum[:, -1:]
    q = torch.minimum((s[:, None] + ahead[None]).clamp(min=0.0), total)
    k = torch.searchsorted(cum[:, 1:].contiguous(), q.contiguous())
    k = torch.minimum(k, (route_n[:, None] - 2).clamp(min=0))
    k = k.clamp(max=seg.shape[1] - 1)
    a = route.gather(1, k[..., None].expand(-1, -1, 2))
    d = seg.gather(1, k[..., None].expand(-1, -1, 2))
    u = (q - cum.gather(1, k)) / seg_len.gather(1, k).clamp(min=1e-9)
    return a + u.clamp(0.0, 1.0)[..., None] * d


def ray_distances(p: torch.Tensor, yaw: torch.Tensor, segs: torch.Tensor, n: torch.Tensor,
                  angles: torch.Tensor, max_range: float) -> torch.Tensor:
    """[W, R]: the distance (m) from p [W, 2] along each ray yaw [W] +
    angles [R] to the nearest of the segments segs [W, N, 4] (the first n [W]
    valid), max_range when none is closer -- MetaDrive's side detector, on
    the lines that end an episode."""
    th = yaw[:, None] + angles[None]
    dx, dy = torch.cos(th)[..., None], torch.sin(th)[..., None]  # [W, R, 1]
    ax, ay = (segs[..., 0] - p[:, None, 0])[:, None], (segs[..., 1] - p[:, None, 1])[:, None]  # [W, 1, N]
    ex, ey = (segs[..., 2] - segs[..., 0])[:, None], (segs[..., 3] - segs[..., 1])[:, None]
    denom = dx * ey - dy * ex  # d x e
    ok = denom.abs() > 1e-9
    denom = torch.where(ok, denom, torch.ones_like(denom))
    r = (ax * ey - ay * ex) / denom  # (a x e) / (d x e): distance along the ray
    u = (ax * dy - ay * dx) / denom  # (a x d) / (d x e): position on the segment
    valid = (torch.arange(segs.shape[1], device=p.device)[None] < n[:, None])[:, None]
    hit = ok & valid & (r >= 0.0) & (u >= 0.0) & (u <= 1.0)
    r = torch.where(hit, r, torch.full_like(r, float("inf")))
    return r.min(-1).values.clamp(max=max_range)


def box_corners(c: torch.Tensor, yaw: torch.Tensor, length: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    """[..., 4, 2] corners (counter-clockwise) of boxes centred at c [..., 2]."""
    d = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1) * (0.5 * length)[..., None]
    n = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1) * (0.5 * width)[..., None]
    return torch.stack([c + d + n, c - d + n, c - d - n, c + d - n], dim=-2)


def segments_touch_box(c: torch.Tensor, yaw: torch.Tensor, length: torch.Tensor, width: torch.Tensor,
                       segs: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Whether any of the segments segs [W, N, 4] (the first n [W] valid)
    meets the box (centre c [W, 2], yaw [W], length / width [W]): each
    segment is moved into the box's frame and clipped against it
    (Liang-Barsky)."""
    cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]

    def local(x, y):
        dx, dy = x - c[:, None, 0], y - c[:, None, 1]
        return dx * cos + dy * sin, -dx * sin + dy * cos

    x0, y0 = local(segs[..., 0], segs[..., 1])
    x1, y1 = local(segs[..., 2], segs[..., 3])
    hx, hy = (0.5 * length)[:, None], (0.5 * width)[:, None]
    dx, dy = x1 - x0, y1 - y0
    lo, hi = torch.zeros_like(x0), torch.ones_like(x0)
    ok = torch.ones_like(x0, dtype=torch.bool)
    for p, q in ((-dx, x0 + hx), (dx, hx - x0), (-dy, y0 + hy), (dy, hy - y0)):
        parallel = p.abs() < 1e-12
        ok &= ~(parallel & (q < 0))
        r = q / torch.where(parallel, torch.ones_like(p), p)
        lo = torch.where(~parallel & (p < 0), torch.maximum(lo, r), lo)
        hi = torch.where(~parallel & (p > 0), torch.minimum(hi, r), hi)
    hit = ok & (lo <= hi)
    valid = torch.arange(segs.shape[1], device=c.device)[None] < n[:, None]
    return (hit & valid).any(-1)


def boxes_overlap(c: torch.Tensor, yaw: torch.Tensor, size: torch.Tensor,
                  oc: torch.Tensor, oyaw: torch.Tensor, osize: torch.Tensor) -> torch.Tensor:
    """[W, A]: whether the box (c [W, 2], yaw [W], size [W, 2] length/width)
    overlaps each other box (oc [W, A, 2], oyaw [W, A], osize [W, A, 2]),
    by the separating axis theorem."""
    a = box_corners(c, yaw, size[:, 0], size[:, 1])[:, None]  # [W, 1, 4, 2]
    b = box_corners(oc, oyaw, osize[..., 0], osize[..., 1])  # [W, A, 4, 2]
    axes = torch.stack([torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)[:, None].expand_as(oc),
                        torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1)[:, None].expand_as(oc),
                        torch.stack([torch.cos(oyaw), torch.sin(oyaw)], -1),
                        torch.stack([-torch.sin(oyaw), torch.cos(oyaw)], -1)], dim=-2)  # [W, A, 4, 2]
    pa = (a[:, :, None] * axes[:, :, :, None]).sum(-1)  # [W, A, 4 axes, 4 corners]
    pb = (b[:, :, None] * axes[:, :, :, None]).sum(-1)
    separated = (pa.max(-1).values < pb.min(-1).values) | (pb.max(-1).values < pa.min(-1).values)
    return ~separated.any(-1)
