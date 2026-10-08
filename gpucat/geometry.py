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
