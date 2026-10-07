"""CAT's adversary selection, batched in torch over worlds (gpucat/PLAN.md, M1).

CAT (advgen/adv_generator.py, AdvGenerator.generate) runs DenseTNT once per
adversarial episode on inputs taken from the scene's log only (the first 11
steps, parsed and cached per scene), so every call for a scene yields the
same 32 adversary candidates; the closed loop enters only through the ego's
last AV_traj_num trajectories in that scene, against which the candidates are
scored. Here the candidates (and what the fair rule needs) are computed once
per scene (scripts/gpucat/precompute_candidates.py) and the scoring and choice
run per episode for all worlds at once on the GPU:

  cat_scores        score [B, K] and min_dist [B, K], exactly as
                    responsibility.adversarial.cat_collision_scores (CAT's
                    loop): boxes every 5th step with CAT's smoothed path yaw,
                    centre distance against the sum of half-diagonals, strict
                    edge-segment intersection, 0.99 ** (t + 1) for a first
                    collision at subsampled step t, min_dist over the steps up
                    to the first collision, truncated to whole metres (CAT
                    keeps it in an integer array)
  adversary_beta    beta [B, K], the adversary's safety responsibility toward
                    each ego trajectory if it drove candidate j, weighted by
                    P(AV_i) (responsibility.adversarial.adversary_responsibility)
  select            CAT's rule and the fair rule (responsibility.adversarial.select)
  plan              the 91-row plan (x, y, vx, vy, yaw): the logged history up
                    to step 10, then the chosen future, with CAT's yaw and
                    velocity (get_polyline_yaw, get_polyline_vel)
"""

import math
from typing import Dict, Optional, Tuple

import torch

SUB = 5  # CAT compares boxes every 5th step (0.5 s)
DECAY = 0.99  # CAT's per-step decay of later collisions
TWO_PI = 2 * math.pi


def polyline_yaw(traj: torch.Tensor) -> torch.Tensor:
    """advgen's get_polyline_yaw on [..., T, 2]: the direction to the next
    point (the last repeats the one before), unwrapped CAT's way (each value
    shifted by 2 pi once when it jumps by more than 1.5 pi from the previous,
    already shifted one), then a 5-point moving average with edge padding."""
    diff = torch.roll(traj, -1, dims=-2) - traj
    yaw = torch.atan2(diff[..., 1], diff[..., 0])
    yaw = torch.cat([yaw[..., :-1], yaw[..., -2:-1]], dim=-1)
    out = [yaw[..., 0]]
    for i in range(1, yaw.shape[-1]):
        y = yaw[..., i]
        y = torch.where(y - out[-1] > 1.5 * math.pi, y - TWO_PI, torch.where(out[-1] - y > 1.5 * math.pi, y + TWO_PI, y))
        out.append(y)
    yaw = torch.stack(out, dim=-1)
    padded = torch.cat([yaw[..., :1].expand(*yaw.shape[:-1], 2), yaw, yaw[..., -1:].expand(*yaw.shape[:-1], 2)], dim=-1)
    return padded.unfold(-1, 5, 1).mean(-1)


def polyline_vel(traj: torch.Tensor) -> torch.Tensor:
    """advgen's get_polyline_vel: (next - this) / 0.1, the last point 0."""
    nxt = torch.cat([traj[..., 1:, :], traj[..., -1:, :]], dim=-2)
    return (nxt - traj) / 0.1


def _corners(traj: torch.Tensor, yaw: torch.Tensor, length: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    """CAT's box corners A, B, C, D [..., T, 4, 2] (front-right, front-left,
    rear-left, rear-right in CAT's order) for centres [..., T, 2]; length and
    width broadcast against yaw [..., T]."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    l, w = 0.5 * length, 0.5 * width
    x, y = traj[..., 0], traj[..., 1]
    return torch.stack([
        torch.stack([x + l * c + w * s, y + l * s - w * c], -1),
        torch.stack([x + l * c - w * s, y + l * s + w * c], -1),
        torch.stack([x - l * c - w * s, y - l * s + w * c], -1),
        torch.stack([x - l * c + w * s, y - l * s - w * c], -1)], dim=-2)


def _edges_cross(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Whether any edge of quad p crosses any edge of quad q ([..., 4, 2]
    each), with advgen's Intersect (strictly opposite sides both ways)."""
    p0, p1 = p, torch.roll(p, -1, dims=-2)  # edges AB, BC, CD, DA
    q0, q1 = q, torch.roll(q, -1, dims=-2)
    a0, a1 = p0[..., :, None, :], p1[..., :, None, :]  # [..., 4, 1, 2]
    b0, b1 = q0[..., None, :, :], q1[..., None, :, :]  # [..., 1, 4, 2]

    def side(o, d, pt):  # Intersect's cross products: (o - d) x (o - pt)
        v0, v = o - d, o - pt
        return v0[..., 0] * v[..., 1] - v0[..., 1] * v[..., 0]

    a = side(a0, a1, b0) * side(a0, a1, b1)
    c = side(b0, b1, a0) * side(b0, b1, a1)
    return ((a < 0) & (c < 0)).flatten(-2).any(-1)


def subsampled_yaw(trajs: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """CAT's box yaw every SUB-th step [..., ceil(T / SUB)]: polyline_yaw over
    each trajectory's own length (lengths [...], default the full T; CAT
    smooths over the trajectory it has), zero past it. Depends on the
    trajectory alone, so it is computed once: for candidates when they are
    loaded, for an ego trajectory when it is stored."""
    T = trajs.shape[-2]
    if lengths is None:
        return polyline_yaw(trajs)[..., ::SUB]
    yaw = torch.zeros(trajs.shape[:-1], device=trajs.device, dtype=trajs.dtype)
    for n in torch.unique(lengths).tolist():
        if n < 2:
            continue
        # integer indices: with a 2-D boolean mask, x[mask, :n] does not slice the time axis
        idx = torch.nonzero(lengths == n, as_tuple=True)
        yaw[idx + (slice(0, n),)] = polyline_yaw(trajs[idx + (slice(0, n),)])
    return yaw[..., ::SUB][..., : (T + SUB - 1) // SUB]


def cat_scores(cand: torch.Tensor, probs_ov: torch.Tensor, trajs_av: torch.Tensor, av_len: torch.Tensor,
               probs_av: torch.Tensor, ov_size: torch.Tensor, av_size: torch.Tensor,
               yaw_ov: Optional[torch.Tensor] = None, yaw_av: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """CAT's ``res`` and ``min_dist`` for a batch of B scenes.

    cand [B, K, 80, 2] candidate futures, probs_ov [B, K] (cat_candidate_probs),
    trajs_av [B, M, 80, 2] the ego trajectories (steps 11.. of earlier episodes,
    padded), av_len [B, M] their lengths in steps (0: absent), probs_av [B, M],
    ov_size / av_size [B, 2] (length, width); yaw_ov [B, K, S] and yaw_av [B, M, S]
    from subsampled_yaw if cached (computed here otherwise). Returns score
    [B, K] and min_dist [B, K] (whole metres, as float)."""
    if yaw_ov is None:
        yaw_ov = subsampled_yaw(cand)  # [B, K, S]
    if yaw_av is None:
        yaw_av = subsampled_yaw(trajs_av, av_len)  # [B, M, S]
    box_ov = _corners(cand[..., ::SUB, :], yaw_ov, ov_size[:, None, None, 0], ov_size[:, None, None, 1])
    box_av = _corners(trajs_av[..., ::SUB, :], yaw_av, av_size[:, None, None, 0], av_size[:, None, None, 1])
    S = box_ov.shape[2]
    steps_av = (av_len + SUB - 1) // SUB  # the subsampled steps zip() pairs
    t = torch.arange(S, device=cand.device)
    active = t[None, None, :] < steps_av[:, :, None]  # [B, M, S]
    centre_ov, centre_av = cand[..., ::SUB, :], trajs_av[..., ::SUB, :]
    dist = (centre_av[:, None] - centre_ov[:, :, None]).norm(dim=-1)  # [B, K, M, S]
    reach = (0.5 * av_size).norm(dim=-1) + (0.5 * ov_size).norm(dim=-1)  # [B]
    near = dist < reach[:, None, None, None]
    cross = _edges_cross(box_av[:, None], box_ov[:, :, None])  # [B, K, M, S]
    hit = near & cross & active[:, None]
    first = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(hit[..., 0], S, dtype=torch.long))
    p3 = torch.where(first < S, DECAY ** (first.to(cand.dtype) + 1), torch.zeros_like(first, dtype=cand.dtype))
    present = (av_len > 0)[:, None, :]
    score = (probs_ov[:, :, None] * probs_av[:, None, :] * p3 * present).sum(-1)  # [B, K]
    upto = (t[None, None, None, :] <= first[..., None]) & active[:, None] & present[..., None]  # steps CAT visits
    min_dist = torch.where(upto, dist, torch.full_like(dist, 1e6)).flatten(-2).min(-1).values
    return score, torch.floor(min_dist)


def candidate_probs(log_scores: torch.Tensor) -> torch.Tensor:
    """cat_candidate_probs: the scores from the 7th on set to the 7th, softmax."""
    s = log_scores.clone()
    s[..., 6:] = s[..., 6:7]
    return torch.softmax(s, dim=-1)


def _cvar(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """risk.cvar on the last dim: the mean of the values >= the alpha-quantile."""
    var = torch.quantile(x, alpha, dim=-1, keepdim=True)
    tail = x >= var
    tail = tail | (~tail.any(-1, keepdim=True) & (x == x.max(-1, keepdim=True).values))
    return (x * tail).sum(-1) / tail.sum(-1).clamp(min=1)


def adversary_beta(samples: torch.Tensor, cand: torch.Tensor, trajs_av: torch.Tensor, av_len: torch.Tensor,
                   probs_av: torch.Tensor, horizon: int = 80, d_sat: Optional[float] = 10.0,
                   alpha: float = 0.1) -> torch.Tensor:
    """beta [B, K]: per ego trajectory i, CVaR over the adversary's motion set
    samples [B, N, 80, 2] of D(sample, ego_i) - D(candidate j, ego_i), D the
    closest approach over the steps ego_i covers (saturated at d_sat), then
    weighted by P(AV_i) normalised over the trajectories present."""
    T = min(horizon, cand.shape[2])
    t = torch.arange(T, device=cand.device)
    valid = t[None, None, :] < av_len[:, :, None].clamp(max=T)  # [B, M, T]

    def closest(x):  # x [B, X, T, 2] -> [B, M, X]
        d = (x[:, None, :, :T] - trajs_av[:, :, None, :T]).norm(dim=-1)  # [B, M, X, T]
        d = torch.where(valid[:, :, None], d, torch.full_like(d, float("inf"))).min(-1).values
        d = torch.nan_to_num(d, posinf=0.0)
        return d.clamp(max=d_sat) if d_sat is not None else d

    c = closest(samples)[:, :, None, :] - closest(cand)[..., None]  # [B, M, K, N]
    beta = _cvar(c, alpha)  # [B, M, K]
    present = (av_len > 0).to(cand.dtype)
    w = probs_av * present
    w = torch.where(w.sum(-1, keepdim=True) > 0, w / w.sum(-1, keepdim=True).clamp(min=1e-12),
                    present / present.sum(-1, keepdim=True).clamp(min=1))
    return (w[:, :, None] * beta).sum(1)


def select(rule: str, score: torch.Tensor, min_dist: torch.Tensor, beta: Optional[torch.Tensor] = None,
           avoid: Optional[torch.Tensor] = None, threshold: float = 1.0, min_avoid: float = 0.3) -> torch.Tensor:
    """responsibility.adversarial.select for "cat" and "fair", per row: the
    chosen candidate [B]. Ties go to the first index, as numpy's argmax/argmin."""
    def first_max(x, mask):
        x = torch.where(mask, x, torch.full_like(x, -float("inf")))
        return x.argmax(-1)

    def first_min(x, mask):
        x = torch.where(mask, x, torch.full_like(x, float("inf")))
        return x.argmin(-1)

    every = torch.ones_like(score, dtype=torch.bool)
    hit = score > 0
    cat = torch.where(hit.any(-1), first_max(score, every), first_min(min_dist, every))
    if rule == "cat":
        return cat
    if rule != "fair":
        raise ValueError("rule must be cat or fair")
    within = beta <= threshold
    ok = within & (avoid >= min_avoid)
    in_ok = torch.where((hit & ok).any(-1), first_max(score, ok), first_min(min_dist, ok))
    pool = torch.where(within.any(-1, keepdim=True), within, every)
    best_avoid = torch.where(pool, avoid, torch.full_like(avoid, -float("inf"))).max(-1, keepdim=True).values
    best = pool & (avoid == best_avoid)
    fallback = first_min(min_dist, best)
    return torch.where(ok.any(-1), in_ok, fallback)


def plan(adv_past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """[B, 91, 5] (x, y, vx, vy, yaw) as AdvGenerator.generate builds adv_traj."""
    pos = torch.cat([adv_past, future], dim=-2)
    return torch.cat([pos, polyline_vel(pos), polyline_yaw(pos)[..., None]], dim=-1)


class CandidateBank:
    """The per-scene precomputation (precompute_candidates.py's .npz) on a
    device: candidates [S, K, 80, 2], probs_ov [S, K], adv_past [S, 11, 2],
    ov_size / av_size [S, 2], ego_route [S, 80, 2], and for the fair rule the
    adversary's motion-set samples [S, N, 80, 2] and the ego avoidability of
    each candidate [S, K]. ``index`` maps scene stem -> row."""

    def __init__(self, path: str, device: str = "cuda", dtype=torch.float32):
        import numpy as np

        z = np.load(path, allow_pickle=False)
        self.stems = [str(s) for s in z["stems"]]
        self.index = {s: i for i, s in enumerate(self.stems)}
        t = lambda k: torch.as_tensor(z[k], dtype=dtype, device=device)  # noqa: E731
        self.cand, self.log_scores = t("candidates"), t("log_scores")
        self.probs_ov = candidate_probs(self.log_scores.double()).to(dtype)
        self.adv_past, self.ego_route = t("adv_past"), t("ego_route")
        self.ov_size, self.av_size = t("ov_size"), t("av_size")
        self.samples = t("samples") if "samples" in z.files else None
        self.avoid = t("avoid") if "avoid" in z.files else None
        self.adv_id = [str(x) for x in z["adv_id"]]
        self.yaw_ov = subsampled_yaw(self.cand.double()).to(dtype)  # [S, K, S_sub], CAT's box yaw
