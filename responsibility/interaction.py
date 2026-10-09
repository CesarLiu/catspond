"""Which agents actually interact with the queried agent (the neighbour set
of Eq. 5), from the log alone -- ported from catk's geometric selection.

A plain radius both misses real conflicts (a fast car 40 m away that crosses
the ego's path a second later) and admits non-interactions (a car ahead in
the same lane that never closes in). Candidates within ``prefilter_radius``
are kept if any of these holds over the window [k, k + horizon]:

  min_gap  closest approach of the two footprints (3-circle cover) <= gap_threshold
  pet      post-encroachment time: smallest time difference with which the two
           occupy the same place (within ``pet_tolerance``), history included
           but at least one occupation in the window <= pet_threshold
  ttc      first contact of the footprints under constant velocity and
           heading from step k <= ttc_threshold

ooi_neighbours is the simplified alternative (``--use-ooi``): the other
objects of interest of the scenario, whatever the evidence, so that one
scene is measured for one pair (in CAT's scenes: the self-driving car and
the adversary) in every window.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from responsibility.geometry import circle_centers, circle_decomposition
from responsibility.scene import HISTORY_STEPS, Scene

DT = 0.1


@dataclass
class InteractionConfig:
    prefilter_radius: float = 50.0
    gap_threshold: float = 10.0
    pet_threshold: float = 2.0
    pet_tolerance: float = 2.0
    ttc_threshold: float = 4.0
    ttc_horizon: float = 6.0
    max_neighbors: Optional[int] = 8


def _footprint_gap(pos_a, head_a, shape_a, pos_b, head_b, shape_b, n_circles=3):
    """[Ta] x [M, Ta] -> [M, Ta] gap between circle covers (negative = overlap)."""
    off_a, r_a = circle_decomposition(shape_a[:1, 0], shape_a[:1, 1], n_circles)
    off_b, r_b = circle_decomposition(shape_b[:, 0], shape_b[:, 1], n_circles)
    ca = circle_centers(pos_a.unsqueeze(0), head_a.unsqueeze(0), off_a)  # [1, T, C, 2]
    cb = circle_centers(pos_b, head_b, off_b)  # [M, T, C, 2]
    centre = torch.linalg.norm(ca[:, :, :, None, :] - cb[:, :, None, :, :], dim=-1)  # [M, T, C, C]
    return centre.amin(dim=(-2, -1)) - (r_a[0] + r_b)[:, None]


def interaction_scores(scene: Scene, query: int, candidates: Sequence[int], step: int, horizon: int,
                       cfg: InteractionConfig) -> Dict[str, np.ndarray]:
    """min_gap [M] (m), pet [M] (s), ttc [M] (s); inf where never applicable."""
    m = len(candidates)
    inf = np.full(m, np.inf)
    if m == 0:
        return {"min_gap": inf, "pet": inf, "ttc": inf}
    cand = list(candidates)
    last = min(step + horizon, scene.n_steps - 1)
    t = slice(step, last + 1)
    f = torch.float32
    pos = torch.as_tensor(scene.position[..., :2], dtype=f)
    head = torch.as_tensor(scene.heading, dtype=f)
    valid = torch.as_tensor(scene.valid)
    shape = torch.as_tensor(np.stack([scene.shape_at(i, step) for i in [query] + cand]), dtype=f)

    gap = _footprint_gap(pos[query, t], head[query, t], shape[:1], pos[cand, t], head[cand, t], shape[1:])
    both = valid[query, t].unsqueeze(0) & valid[cand, t]
    min_gap = gap.masked_fill(~both, float("inf")).amin(-1).clamp(min=0.0).numpy()

    first = max(0, step - (HISTORY_STEPS - 1))
    h = slice(first, last + 1)
    qa, ca = pos[query, h], pos[cand, h]  # [T, 2], [M, T, 2]
    close = torch.cdist(qa.unsqueeze(0).expand(m, -1, -1), ca) < cfg.pet_tolerance  # [M, T, T]
    close &= valid[query, h][None, :, None] & valid[cand, h][:, None, :]
    idx = torch.arange(qa.shape[0])
    future = (idx[:, None] >= step - first) | (idx[None, :] >= step - first)
    dt = (idx[:, None] - idx[None, :]).abs().to(f) * DT
    pet = dt.expand(m, -1, -1).masked_fill(~(close & future), float("inf")).amin(dim=(-2, -1)).numpy()

    vel = torch.as_tensor(scene.velocity, dtype=f)
    s = torch.arange(int(round(cfg.ttc_horizon / DT)) + 1, dtype=f) * DT
    q_ext = pos[query, step][None] + vel[query, step][None] * s[:, None]  # [S, 2]
    c_ext = pos[cand, step][:, None] + vel[cand, step][:, None] * s[None, :, None]  # [M, S, 2]
    touching = _footprint_gap(q_ext, head[query, step].expand(len(s)), shape[:1],
                              c_ext, head[cand, step][:, None].expand(-1, len(s)), shape[1:]) <= 0
    first_touch = torch.where(touching, s[None], torch.full_like(s[None], float("inf"))).amin(-1)
    now_valid = valid[cand, step] & valid[query, step]
    ttc = first_touch.masked_fill(~now_valid, float("inf")).numpy()
    return {"min_gap": min_gap, "pet": pet, "ttc": ttc}


def interacting_neighbours(scene: Scene, query: int, step: int, horizon: int,
                           cfg: Optional[InteractionConfig] = None) -> Dict[int, Dict[str, float]]:
    """{neighbour: its interaction scores} for the agents that interact with
    ``query`` over [step, step + horizon], closest approach first."""
    cfg = cfg or InteractionConfig()
    if not scene.valid[query, step]:
        return {}
    here = scene.position[:, step, :2]
    near = [i for i in range(scene.n_agents)
            if i != query and scene.valid[i, step]
            and np.linalg.norm(here[i] - here[query]) <= cfg.prefilter_radius]
    scores = interaction_scores(scene, query, near, step, horizon, cfg)
    keep = [j for j in range(len(near))
            if scores["min_gap"][j] <= cfg.gap_threshold
            or scores["pet"][j] <= cfg.pet_threshold
            or scores["ttc"][j] <= cfg.ttc_threshold]
    keep.sort(key=lambda j: scores["min_gap"][j])
    if cfg.max_neighbors is not None:
        keep = keep[: cfg.max_neighbors]
    return {near[j]: {k: float(v[j]) for k, v in scores.items()} for j in keep}


def ooi_neighbours(scene: Scene, query: int, step: int, horizon: int,
                   cfg: Optional[InteractionConfig] = None) -> Dict[int, Dict[str, float]]:
    """{neighbour: its interaction scores} for the scenario's other objects
    of interest present at ``step``, interacting or not."""
    cfg = cfg or InteractionConfig()
    if query not in scene.objects_of_interest:
        raise ValueError(f"agent {scene.track_ids[query]} is not an object of interest of the scenario")
    if not scene.valid[query, step]:
        return {}
    others = [i for i in scene.objects_of_interest if i != query and scene.valid[i, step]]
    if not others:
        return {}
    scores = interaction_scores(scene, query, others, step, horizon, cfg)
    return {b: {k: float(v[j]) for k, v in scores.items()} for j, b in enumerate(others)}


def all_neighbours_within(scene: Scene, query: int, step: int, radius: float) -> List[int]:
    """The paper's plain radius selection, for comparison."""
    here = scene.position[:, step, :2]
    return [i for i in range(scene.n_agents)
            if i != query and scene.valid[i, step] and np.linalg.norm(here[i] - here[query]) <= radius]
