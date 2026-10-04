"""Who caused a collision, by counterfactual safety responsibility.

For a collision of the ego with agent b at step c, both sides are measured
in the window that ends with the collision (context step k = c - lookback,
at least the first step DenseTNT can predict from):

    beta_ego    = beta_s(ego, b): how much closer to b the ego came than its
                  own alternatives at k would have
    beta_other  = beta_s(b, ego): the same for b toward the ego

each from that agent's DenseTNT motion set, against the other's actual
trajectory (metrics.safety_responsibility). A car that was rear-ended did
what its alternatives do (beta_ego ~ 0) while the follower closed in far more
than its alternatives (beta_other > 0); a car that cut in is the reverse.

    share    w = beta_ego+ / (beta_ego+ + beta_other+), 0.5 when neither is
             positive (nothing tells them apart)
    verdict  "ego" / "other" when one side's positive part exceeds the other's
             by more than ``margin`` metres, "shared" otherwise, "ego-only"
             when b is not predicted (DenseTNT predicts vehicles only) --
             then only beta_ego is known

The rule baseline (``rear_end_rule``) is the traffic-law reading of the
most common collision: in a rear-end collision the follower is at fault.
It applies when both travel the same way (headings within 30 degrees) and
they meet end to end rather than side by side, and says nothing otherwise.
The counterfactual verdict is compared against it where both give one, and
against RSS (responsibility/rss.py), the formal rule-based baseline.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from responsibility.interaction import InteractionConfig, interaction_scores
from responsibility.metrics import FIRST_STEP, ResponsibilityConfig, effective_horizon, motion_set, safety_responsibility
from responsibility.scene import Scene

LOOKBACK = 20  # steps: the window ending with the collision (the metric horizon)
MARGIN = 0.1  # m: smaller differences do not tell the sides apart
REAR_END_HEADING = np.deg2rad(30.0)  # rad: "travelling the same way"


@dataclass
class Blame:
    crash_step: int
    window: int
    other: int
    other_id: str
    other_type: str
    beta_ego: float
    beta_other: Optional[float]
    share: Optional[float]
    verdict: str
    rule: str = "n/a"  # rear_end_rule's verdict

    def as_row(self) -> Dict:
        return asdict(self)


def crash_partner(scene: Scene, ego: int, step: int) -> Optional[int]:
    """The agent closest to the ego (footprint gap) at ``step``."""
    others = [i for i in range(scene.n_agents) if i != ego and scene.valid[i, step]]
    if not others or not scene.valid[ego, step]:
        return None
    gap = interaction_scores(scene, ego, others, step, 0, InteractionConfig())["min_gap"]
    return others[int(np.argmin(gap))]


def pair_safety(model, scene: Scene, a: int, b: int, step: int, cfg: ResponsibilityConfig,
                generator: Optional[torch.Generator] = None) -> Optional[float]:
    """beta_s(a, b) at context step ``step``, or None when DenseTNT does not
    predict ``a`` there."""
    horizon = effective_horizon(scene, step, cfg)
    if horizon == 0:
        return None
    dist = model.distribution(scene, step, a)
    if dist is None:
        return None
    trajs, _, weights = motion_set(model, dist, cfg, generator)
    fut = slice(step + 1, step + 1 + horizon)
    return safety_responsibility(trajs[:, :horizon, :2], scene.position[a, fut, :2], scene.valid[a, fut],
                                 scene.position[b, fut, :2], scene.valid[b, fut], cfg, weights)


def split(beta_ego: float, beta_other: Optional[float], margin: float = MARGIN):
    """(share, verdict) from the two responsibilities (module docstring)."""
    pe = max(beta_ego, 0.0)
    if beta_other is None:
        return None, "ego" if pe > margin else "ego-only"
    po = max(beta_other, 0.0)
    share = pe / (pe + po) if pe + po > 0 else 0.5
    verdict = "ego" if pe - po > margin else "other" if po - pe > margin else "shared"
    return share, verdict


def rear_end_rule(scene: Scene, ego: int, other: int, crash_step: int) -> str:
    """The follower is at fault in a rear-end collision: "ego" when the ego
    ran into the other from behind, "other" when the other ran into the
    ego, "n/a" when the collision is not rear-end (different directions,
    side by side) or the two are never seen together. The geometry is read
    at the last step before the collision with both present (at the
    collision itself they already overlap)."""
    steps = [k for k in range(crash_step, -1, -1) if scene.valid[ego, k] and scene.valid[other, k]]
    if not steps:
        return "n/a"
    k = next((k for k in steps if k < crash_step), steps[0])
    turn = np.angle(np.exp(1j * (float(scene.heading[other, k]) - float(scene.heading[ego, k]))))
    if abs(turn) > REAR_END_HEADING:
        return "n/a"
    heading = float(scene.heading[ego, k]) + turn / 2  # the common direction of travel
    d = scene.position[other, k, :2] - scene.position[ego, k, :2]
    lon = float(d[0] * np.cos(heading) + d[1] * np.sin(heading))
    lat = float(-d[0] * np.sin(heading) + d[1] * np.cos(heading))
    (l_e, w_e), (l_o, w_o) = scene.shape_at(ego, k), scene.shape_at(other, k)
    # end to end: the other lies off the ego's front or back rather than its side
    # (offsets in half-extents of the two boxes, so it also holds once they overlap)
    if abs(lon) / ((l_e + l_o) / 2) <= abs(lat) / ((w_e + w_o) / 2):
        return "n/a"
    return "ego" if lon > 0 else "other"


def rollout_collision(scene: Scene, rollout: Dict) -> Optional[Tuple[int, int]]:
    """(step, partner) of a rollout's vehicle collision, in the scene rebuilt
    from it (rollouts.scene_from_rollout): the last recorded step and the
    recorded partner (``crash_with``, else the agent closest to the ego);
    None after the end of the log or with nobody there."""
    step = rollout["end"]["step"]
    if step >= scene.n_steps:  # nothing logged left to collide with
        return None
    with_id = rollout["end"].get("crash_with")
    other = scene.index(with_id) if with_id is not None and str(with_id) in scene.track_ids else None
    if other is None:
        other = crash_partner(scene, scene.sdc, step)
    return None if other is None else (step, other)


def rollout_blame(model, scene: Scene, rollout: Dict, cfg: Optional[ResponsibilityConfig] = None,
                  lookback: int = LOOKBACK, margin: float = MARGIN) -> Optional[Blame]:
    """crash_blame for the collision a rollout ended in (rollout_collision);
    None after the end of the log."""
    collision = rollout_collision(scene, rollout)
    if collision is None:
        return None
    step, other = collision
    return crash_blame(model, scene, scene.sdc, step, other, cfg, lookback, margin)


def crash_blame(model, scene: Scene, ego: int, crash_step: int, other: Optional[int] = None,
                cfg: Optional[ResponsibilityConfig] = None, lookback: int = LOOKBACK,
                margin: float = MARGIN) -> Optional[Blame]:
    """The ego's and the other's responsibility for a collision at
    ``crash_step`` (other: the colliding agent, default the closest one), or
    None when there is no window before the collision in which both exist."""
    cfg = cfg or ResponsibilityConfig()
    if other is None:
        other = crash_partner(scene, ego, crash_step)
        if other is None:
            return None
    window = next((k for k in range(max(FIRST_STEP, crash_step - lookback), crash_step)
                   if scene.valid[ego, k] and scene.valid[other, k]), None)
    if window is None:
        return None
    generator = torch.Generator().manual_seed(cfg.seed * 1000003 + crash_step)
    beta_ego = pair_safety(model, scene, ego, other, window, cfg, generator)
    if beta_ego is None:
        return None
    beta_other = pair_safety(model, scene, other, ego, window, cfg, generator)
    share, verdict = split(beta_ego, beta_other, margin)
    return Blame(crash_step, window, other, scene.track_ids[other], scene.types[other],
                 float(beta_ego), None if beta_other is None else float(beta_other), share, verdict,
                 rear_end_rule(scene, ego, other, crash_step))
