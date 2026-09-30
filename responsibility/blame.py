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
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch

from responsibility.interaction import InteractionConfig, interaction_scores
from responsibility.metrics import FIRST_STEP, ResponsibilityConfig, effective_horizon, safety_responsibility
from responsibility.scene import Scene

LOOKBACK = 20  # steps: the window ending with the collision (the metric horizon)
MARGIN = 0.1  # m: smaller differences do not tell the sides apart


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
    _, _, trajs = model.sample(dist, cfg.n_safety_samples, generator=generator)
    fut = slice(step + 1, step + 1 + horizon)
    return safety_responsibility(trajs[:, :horizon, :2], scene.position[a, fut, :2], scene.valid[a, fut],
                                 scene.position[b, fut, :2], scene.valid[b, fut], cfg)


def split(beta_ego: float, beta_other: Optional[float], margin: float = MARGIN):
    """(share, verdict) from the two responsibilities (module docstring)."""
    pe = max(beta_ego, 0.0)
    if beta_other is None:
        return None, "ego" if pe > margin else "ego-only"
    po = max(beta_other, 0.0)
    share = pe / (pe + po) if pe + po > 0 else 0.5
    verdict = "ego" if pe - po > margin else "other" if po - pe > margin else "shared"
    return share, verdict


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
                 float(beta_ego), None if beta_other is None else float(beta_other), share, verdict)
