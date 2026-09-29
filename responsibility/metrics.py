"""Safety and courtesy responsibility (Hsu et al., IROS 2023, Eq. 2-5) with a
DenseTNT motion model.

**Safety responsibility** of agent a at context step k:

    C(a, b)       = D_g(xi, xi_b) - D_g(xi_a, xi_b)                (Eq. 2)
    beta_s(a, b)  = CVaR_alpha[ C(a, b) ],  xi ~ motion set of a    (Eq. 3)
    beta_s(a)     = max_b beta_s(a, b)                              (Eq. 5)

D_g is the closest approach over the metric horizon (saturated at d_sat),
xi_a / xi_b the logged futures, and the motion set N trajectories DenseTNT
samples for a from its goal distribution at k. beta_s > 0: most of what a
could have done would have kept more distance to b than what it actually
did, so a gave up safety margin (drove more aggressively than its
alternatives); beta_s <= 0: a kept at least as much distance as usual.

**Courtesy responsibility** of a toward b (Eq. 4):

    beta_c(a, b) = KL( pi_b(. | x_k) || pi_b(. | x_k without a) )
    beta_c(a)    = max_b beta_c(a, b)

computed exactly over b's goal distribution (DenseTNT's dense goal grid): how
much a's presence changes what b intends to do. It is measured on goals,
i.e. intent; the completion of a trajectory through a given goal does not
enter. DenseTNT predicts vehicles only, so courtesy is measured toward
vehicle neighbours; safety toward every neighbour.

Both are open-loop: DenseTNT conditions on the 1.1 s of history up to k and
never on anyone's future.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from responsibility.geometry import pairwise_min_distance_over_time
from responsibility.interaction import InteractionConfig, interacting_neighbours
from responsibility.risk import cvar
from responsibility.scene import Scene

FIRST_STEP = 10  # DenseTNT needs 10 steps of history before the current one


@dataclass
class ResponsibilityConfig:
    n_safety_samples: int = 40
    cvar_alpha: float = 0.1  # upper-tail confidence level (catk convention; 0.1 ~ close to the mean)
    d_sat: Optional[float] = 10.0  # D_g saturation (m): responsibility is about interaction range
    metric_horizon: int = 20  # 10 Hz steps scored (2 s)
    window_stride: int = 5  # steps between evaluated context steps
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    courtesy: bool = True
    seed: int = 0


@dataclass
class Observation:
    """Responsibility of one agent at one context step."""

    scenario_id: str
    agent: int
    agent_id: str
    step: int
    safety: float
    courtesy: float
    speed: float
    # neighbour track id -> {"safety", "courtesy", "min_gap", "pet", "ttc", "type"}
    per_neighbour: Dict[str, Dict[str, float]]

    def as_row(self) -> Dict:
        worst_s = max(self.per_neighbour.items(), key=lambda kv: kv[1]["safety"], default=(None, None))[0]
        worst_c = max(((k, v) for k, v in self.per_neighbour.items() if v.get("courtesy") is not None),
                      key=lambda kv: kv[1]["courtesy"], default=(None, None))[0]
        return {
            "scenario_id": self.scenario_id, "agent_id": self.agent_id, "step": self.step,
            "time": round(self.step / 10.0, 1), "speed": round(self.speed, 3),
            "safety": round(self.safety, 5), "courtesy": round(self.courtesy, 6),
            "n_neighbours": len(self.per_neighbour), "safety_against": worst_s, "courtesy_toward": worst_c,
        }


def effective_horizon(scene: Scene, step: int, cfg: ResponsibilityConfig) -> int:
    return max(0, min(cfg.metric_horizon, scene.n_steps - 1 - step))


def window_steps(scene: Scene, agent: int, cfg: ResponsibilityConfig) -> List[int]:
    """Context steps at which ``agent`` is observed and a full metric horizon
    of log remains."""
    last = scene.n_steps - 1 - cfg.metric_horizon
    return [k for k in range(FIRST_STEP, last + 1, cfg.window_stride) if scene.valid[agent, k]]


def safety_responsibility(samples: np.ndarray, actual: np.ndarray, actual_valid: np.ndarray,
                          neighbour: np.ndarray, neighbour_valid: np.ndarray, cfg: ResponsibilityConfig) -> float:
    """beta_s(a, b), Eq. 3, over the first H steps: samples [N, H, 2] of a's
    motion, a's logged [H, 2] and b's logged [H, 2] futures with validity."""
    f = torch.float32
    s = torch.as_tensor(samples, dtype=f)
    a = torch.as_tensor(actual, dtype=f)
    b = torch.as_tensor(neighbour, dtype=f)
    vb = torch.as_tensor(neighbour_valid) & torch.as_tensor(actual_valid)  # compare where both are logged
    d_samples = pairwise_min_distance_over_time(s, b, valid_b=vb, d_sat=cfg.d_sat)
    d_actual = pairwise_min_distance_over_time(a.unsqueeze(0), b, valid_b=vb, d_sat=cfg.d_sat).squeeze(0)
    return float(cvar(d_samples - d_actual, cfg.cvar_alpha))


def goal_kl(log_p: torch.Tensor, log_q: torch.Tensor) -> float:
    """KL(p || q) over a shared goal grid, in nats (clamped at 0 against rounding)."""
    return max(0.0, float((log_p.exp() * (log_p - log_q)).sum()))


def responsibility_at(model, scene: Scene, agent: int, step: int, cfg: ResponsibilityConfig,
                      generator: Optional[torch.Generator] = None) -> Optional[Observation]:
    """Safety and courtesy responsibility of ``agent`` at context step
    ``step``, or None when DenseTNT has no prediction for it there.

    ``model`` provides ``distribution(scene, step, agent)``,
    ``sample(dist, n, generator) -> (goal idx, log prob, trajectories [n, 80, 2])``
    and ``with_and_without(scene, step, b, a) -> (dist, dist) | None``
    (responsibility.densetnt.DenseTNT)."""
    horizon = effective_horizon(scene, step, cfg)
    if horizon == 0:
        return None
    dist = model.distribution(scene, step, agent)
    if dist is None:
        return None
    neighbours = interacting_neighbours(scene, agent, step, horizon, cfg.interaction)
    fut = slice(step + 1, step + 1 + horizon)
    speed = float(np.linalg.norm(scene.velocity[agent, step]))
    per_neighbour: Dict[str, Dict[str, float]] = {}
    if neighbours:
        _, _, trajs = model.sample(dist, cfg.n_safety_samples, generator=generator)
        samples = trajs[:, :horizon, :2]
        actual, actual_valid = scene.position[agent, fut, :2], scene.valid[agent, fut]
        for b, evidence in neighbours.items():
            entry = dict(evidence, type=scene.types[b])
            entry["safety"] = safety_responsibility(
                samples, actual, actual_valid, scene.position[b, fut, :2], scene.valid[b, fut], cfg)
            entry["courtesy"] = None
            if cfg.courtesy:
                pair = model.with_and_without(scene, step, b, agent)
                if pair is not None:
                    entry["courtesy"] = goal_kl(pair[0].log_prob, pair[1].log_prob)
            per_neighbour[scene.track_ids[b]] = entry
    safety = max((v["safety"] for v in per_neighbour.values()), default=0.0)
    courtesy = max((v["courtesy"] for v in per_neighbour.values() if v["courtesy"] is not None), default=0.0)
    return Observation(scene.scenario_id, agent, scene.track_ids[agent], step, safety, courtesy, speed, per_neighbour)


def scene_responsibility(model, scene: Scene, agent: Optional[int] = None,
                         cfg: Optional[ResponsibilityConfig] = None) -> List[Observation]:
    """Responsibility of ``agent`` (default: the self-driving car) at every
    evaluated context step of the scene."""
    cfg = cfg or ResponsibilityConfig()
    agent = scene.sdc if agent is None else agent
    generator = torch.Generator().manual_seed(cfg.seed)
    out = []
    for step in window_steps(scene, agent, cfg):
        obs = responsibility_at(model, scene, agent, step, cfg, generator)
        if obs is not None:
            out.append(obs)
    return out


def config_dict(cfg: ResponsibilityConfig) -> Dict:
    return asdict(cfg)
