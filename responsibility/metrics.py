"""Safety and courtesy responsibility (Hsu et al., IROS 2023, Eq. 2-5) with a
DenseTNT motion model.

**Safety responsibility** of agent a at context step k:

    C(a, b)       = D_g(xi, xi_b) - D_g(xi_a, xi_b)                (Eq. 2)
    beta_s(a, b)  = CVaR_alpha[ C(a, b) ],  xi ~ motion set of a    (Eq. 3)
    beta_s(a)     = max_b beta_s(a, b)                              (Eq. 5)

D_g is the closest approach over the metric horizon (saturated at d_sat),
xi_a / xi_b the logged futures, and the motion set N trajectories the motion
model samples for a at k (DenseTNT: from its goal distribution) -- or, with
``motion_set="weighted"`` and a model with a finite motion set (UniTraj's MTR:
64 intentions), all of them weighted by their probabilities, which makes the
CVaR exact; with ``motion_set="topk"``, the N most probable of them, weighted
by their probabilities renormalised over the N; with ``motion_set="nms"``, N
of them spread over the distribution by CAT's goal NMS, each weighted by the
probability of the goals nearest to it (responsibility/modes.py). beta_s > 0: most of what a
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

The neighbours b are the agents interaction evidence selects
(responsibility.interaction), or with ``use_ooi`` the scenario's other
objects of interest, in every window, so that a scene is measured for one
pair (a itself must be an object of interest).

With ``filter`` (responsibility.motion_filter), beta_s(a, b) is taken over
the valid part of a's motion set only: same route, on the road,
kinematically feasible, through no third agent. The per-neighbour entries
then also hold "kept" and "kept_mass" (what survived) and, with the lane
route, "route_goal_mass" (the goal mass on a's route lanes).

With ``courtesy_valid_goals``, the KL of beta_c(a, b) is taken over b's
valid goals only -- those on the lanes b can reach from where it is
(responsibility.lanes.reachable_lanes) -- both distributions renormalised
there; "courtesy_goal_mass" holds the mass of b's distribution (with a)
on them. With ``courtesy_same_mode``, the KL is taken over the goals of b's
own logged drive mode instead (same_mode_support): "lanes", those on b's
lane route (responsibility.lanes.route_lanes, as --lane-route for a);
"path", those within ``courtesy_path_lateral`` (m) of b's logged path, with
no map needed. beta_c then measures how a changes b's plan within the mode
b drove, not a change of mode.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from responsibility.geometry import pairwise_min_distance_over_time
from responsibility.interaction import InteractionConfig, interacting_neighbours, ooi_neighbours
from responsibility.intent import distance_to_logged_path
from responsibility.lanes import lane_graph, reachable_lanes, route_lanes
from responsibility.motion_filter import MotionFilter, MotionFilterConfig, goals_on_lanes, restrict_to_route
from responsibility.risk import cvar
from responsibility.scene import Scene

FIRST_STEP = 10  # DenseTNT needs 10 steps of history before the current one
MOTION_SETS = ("sampled", "weighted", "topk", "nms")  # ResponsibilityConfig.motion_set
MIN_VALID_GOAL_MASS = 0.5  # below this share on b's reachable lanes, courtesy is not restricted to them
MIN_MODE_MASS = 1e-3  # below this share in b's own drive mode there is nothing to renormalise: not restricted


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
    # "weighted": the whole motion set; "topk": its n_safety_samples most probable; "nms": n_safety_samples
    # spread by goal NMS (modes.py)
    motion_set: str = "sampled"
    filter: MotionFilterConfig = field(default_factory=MotionFilterConfig)  # valid counterfactuals for beta_s
    courtesy_valid_goals: bool = False  # beta_c over the goals on lanes b can reach
    courtesy_same_mode: Optional[str] = None  # beta_c over b's own drive mode: "lanes" (HD) or "path" (map-free)
    courtesy_path_lateral: float = 6.0  # m from b's logged path, for courtesy_same_mode="path"
    use_ooi: bool = False  # neighbours: the other objects of interest, always (one pair per scene)


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


def window_steps(scene: Scene, agent: int, cfg: ResponsibilityConfig, last_step: Optional[int] = None) -> List[int]:
    """Context steps at which ``agent`` is observed and a full metric horizon
    of log remains (and, if given, not after ``last_step``)."""
    last = scene.n_steps - 1 - cfg.metric_horizon
    if last_step is not None:
        last = min(last, last_step)
    return [k for k in range(FIRST_STEP, last + 1, cfg.window_stride) if scene.valid[agent, k]]


def weighted_cvar(values: torch.Tensor, weights: torch.Tensor, alpha: float) -> torch.Tensor:
    """CVaR_alpha of a discrete distribution (values [..., K] with
    probabilities weights [K]): the mean of its upper (1 - alpha) tail, the
    boundary value counted with the fraction of its mass inside the tail --
    risk.cvar's convention for a weighted, finite set."""
    w = weights / weights.sum()
    order = torch.argsort(values, dim=-1, descending=True)
    v = torch.gather(values, -1, order)
    w = w.expand_as(values).gather(-1, order)
    tail = 1.0 - alpha
    if tail <= 0:
        return v[..., 0]
    before = torch.cumsum(w, dim=-1) - w
    inside = (tail - before).clamp(min=0.0)
    inside = torch.minimum(inside, w)
    return (inside * v).sum(-1) / tail


def safety_responsibility(samples: np.ndarray, actual: np.ndarray, actual_valid: np.ndarray,
                          neighbour: np.ndarray, neighbour_valid: np.ndarray, cfg: ResponsibilityConfig,
                          weights: Optional[np.ndarray] = None) -> float:
    """beta_s(a, b), Eq. 3, over the first H steps: samples [N, H, 2] of a's
    motion (probability ``weights`` [N] if it is a weighted motion set), a's
    logged [H, 2] and b's logged [H, 2] futures with validity."""
    f = torch.float32
    s = torch.as_tensor(samples, dtype=f)
    a = torch.as_tensor(actual, dtype=f)
    b = torch.as_tensor(neighbour, dtype=f)
    vb = torch.as_tensor(neighbour_valid) & torch.as_tensor(actual_valid)  # compare where both are logged
    d_samples = pairwise_min_distance_over_time(s, b, valid_b=vb, d_sat=cfg.d_sat)
    d_actual = pairwise_min_distance_over_time(a.unsqueeze(0), b, valid_b=vb, d_sat=cfg.d_sat).squeeze(0)
    if weights is not None:
        return float(weighted_cvar(d_samples - d_actual, torch.as_tensor(weights, dtype=f), cfg.cvar_alpha))
    return float(cvar(d_samples - d_actual, cfg.cvar_alpha))


def motion_set(model, dist, cfg: ResponsibilityConfig, generator: Optional[torch.Generator] = None):
    """The agent's motion set: (trajectories [N, T, 2], their log
    probabilities [N], weights [N] or None) -- N samples, or with
    cfg.motion_set == "weighted" the model's whole, probability-weighted set,
    or with "topk" its N most probable members, probability-weighted (the
    weights are not renormalised here; weighted_cvar does that), or with
    "nms" N members spread by CAT's goal NMS, each weighted by the
    probability nearest to it (responsibility/modes.py)."""
    if cfg.motion_set == "nms":
        if not hasattr(model, "nms_motion_set"):
            raise ValueError("motion_set='nms' needs a model with an NMS motion set (DenseTNT or UniTraj's MTR)")
        trajs, weights = model.nms_motion_set(dist, cfg.n_safety_samples)
        weights = np.asarray(weights, dtype=np.float64)
        return trajs, torch.as_tensor(np.log(np.clip(weights, 1e-300, None))), weights
    if cfg.motion_set in ("weighted", "topk"):
        if not hasattr(model, "motion_set"):
            raise ValueError(f"motion_set='{cfg.motion_set}' needs a model with a finite motion set "
                             "(e.g. UniTraj's MTR)")
        trajs, probs = (model.motion_set(dist, top_k=cfg.n_safety_samples) if cfg.motion_set == "topk"
                        else model.motion_set(dist))
        probs = np.asarray(probs, dtype=np.float64)
        return trajs, torch.as_tensor(np.log(np.clip(probs, 1e-300, None))), probs
    _, log_prob, trajs = model.sample(dist, cfg.n_safety_samples, generator=generator)
    return trajs, log_prob, None


def goal_kl(log_p: torch.Tensor, log_q: torch.Tensor, support: Optional[torch.Tensor] = None) -> float:
    """KL(p || q) over a shared goal grid, in nats (clamped at 0 against
    rounding); with ``support`` (bool [G]), over those goals only, both
    distributions renormalised there."""
    if support is not None:
        log_p = log_p[support] - torch.logsumexp(log_p[support], 0)
        log_q = log_q[support] - torch.logsumexp(log_q[support], 0)
    return max(0.0, float((log_p.exp() * (log_p - log_q)).sum()))


def valid_goal_support(scene: Scene, b: int, step: int, dist, radius: float) -> Optional[torch.Tensor]:
    """b's valid goals (bool [G]): those on the lanes b can reach, or None
    (no restriction) for a model without goals, b on no lane, or less than
    MIN_VALID_GOAL_MASS of b's goal mass on those lanes."""
    if getattr(dist, "goals_global", None) is None:
        return None
    lanes = reachable_lanes(scene, b, step)
    if lanes is None:
        return None
    on = torch.as_tensor(goals_on_lanes(dist, lanes, radius, scene), device=dist.log_prob.device)
    # most of b's predicted mass off its lanes: the map misses where it goes (e.g. a driveway)
    return on if float(dist.log_prob.exp()[on].sum()) >= MIN_VALID_GOAL_MASS else None


def goals_in_scene(dist) -> Optional[np.ndarray]:
    """A goal distribution's goals [G, 2] in the scene frame (DenseTNT's
    grid; MTR's intention end points), or None."""
    goals = getattr(dist, "goals_global", None)
    if goals is None and hasattr(dist, "goals") and hasattr(dist, "to_global"):
        goals = dist.to_global(dist.goals)
    return None if goals is None else np.asarray(goals)


def same_mode_support(scene: Scene, b: int, step: int, dist, cfg: ResponsibilityConfig) -> Optional[torch.Tensor]:
    """b's goals in its own logged drive mode (bool [G]), or None (no
    restriction) for a model without goals, b on no lane ("lanes"), or less
    than MIN_MODE_MASS of b's goal mass in that mode."""
    goals = goals_in_scene(dist)
    if goals is None:
        return None
    if cfg.courtesy_same_mode == "lanes":
        lanes = route_lanes(scene, b, step)
        if lanes is None:
            return None
        on = lane_graph(scene).near(goals, lanes, cfg.filter.lane_radius)
    elif cfg.courtesy_same_mode == "path":
        on = distance_to_logged_path(scene, b, step, goals) <= cfg.courtesy_path_lateral
    else:
        raise ValueError(f"courtesy_same_mode must be 'lanes' or 'path', not {cfg.courtesy_same_mode!r}")
    on = torch.as_tensor(on, device=dist.log_prob.device)
    return on if float(dist.log_prob.exp()[on].sum()) >= MIN_MODE_MASS else None


def responsibility_at(model, scene: Scene, agent: int, step: int, cfg: ResponsibilityConfig,
                      generator: Optional[torch.Generator] = None,
                      record: Optional[Dict] = None) -> Optional[Observation]:
    """Safety and courtesy responsibility of ``agent`` at context step
    ``step``, or None when DenseTNT has no prediction for it there.

    ``model`` provides ``distribution(scene, step, agent)``,
    ``sample(dist, n, generator) -> (goal idx, log prob, trajectories [n, 80, 2])``
    and ``with_and_without(scene, step, b, a) -> (dist, dist) | None``
    (responsibility.densetnt.DenseTNT).

    ``record``, if given, receives what the values were computed from (for
    visualisation): "distribution" (the agent's goal distribution),
    "samples" [N, 80, 2] and "sample_log_prob" [N] (None without neighbours),
    "horizon", "neighbours" {index: evidence}, "courtesy" {neighbour
    index: (distribution with, without the agent)}, "courtesy_support"
    {neighbour index: the goals the KL was taken over (bool [G]) or None}
    and, with a filter, "kept" {neighbour index: indices of the samples
    beta_s used}."""
    horizon = effective_horizon(scene, step, cfg)
    if horizon == 0:
        return None
    dist = model.distribution(scene, step, agent)
    if dist is None:
        return None
    select = ooi_neighbours if cfg.use_ooi else interacting_neighbours
    neighbours = select(scene, agent, step, horizon, cfg.interaction)
    fut = slice(step + 1, step + 1 + horizon)
    speed = float(np.linalg.norm(scene.velocity[agent, step]))
    per_neighbour: Dict[str, Dict[str, float]] = {}
    if record is not None:
        record.update(distribution=dist, samples=None, sample_log_prob=None, horizon=horizon,
                      neighbours=neighbours, courtesy={}, courtesy_support={}, kept={})
    if neighbours:
        route_mass = None
        if cfg.filter.lane_route:  # a goal-based model: restrict its goals to a's route lanes before sampling
            dist, route_mass = restrict_to_route(scene, agent, step, dist, cfg.filter)
        trajs, sample_log_prob, weights = motion_set(model, dist, cfg, generator)
        if record is not None:
            record["samples"], record["sample_log_prob"] = trajs, sample_log_prob
        samples = trajs[:, :horizon, :2]
        actual, actual_valid = scene.position[agent, fut, :2], scene.valid[agent, fut]
        valid_set = (MotionFilter(scene, agent, step, np.asarray(trajs)[..., :2], horizon, list(neighbours),
                                  cfg.filter, weights, goals_restricted=route_mass is not None)
                     if cfg.filter.active else None)
        for b, evidence in neighbours.items():
            entry = dict(evidence, type=scene.types[b])
            s_b, w_b = samples, weights
            if valid_set is not None:
                idx = valid_set.keep(b)
                s_b, w_b = samples[idx], None if weights is None else weights[idx]
                entry.update(valid_set.stats(idx))
                if route_mass is not None:
                    entry["route_goal_mass"] = round(route_mass, 4)
                if record is not None:
                    record["kept"][b] = idx
            entry["safety"] = safety_responsibility(
                s_b, actual, actual_valid, scene.position[b, fut, :2], scene.valid[b, fut], cfg, w_b)
            entry["courtesy"] = None
            if cfg.courtesy:
                pair = model.with_and_without(scene, step, b, agent)
                if pair is not None:
                    if cfg.courtesy_same_mode:
                        support = same_mode_support(scene, b, step, pair[0], cfg)
                    elif cfg.courtesy_valid_goals:
                        support = valid_goal_support(scene, b, step, pair[0], cfg.filter.lane_radius)
                    else:
                        support = None
                    entry["courtesy"] = goal_kl(pair[0].log_prob, pair[1].log_prob, support)
                    if support is not None:
                        entry["courtesy_goal_mass"] = round(float(pair[0].log_prob.exp()[support].sum()), 4)
                    if record is not None:
                        record["courtesy"][b] = pair
                        record["courtesy_support"][b] = support
            per_neighbour[scene.track_ids[b]] = entry
    safety = max((v["safety"] for v in per_neighbour.values()), default=0.0)
    courtesy = max((v["courtesy"] for v in per_neighbour.values() if v["courtesy"] is not None), default=0.0)
    return Observation(scene.scenario_id, agent, scene.track_ids[agent], step, safety, courtesy, speed, per_neighbour)


def scene_responsibility(model, scene: Scene, agent: Optional[int] = None,
                         cfg: Optional[ResponsibilityConfig] = None,
                         last_step: Optional[int] = None) -> List[Observation]:
    """Responsibility of ``agent`` (default: the self-driving car) at every
    evaluated context step of the scene (up to ``last_step``, e.g. the end of
    a simulated episode: responsibility.rollouts.last_window_step)."""
    cfg = cfg or ResponsibilityConfig()
    agent = scene.sdc if agent is None else agent
    generator = torch.Generator().manual_seed(cfg.seed)
    out = []
    for step in window_steps(scene, agent, cfg, last_step):
        obs = responsibility_at(model, scene, agent, step, cfg, generator)
        if obs is not None:
            out.append(obs)
    return out


def config_dict(cfg: ResponsibilityConfig) -> Dict:
    return asdict(cfg)
