"""Responsibility-constrained adversarial scenario generation for CAT.

CAT (``advgen/adv_generator.py``) predicts 32 trajectories for the adversary
with DenseTNT and picks the one maximising

    score_j = sum_i P(OV_j) P(AV_i) P(collision of OV_j with AV_i)

over the ego's recent trajectories AV_i -- i.e. whichever likely trajectory
hits the ego. Nothing stops that trajectory from being the adversary simply
driving into the ego, a crash the ego could not have prevented and the
adversary is to blame for.

Here each candidate additionally gets the adversary's **safety
responsibility** toward the ego (catk / Hsu et al., Eq. 2-3): the CVaR, over
the adversary's own DenseTNT motion set, of how much more distance to the ego
its alternatives would have kept than the candidate does,

    beta_j = sum_i P(AV_i) CVaR_alpha[ D_g(xi, AV_i) - D_g(OV_j, AV_i) ],  xi ~ adversary's motion set

measured over the whole 8 s CAT checks. A candidate with low beta is one the
adversary would plausibly drive anyway; a collision it produces is one the
ego has to handle. Selection rules:

  cat          CAT's own rule (argmax score, else the closest approach)
  constrained  the highest-scoring candidate with beta <= threshold; among
               those, the closest approach if none collides; the least
               responsible candidate if none qualifies
  penalized    argmax score * exp(-max(beta, 0) / penalty), then as cat
  fair         like constrained, but a candidate must also be avoidable:
               avoid_j >= rho (below); if no candidate qualifies, the most
               avoidable one among those within the threshold

**Ego avoidability** of candidate j: the share of the ego's own DenseTNT
motion set at the generation step -- what drivers with the ego's history
would do -- that never touches the adversary driving j (footprints covered by
circles, over the whole 8 s). avoid_j ~ 0: whatever a human in the ego's
place does, j hits it -- an unavoidable crash, useless to learn from;
avoid_j ~ 1: nearly every plausible ego motion escapes it. The ego's motion
is open-loop (it does not react to j), so this underestimates what a
reacting driver could avoid.

Courtesy responsibility is not used: DenseTNT conditions on history only, so
the adversary's influence on the ego's predicted goals is the same for every
candidate.

Candidates come from the wrapper (``responsibility.densetnt``), i.e. the
adversary predicted on its own; CAT batches it with the ego, which shifts the
prediction slightly (see densetnt.py). All rules select among the same
candidates, so they compare cleanly.
"""

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from responsibility.densetnt import DenseTNT, cat_instance
from responsibility.geometry import BoxTrajectories, collision_cost_matrix
from responsibility.metrics import ResponsibilityConfig, safety_responsibility
from responsibility.scene import Scene

RULES = ("cat", "constrained", "penalized", "fair")
GENERATION_STEP = 10  # CAT plans the adversary from the first 11 logged steps


def cat_candidate_probs(log_scores: np.ndarray) -> np.ndarray:
    """P(OV_j) as AdvGenerator.generate derives it from the NMS scores."""
    s = np.array(log_scores, dtype=np.float64)
    s[6:] = s[6]
    p = np.exp(s)
    return p / p.sum()


def cat_collision_scores(trajs_ov: np.ndarray, probs_ov: np.ndarray, trajs_av: Sequence[np.ndarray],
                         probs_av: Sequence[float], ov_size: Dict, av_size: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """``score`` [K] and ``min_dist`` [K] exactly as AdvGenerator.generate
    computes them (oriented boxes every 0.5 s, segment intersection, a
    0.99-per-step decay of later collisions)."""
    from advgen.adv_generator import Intersect, get_polyline_yaw

    def boxes(traj, w, l):
        t = traj[::5]
        yaw = get_polyline_yaw(traj)[::5].reshape(-1, 1)
        c, s = np.cos(yaw), np.sin(yaw)
        return np.concatenate((
            t, yaw,
            t[:, 0:1] + 0.5 * l * c + 0.5 * w * s, t[:, 1:2] + 0.5 * l * s - 0.5 * w * c,
            t[:, 0:1] + 0.5 * l * c - 0.5 * w * s, t[:, 1:2] + 0.5 * l * s + 0.5 * w * c,
            t[:, 0:1] - 0.5 * l * c - 0.5 * w * s, t[:, 1:2] - 0.5 * l * s + 0.5 * w * c,
            t[:, 0:1] - 0.5 * l * c + 0.5 * w * s, t[:, 1:2] - 0.5 * l * s - 0.5 * w * c), axis=1)

    k = len(trajs_ov)
    res = np.zeros(k)
    # an integer array as in CAT (np.full(32, fill_value=1000000)): distances
    # are truncated to whole metres, which decides CAT's closest-approach ties
    min_dist = np.full(k, 1000000)
    reach = np.linalg.norm([0.5 * av_size["l"], 0.5 * av_size["w"]]) + np.linalg.norm(
        [0.5 * ov_size["l"], 0.5 * ov_size["w"]])
    for j in range(k):
        bbox_ov = boxes(trajs_ov[j], ov_size["w"], ov_size["l"])
        for traj_av, p_av in zip(trajs_av, probs_av):
            bbox_av = boxes(np.asarray(traj_av), av_size["w"], av_size["l"])
            p3, uncertainty = 0.0, 1.0
            for (cx1, cy1, _, xa, ya, xb, yb, xc, yc, xd, yd), (cx2, cy2, _, xe, ye, xf, yf, xg, yg, xh, yh) in zip(
                    bbox_av, bbox_ov):
                uncertainty *= 0.99
                d = np.linalg.norm([cx1 - cx2, cy1 - cy2])
                min_dist[j] = min(min_dist[j], d)
                if d >= reach:
                    continue
                av_edges = [[xa, ya, xb, yb], [xb, yb, xc, yc], [xc, yc, xd, yd], [xd, yd, xa, ya]]
                ov_edges = [[xe, ye, xf, yf], [xf, yf, xg, yg], [xg, yg, xh, yh], [xh, yh, xe, ye]]
                if any(Intersect(a, o) for a in av_edges for o in ov_edges):
                    p3 = uncertainty
                    break
            res[j] += probs_ov[j] * p_av * p3
    return res, min_dist


def adversary_responsibility(samples: np.ndarray, candidates: np.ndarray, trajs_av: Sequence[np.ndarray],
                             probs_av: Sequence[float], cfg: ResponsibilityConfig, horizon: int) -> np.ndarray:
    """beta_j [K]: the adversary's safety responsibility toward the ego if it
    drove candidate j, averaged over the ego trajectories by P(AV_i)."""
    w = np.asarray(probs_av, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
    beta = np.zeros(len(candidates))
    for traj_av, wi in zip(trajs_av, w):
        traj_av = np.asarray(traj_av)[:, :2]
        h = min(horizon, len(traj_av), candidates.shape[1])
        ego = np.zeros((horizon, 2))
        ego[:h] = traj_av[:h]
        ego_valid = np.arange(horizon) < h
        for j, cand in enumerate(candidates):
            beta[j] += wi * safety_responsibility(samples[:, :horizon], cand[:horizon], np.ones(horizon, dtype=bool),
                                                  ego, ego_valid, cfg)
    return beta


def path_headings(traj: np.ndarray, origin: np.ndarray, start_heading: float, min_step: float = 0.05) -> np.ndarray:
    """Headings [..., T] along trajectories [..., T, 2] that continue from
    ``origin`` (the position one step before their first point): the
    direction of each step's motion, held from the previous step while the
    agent moves less than ``min_step`` m (standing), ``start_heading`` before
    it first moves."""
    traj = np.asarray(traj, dtype=np.float64)
    prev = np.concatenate([np.broadcast_to(np.asarray(origin, dtype=np.float64), traj[..., :1, :].shape),
                           traj[..., :-1, :]], axis=-2)
    step = traj - prev
    moving = np.linalg.norm(step, axis=-1) > min_step
    angle = np.arctan2(step[..., 1], step[..., 0])
    out = np.empty(traj.shape[:-1])
    current = np.full(traj.shape[:-2], float(start_heading))
    for t in range(traj.shape[-2]):
        current = np.where(moving[..., t], angle[..., t], current)
        out[..., t] = current
    return out


def ego_avoidability(ego_samples: np.ndarray, candidates: np.ndarray, ego_start: Tuple[np.ndarray, float],
                     adv_start: Tuple[np.ndarray, float], ego_size: Dict, adv_size: Dict,
                     horizon: int = 80, n_circles: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """``avoid`` [K]: the share of the ego's motion set that never touches
    candidate j, and ``hits`` [N, K] (sample n touches candidate j).

    ego_samples [N, T, 2] and candidates [K, T, 2] cover the same steps after
    the generation step; ``*_start`` are (position, heading) at that step and
    the sizes CAT's {"w", "l"} dicts. Footprints are covered by
    ``n_circles`` circles (geometry.collision_cost_matrix), headings follow
    the paths (path_headings)."""
    f = torch.float32
    h = min(horizon, ego_samples.shape[1], candidates.shape[1])

    def boxes(trajs, start, size):
        trajs = np.asarray(trajs, dtype=np.float64)[:, :h, :2]
        heading = path_headings(trajs, start[0], start[1])
        n = len(trajs)
        return BoxTrajectories(pos=torch.as_tensor(trajs, dtype=f), heading=torch.as_tensor(heading, dtype=f),
                               valid=torch.ones(n, h, dtype=torch.bool),
                               length=torch.full((n,), float(size["l"]), dtype=f),
                               width=torch.full((n,), float(size["w"]), dtype=f))

    cost = collision_cost_matrix(boxes(ego_samples, ego_start, ego_size), boxes(candidates, adv_start, adv_size),
                                 n_circles=n_circles)
    hits = (cost > 0).numpy()
    return 1.0 - hits.mean(axis=0), hits


def select(rule: str, score: np.ndarray, min_dist: np.ndarray, beta: Optional[np.ndarray],
           threshold: float = 1.0, penalty: float = 1.0, avoid: Optional[np.ndarray] = None,
           min_avoid: float = 0.3) -> Tuple[int, str]:
    """The chosen candidate and why."""
    if rule == "cat" or beta is None:
        return (int(np.argmax(score)), "collision") if np.any(score) else (int(np.argmin(min_dist)), "closest")
    if rule == "constrained":
        ok = beta <= threshold
        if not ok.any():
            return int(np.argmin(beta)), "least responsible (none within threshold)"
        if np.any(score[ok]):
            return int(np.flatnonzero(ok)[np.argmax(score[ok])]), "collision within threshold"
        return int(np.flatnonzero(ok)[np.argmin(min_dist[ok])]), "closest within threshold"
    if rule == "penalized":
        weighted = score * np.exp(-np.maximum(beta, 0.0) / penalty)
        if np.any(weighted):
            return int(np.argmax(weighted)), "penalized collision"
        return int(np.argmin(min_dist)), "closest"
    if rule == "fair":
        if avoid is None:
            raise ValueError("the fair rule needs the candidates' ego avoidability")
        within = beta <= threshold
        ok = within & (avoid >= min_avoid)
        if ok.any():
            if np.any(score[ok]):
                return int(np.flatnonzero(ok)[np.argmax(score[ok])]), "avoidable collision within threshold"
            return int(np.flatnonzero(ok)[np.argmin(min_dist[ok])]), "closest avoidable within threshold"
        pool = within if within.any() else np.ones_like(within)
        idx = np.flatnonzero(pool)
        best = idx[avoid[idx] == avoid[idx].max()]
        return int(best[np.argmin(min_dist[best])]), "most avoidable (none avoidable enough)"
    raise ValueError(f"rule must be one of {RULES}")


def selection_name(args) -> str:
    """How runs with this adversary are named: cat, constrained<tau>,
    penalized<p>, fair<tau>_<rho>."""
    rule = getattr(args, "adv_selection", "cat")
    if rule == "cat":
        return "cat"
    if rule == "constrained":
        return f"constrained{args.resp_threshold:g}"
    if rule == "penalized":
        return f"penalized{args.resp_penalty:g}"
    return f"fair{args.resp_threshold:g}_{args.resp_avoid:g}"


def add_arguments(parser) -> None:
    g = parser.add_argument_group("responsibility-constrained adversary")
    g.add_argument("--adv_selection", default="cat", choices=RULES,
                   help="cat: CAT's original generator; constrained / penalized: limit the adversary's "
                        "safety responsibility toward the ego; fair: also require that the ego can "
                        "avoid it (responsibility/adversarial.py).")
    g.add_argument("--resp_threshold", type=float, default=1.0,
                   help="m; constrained / fair: highest beta a chosen adversary trajectory may have.")
    g.add_argument("--resp_avoid", type=float, default=0.3,
                   help="fair: lowest share of the ego's motion set that must escape the adversary.")
    g.add_argument("--resp_penalty", type=float, default=1.0, help="m; penalized: exp(-beta / penalty).")
    g.add_argument("--resp_samples", type=int, default=40, help="Adversary motion-set size for beta.")
    g.add_argument("--resp_horizon", type=int, default=80, help="10 Hz steps over which beta is measured.")
    g.add_argument("--resp_d_sat", type=float, default=10.0, help="m; D_g saturation (<= 0: none).")
    g.add_argument("--resp_device", default=None, help="Default: cuda if available.")


def make_adv_generator(parser):
    """CAT's AdvGenerator, or the responsibility-constrained one, per
    --adv_selection (defaults to CAT's)."""
    add_arguments(parser)
    known, _ = parser.parse_known_args()
    if known.adv_selection == "cat":
        from advgen.adv_generator import AdvGenerator

        return AdvGenerator(parser)
    return ResponsibleAdvGenerator(parser)


def cat_adversary(scene: Scene) -> Optional[int]:
    """The adversary CAT uses: the object of interest other than the ego."""
    others = [i for i in scene.objects_of_interest if i != scene.sdc]
    return others[0] if others else None


def _base():
    from advgen.adv_generator import AdvGenerator

    return AdvGenerator


class ResponsibleAdvGenerator(_base()):
    """Drop-in replacement for CAT's AdvGenerator (same calls: before_episode,
    log_AV_history, after_episode, generate, adv_agent, adv_traj) choosing the
    adversary trajectory with a responsibility rule. Every choice is logged in
    ``selections`` (see ``report``)."""

    def __init__(self, parser, argv: Optional[Sequence[str]] = None):
        """Like AdvGenerator(parser): adds CAT's and this generator's options to
        ``parser`` and parses the command line (or ``argv``); the DenseTNT
        wrapper is loaded instead of AdvGenerator's model on cuda:0."""
        import advgen.utils

        from responsibility.densetnt import CAT_OTHER_PARAMS

        if not any(a.dest == "adv_selection" for a in parser._actions):
            add_arguments(parser)
        advgen.utils.add_argument(parser)
        parser.set_defaults(other_params=list(CAT_OTHER_PARAMS), mode_num=32)
        self.args = parser.parse_args(argv)
        device = self.args.resp_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtnt = DenseTNT(device=device, mode_num=32)
        self.model = self.dtnt.model
        self.cfg = ResponsibilityConfig(n_safety_samples=self.args.resp_samples,
                                        d_sat=self.args.resp_d_sat if self.args.resp_d_sat > 0 else None)
        self.storage = {}
        self.selections: List[Dict] = []

    def before_episode(self, env):
        seed = env.current_seed
        if seed not in self.storage:
            # before AdvGenerator._parse, which removes the ego from the scenario's objects of interest
            scene = Scene.from_description(copy.deepcopy(env.engine.data_manager._scenario[seed]))
        super().before_episode(env)
        if "scene" not in self.storage[seed]:
            self.storage[seed]["scene"] = scene

    def generate(self, mode="train"):
        from advgen.adv_generator import StepAlignedPlan, get_polyline_vel, get_polyline_yaw

        st = self.storage[self.env.current_seed]
        if mode == "train":
            trajs_av, probs_av = list(st["AV_trajs"]), list(st["AV_probs"])
        else:
            trajs_av, probs_av = list(st["AV_trajs_eval"]), [1.0]
        choice = self.choose(st["scene"], trajs_av, probs_av, st["adv_info"], st["ego_info"],
                             seed=int(self.env.current_seed))
        adv_pos = np.concatenate((st["adv_past"], choice["trajectory"]), axis=0)
        self.adv_traj = StepAlignedPlan(np.concatenate(
            (adv_pos, get_polyline_vel(adv_pos), get_polyline_yaw(adv_pos).reshape(-1, 1)), axis=1), self.env)
        # scores only: candidates and samples of every episode would pile up over a training run
        self.selections.append({k: choice[k] for k in ("rule", "chosen", "why", "score", "min_dist", "beta", "avoid")})
        return st["traffic_motion_feat"], self.adv_traj, np.array(trajs_av), bool(np.any(choice["score"]))

    def choose(self, scene: Scene, trajs_av, probs_av, ov_size, av_size, seed: int = 0,
               rules: Optional[Sequence[str]] = None) -> Dict:
        """Candidates, their scores and responsibilities, and the choice of
        ``self.args.adv_selection`` (plus the choice of every rule in ``rules``)."""
        dist = self.dtnt.goal_distributions([cat_instance(self.dtnt, scene, select=1)])[0]
        candidates, log_scores = self.dtnt.nms_modes(dist)
        probs = cat_candidate_probs(log_scores)
        score, min_dist = cat_collision_scores(candidates, probs, trajs_av, probs_av, ov_size, av_size)
        generator = torch.Generator().manual_seed(seed)
        _, _, samples = self.dtnt.sample(dist, self.cfg.n_safety_samples, generator=generator)
        beta = adversary_responsibility(samples, candidates, trajs_av, probs_av, self.cfg, self.args.resp_horizon)
        rule = self.args.adv_selection
        avoid, ego_samples = None, None
        if rule == "fair" or "fair" in (rules or ()):
            avoid, ego_samples = self.avoidability(scene, candidates, ov_size, av_size, seed)
        chosen, why = select(rule, score, min_dist, beta, self.args.resp_threshold, self.args.resp_penalty,
                             avoid, self.args.resp_avoid)
        out = {"rule": rule, "chosen": chosen, "why": why, "trajectory": candidates[chosen],
               "score": score, "min_dist": min_dist, "beta": beta, "avoid": avoid, "candidates": candidates,
               "log_scores": log_scores, "samples": samples, "ego_samples": ego_samples}
        for other in rules or ():
            out[f"chosen_{other}"] = select(other, score, min_dist, beta, self.args.resp_threshold,
                                            self.args.resp_penalty, avoid, self.args.resp_avoid)[0]
        return out

    def avoidability(self, scene: Scene, candidates: np.ndarray, ov_size: Dict, av_size: Dict,
                     seed: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Ego avoidability [K] of the candidates, from the ego's DenseTNT
        motion set at the generation step (its own random stream, so the
        adversary's samples -- and every other rule's choice -- are the same
        with or without it), and the ego samples; all ones (no filtering)
        when DenseTNT has no prediction for the ego there."""
        step, ego = GENERATION_STEP, scene.sdc
        adv = cat_adversary(scene)
        dist = self.dtnt.distribution(scene, step, ego)
        if dist is None or adv is None:
            return np.ones(len(candidates)), None
        generator = torch.Generator().manual_seed(seed * 7919 + 1)
        _, _, samples = self.dtnt.sample(dist, self.cfg.n_safety_samples, generator=generator)
        samples = samples.detach().cpu().numpy() if torch.is_tensor(samples) else np.asarray(samples)
        avoid, _ = ego_avoidability(
            samples, candidates,
            (scene.position[ego, step, :2], float(scene.heading[ego, step])),
            (scene.position[adv, step, :2], float(scene.heading[adv, step])),
            av_size, ov_size, horizon=self.args.resp_horizon)
        return avoid, samples

    def report(self) -> None:
        if not self.selections:
            return
        collide = np.mean([s["score"][s["chosen"]] > 0 for s in self.selections])
        beta = np.mean([s["beta"][s["chosen"]] for s in self.selections])
        line = (f"[{self.args.adv_selection}] {len(self.selections)} adversaries: predicted collision in "
                f"{100 * collide:.0f}%, mean adversary safety responsibility {beta:+.2f} m")
        avoid = [s["avoid"][s["chosen"]] for s in self.selections if s.get("avoid") is not None]
        if avoid:
            line += f", mean ego avoidability {np.mean(avoid):.2f}"
        print(line)

