"""CAT's pretrained DenseTNT as a probabilistic motion model.

DenseTNT scores a dense grid of candidate goals (224 x 224 points in the
predicted agent's frame, CAT's ``raster`` setting) with a log-softmax, and a
completion head turns any goal into an 8 s trajectory. CAT only uses the
32 goals that survive non-maximum suppression. Responsibility needs the
distribution itself:

- **samples** of an agent's motion (the motion set of Eq. 1a): goals drawn
  i.i.d. from the categorical goal distribution, each completed into a
  trajectory. NMS modes would be a mode-seeking, non-random selection.
- **the distribution with and without another agent**: two forward passes,
  the second with that agent left out of the scene. The candidate grid only
  depends on the predicted agent's own pose, so the two goal distributions
  live on the same support and their KL is exact (courtesy, Eq. 4).

The model is only ever conditioned on the 1.1 s of history before the
current step, never on anyone's future: DenseTNT's counterfactuals are
open-loop by construction (catk's ``neighbor_future="hidden"``).
"""

import argparse
import logging
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

sys.modules.setdefault("pickle5", pickle)  # advgen.structs imports pickle5; plain pickle does on py>=3.8
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf  # noqa: E402

import advgen.adv_utils  # noqa: E402
import advgen.utils  # noqa: E402
import advgen.utils_cython  # noqa: E402
from advgen.modeling.vectornet import VectorNet  # noqa: E402
from responsibility.modes import nms_select  # noqa: E402
from responsibility.scene import Scene, cat_agent_order, womd_features  # noqa: E402

# advgen/adv_generator.py's configuration of the pretrained model
CAT_OTHER_PARAMS = [
    "l1_loss", "densetnt", "goals_2D", "enhance_global_graph", "laneGCN", "point_sub_graph",
    "laneGCN-4", "stride_10_2", "raster", "train_pair_interest",
]
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "advgen", "pretrained", "densetnt.bin")


def cat_args(mode_num: int = 32, output_dir: Optional[str] = None):
    """The argument namespace AdvGenerator builds, without reading sys.argv."""
    parser = argparse.ArgumentParser()
    advgen.utils.add_argument(parser)
    parser.set_defaults(other_params=list(CAT_OTHER_PARAMS), mode_num=mode_num,
                        output_dir=output_dir or os.path.join(tempfile.gettempdir(), "densetnt_responsibility"))
    args = parser.parse_args([])
    advgen.utils.init(args, logging.getLogger("densetnt"))
    return args


def _last_valid_xy(scene: Scene, agents: Sequence[int], step: int) -> np.ndarray:
    """Each agent's position at its last observed step up to ``step``."""
    agents = list(agents)
    if not agents:
        return np.zeros((0, 2))
    valid = scene.valid[agents, : step + 1]
    last = np.where(valid.any(-1), step - np.argmax(valid[:, ::-1], axis=-1), 0)
    return scene.position[agents, last, :2]


@dataclass
class GoalDistribution:
    """One agent's predicted goal distribution in one scene context."""

    agent: int  # index of the predicted agent in the Scene
    step: int  # context (current) step
    excluded: Tuple[int, ...]  # agents left out of the scene
    goals: np.ndarray  # [G, 2] candidate goals, in the agent's own frame
    log_prob: torch.Tensor  # [G] log-softmax scores
    origin: np.ndarray  # [2] agent position at ``step`` (frame origin)
    yaw: float  # frame rotation used by advgen.utils.Normalizer
    context: Tuple[torch.Tensor, torch.Tensor]  # (element inputs [L, H], global states [L, H])
    mapping: dict  # the advgen instance, kept for NMS mode selection

    def to_global(self, points: np.ndarray) -> np.ndarray:
        """Agent frame -> scene frame (vectorised advgen.utils.Normalizer(reverse=True))."""
        c, s = np.cos(-self.yaw), np.sin(-self.yaw)
        local = np.asarray(points, dtype=np.float64)
        x = local[..., 0] * c - local[..., 1] * s
        y = local[..., 0] * s + local[..., 1] * c
        return np.stack([x, y], axis=-1) + self.origin

    @property
    def goals_global(self) -> np.ndarray:
        return self.to_global(self.goals)


class DenseTNT:
    """Wraps advgen's VectorNet for goal distributions, sampling and completion."""

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str = "cpu", mode_num: int = 32):
        self.args = cat_args(mode_num=mode_num)
        self.device = device
        self.model = VectorNet(self.args).to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model.eval()
        self.decoder = self.model.decoder

    # ------------------------------------------------------------------
    # instances
    # ------------------------------------------------------------------

    def _partner(self, scene: Scene, step: int, target: int, avoid: Sequence[int]) -> Optional[int]:
        """DenseTNT's pair model takes two objects of interest; the second
        only fills the second slot of the agent list. The nearest other
        vehicle present at ``step`` is used (any agent if there is none),
        never one of ``avoid``: a counterfactual that removes an agent must
        not also change the partner, or the comparison would measure that too."""
        present = [i for i in range(scene.n_agents)
                   if i != target and i not in avoid and scene.valid[i, step]]
        if not present:
            return None
        vehicles = [i for i in present if scene.types[i] == "VEHICLE"] or present
        d = np.linalg.norm(scene.position[vehicles, step, :2] - scene.position[target, step, :2], axis=-1)
        return vehicles[int(np.argmin(d))]

    def instance(self, scene: Scene, step: int, target: int, excluded: Sequence[int] = (),
                 order: Optional[Sequence[int]] = None, select: int = 0,
                 avoid_partner: Sequence[int] = ()) -> Optional[dict]:
        """advgen's input mapping for predicting ``target`` at ``step`` with
        ``excluded`` agents removed from the scene; None where DenseTNT has no
        prediction (target not a vehicle, or not observed at ``step``).
        Pass the same ``avoid_partner`` to both sides of a with/without
        comparison (the agent being removed) so they share the partner.
        ``order`` / ``select`` override the agent order and which of the first
        two agents is predicted (for reproducing CAT's own instances)."""
        if target in excluded:
            raise ValueError("the predicted agent cannot be excluded from its own scene")
        if not scene.valid[target, step] or scene.types[target] != "VEHICLE":
            return None
        if order is None:
            partner = self._partner(scene, step, target, tuple(excluded) + tuple(avoid_partner))
            if partner is None:
                return None
            # nearest first: the input holds 128 agents, so any beyond that
            # must be the far ones (scene order would drop arbitrary agents,
            # possibly the one whose influence is being measured)
            rest = [i for i in range(scene.n_agents) if i not in (target, partner) and i not in excluded]
            anchor = scene.position[target, step, :2]
            last_seen = np.where(scene.valid[rest][:, : step + 1].any(-1),
                                 np.linalg.norm(_last_valid_xy(scene, rest, step) - anchor, axis=-1), np.inf)
            order = [target, partner] + [rest[j] for j in np.argsort(last_seen, kind="stable")]
        features = {k: tf.convert_to_tensor(v) for k, v in womd_features(scene, step, order).items()}
        inputs, decoded = advgen.adv_utils._parse(features)
        mapping = advgen.adv_utils.get_instance(self.args, inputs, decoded, "tmp", select=select)
        if mapping is not None:
            mapping["responsibility"] = dict(agent=target, step=step, excluded=tuple(excluded))
        return mapping

    # ------------------------------------------------------------------
    # model passes
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(self, mapping: dict):
        """VectorNet's encoder on one instance (what ``VectorNet.forward``
        does before handing over to the decoder).

        Instances are encoded one at a time on purpose: in a batch, advgen pads
        every instance's elements to the longest one and its attention mask
        (``attention_mask[i][:length][:length]``) only limits the rows, so the
        shorter instances also attend to the zero padding and a prediction
        depends on what it was batched with (up to ~1 cm and 4e-3 in log-score
        on CAT's ego+adversary pairs). Alone, there is no padding."""
        model, device = self.model, self.device
        batch = [mapping]
        matrix = advgen.utils.get_from_mapping(batch, "matrix")
        spans = advgen.utils.get_from_mapping(batch, "polyline_spans")
        element_states, _ = model.forward_encode_sub_graph(batch, matrix, spans, device, 1)
        inputs, lengths = advgen.utils.merge_tensors(element_states, device=device)
        mask = torch.ones([1, lengths[0], lengths[0]], device=device)
        hidden = model.global_graph(inputs, mask, batch)
        return inputs, lengths, hidden

    @torch.no_grad()
    def goal_distributions(self, mappings: List[dict]) -> List[GoalDistribution]:
        """The goal distribution of every instance. The raster CNN leaves its
        output on ``args``, so each instance is scored right after encoding."""
        out = []
        for m in mappings:
            inputs, lengths, hidden = self.encode(m)
            goals = m["goals_2D"]
            log_prob = self.decoder.get_scores(
                torch.tensor(goals, device=self.device, dtype=torch.float), inputs, hidden, lengths, 0, [m],
                self.device,
            )
            meta = m["responsibility"]
            normalizer = m["normalizer"]
            out.append(GoalDistribution(
                agent=meta["agent"], step=meta["step"], excluded=meta["excluded"],
                goals=goals, log_prob=log_prob,
                origin=np.array([normalizer.x, normalizer.y], dtype=np.float64), yaw=float(normalizer.yaw),
                context=(inputs[0, : lengths[0]], hidden[0]), mapping=m,
            ))
        return out

    def distribution(self, scene: Scene, step: int, target: int, excluded: Sequence[int] = (),
                     avoid_partner: Sequence[int] = ()) -> Optional[GoalDistribution]:
        mapping = self.instance(scene, step, target, excluded, avoid_partner=avoid_partner)
        return None if mapping is None else self.goal_distributions([mapping])[0]

    def with_and_without(self, scene: Scene, step: int, target: int, removed: int):
        """``target``'s goal distribution with and without ``removed`` in the
        scene (same partner, same candidate goals), or None if DenseTNT does
        not predict ``target`` at ``step``."""
        with_a = self.distribution(scene, step, target, avoid_partner=(removed,))
        if with_a is None:
            return None
        without_a = self.distribution(scene, step, target, excluded=(removed,))
        if without_a is None or not np.array_equal(with_a.goals, without_a.goals):
            raise RuntimeError("removing an agent changed the predicted agent's candidate goals")
        return with_a, without_a

    @torch.no_grad()
    def complete(self, dist: GoalDistribution, goals_local: np.ndarray) -> np.ndarray:
        """Trajectories [M, 80, 2] (scene frame) through goals given in the
        agent's frame, with the completion head exactly as
        ``Decoder.goals_2D_eval`` applies it to the NMS goals."""
        dec = self.decoder
        inputs_i, hidden_i = dist.context
        goals = torch.tensor(np.asarray(goals_local), dtype=torch.float, device=self.device)
        feature = dec.goals_2D_mlps(goals)
        attention = dec.tnt_cross_attention(feature.unsqueeze(0), inputs_i.unsqueeze(0)).squeeze(0)
        trajs = dec.tnt_decoder(torch.cat([hidden_i[0].unsqueeze(0).expand(len(feature), -1), feature, attention], -1))
        trajs = trajs.view(len(feature), dec.future_frame_num, 2).cpu().numpy()
        return dist.to_global(trajs)

    def sample(self, dist: GoalDistribution, n: int, generator: Optional[torch.Generator] = None):
        """``n`` i.i.d. draws of the agent's motion: goal indices [n], their
        log-probabilities [n] and trajectories [n, 80, 2] (scene frame)."""
        probs = dist.log_prob.exp().cpu()
        idx = torch.multinomial(probs, n, replacement=True, generator=generator)
        trajs = self.complete(dist, dist.goals[idx.numpy()])
        return idx, dist.log_prob.cpu()[idx], trajs

    def motion_set(self, dist: GoalDistribution, top_mass: float = 0.999, top_k: Optional[int] = None):
        """The whole goal distribution as a weighted motion set: every goal
        of the dense grid among the most probable ones covering ``top_mass``,
        completed (trajectories [G, 80, 2], scene frame), with its
        probability [G] (``--motion-set weighted``). Goals closer along the
        route stand for slower, braking executions, which 40 samples of a
        peaked distribution rarely draw. With ``top_k``, the ``top_k`` most
        probable goals instead (``--motion-set topk``): G is a median 944 on
        the first 30 scenes, so this completes far fewer trajectories."""
        p = dist.log_prob.double().exp().cpu().numpy()
        order = np.argsort(-p, kind="stable")
        n = top_k if top_k is not None else int(np.searchsorted(np.cumsum(p[order]), top_mass)) + 1
        keep = np.sort(order[:min(n, len(order))])
        return self.complete(dist, dist.goals[keep]), p[keep]

    def nms_motion_set(self, dist: GoalDistribution, n: int):
        """``n`` goals spread over the grid by CAT's NMS rule (its
        nms_threshold times the speed scale factor of the agent's speed),
        completed (trajectories [n, 80, 2], scene frame), each weighted by
        the probability of the goals nearest to it (``--motion-set nms``,
        responsibility/modes.py)."""
        threshold = self.args.nms_threshold * advgen.utils_cython.speed_scale_factor(dist.mapping["speed"])
        idx, weights = nms_select(dist.goals, dist.log_prob.double().exp().cpu().numpy(), n, threshold)
        return self.complete(dist, dist.goals[idx]), weights

    def nms_modes(self, dist: GoalDistribution, mode_num: Optional[int] = None):
        """CAT's prediction: the top goals after non-maximum suppression,
        completed. Returns (trajectories [K, 80, 2], log scores [K])."""
        m = dist.mapping
        scores = dist.log_prob.cpu().numpy()
        advgen.utils.select_goals_by_NMS(m, dist.goals, scores, self.args.nms_threshold, m["speed"],
                                         mode_num=mode_num or self.args.mode_num)
        return self.complete(dist, m["pred_goals"]), np.asarray(m["pred_probs"])


def cat_instance(model: DenseTNT, scene: Scene, select: int) -> Optional[dict]:
    """The instance AdvGenerator builds for agent ``select`` (0 = the
    self-driving car, 1 = the adversary) of CAT's pair, for comparisons."""
    order = cat_agent_order(scene)
    return model.instance(scene, 10, order[select], order=order, select=select)
