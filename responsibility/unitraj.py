"""UniTraj's MTR as the motion model of the responsibility metrics, next to
CAT's DenseTNT (plan and design decisions: responsibility/UNITRAJ_PLAN.md).

The metrics (responsibility.metrics, .blame) need three things of a model:
``distribution(scene, step, agent)``, ``sample(dist, n, generator)`` and
``with_and_without(scene, step, b, a)``. Here:

  distribution  MTR's 64 intention queries without its final NMS: a
                categorical distribution over 64 fixed intention points
                (k-means centres of 8 s endpoints per agent type, the same for
                every prediction) and one trajectory per intention. MTR's
                classification loss is a cross-entropy over these points with
                the one nearest the true endpoint as the target, so the scores
                are a distribution over intentions.
  sample        intentions drawn from it, each with its trajectory
  motion_set    all 64 trajectories and their probabilities (for the exact,
                probability-weighted safety CVaR)
  nms_motion_set  n intentions spread by CAT's NMS rule over their endpoints,
                weighted by the probability nearest to each (modes.py)
  with/without  the neighbour's inputs with the agent, and the same inputs with
                the agent's slot masked out: the other agents, their slots and
                the map are unchanged, and the 64 intentions are the same, so
                courtesy is an exact KL over them

Inputs are built with UniTraj's own preprocessing (BaseDataset.preprocess /
process), from the configuration the model was trained with, on a scenario
description cut to the window [step - past + 1, step + future]: the scene is
cropped, padded with invalid steps where the clip ends (UniTraj pads at the
front, which would shift the time axis of any later step) and its timestamps
restart at 0 as in training. The model itself sits behind ``predictor``, a
callable batch -> (scores [B, K], trajectories [B, K, T, 2]) in the centre
agent's frame: ``mtr_predictor`` for a trained MTR (needs UniTraj's CUDA ops,
i.e. a GPU), anything else for tests.

UniTraj is imported from ``UNITRAJ_ROOT`` (default: a UniTraj checkout next
to this repository) unless it is installed. Its dataset code imports
metadrive and scenarionet only for type constants and file readers the
adapter does not use; when they are not already imported, stand-ins are
registered (MetaDriveType from CAT's metadrive/type.py, which has no
dependencies), so neither CAT's simulator nor ScenarioNet is needed here.
"""

import copy
import importlib
import importlib.util
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from responsibility.modes import NMS_THRESHOLD, nms_select, speed_scale_factor
from responsibility.scene import Scene

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = REPO / "responsibility" / "unitraj_configs"
DEFAULT_METHOD = "MTR_womd"


# ----------------------------------------------------------------------
# importing UniTraj
# ----------------------------------------------------------------------

def unitraj_root(root=None) -> Path:
    """The UniTraj checkout (the directory holding ``unitraj/``)."""
    for candidate in (root, os.environ.get("UNITRAJ_ROOT"), REPO.parent / "UniTraj"):
        if candidate and (Path(candidate) / "unitraj").is_dir():
            return Path(candidate)
    spec = importlib.util.find_spec("unitraj")
    if spec is not None and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0]).parent
    raise ImportError("UniTraj not found: set UNITRAJ_ROOT to the UniTraj checkout")


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def _install_stand_ins() -> None:
    if "metadrive" not in sys.modules:
        spec = importlib.util.spec_from_file_location("_cat_metadrive_type", REPO / "metadrive" / "type.py")
        mdtype = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mdtype)
        _module("metadrive", __path__=[])
        _module("metadrive.scenario", __path__=[])
        _module("metadrive.scenario.scenario_description", MetaDriveType=mdtype.MetaDriveType)
    if "scenarionet" not in sys.modules:
        def unavailable(*args, **kwargs):
            raise RuntimeError("scenarionet is not loaded (the responsibility adapter does not read datasets)")

        _module("scenarionet", __path__=[])
        _module("scenarionet.common_utils", read_scenario=unavailable, read_dataset_summary=unavailable)


def import_real_scenarionet() -> None:
    """Imports the installed scenarionet and metadrive (for reading
    ScenarioNet datasets, i.e. training), which CAT's own ``metadrive/`` at
    the repository root would otherwise shadow; call it before anything
    imports UniTraj, so no stand-ins are registered."""
    here = {REPO.resolve()}
    saved = list(sys.path)
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in here]
    try:
        importlib.import_module("scenarionet.common_utils")
        importlib.import_module("metadrive.scenario.scenario_description")
    finally:
        sys.path[:] = saved


def import_unitraj(module: str, root=None):
    """``unitraj.<module>`` without running the package initialisers of
    ``unitraj.datasets`` / ``unitraj.models``, which import every model and
    dataset (and their dependencies) UniTraj supports."""
    base = unitraj_root(root)
    if str(base) not in sys.path:
        sys.path.append(str(base))
    _install_stand_ins()
    for package in ("unitraj.datasets", "unitraj.models"):
        if package not in sys.modules:
            _module(package, __path__=[str(base / Path(*package.split(".")))])
    return importlib.import_module(f"unitraj.{module}")


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------

class AttrDict(dict):
    """dict with attribute access, as UniTraj reads its (OmegaConf) config."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value


def _attr(value):
    if isinstance(value, dict):
        return AttrDict({k: _attr(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_attr(v) for v in value]
    return value


def load_config(method: str = DEFAULT_METHOD, config_dir=DEFAULT_CONFIG_DIR, root=None) -> AttrDict:
    """UniTraj's config.yaml with ``method/<method>.yaml`` merged over it, as
    train.py merges them (``OmegaConf.merge(cfg, cfg.method)``); the method
    file is looked up in ``config_dir`` first, then in UniTraj's configs."""
    import yaml

    base_dir = unitraj_root(root) / "unitraj" / "configs"
    with open(base_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg.pop("defaults", None)
    path = Path(config_dir) / "method" / f"{method}.yaml"
    if not path.exists():
        path = base_dir / "method" / f"{method}.yaml"
    with open(path) as f:
        method_cfg = yaml.safe_load(f)
    cfg["method"] = method_cfg
    cfg.update(copy.deepcopy(method_cfg))
    cfg.setdefault("remove_outliers", False)
    return _attr(cfg)


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------

def window_description(scene: Scene, step: int, target_id: str, past: int, future: int) -> Dict:
    """The scenario description of the window around context step ``step``:
    ``past`` steps up to and including it and ``future`` after it, steps
    outside the clip invalid, timestamps from 0, and ``target_id`` the only
    track to predict."""
    first = step - past + 1
    total = past + future
    src = np.arange(first, first + total)
    inside = (src >= 0) & (src < scene.n_steps)
    idx = np.clip(src, 0, scene.n_steps - 1)
    desc = scene.to_description()
    for track in desc["tracks"].values():
        state = track["state"]
        for key, value in state.items():
            value = value[idx]
            if key == "valid":
                value = value & inside
            else:
                value = np.where(inside.reshape((-1,) + (1,) * (value.ndim - 1)), value, 0)
            state[key] = value
        track["metadata"]["track_length"] = total
    lights = {}
    for key, light in (desc["dynamic_map_states"] or {}).items():
        light = copy.copy(light)
        states = list(light.get("state", {}).get("object_state", []))
        light["state"] = dict(light.get("state", {}),
                              object_state=[states[s] if 0 <= s < len(states) else None for s in src])
        lights[key] = light
    desc["dynamic_map_states"] = lights
    meta = desc["metadata"]
    meta["ts"] = np.arange(total, dtype=np.float64) * 0.1
    meta["track_length"] = total
    meta["tracks_to_predict"] = {str(target_id): {"track_id": str(target_id), "difficulty": 0,
                                                  "object_type": scene.types[scene.index(target_id)]}}
    return desc


@dataclass
class Instance:
    """One prediction input: the sample (UniTraj's per-agent dict, before
    collation), which scene track occupies each agent slot, and the centre
    agent's pose (origin, yaw) at the context step."""

    agent: int
    step: int
    sample: Dict
    slots: List[Optional[str]]
    origin: np.ndarray
    yaw: float
    excluded: Tuple[int, ...] = ()


class Inputs:
    """Builds model inputs with UniTraj's preprocessing, from its config."""

    def __init__(self, config: Optional[AttrDict] = None, root=None):
        self.config = config or load_config(root=root)
        base = import_unitraj("datasets.base_dataset", root)
        self._common = import_unitraj("datasets.common_utils", root)
        dataset = object.__new__(base.BaseDataset)  # no data loading
        dataset.config = self.config
        dataset.is_validation = True
        dataset.starting_frame = 0
        self.dataset = dataset
        self.past = int(self.config["past_len"])
        self.future = int(self.config["future_len"])
        self.types = set(self.config["object_type"])

    def predicts(self, scene: Scene, agent: int, step: int) -> bool:
        return bool(scene.valid[agent, step]) and scene.types[agent] in self.types

    def instance(self, scene: Scene, step: int, agent: int) -> Optional[Instance]:
        """``agent``'s input at ``step``, or None where the model has no
        prediction for it (not observed then, or a type it was not trained on)."""
        if not self.predicts(scene, agent, step):
            return None
        tid = scene.track_ids[agent]
        internal = self.dataset.preprocess(window_description(scene, step, tid, self.past, self.future))
        out = self.dataset.process(internal)
        if not out:
            return None
        sample = out[0]
        slots = self.slot_ids(internal)
        if slots[int(sample["track_index_to_predict"])] != tid:
            raise RuntimeError(f"agent slot {int(sample['track_index_to_predict'])} is not the predicted agent {tid}")
        center = sample["center_objects_world"]
        return Instance(agent, step, sample, slots, np.asarray(center[:2], dtype=np.float64), float(center[6]))

    def slot_ids(self, internal: Dict) -> List[Optional[str]]:
        """The scene track in each of the model's agent slots, as
        BaseDataset.get_agent_data fills them (tracks observed at some past
        step, nearest to the centre agent at the current step first), by the
        same computations, so ties break identically; None for padding."""
        info = internal
        cur = info["current_time_index"]
        trajs = info["track_infos"]["trajs"]
        ids = info["track_infos"]["object_id"]
        target = info["tracks_to_predict"]["track_index"][0]
        past = trajs[:, :cur + 1]
        center = trajs[target, cur][None]
        local = self.dataset.transform_trajs_to_center_coords(
            obj_trajs=past, center_xyz=center[:, 0:3], center_heading=center[:, 6], heading_index=6,
            rot_vel_index=[7, 8])
        mask = local[:, :, :, -1]
        xy = local[:, :, :, 0:2].copy()
        xy[mask == 0] = 0
        keep = np.logical_not(past[:, :, -1].sum(axis=-1) == 0)
        xy, mask = xy[:, keep], mask[:, keep]
        dist = np.linalg.norm(xy[:, :, -1], axis=-1)
        dist[mask[..., -1] == 0] = 1e10
        order = np.argsort(dist, axis=-1)[0, : int(self.config["max_num_agents"])]
        kept = [ids[i] for i in np.flatnonzero(keep)]
        slots = [str(kept[j]) for j in order]
        return slots + [None] * (int(self.config["max_num_agents"]) - len(slots))

    @staticmethod
    def without(inst: Instance, scene: Scene, removed: Sequence[int]) -> Instance:
        """The same input with ``removed`` agents' slots masked out (absent
        agents are ignored: they were not in the input anyway)."""
        sample = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in inst.sample.items()}
        for agent in removed:
            if agent == inst.agent:
                raise ValueError("the predicted agent cannot be removed from its own input")
            tid = scene.track_ids[agent]
            if tid not in inst.slots:
                continue
            s = inst.slots.index(tid)
            sample["obj_trajs_mask"][s] = False
            for key in ("obj_trajs", "obj_trajs_pos", "obj_trajs_last_pos", "obj_trajs_future_state"):
                sample[key][s] = 0
            sample["obj_trajs_future_mask"][s] = False
        return Instance(inst.agent, inst.step, sample, inst.slots, inst.origin, inst.yaw,
                        tuple(inst.excluded) + tuple(removed))

    def collate(self, instances: Sequence[Instance]) -> Dict:
        return self.dataset.collate_fn([i.sample for i in instances])


def to_global(points: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    """[..., 2] from the centre agent's frame to the scene frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = points[..., 0], points[..., 1]
    return np.stack([c * x - s * y + origin[0], s * x + c * y + origin[1]], axis=-1)


# ----------------------------------------------------------------------
# distributions and the model interface
# ----------------------------------------------------------------------

@dataclass
class IntentionDistribution:
    """An agent's predicted intentions at a context step: ``log_prob`` [K]
    over K fixed intention points and one trajectory [K, T, 2] per
    intention (centre agent's frame). ``goals`` (the trajectories'
    endpoints) and ``to_global`` mirror densetnt.GoalDistribution, so
    records and plots handle both."""

    agent: int
    step: int
    excluded: Tuple[int, ...]
    log_prob: torch.Tensor
    trajectories: np.ndarray
    origin: np.ndarray
    yaw: float
    extras: Dict = field(default_factory=dict)

    @property
    def goals(self) -> np.ndarray:
        return self.trajectories[:, -1, :]

    def to_global(self, points: np.ndarray) -> np.ndarray:
        return to_global(np.asarray(points, dtype=np.float64), self.origin, self.yaw)

    def trajectories_global(self) -> np.ndarray:
        return self.to_global(self.trajectories)


Predictor = Callable[[Dict], Tuple[torch.Tensor, torch.Tensor]]


class UniTrajModel:
    """The metrics' model interface over a UniTraj predictor (module docstring)."""

    def __init__(self, predictor: Predictor, inputs: Optional[Inputs] = None, temperature: float = 1.0,
                 batch_size: int = 64):
        self.predictor = predictor
        self.inputs = inputs or Inputs()
        self.temperature = float(temperature)
        self.batch_size = batch_size

    @torch.no_grad()
    def distributions(self, instances: Sequence[Instance]) -> List[IntentionDistribution]:
        out = []
        for i in range(0, len(instances), self.batch_size):
            chunk = list(instances[i: i + self.batch_size])
            scores, trajs = self.predictor(self.inputs.collate(chunk))
            scores = torch.as_tensor(scores, dtype=torch.float64).cpu()
            log_prob = torch.log_softmax(torch.log(scores.clamp_min(1e-12)) / self.temperature, dim=-1)
            trajs = torch.as_tensor(trajs).detach().cpu().numpy()[..., :2].astype(np.float64)
            for inst, lp, tr in zip(chunk, log_prob, trajs):
                out.append(IntentionDistribution(inst.agent, inst.step, tuple(inst.excluded), lp.float(), tr,
                                                 inst.origin, inst.yaw))
        return out

    def distribution(self, scene: Scene, step: int, target: int, excluded: Sequence[int] = (),
                     avoid_partner: Sequence[int] = ()) -> Optional[IntentionDistribution]:
        inst = self.inputs.instance(scene, step, target)
        if inst is None:
            return None
        if excluded:
            inst = self.inputs.without(inst, scene, excluded)
        dist = self.distributions([inst])[0]
        dist.extras["speed"] = float(np.linalg.norm(scene.velocity[target, step]))  # m/s, for the NMS threshold
        return dist

    def with_and_without(self, scene: Scene, step: int, target: int, removed: int):
        """``target``'s intention distribution with and without ``removed``
        (one batch; identical inputs but for the removed agent's slot), or
        None if the model does not predict ``target`` at ``step``."""
        inst = self.inputs.instance(scene, step, target)
        if inst is None:
            return None
        return tuple(self.distributions([inst, self.inputs.without(inst, scene, [removed])]))

    def sample(self, dist: IntentionDistribution, n: int, generator: Optional[torch.Generator] = None):
        """n intentions drawn from the distribution: (index [n], log prob [n],
        trajectories [n, T, 2] in the scene frame)."""
        probs = dist.log_prob.double().exp()
        idx = torch.multinomial(probs, n, replacement=True, generator=generator)
        trajs = dist.trajectories_global()[idx.numpy()]
        return idx, dist.log_prob[idx], trajs

    def motion_set(self, dist: IntentionDistribution, top_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """All K trajectories [K, T, 2] (scene frame) and their probabilities
        [K], or with ``top_k`` the ``top_k`` most probable of them."""
        trajs, probs = dist.trajectories_global(), dist.log_prob.double().exp().numpy()
        if top_k is None or top_k >= len(probs):
            return trajs, probs
        keep = np.sort(np.argsort(-probs, kind="stable")[:top_k])
        return trajs[keep], probs[keep]

    def nms_motion_set(self, dist: IntentionDistribution, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """``n`` of the intentions, spread by CAT's NMS rule over their
        endpoints (7.2 m times the speed scale factor of the agent's speed),
        and their weights [n] (the probability of the intentions nearest to
        each; ``--motion-set nms``)."""
        if "speed" not in dist.extras:
            raise ValueError("the NMS motion set needs the agent's speed: use a distribution from distribution()")
        threshold = NMS_THRESHOLD * speed_scale_factor(dist.extras["speed"])
        idx, weights = nms_select(dist.goals, dist.log_prob.double().exp().numpy(), n, threshold)
        return dist.trajectories_global()[idx], weights


# ----------------------------------------------------------------------
# MTR (needs UniTraj's CUDA ops, i.e. a GPU)
# ----------------------------------------------------------------------

def mtr_predictor(checkpoint, config: Optional[AttrDict] = None, device: str = "cuda", root=None,
                  num_modes: int = 64) -> Predictor:
    """A trained MTR (a UniTraj / Lightning checkpoint) as a predictor that
    keeps all intention queries (``num_modes`` = 64: no NMS). The encoder
    and decoder are called directly: MotionTransformer.forward also computes
    the training loss, which needs a valid future the metrics do not have."""
    config = config or load_config(root=root)
    cwd = os.getcwd()
    os.chdir(unitraj_root(root) / "unitraj")  # the intention points file is relative to it
    try:
        mtr = import_unitraj("models.mtr.MTR", root)
        model = mtr.MotionTransformer(config)
    finally:
        os.chdir(cwd)
    if str(checkpoint) != "random":  # "random": untrained weights, to test the environment
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
    model.motion_decoder.num_motion_modes = num_modes
    model.to(device).eval()

    @torch.no_grad()
    def raw(batch):
        """MTR's final scores [B, M] and full trajectories [B, M, T, 7]."""
        inputs = batch["input_dict"]
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        out = model.motion_decoder(model.context_encoder(batch))
        return out["pred_scores"], out["pred_trajs"]

    def predict(batch):
        scores, trajs = raw(batch)
        return scores, trajs[..., :2]

    predict.model, predict.raw = model, raw  # for verify_unitraj.py
    return predict


def load_calibration(checkpoint) -> float:
    """The softmax temperature fitted for a checkpoint
    (scripts/responsibility/calibrate_unitraj.py), 1.0 if none."""
    import json

    path = Path(str(checkpoint) + ".calibration.json")
    return float(json.loads(path.read_text())["temperature"]) if path.exists() else 1.0
