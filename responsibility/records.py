"""Per-scene records of a responsibility run: everything needed to inspect or
visualise it offline, without DenseTNT, TensorFlow or the scene files.

A record (one pickle per scene and agent) holds

  scene       the Scene's arrays and map (Scene(**record["scene"]) rebuilds it)
  agent       index and track id of the queried agent, and the run's config
  frames      one per context step t_k:
                step, observation (the values, per neighbour included)
                samples [N, 80, 2] and sample_log_prob [N]: the agent's motion set
                goals: the agent's goal distribution
                courtesy: per vehicle neighbour, its goal distribution with and
                          without the agent, and their KL

Goal distributions are stored sparsely -- the most probable goals covering
``top_mass`` of the probability, in scene coordinates -- since DenseTNT's grid
has 50176 goals. In windows without neighbours the metric draws no samples;
a small display-only motion set is then drawn from a generator of its own, so
the metric's random stream (and every later value) stays as in a run without
records.
"""

import os
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from responsibility.metrics import Observation, ResponsibilityConfig, responsibility_at, window_steps
from responsibility.scene import Scene

VERSION = 1
DISPLAY_SAMPLES = 24


def sparse_goals(dist, top_mass: float = 0.99, max_points: int = 6000) -> Dict[str, np.ndarray]:
    """The most probable goals covering ``top_mass`` (at most ``max_points``),
    in the scene frame, with their probabilities."""
    p = dist.log_prob.exp().detach().cpu().numpy()
    order = np.argsort(-p)
    n = min(int(np.searchsorted(np.cumsum(p[order]), top_mass)) + 1, max_points, len(p))
    keep = order[:n]
    return {"points": dist.to_global(dist.goals[keep]).astype(np.float32),
            "prob": p[keep].astype(np.float32), "mass": float(p[keep].sum())}


def _numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def capture_frame(model, scene: Scene, agent: int, step: int, cfg: ResponsibilityConfig,
                  generator: Optional[torch.Generator] = None, top_mass: float = 0.99
                  ) -> Tuple[Optional[Observation], Optional[Dict]]:
    """The observation at ``step`` and its record frame (None, None when
    DenseTNT has no prediction for the agent there)."""
    record: Dict = {}
    obs = responsibility_at(model, scene, agent, step, cfg, generator, record=record)
    if obs is None:
        return None, None
    samples, log_prob = record["samples"], record["sample_log_prob"]
    if samples is None:  # no neighbours: display-only samples, off the metric's random stream
        display = torch.Generator().manual_seed(cfg.seed * 100003 + step)
        _, log_prob, samples = model.sample(record["distribution"], DISPLAY_SAMPLES, generator=display)
    courtesy = {}
    for b, (with_a, without_a) in record["courtesy"].items():
        tid = scene.track_ids[b]
        courtesy[tid] = {"with": sparse_goals(with_a, top_mass), "without": sparse_goals(without_a, top_mass),
                         "kl": obs.per_neighbour[tid]["courtesy"]}
    frame = {
        "step": step,
        "observation": asdict(obs),
        "samples": _numpy(samples),
        "sample_log_prob": _numpy(log_prob),
        "metric_samples": record["samples"] is not None,
        "goals": sparse_goals(record["distribution"], top_mass),
        "courtesy": courtesy,
        # with a motion filter: per neighbour track id, the samples beta_s used
        "kept": {scene.track_ids[b]: np.asarray(idx) for b, idx in record.get("kept", {}).items()},
    }
    return obs, frame


def run_scene(model, scene: Scene, agent: int, cfg: ResponsibilityConfig,
              top_mass: float = 0.99, last_step: Optional[int] = None) -> Tuple[List[Observation], Dict]:
    """Every window of the scene, as ``metrics.scene_responsibility`` computes
    them (same random stream, same values), plus the scene's record."""
    generator = torch.Generator().manual_seed(cfg.seed)
    observations, frames = [], []
    for step in window_steps(scene, agent, cfg, last_step):
        obs, frame = capture_frame(model, scene, agent, step, cfg, generator, top_mass)
        if obs is not None:
            observations.append(obs)
            frames.append(frame)
    record = {
        "version": VERSION,
        "scene": dict(scene.__dict__),
        "agent": agent,
        "agent_id": scene.track_ids[agent],
        "config": asdict(cfg),
        "top_mass": top_mass,
        "frames": frames,
    }
    return observations, record


def scene_of(record: Dict) -> Scene:
    return Scene(**record["scene"])


def save_record(record: Dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(record, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_record(path) -> Dict:
    with open(path, "rb") as f:
        record = pickle.load(f)
    if record.get("version") != VERSION:
        raise ValueError(f"{path}: record version {record.get('version')}, expected {VERSION}")
    return record
