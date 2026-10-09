"""Synthetic scenes and a stand-in motion model, so the responsibility
metrics can be tested without TensorFlow, the checkpoint or scene files."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.scene import Scene  # noqa: E402

N_STEPS = 91


def track(xy0, vel, heading=0.0, kind="VEHICLE", valid=None, length=4.8, width=2.0):
    """A constant-velocity track over the 91-step clip; xy0 is its position at step 10."""
    t = (np.arange(N_STEPS) - 10) * 0.1
    pos = np.zeros((N_STEPS, 3), dtype=np.float32)
    pos[:, 0] = xy0[0] + vel[0] * t
    pos[:, 1] = xy0[1] + vel[1] * t
    return {
        "type": kind,
        "state": {
            "position": pos,
            "heading": np.full(N_STEPS, heading, dtype=np.float32),
            "velocity": np.tile(np.asarray(vel, dtype=np.float32), (N_STEPS, 1)),
            "length": np.full(N_STEPS, length, dtype=np.float32),
            "width": np.full(N_STEPS, width, dtype=np.float32),
            "height": np.full(N_STEPS, 1.5, dtype=np.float32),
            "valid": np.ones(N_STEPS, dtype=bool) if valid is None else np.asarray(valid, dtype=bool),
        },
        "metadata": {},
    }


def make_scene(tracks, sdc="0", ooi=("0", "1"), map_features=None, lights=None):
    description = {
        "id": "test",
        "tracks": {str(k): v for k, v in tracks.items()},
        "map_features": map_features or {
            "100": {"type": "LANE_SURFACE_STREET", "polyline": np.array([[0.0, 0, 0], [50, 0, 0], [100, 0, 0]])},
            "101": {"type": "STOP_SIGN", "position": np.array([5.0, 5.0, 0.0]), "lane": ["100"]},
        },
        "dynamic_map_states": lights if lights is not None else {
            "200": {"type": "TRAFFIC_LIGHT", "lane": "100", "stop_point": np.array([20.0, 0.0, 0.0]),
                    "state": {"object_state": ["LANE_STATE_GO"] * 50 + ["LANE_STATE_STOP"] * 41}},
        },
        "metadata": {"sdc_id": sdc, "objects_of_interest": list(ooi), "scenario_id": "synthetic"},
    }
    return Scene.from_description(description)


class FakeModel:
    """Duck-typed stand-in for DenseTNT: ``samples(n)`` gives the queried
    agent's motion set [n, 80, 2] (``samples_by_agent[i](n)`` agent i's, if
    given); ``courtesy_logits[b]`` gives b's goal
    logits (with, without the queried agent); agents not listed there are
    'not predicted' (as DenseTNT does for pedestrians)."""

    def __init__(self, samples, courtesy_logits=None, predicted=None, samples_by_agent=None):
        self.samples = samples
        self.samples_by_agent = samples_by_agent or {}
        self.courtesy_logits = courtesy_logits or {}
        self.predicted = predicted
        self.calls = []

    def distribution(self, scene, step, agent, excluded=(), avoid_partner=()):
        if self.predicted is not None and agent not in self.predicted:
            return None
        return SimpleNamespace(log_prob=torch.log_softmax(torch.zeros(4), -1), agent=agent)

    def sample(self, dist, n, generator=None):
        return None, None, self.samples_by_agent.get(dist.agent, self.samples)(n)

    def with_and_without(self, scene, step, b, a):
        self.calls.append((step, b, a))
        if b not in self.courtesy_logits:
            return None
        p, q = self.courtesy_logits[b]
        return (SimpleNamespace(log_prob=torch.log_softmax(torch.tensor(p, dtype=torch.float), -1)),
                SimpleNamespace(log_prob=torch.log_softmax(torch.tensor(q, dtype=torch.float), -1)))
