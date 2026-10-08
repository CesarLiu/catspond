"""Model selection, the weighted, top-k and NMS motion sets and run-settings compatibility."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from responsibility.metrics import ResponsibilityConfig, responsibility_at, weighted_cvar
from responsibility.models import load_model, model_settings
from responsibility.modes import nms_select, speed_scale_factor
from responsibility.motion_filter import MotionFilterConfig
from responsibility.risk import cvar
from scripts.responsibility.compute_responsibility import same_settings
from tests.responsibility.conftest import FakeModel, make_scene, track


def test_weighted_cvar_matches_the_sample_cvar_and_its_limits():
    values = torch.tensor([3.0, -1.0, 0.5, 2.0, 0.0])
    equal = torch.full((5,), 0.2)
    assert float(weighted_cvar(values, equal, 0.0)) == pytest.approx(float(values.mean()))
    assert float(weighted_cvar(values, equal, 0.6)) == pytest.approx((3.0 + 2.0) / 2)  # upper 40% = 2 of 5
    assert float(weighted_cvar(values, equal, 0.999)) == pytest.approx(3.0, abs=1e-3)
    # unequal weights: the upper half of the mass is 3 (0.1), 2 (0.2) and 0.5 (0.2)
    weights = torch.tensor([0.1, 0.4, 0.2, 0.2, 0.1])
    assert float(weighted_cvar(values, weights, 0.5)) == pytest.approx((0.1 * 3.0 + 0.2 * 2.0 + 0.2 * 0.5) / 0.5)
    assert float(weighted_cvar(values, weights, 0.0)) == pytest.approx(float((values * weights).sum()))
    # distinct values with equal weights agree with the sample CVaR
    distinct = torch.tensor([4.0, 1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.0, -3.0, 5.0])
    assert float(weighted_cvar(distinct, torch.full((10,), 0.1), 0.8)) == pytest.approx(float(cvar(distinct, 0.8)))
    # the boundary value counts with the part of its mass inside the tail
    assert float(weighted_cvar(values, weights, 0.85)) == pytest.approx((0.1 * 3.0 + 0.05 * 2.0) / 0.15)


class WeightedModel(FakeModel):
    def __init__(self, trajs, probs, **kw):
        super().__init__(lambda n: None, **kw)
        self.trajs, self.probs = trajs, probs

    def motion_set(self, dist, top_k=None):
        if top_k is None:
            return self.trajs, self.probs
        keep = np.sort(np.argsort(-self.probs, kind="stable")[:top_k])
        return self.trajs[keep], self.probs[keep]


def test_weighted_motion_set_in_the_metric():
    scene = make_scene({"0": track((0, 0), (10, 0)), "1": track((0, 4.0), (10, 0))})  # alongside, 4 m
    t = np.arange(1, 81) * 0.1
    keep = np.stack([10 * t, np.zeros(80)], -1)  # what the ego did (from step 10)
    away = np.stack([10 * t, np.full(80, -3.0)], -1)  # 3 m further from the neighbour
    trajs = np.stack([keep, away])
    cfg = ResponsibilityConfig(motion_set="weighted", d_sat=None, cvar_alpha=0.0, courtesy=False)
    obs = responsibility_at(WeightedModel(trajs, np.array([0.25, 0.75])), scene, 0, 10, cfg)
    assert obs.safety == pytest.approx(0.75 * 3.0, abs=1e-4)  # the mean over the weighted set
    with pytest.raises(ValueError, match="finite motion set"):
        responsibility_at(FakeModel(lambda n: np.repeat(keep[None], n, 0)), scene, 0, 10, cfg)


def test_top_k_motion_set_keeps_the_most_probable_and_renormalises():
    scene = make_scene({"0": track((0, 0), (10, 0)), "1": track((0, 4.0), (10, 0))})  # alongside, 4 m
    t = np.arange(1, 81) * 0.1
    keep = np.stack([10 * t, np.zeros(80)], -1)  # what the ego did: C = 0
    away = np.stack([10 * t, np.full(80, -3.0)], -1)  # C = +3
    closer = np.stack([10 * t, np.full(80, 2.0)], -1)  # 2 m from the neighbour: C = -2
    model = WeightedModel(np.stack([keep, away, closer]), np.array([0.2, 0.5, 0.3]))
    cfg = ResponsibilityConfig(motion_set="topk", n_safety_samples=2, d_sat=None, cvar_alpha=0.0, courtesy=False)
    obs = responsibility_at(model, scene, 0, 10, cfg)
    assert obs.safety == pytest.approx((0.5 * 3.0 - 0.3 * 2.0) / 0.8, abs=1e-4)  # "keep" (0.2) is dropped
    cfg.n_safety_samples = 40  # more than the set: all of it, as "weighted"
    whole = responsibility_at(model, scene, 0, 10, ResponsibilityConfig(motion_set="weighted", d_sat=None,
                                                                         cvar_alpha=0.0, courtesy=False))
    assert responsibility_at(model, scene, 0, 10, cfg).safety == pytest.approx(whole.safety)



def test_nms_select_spreads_fills_and_weights_by_nearest_mass():
    points = np.array([[0.0, 0], [1, 0], [10, 0], [11, 0], [30, 0], [50, 0]])
    probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1, 0.0])
    idx, w = nms_select(points, probs, 3, threshold=5.0)
    assert idx.tolist() == [0, 2, 4]  # 1 and 3 lie within 5 m of a picked goal
    assert w == pytest.approx([0.55, 0.35, 0.1])  # each goal's mass goes to its nearest pick
    idx, w = nms_select(points, probs, 5, threshold=5.0)  # three survive: fill with the most probable left
    assert idx.tolist() == [0, 2, 4, 1, 3] and w == pytest.approx(probs[idx])  # weights in pick order
    idx, _ = nms_select(points, probs, 10, threshold=5.0)
    assert 5 not in idx.tolist() and len(idx) == 5  # zero-probability goals are never picked


def test_speed_scale_factor_matches_cat():
    utils_cython = pytest.importorskip("advgen.utils_cython")
    for speed in [0.0, 1.4, 3.0, 7.5, 11.0, 20.0]:
        assert speed_scale_factor(speed) == pytest.approx(utils_cython.speed_scale_factor(speed), abs=1e-6)


class NMSModel(WeightedModel):
    def nms_motion_set(self, dist, n):
        idx, weights = nms_select(self.trajs[:, -1], self.probs, n, threshold=2.0)
        return self.trajs[idx], weights


def test_nms_motion_set_in_the_metric():
    scene = make_scene({"0": track((0, 0), (10, 0)), "1": track((0, 4.0), (10, 0))})  # alongside, 4 m
    t = np.arange(1, 81) * 0.1
    keep = np.stack([10 * t, np.zeros(80)], -1)  # what the ego did: C = 0
    nearby = np.stack([10 * t, np.full(80, 0.5)], -1)  # 0.5 m toward the neighbour: C = -0.5
    away = np.stack([10 * t, np.full(80, -3.0)], -1)  # C = +3
    model = NMSModel(np.stack([keep, nearby, away]), np.array([0.4, 0.35, 0.25]))
    cfg = ResponsibilityConfig(motion_set="nms", n_safety_samples=2, d_sat=None, cvar_alpha=0.0, courtesy=False)
    # "nearby" is within 2 m of "keep": the set is {keep, away}, keep carrying nearby's mass
    assert responsibility_at(model, scene, 0, 10, cfg).safety == pytest.approx(0.75 * 0.0 + 0.25 * 3.0, abs=1e-4)
    top = ResponsibilityConfig(motion_set="topk", n_safety_samples=2, d_sat=None, cvar_alpha=0.0, courtesy=False)
    assert responsibility_at(model, scene, 0, 10, top).safety == pytest.approx(-0.35 * 0.5 / 0.75, abs=1e-4)
    with pytest.raises(ValueError, match="NMS motion set"):
        responsibility_at(WeightedModel(model.trajs, model.probs), scene, 0, 10, cfg)

def test_runs_made_before_model_choice_still_resume():
    off = asdict(MotionFilterConfig())
    settings = {"responsibility": {"n_safety_samples": 40, "motion_set": "sampled", "filter": off,
                                   "courtesy_valid_goals": False, "use_ooi": False}, "agent": "sdc",
                "scenes": "/x", "model": {"name": "densetnt"}}
    old = {"responsibility": {"n_safety_samples": 40}, "agent": "sdc", "scenes": "/x"}
    assert same_settings(json.loads(json.dumps(old)), settings)
    filtered = dict(settings, responsibility=dict(settings["responsibility"], filter=dict(off, route_tolerance=2.0)))
    assert not same_settings(old, filtered)
    mtr = dict(settings, model={"name": "mtr", "checkpoint": "/c.ckpt", "method": "MTR_womd", "temperature": 1.0})
    assert not same_settings(old, mtr)
    assert not same_settings(old, dict(settings, responsibility={"n_safety_samples": 40, "motion_set": "weighted",
                                                                 "filter": off, "courtesy_valid_goals": False,
                                                                 "use_ooi": False}))
    assert not same_settings(old, dict(settings, responsibility=dict(settings["responsibility"], use_ooi=True)))


def test_model_options():
    args = SimpleNamespace(model="densetnt", checkpoint=None, unitraj_root=None, unitraj_method=None, temperature=None)
    assert model_settings(args) == {"name": "densetnt"}
    mtr = SimpleNamespace(**{**vars(args), "model": "mtr", "temperature": 1.5})
    assert model_settings(mtr)["temperature"] == 1.5 and model_settings(mtr)["method"] == "MTR_womd"
    with pytest.raises(SystemExit, match="checkpoint"):
        load_model(mtr, "cpu")
