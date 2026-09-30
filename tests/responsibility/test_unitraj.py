"""The UniTraj adapter's inputs (UniTraj's own preprocessing, no model) and
its model interface over a stand-in predictor."""

import numpy as np
import pytest
import torch

from responsibility.metrics import ResponsibilityConfig, scene_responsibility
from responsibility.records import run_scene
from responsibility.rollouts import make_rollout, scene_from_rollout
from responsibility.scene import Scene
from tests.responsibility.conftest import make_scene, track

unitraj = pytest.importorskip("responsibility.unitraj")
try:
    INPUTS = unitraj.Inputs()
except ImportError as e:  # no UniTraj checkout
    pytest.skip(f"UniTraj not available: {e}", allow_module_level=True)

K = 64


def _scene():
    return make_scene({
        "0": track((0, 0), (10, 0)),  # ego
        "1": track((20, 3.5), (8, 0)),  # vehicle ahead in the next lane
        "2": track((-15, 0), (9, 0)),  # follower
        "3": track((300, 300), (0, 0)),  # far away
        "4": track((5, 8), (1, 0), kind="PEDESTRIAN"),
    })


def fake_predictor(batch):
    """Scores that depend on every observed agent (so masking one changes
    them) and straight trajectories along the centre agent's heading."""
    d = batch["input_dict"]
    pos = d["obj_trajs_last_pos"][..., :2]  # [B, A, 2]
    valid = d["obj_trajs_mask"].any(-1)  # [B, A]
    near = (torch.exp(-pos.norm(dim=-1) / 20.0) * valid).sum(-1)  # [B]
    k = torch.arange(K, dtype=torch.float32)
    logits = -((k[None] - 20.0 * near[:, None]) ** 2) / 50.0
    t = torch.arange(1, 81, dtype=torch.float32) * 0.1
    x = (k[:, None] / 4.0) * t[None]
    y = ((k[:, None] - 32) / 16.0) * t[None]
    trajs = torch.stack([x, y], -1)[None].expand(len(near), -1, -1, -1)
    return torch.softmax(logits, -1), trajs


def _model(**kw):
    return unitraj.UniTrajModel(fake_predictor, INPUTS, **kw)


def _global(inst, points):
    return unitraj.to_global(np.asarray(points, dtype=np.float64), inst.origin, inst.yaw)


def test_scene_description_round_trip():
    scene = _scene()
    back = Scene.from_description(scene.to_description())
    for key in ("position", "heading", "velocity", "length", "width", "height", "valid"):
        np.testing.assert_array_equal(getattr(back, key), getattr(scene, key))
    assert back.track_ids == scene.track_ids and back.sdc == scene.sdc
    assert back.objects_of_interest == scene.objects_of_interest


@pytest.mark.parametrize("step", [10, 40, 70])
def test_window_aligns_history_and_future_with_the_scene(step):
    scene = _scene()
    inst = INPUTS.instance(scene, step, scene.sdc)
    s = inst.sample
    assert s["obj_trajs"].shape[1:] == (INPUTS.past, 29)
    slot = int(s["track_index_to_predict"])
    assert inst.slots[slot] == "0"
    history = _global(inst, s["obj_trajs_pos"][slot, :, :2])
    np.testing.assert_allclose(history, scene.position[0, step - 10: step + 1, :2], atol=1e-4)
    # the future inside the clip, invalid beyond it (not shifted in time)
    n = min(80, scene.n_steps - 1 - step)
    mask = s["center_gt_trajs_mask"].astype(bool)
    assert mask[:n].all() and not mask[n:].any()
    np.testing.assert_allclose(_global(inst, s["center_gt_trajs"][:n, :2]),
                               scene.position[0, step + 1: step + 1 + n, :2], atol=1e-4)
    # the centre agent's pose is the scene's at the step
    np.testing.assert_allclose(inst.origin, scene.position[0, step, :2], atol=1e-5)


def test_slots_name_the_tracks_they_hold():
    scene = _scene()
    inst = INPUTS.instance(scene, 30, scene.sdc)
    held = [t for t in inst.slots if t is not None]
    assert held[0] == "0" and set(held) == {"0", "1", "2", "3", "4"}
    for j, tid in enumerate(inst.slots[: len(held)]):
        a = scene.index(tid)
        np.testing.assert_allclose(_global(inst, inst.sample["obj_trajs_last_pos"][j, :2]),
                                   scene.position[a, 30, :2], atol=1e-3)


def test_removal_masks_one_slot_and_nothing_else():
    scene = _scene()
    inst = INPUTS.instance(scene, 30, scene.sdc)
    without = INPUTS.without(inst, scene, [scene.index("1")])
    s = inst.slots.index("1")
    assert not without.sample["obj_trajs_mask"][s].any() and inst.sample["obj_trajs_mask"][s].any()
    others = [j for j in range(len(inst.slots)) if j != s]
    for key in ("obj_trajs", "obj_trajs_mask", "obj_trajs_last_pos"):
        np.testing.assert_array_equal(without.sample[key][others], inst.sample[key][others])
    np.testing.assert_array_equal(without.sample["map_polylines"], inst.sample["map_polylines"])
    assert without.excluded == (scene.index("1"),)
    with pytest.raises(ValueError):
        INPUTS.without(inst, scene, [scene.sdc])


def test_which_agents_are_predicted():
    scene = _scene()
    assert INPUTS.instance(scene, 30, scene.index("4")) is not None  # pedestrians too (unlike DenseTNT)
    valid = np.ones(91, dtype=bool)
    valid[25:] = False
    gone = make_scene({"0": track((0, 0), (10, 0)), "1": track((20, 3.5), (8, 0), valid=valid)})
    assert INPUTS.instance(gone, 30, 1) is None


def test_distributions_samples_and_counterfactuals():
    scene = _scene()
    model = _model()
    dist = model.distribution(scene, 30, scene.sdc)
    assert dist.log_prob.shape == (K,) and float(dist.log_prob.exp().sum()) == pytest.approx(1.0, abs=1e-5)
    assert dist.trajectories.shape == (K, 80, 2)
    idx, lp, trajs = model.sample(dist, 40, generator=torch.Generator().manual_seed(0))
    assert trajs.shape == (40, 80, 2)
    np.testing.assert_allclose(trajs, dist.trajectories_global()[idx.numpy()])
    # removing a near agent moves the prediction, removing a far one hardly, an absent one not at all
    from responsibility.metrics import goal_kl

    near = goal_kl(*[d.log_prob for d in model.with_and_without(scene, 30, scene.sdc, scene.index("2"))])
    far = goal_kl(*[d.log_prob for d in model.with_and_without(scene, 30, scene.sdc, scene.index("3"))])
    assert near > 1e-3 and far < near / 10
    # the motion set and a flatter temperature
    trajs_all, probs = model.motion_set(dist)
    assert trajs_all.shape == (K, 80, 2) and probs.sum() == pytest.approx(1.0)
    hot = _model(temperature=3.0).distribution(scene, 30, scene.sdc)
    entropy = lambda d: float(-(d.log_prob.exp() * d.log_prob).sum())  # noqa: E731
    assert entropy(hot) > entropy(dist)


def test_the_metrics_and_records_run_on_the_adapter():
    scene = _scene()
    cfg = ResponsibilityConfig(window_stride=20)
    obs = scene_responsibility(_model(), scene, scene.sdc, cfg)
    assert [o.step for o in obs] == [10, 30, 50, 70]
    assert any(o.per_neighbour for o in obs)
    obs2, record = run_scene(_model(), scene, scene.sdc, cfg)
    assert [(o.safety, o.courtesy) for o in obs2] == [(o.safety, o.courtesy) for o in obs]
    assert record["frames"][0]["goals"]["points"].shape[1] == 2


def test_rollout_scenes_build_inputs():
    scene = _scene()
    n = 41
    rollout = make_rollout("7", scene.scenario_id, "td3", "none",
                           ego={"position": scene.position[0, :n, :2] + np.array([0.0, 1.0]),
                                "heading": scene.heading[0, :n]},
                           end={"step": n - 1, "reason": "out_of_road", "out_of_road": True})
    played = scene_from_rollout(scene, rollout)
    inst = INPUTS.instance(played, 30, played.sdc)
    slot = int(inst.sample["track_index_to_predict"])
    np.testing.assert_allclose(_global(inst, inst.sample["obj_trajs_pos"][slot, -1, :2]),
                               scene.position[0, 30, :2] + np.array([0.0, 1.0]), atol=1e-4)
    mask = inst.sample["center_gt_trajs_mask"].astype(bool)
    assert mask[:10].all() and not mask[10:].any()  # the episode ended at step 40


def test_the_weighted_motion_set_runs_on_the_adapter():
    scene = _scene()
    sampled = scene_responsibility(_model(), scene, scene.sdc, ResponsibilityConfig(window_stride=20))
    exact = scene_responsibility(_model(), scene, scene.sdc,
                                 ResponsibilityConfig(window_stride=20, motion_set="weighted"))
    assert [o.step for o in exact] == [o.step for o in sampled]
    assert [o.courtesy for o in exact] == [o.courtesy for o in sampled]  # courtesy does not use the motion set
    _, record = run_scene(_model(), scene, scene.sdc, ResponsibilityConfig(window_stride=20, motion_set="weighted"))
    frame = next(f for f in record["frames"] if f["metric_samples"])
    assert frame["samples"].shape == (K, 80, 2)
