"""Checks the UniTraj MTR adapter on real scenes (server, GPU), the way
verify_densetnt.py checks the DenseTNT wrapper:

  repeat     the same input twice gives the same distribution (KL 0)
  batch      an input predicted in a batch with others gives what it gives
             alone (UniTraj pads to fixed sizes with masks; DenseTNT's
             padding leaked into predictions, which is why it is run one
             instance at a time)
  removal    masking an agent's slot gives the prediction of the scene with
             the agent's track deleted (scenes where no other agent then
             enters the model's 64 slots); a near agent moves the
             prediction more than a far one
  nms        applying MTR's NMS to the 64 intentions reproduces UniTraj's own
             6-mode output, so the adapter reads the model as UniTraj does
  speed      inputs per second: building them (CPU) and predicting (GPU)

It also prints the checkpoint's calibration (calibrate_unitraj.py) if there
is one.

Example (from the cat repository root, UniTraj environment):
    python -m scripts.responsibility.verify_unitraj --checkpoint ckpt/mtr_womd.ckpt --n 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.metrics import goal_kl  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402


class Report:
    def __init__(self):
        self.ok = True

    def check(self, name, passed, detail):
        self.ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--unitraj-root", default=None)
    p.add_argument("--unitraj-method", default="MTR_womd")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tol-logp", type=float, default=1e-3, help="max |log p| difference counted as equal")
    p.add_argument("--tol-traj", type=float, default=1e-2, help="m; max trajectory difference counted as equal")
    return p.parse_args()


def compare(a, b):
    return (float((a.log_prob - b.log_prob).abs().max()), float(np.abs(a.trajectories - b.trajectories).max()),
            goal_kl(a.log_prob, b.log_prob))


def observed_agents(scene, step, past):
    """Agents observed at some step of the model's history window."""
    lo = max(0, step - past + 1)
    return [i for i in range(scene.n_agents) if scene.valid[i, lo: step + 1].any()]


def check_scene(model, scene, step, report, args):
    inputs = model.inputs
    ego = scene.sdc
    inst = inputs.instance(scene, step, ego)
    if inst is None:
        print("  (ego not predicted here)")
        return
    # repeat
    a, b = model.distributions([inst, inst])
    dl, dt, kl = compare(a, b)
    report.check("repeat", kl < 1e-9 and dl < 1e-6, f"KL {kl:.2e}, |dlogp| {dl:.1e}, |dtraj| {dt:.1e} m")

    # batch vs alone
    others = [i for i in observed_agents(scene, step, inputs.past) if inputs.predicts(scene, i, step)][:16]
    batch = [x for x in (inputs.instance(scene, step, i) for i in others) if x is not None]
    together = model.distributions(batch)
    worst = max((compare(together[j], model.distributions([batch[j]])[0]) for j in range(len(batch))),
                key=lambda c: c[0])
    report.check("batch", worst[0] < args.tol_logp and worst[1] < args.tol_traj,
                 f"{len(batch)} inputs: worst |dlogp| {worst[0]:.1e}, |dtraj| {worst[1]:.1e} m")

    # removal: masking vs deleting, near vs far
    slots = [t for t in inst.slots if t is not None]
    near_id, far_id = slots[1] if len(slots) > 1 else None, slots[-1] if len(slots) > 2 else None
    if near_id is None:
        print("  (no other agent for the removal checks)")
        return
    near, far = scene.index(near_id), scene.index(far_id) if far_id else None
    masked_near = model.distributions([inputs.without(inst, scene, [near])])[0]
    if len(observed_agents(scene, step, inputs.past)) <= int(inputs.config["max_num_agents"]):
        deleted = model.distribution(scene.drop([near]), step, scene.drop([near]).sdc)
        dl, dt, kl = compare(masked_near, deleted)
        report.check("removal", dl < args.tol_logp and dt < args.tol_traj,
                     f"masked slot vs deleted track: |dlogp| {dl:.1e}, |dtraj| {dt:.1e} m, KL {kl:.1e}")
    else:
        print(f"  (more than {inputs.config['max_num_agents']} observed agents: deleting one lets another in; "
              f"masking vs deleting not comparable here)")
    kl_near = goal_kl(a.log_prob, masked_near.log_prob)
    if far is not None:
        kl_far = goal_kl(a.log_prob, model.distributions([inputs.without(inst, scene, [far])])[0].log_prob)
        d_near = np.linalg.norm(scene.position[near, step, :2] - scene.position[ego, step, :2])
        d_far = np.linalg.norm(scene.position[far, step, :2] - scene.position[ego, step, :2])
        report.check("removal", kl_near >= kl_far,
                     f"KL removing the nearest agent ({d_near:.0f} m) {kl_near:.4f} >= farthest in the input "
                     f"({d_far:.0f} m) {kl_far:.4f} nats")

    # NMS of our 64 = UniTraj's 6 modes
    predictor = model.predictor
    if hasattr(predictor, "raw"):
        from responsibility.unitraj import import_unitraj

        motion_utils = import_unitraj("models.mtr.motion_utils")
        decoder = predictor.model.motion_decoder
        scores64, trajs64 = predictor.raw(inputs.collate([inst]))
        decoder.num_motion_modes = 6
        try:
            scores6, trajs6 = predictor.raw(inputs.collate([inst]))
        finally:
            decoder.num_motion_modes = 64
        ours_t, ours_s, _ = motion_utils.batch_nms(pred_trajs=trajs64, pred_scores=scores64,
                                                   dist_thresh=inputs.config["MOTION_DECODER"]["NMS_DIST_THRESH"],
                                                   num_ret_modes=6)
        ds = float((ours_s - scores6).abs().max())
        dtr = float((ours_t[..., :2] - trajs6[..., :2]).abs().max())
        report.check("nms", ds < 1e-5 and dtr < 1e-4, f"|dscore| {ds:.1e}, |dtraj| {dtr:.1e} m")


def main():
    args = parse_args()
    from responsibility.unitraj import Inputs, UniTrajModel, load_calibration, load_config, mtr_predictor

    config = load_config(method=args.unitraj_method, root=args.unitraj_root)
    model = UniTrajModel(mtr_predictor(args.checkpoint, config, device=args.device, root=args.unitraj_root),
                         Inputs(config, root=args.unitraj_root), temperature=load_calibration(args.checkpoint))
    report = Report()
    files = [p for p in scene_files(args.scenes) if p.stem.isdigit()][: args.n]
    for path in files:
        scene = Scene.load(path)
        print(f"scene {path.name} ({scene.scenario_id}): {scene.n_agents} agents", flush=True)
        check_scene(model, scene, args.step, report, args)

    scene = Scene.load(files[0])
    agents = [i for i in range(scene.n_agents) if model.inputs.predicts(scene, i, args.step)][:64]
    t0 = time.time()
    insts = [model.inputs.instance(scene, args.step, i) for i in agents]
    t1 = time.time()
    model.distributions(insts)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    t2 = time.time()
    print(f"  [info] speed: building {len(insts) / (t1 - t0):.0f} inputs/s (CPU), "
          f"predicting {len(insts) / (t2 - t1):.0f} inputs/s (batch {model.batch_size})")
    calib = Path(str(args.checkpoint) + ".calibration.json")
    if calib.exists():
        c = json.loads(calib.read_text())
        print(f"  [info] calibration: T = {c['temperature']:.3f}; target-intention NLL {c['before']['nll']:.3f} -> "
              f"{c['after']['nll']:.3f}, ECE {c['before']['ece']:.3f} -> {c['after']['ece']:.3f}")
    else:
        print("  [info] no calibration file: run calibrate_unitraj.py (temperature 1 is used)")
    print("\nALL CHECKS PASSED" if report.ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
