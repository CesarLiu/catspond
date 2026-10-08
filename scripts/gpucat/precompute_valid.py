"""The fair_valid rule's per-scene inputs: which of the adversary's 40
DenseTNT samples in precompute_candidates.py's bank are valid alternatives,
and each candidate's ego avoidability over the ego's valid samples only
(gpucat/PLAN.md, "M3 之前").

  adversary  responsibility/motion_filter.py with --valid-counterfactuals'
             rules: lane_route, drivable (3 m), kinematics, and no drive
             through a third agent's logged future (the ego excepted, its
             distance is what beta measures)
  ego        drivable (3 m) and kinematics: a physically valid escape (one
             onto another route still avoids the collision)

Both depend on the scene's log only, so they are computed once per scene.
The ego's samples are drawn again from the same random stream as the bank's
(ResponsibleAdvGenerator.avoidability). Writes an .npz with stems,
sample_keep [S, 40] (bool) and avoid_valid [S, 32].

    python -m scripts.gpucat.precompute_valid --bank logs/gpucat/candidates.npz --out logs/gpucat/valid_samples.npz
"""

import argparse
import copy
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from responsibility.adversarial import GENERATION_STEP, cat_adversary, ego_avoidability  # noqa: E402
from responsibility.motion_filter import MotionFilter, MotionFilterConfig  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

TAU, RHO, HORIZON = 2.0, 0.1, 80
ADV_FILTER = MotionFilterConfig(lane_route=True, drivable_half_width=3.0, kinematics=True, collision=True)
EGO_FILTER = MotionFilterConfig(drivable_half_width=3.0, kinematics=True)


def make_generator(device: str):
    from responsibility.adversarial import ResponsibleAdvGenerator

    return ResponsibleAdvGenerator(argparse.ArgumentParser(), argv=[
        "--adv_selection", "fair", "--resp_threshold", str(TAU), "--resp_avoid", str(RHO), "--resp_device", device])


def load_scene(scenes: Path, stem: str) -> Scene:
    with open(scenes / f"{stem}.pkl", "rb") as f:
        return Scene.from_description(copy.deepcopy(pickle.load(f)))


def valid_sets(gen, scene: Scene, z, i: int, stem: str):
    """(adversary samples kept [indices], ego avoidability of the raw samples
    [K], over the valid ones [K], ego samples kept [indices])."""
    adv, ego, k = cat_adversary(scene), scene.sdc, GENERATION_STEP
    cand = z["candidates"][i].astype(np.float64)
    samples = z["samples"][i].astype(np.float64)
    ov = {"l": z["ov_size"][i][0], "w": z["ov_size"][i][1]}
    av = {"l": z["av_size"][i][0], "w": z["av_size"][i][1]}
    a_keep = MotionFilter(scene, adv, k, samples, HORIZON, [ego], ADV_FILTER).keep(ego)
    avoid_raw, ego_samples = gen.avoidability(scene, cand, ov, av, seed=int(stem))
    if ego_samples is None:  # no ego prediction: avoidability is all ones either way
        return a_keep, avoid_raw, avoid_raw, np.arange(0)
    e_keep = MotionFilter(scene, ego, k, ego_samples, HORIZON, [], EGO_FILTER).keep()
    avoid_valid, _ = ego_avoidability(ego_samples[e_keep], cand,
                                      (scene.position[ego, k, :2], float(scene.heading[ego, k])),
                                      (scene.position[adv, k, :2], float(scene.heading[adv, k])),
                                      av, ov, horizon=HORIZON)
    return a_keep, avoid_raw, avoid_valid, e_keep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="logs/gpucat/valid_samples.npz")
    args = p.parse_args(argv)
    z = np.load(REPO / args.bank, allow_pickle=True)
    stems = [str(s) for s in z["stems"]]
    gen = make_generator(args.device)
    keep = np.zeros(z["samples"].shape[:2], dtype=bool)
    avoid = np.zeros(z["avoid"].shape)
    for i, stem in enumerate(stems):
        a_keep, _, avoid[i], _ = valid_sets(gen, load_scene(REPO / args.scenes, stem), z, i, stem)
        keep[i, a_keep] = True
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(stems)}] adversary samples kept: median {np.median(keep[:i + 1].sum(1)):.0f}", flush=True)
    np.savez(REPO / args.out, stems=np.array(stems), sample_keep=keep, avoid_valid=avoid)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
