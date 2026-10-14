"""How stable the evaluation's crash verdicts (crashes.csv of
compute_responsibility --rollouts) are to choices that should not matter:

  repro     the stored verdict recomputed exactly (seed 0, 40 sampled
            alternatives): checks that the stored rows are what the code gives
  seed1/2   the same with ResponsibilityConfig.seed 1 and 2 (other 40 samples)
  nms       --motion-set nms (40 goals spread by CAT's NMS, probability-weighted;
            the README's recommended set)
  footprint the partner replaced by the agent whose footprint is closest at the
            crash step, where that differs from the recorded centre-nearest one

It reads logs/responsibility/eval/<run>/crashes.csv, the rollouts in
rollouts_eval/<run>/ and the scenes in raw_scenes_500/, and appends one row per
(run, crash, variant) to OUT.csv with the stored and the recomputed verdict
and beta_s of both sides; a rerun skips the rows already in OUT.csv. Part of
the collision-attribution audit of 2026-10-09 (A4), which matters because the
0.1 m verdict margin is close to the Monte Carlo noise of 40 samples. On
td3_cat_fairinf_0.5_s0/none (17 crashes, 21 s on an L4) the stored verdicts
reproduce exactly (beta_ego differs by 0.0 m), and seeds 1 and 2 and the NMS
set change beta_ego by at most 0.28 m and no verdict.

Runs DenseTNT (about 1-2 GB of GPU memory). Imports the responsibility package
from the directory given by --code (default: this repository, which has the
NMS motion set since the merge e1e89fd).

usage (repo root, cat39 env):
  python -m scripts.gpucat.attribution_sensitivity --runs td3_cat_s0/none td3_cat_s0/cat ... --out OUT.csv
"""

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", default=str(REPO))
    p.add_argument("--runs", nargs="+", required=True, help="<policy>/<adv_mode> under logs/responsibility/eval")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--variants", nargs="+", default=["repro", "seed1", "seed2", "nms", "footprint"])
    args = p.parse_args()
    sys.path.insert(0, args.code)
    import torch  # noqa: F401
    from responsibility.blame import crash_blame, crash_partner, rollout_collision
    from responsibility.densetnt import DenseTNT
    from responsibility.metrics import ResponsibilityConfig
    from responsibility.rollouts import load_rollout, scene_from_rollout
    from responsibility.scene import Scene

    model = DenseTNT(checkpoint=str(REPO / "advgen/pretrained/densetnt.bin"), device=args.device)
    base = ResponsibilityConfig()
    cfgs = {"repro": base, "seed1": replace(base, seed=1), "seed2": replace(base, seed=2),
            "nms": replace(base, motion_set="nms")}
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {(r["run"], r["scene"], r["variant"]) for r in csv.DictReader(open(out))}
    fields = ["run", "scene", "variant", "crash_step", "other_id", "stored_verdict", "verdict", "beta_ego",
              "beta_other", "share", "stored_beta_ego", "stored_beta_other"]
    new = not out.exists()
    scenes = {}
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for run in args.runs:
            rdir = REPO / "logs/responsibility/eval" / run
            rows = list(csv.DictReader(open(rdir / "crashes.csv")))
            t0 = time.time()
            for row in rows:
                stem = row["scene"]
                if stem not in scenes:
                    scenes[stem] = Scene.load(REPO / "raw_scenes_500" / f"{stem}.pkl")
                rollout = load_rollout(REPO / "rollouts_eval" / run / f"{stem}.pkl")
                scene = scene_from_rollout(scenes[stem], rollout)
                col = rollout_collision(scene, rollout)
                if col is None:
                    continue
                step, other = col
                for v in args.variants:
                    if (run, stem, v) in done:
                        continue
                    o, cfg = other, cfgs.get(v, base)
                    if v == "footprint":
                        o = crash_partner(scene, scene.sdc, step)
                        if o is None or o == other:
                            continue
                    b = crash_blame(model, scene, scene.sdc, step, o, cfg)
                    w.writerow({"run": run, "scene": stem, "variant": v, "crash_step": step,
                                "other_id": scene.track_ids[o], "stored_verdict": row["verdict"],
                                "verdict": "unknown" if b is None else b.verdict,
                                "beta_ego": "" if b is None else b.beta_ego,
                                "beta_other": "" if b is None or b.beta_other is None else b.beta_other,
                                "share": "" if b is None or b.share is None else b.share,
                                "stored_beta_ego": row["beta_ego"], "stored_beta_other": row["beta_other"]})
                f.flush()
            print(f"{run}: {len(rows)} crashes, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
