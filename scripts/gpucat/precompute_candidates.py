"""Precomputes, once per scene, what CAT's adversary selection takes from
DenseTNT (gpucat/adversary.py): the adversary's 32 candidate futures and
their NMS scores, its logged history (steps 0-10), the vehicle sizes (the
log's at step 10, as export_adv_scenes.py), the SDC's logged route (steps
11-90, CAT's initial ego trajectory), and for the fair rule the adversary's
motion-set samples and each candidate's ego avoidability. All of it depends
on the scene's log only (the samples are seeded by the scene index, as
ResponsibleAdvGenerator.generate seeds them), so it equals what CAT's
per-episode inference would produce.

The choices of the numpy implementation against the logged route (rules cat
and fair, with the logged route as the only ego trajectory) are stored too,
for verify_selection.py. Writes one .npz.

Runs in this repository's environment (DenseTNT, TensorFlow), on the GPU:
    python -m scripts.gpucat.precompute_candidates --out logs/gpucat/candidates.npz
"""

import argparse
import copy
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.adversarial import ResponsibleAdvGenerator, select  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402
from scripts.responsibility.benchmark_advgen import cat_storage  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--resp_threshold", default="2", help="tau of the fair rule recorded for verification.")
    p.add_argument("--resp_avoid", default="0.1", help="rho of the fair rule recorded for verification.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    cat_parser = argparse.ArgumentParser()
    cat_parser.add_argument("--OV_traj_num", type=int, default=32)
    cat_parser.add_argument("--AV_traj_num", type=int, default=1)
    gen = ResponsibleAdvGenerator(cat_parser, argv=["--adv_selection", "fair", "--resp_threshold", args.resp_threshold,
                                                    "--resp_avoid", args.resp_avoid, "--resp_device", args.device])
    files = scene_files(args.scenes)[args.first:]
    files = files[: args.n] if args.n is not None else files
    keys = ("candidates", "log_scores", "adv_past", "ov_size", "av_size", "ego_route", "samples", "avoid",
            "score", "min_dist", "beta", "chosen_cat", "chosen_fair")
    rows = {k: [] for k in keys}
    stems, adv_ids = [], []
    for n, path in enumerate(files, 1):
        with open(path, "rb") as f:
            description = pickle.load(f)
        scene = Scene.from_description(copy.deepcopy(description))
        st = cat_storage(description, scene)
        route = np.asarray(st["AV_trajs"][0], dtype=np.float64)
        out = gen.choose(scene, [route], [1.0], st["adv_info"], st["ego_info"], seed=int(path.stem), rules=["cat"])
        rows["candidates"].append(np.asarray(out["candidates"], dtype=np.float32))
        rows["log_scores"].append(np.asarray(out["log_scores"], dtype=np.float64))
        rows["adv_past"].append(np.asarray(st["adv_past"], dtype=np.float64))
        rows["ov_size"].append([st["adv_info"]["l"], st["adv_info"]["w"]])
        rows["av_size"].append([st["ego_info"]["l"], st["ego_info"]["w"]])
        rows["ego_route"].append(route)
        s = out["samples"]
        rows["samples"].append(np.asarray(s.detach().cpu().numpy() if torch.is_tensor(s) else s, dtype=np.float32))
        rows["avoid"].append(np.asarray(out["avoid"], dtype=np.float64))
        rows["score"].append(out["score"])
        rows["min_dist"].append(np.asarray(out["min_dist"], dtype=np.float64))
        rows["beta"].append(out["beta"])
        rows["chosen_cat"].append(out["chosen_cat"])
        rows["chosen_fair"].append(out["chosen"])
        stems.append(path.stem)
        adv_ids.append(str(st["adv_agent"]))
        print(f"[{n}/{len(files)}] {path.stem}: cat -> {out['chosen_cat']}, fair -> {out['chosen']} ({out['why']})",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, stems=np.array(stems), adv_id=np.array(adv_ids),
             **{k: np.asarray(v) for k, v in rows.items()})
    print(f"wrote {len(stems)} scenes to {args.out}")


if __name__ == "__main__":
    main()
