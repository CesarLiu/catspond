"""Checks export_adv_scenes.py against CAT's own generation in MetaDrive:
runs cat_advgen.py's two rounds (the logged ego replayed, then CAT's
AdvGenerator.generate and the adversarial episode) on chosen scenes and
compares, scene by scene, the adversary plan MetaDrive's run produces with
the one exported offline (OUT/<scene>.pkl, the adversary's track from step
11 on).

Reported per scene: the largest position difference between the two plans
over steps 11-90 (below 5 cm: the same candidate; GPU arithmetic differs by millimetres), the vehicle sizes CAT's generator
used in MetaDrive and offline (the log's at step 10), whether the
adversarial episode in MetaDrive ended in a collision of the ego, and the
step the offline plan first overlaps the logged ego. As in cat_advgen.py,
the first round's after_episode() does not update the ego's stored
trajectory, so both sides score the candidates against the logged ego
route.

Needs MetaDrive (the modified package CAT ships) and a GPU (AdvGenerator
puts DenseTNT on cuda:0). Example (from the repository root):
    python -m scripts.responsibility.verify_adv_export --scenes 0 10 20 --offline adv_scenes/cat \\
        --out adv_scenes/verify_cat.csv
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.scene import Scene, sidecar_index  # noqa: E402


def offline_plan(path: Path, adv_id: str) -> np.ndarray:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["tracks"][adv_id]["state"]["position"])[:, :2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, nargs="+", required=True, help="Scene indices (file stems).")
    parser.add_argument("--offline", default="adv_scenes/cat", help="export_adv_scenes.py's output for CAT's rule.")
    parser.add_argument("--data", default="raw_scenes_500")
    parser.add_argument("--out", default=None, help="CSV, one row per scene.")
    parser.add_argument("--plans", default=None, help="Also save MetaDrive's plans [91, 5] and sizes (.npz).")
    parser.add_argument("--OV_traj_num", type=int, default=32)  # as cat_advgen.py
    parser.add_argument("--AV_traj_num", type=int, default=1)
    from metadrive.envs.real_data_envs.waymo_env import WaymoEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy

    from responsibility.adversarial import make_adv_generator

    gen = make_adv_generator(parser)  # CAT's AdvGenerator (--adv_selection cat, the default)
    args = parser.parse_args()
    env = WaymoEnv({"agent_policy": ReplayEgoCarPolicy, "reactive_traffic": False, "use_render": False,
                    "data_directory": args.data, "num_scenarios": 500, "force_reuse_object_name": True,
                    "sequential_seed": True,
                    "vehicle_config": dict(show_navi_mark=False, show_dest_mark=False)})
    index_path = sidecar_index(args.offline)
    if not index_path.exists():  # an export made before the index moved out of the folder
        index_path = Path(args.offline) / "index.json"
    index = json.loads(index_path.read_text())
    rows, plans, sizes = [], {}, {}
    try:
        for i in args.scenes:
            # first round: the logged scenario, the ego replayed (cat_advgen.py)
            env.reset(force_seed=i)
            gen.before_episode(env)
            while True:
                gen.log_AV_history()
                _, _, done, _ = env.step([1.0, 0.0])
                if done:
                    gen.after_episode()
                    break
            # second round: CAT's generation and the adversarial episode
            env.reset(force_seed=i)
            env.vehicle.ego_crash_flag = False
            gen.before_episode(env)
            gen.generate()
            plan = np.array(gen.adv_traj, dtype=float)[:, :2]
            st = gen.storage[env.current_seed]
            env.engine.traffic_manager.set_adv_info(gen.adv_agent, gen.adv_traj)
            crash, steps = False, 0
            while True:
                _, _, done, _ = env.step([1.0, 0.0])
                steps += 1
                crash = bool(env.vehicle.ego_crash_flag)
                if done or crash:
                    gen.after_episode()
                    break
            adv_id = str(gen.adv_agent)
            scene = Scene.load(Path(args.data) / f"{i}.pkl")
            off = offline_plan(Path(args.offline) / f"{i}.pkl", adv_id)
            diff = float(np.abs(plan[11:91] - off[11:91]).max())
            log_adv = scene.shape_at(scene.index(adv_id), 10)
            log_ego = scene.shape_at(scene.sdc, 10)
            row = {"scene": i, "adversary": adv_id, "same_adversary": adv_id == index[str(i)]["adversary"],
                   "max_plan_diff_m": round(diff, 4), "same_plan": diff < 0.05,
                   "md_adv_lw": f"{st['adv_info']['l']:.2f}x{st['adv_info']['w']:.2f}",
                   "log_adv_lw": f"{log_adv[0]:.2f}x{log_adv[1]:.2f}",
                   "md_ego_lw": f"{st['ego_info']['l']:.2f}x{st['ego_info']['w']:.2f}",
                   "log_ego_lw": f"{log_ego[0]:.2f}x{log_ego[1]:.2f}",
                   "md_crash": crash, "md_end_step": steps,
                   "offline_overlap_step": index[str(i)]["first_overlap_with_logged_ego"]}
            rows.append(row)
            plans[str(i)] = np.array(gen.adv_traj, dtype=float)
            sizes[str(i)] = np.array([st["adv_info"]["l"], st["adv_info"]["w"], st["ego_info"]["l"], st["ego_info"]["w"]])
            print(row, flush=True)
    finally:
        env.close()
    same = sum(r["same_plan"] for r in rows)
    md_hit = sum(r["md_crash"] for r in rows)
    off_hit = sum(r["offline_overlap_step"] is not None for r in rows)
    print(f"{len(rows)} scenes: same plan in {same}; collision of the ego in MetaDrive {md_hit}, "
          f"offline overlap with the logged ego {off_hit}")
    if args.plans:
        np.savez(args.plans, **{f"plan_{k}": v for k, v in plans.items()}, **{f"size_{k}": v for k, v in sizes.items()})
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
