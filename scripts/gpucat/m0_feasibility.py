"""M0 of gpucat/PLAN.md: can GPUDrive run CAT's setting? Checks, on CAT's
test scenes exported by export_scenes.py (only the SDC controlled):

  control    every world controls exactly one agent, the SDC (by track id)
  injection  CAT's adversary plan (adv_scenes/cat/<scene>.pkl, written by
             export_adv_scenes.py) written into the simulator's trajectory
             tensor at run time is replayed: the adversary's position after
             each step against the plan, at the best of a 0- or 1-step lag
  ego replay the SDC driven by GPUDrive's inferred expert actions
             (delta_local) against its log
  collision  the first step GPUDrive flags the SDC colliding with a vehicle,
             against the offline first overlap of the plan with the logged
             SDC (index.json of export_adv_scenes.py)
  reset      resetting half the worlds keeps their injected plans and puts
             their adversaries back at the plan's first row, and leaves the
             other worlds where they were

and, with --bench W, the throughput of W worlds with a PPO-sized network
(actor and critic, two 256-unit layers) evaluated on every step's
observations: world steps per second, and GPU memory.

Run with GPUDrive's environment, from this repository's root:
    PYTHONPATH=~/gpudrive/build:~/gpudrive LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.m0_feasibility --out logs/gpucat/m0.json
    ... -m scripts.gpucat.m0_feasibility --bench 256 --out logs/gpucat/m0.json
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAJ_LEN = 91


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpudrive-root", default=str(Path.home() / "gpudrive"))
    p.add_argument("--scenes", default="logs/gpucat/scenes", help="export_scenes.py's output.")
    p.add_argument("--adv", default="adv_scenes/cat", help="export_adv_scenes.py's output for CAT's rule.")
    p.add_argument("--first", type=int, default=400, help="First scene (CAT's test split starts at 400).")
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--bench", type=int, default=None, metavar="W", help="Only the throughput test, with W worlds.")
    p.add_argument("--bench-steps", type=int, default=200)
    p.add_argument("--out", default=None, help="Append the results as a JSON line.")
    return p.parse_args(argv)


def gpu_used_mb():
    rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout.splitlines()
    return sum(int(u) for pid, u in (r.split(", ") for r in rows) if int(pid) == os.getpid())


def scene_dir(args, stems):
    """A directory of links to the chosen scenes (GPUDrive's loader takes a directory)."""
    d = (REPO / "logs/gpucat/m0_scenes").resolve()
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("cat_*.json"):
        f.unlink()
    for s in stems:
        (d / f"cat_{s}.json").symlink_to((REPO / args.scenes / f"cat_{s}.json").resolve())
    return d


def make_env(root, data, worlds):
    from gpudrive.env.config import EnvConfig
    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    loader = SceneDataLoader(root=str(data), batch_size=worlds, dataset_size=worlds, file_prefix="cat", shuffle=False)
    # all_objects: also agents that appear after step 0 (7 of the first 64 test adversaries do)
    config = EnvConfig(dynamics_model="delta_local", collision_behavior="ignore", remove_non_vehicles=False,
                       init_mode="all_objects")
    return GPUDriveTorchEnv(config=config, data_loader=loader, max_cont_agents=64, device="cuda",
                            action_type="continuous")


def check(args, root):
    import numpy as np
    import torch

    index = json.loads((REPO / args.scenes / "index.json").read_text())
    adv_index = json.loads((REPO / args.adv / "index.json").read_text())
    stems = [s for s in map(str, range(args.first, args.first + 10 * args.worlds))
             if s in index and index[s]["adversary_id"] is not None][: args.worlds]
    data = scene_dir(args, stems)
    os.chdir(root)
    env = make_env(root, data, len(stems))
    W = env.num_worlds
    world_stem = [Path(p).stem.split("_", 1)[1] for p in env.data_batch]
    res = {"worlds": W}

    # control: one controlled agent per world, the SDC
    mask = env.cont_agent_mask
    ids = env.sim.absolute_self_observation_tensor().to_torch()[..., 13].long()
    ego_slot = mask.float().argmax(1)
    ok = [int(mask[w].sum()) == 1 and int(ids[w, ego_slot[w]]) == index[world_stem[w]]["sdc_id"] for w in range(W)]
    res["control_ok"] = f"{sum(ok)}/{W}"

    # injection: CAT's plan into the adversary's trajectory
    traj = env.sim.expert_trajectory_tensor().to_torch()  # zero-copy view [W, 64, 1456]
    means = env.sim.world_means_tensor().to_torch()[:, :2]
    adv_slot, plans = [], []
    for w in range(W):
        aid = index[world_stem[w]]["adversary_id"]
        slots = torch.nonzero(ids[w] == aid).flatten()
        a = int(slots[0]) if len(slots) else -1
        adv_slot.append(a)
        with open(REPO / args.adv / f"{world_stem[w]}.pkl", "rb") as f:
            st = pickle.load(f)["tracks"][str(aid)]["state"]
        pos = torch.as_tensor(np.asarray(st["position"])[:, :2], dtype=torch.float32, device="cuda") - means[w]
        vel = torch.as_tensor(np.asarray(st["velocity"]), dtype=torch.float32, device="cuda")
        yaw = torch.as_tensor(np.asarray(st["heading"]), dtype=torch.float32, device="cuda")
        valid = torch.as_tensor(np.asarray(st["valid"]), dtype=torch.bool, device="cuda")  # False only before it appears
        plans.append((pos, valid))
        if a < 0:
            continue
        old = traj[w, a].clone()
        keep = ~valid  # steps before the adversary appears: as GPUDrive loaded them, invalid
        pos = torch.where(keep[:, None], old[0:2 * TRAJ_LEN].view(TRAJ_LEN, 2), pos)
        vel = torch.where(keep[:, None], old[2 * TRAJ_LEN:4 * TRAJ_LEN].view(TRAJ_LEN, 2), vel)
        yaw = torch.where(keep, old[4 * TRAJ_LEN:5 * TRAJ_LEN], yaw)
        traj[w, a, 0:2 * TRAJ_LEN] = pos.flatten()
        traj[w, a, 2 * TRAJ_LEN:4 * TRAJ_LEN] = vel.flatten()
        traj[w, a, 4 * TRAJ_LEN:5 * TRAJ_LEN] = yaw
        traj[w, a, 5 * TRAJ_LEN:6 * TRAJ_LEN] = valid.float()
    res["adversary_found"] = f"{sum(a >= 0 for a in adv_slot)}/{W}"
    env.reset()

    # roll out: the SDC by its expert actions, everyone else replayed
    expert = env.get_expert_actions()[0]  # [W, 64, 91, 3]
    log_pos = env.get_expert_actions()[1]  # [W, 64, 91, 2], demeaned
    adv_xy, ego_xy, hit, ego_done = [], [], [], []
    w_idx = torch.arange(W, device="cuda")
    for t in range(TRAJ_LEN - 1):
        actions = torch.zeros((W, 64, 3), device="cuda")
        actions[w_idx, ego_slot] = expert[w_idx, ego_slot, t]
        env.step_dynamics(actions)
        g = env.sim.absolute_self_observation_tensor().to_torch()
        adv_xy.append(torch.stack([g[w, adv_slot[w], :2] for w in range(W)]))
        ego_xy.append(g[w_idx, ego_slot, :2])
        hit.append(env.sim.info_tensor().to_torch()[w_idx, ego_slot, 1].clone())
        ego_done.append(env.sim.done_tensor().to_torch()[w_idx, ego_slot].flatten().bool().clone())
    adv_xy, ego_xy, hit = torch.stack(adv_xy, 1), torch.stack(ego_xy, 1), torch.stack(hit, 1)  # [W, 90, ...]
    ego_done = torch.stack(ego_done, 1)  # [W, 90]
    found = torch.tensor([a >= 0 for a in adv_slot], device="cuda")
    plan = torch.stack([p for p, _ in plans])  # [W, 91, 2]
    pvalid = torch.stack([v for _, v in plans])  # [W, 91]
    lag_err = {}
    for lag in (0, 1):
        err = (adv_xy - plan[:, 1 - lag:TRAJ_LEN - lag]).norm(dim=-1)
        m = pvalid[:, 1 - lag:TRAJ_LEN - lag] & pvalid[:, 1:] & found[:, None]
        lag_err[f"lag{lag}"] = round(float(err[m].max()), 3)
    res["adversary_vs_plan_max_err_m"] = lag_err
    ego_log = log_pos[w_idx, ego_slot]  # [W, 91, 2]
    ego_err = (ego_xy - ego_log[:, 1:]).norm(dim=-1)
    before_done = torch.cumsum(ego_done.int(), 1) == 0
    res["ego_vs_log_err_m"] = {"median": round(float(ego_err[before_done].median()), 3),
                               "q99": round(float(ego_err[before_done].quantile(0.99)), 3),
                               "max": round(float(ego_err[before_done].max()), 3),
                               "worlds_where_the_ego_reached_its_goal": int(ego_done.any(1).sum())}
    first_hit = [int(torch.nonzero(hit[w]).flatten()[0]) + 1 if bool(hit[w].any()) else None for w in range(W)]
    offline = [adv_index[s]["first_overlap_with_logged_ego"] for s in world_stem]
    both = [(g, o) for g, o in zip(first_hit, offline) if g is not None and o is not None]
    res["collision_per_world"] = {s: [g, o] for s, g, o in zip(world_stem, first_hit, offline)}
    res["collision"] = {"gpudrive": sum(g is not None for g in first_hit), "offline": sum(o is not None for o in offline),
                        "both": len(both),
                        "step_diff_median": float(np.median([g - o for g, o in both])) if both else None,
                        "step_diff_abs_max": int(max(abs(g - o) for g, o in both)) if both else None}

    # reset: half the worlds
    half = list(range(W // 2))
    before = env.sim.absolute_self_observation_tensor().to_torch()[:, :, :2].clone()
    env.reset(env_idx_list=half)
    after = env.sim.absolute_self_observation_tensor().to_torch()[:, :, :2]
    def plan_rows(w):
        v = plans[w][1]
        return traj[w, adv_slot[w], 0:2 * TRAJ_LEN].view(TRAJ_LEN, 2)[v], plans[w][0][v]

    kept = all(torch.allclose(*plan_rows(w)) for w in range(W) if adv_slot[w] >= 0)
    start = max((float((after[w, adv_slot[w]] - plans[w][0][0]).norm()) for w in half
                 if adv_slot[w] >= 0 and bool(plans[w][1][0])), default=0.0)
    others = float((after[W // 2:] - before[W // 2:]).abs().max())
    res["reset"] = {"plans_kept": kept, "reset_adversary_at_plan_start_max_err_m": round(start, 3),
                    "other_worlds_moved_max_m": round(others, 6)}
    return res


def bench(args, root):
    import torch
    import torch.nn as nn

    index = json.loads((REPO / args.scenes / "index.json").read_text())
    stems = [s for s in sorted(index, key=int) if index[s]["adversary_id"] is not None][: args.bench]
    data = scene_dir(args, stems)
    os.chdir(root)
    t0 = time.time()
    env = make_env(root, data, len(stems))
    init_s = time.time() - t0
    W = env.num_worlds
    obs = env.get_obs()
    dim = obs.shape[-1]
    actor = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 6)).cuda()
    critic = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1)).cuda()
    w_idx = torch.arange(W, device="cuda")
    ego_slot = env.cont_agent_mask.float().argmax(1)
    times = []
    for episode in range(3):
        env.reset()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(min(args.bench_steps, TRAJ_LEN - 1)):
            o = env.get_obs()[w_idx, ego_slot]
            with torch.no_grad():
                mu, log_std = actor(o).chunk(2, -1)
                a = torch.tanh(mu + log_std.exp() * torch.randn_like(mu))
                critic(o)
            actions = torch.zeros((W, 64, 3), device="cuda")
            actions[w_idx, ego_slot] = a * torch.tensor([6.0, 6.0, 3.14], device="cuda")
            env.step_dynamics(actions)
            env.get_rewards()
            env.get_dones()
            env.sim.info_tensor().to_torch()
        torch.cuda.synchronize()
        if episode > 0:
            times.append(time.time() - t0)
    steps = min(args.bench_steps, TRAJ_LEN - 1)
    dt = sum(times) / len(times)
    return {"bench_worlds": W, "obs_dim": dim, "init_s": round(init_s, 1),
            "world_steps_per_s": round(W * steps / dt, 1), "gpu_mem_gb": round(gpu_used_mb() / 1024, 2)}


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.gpudrive_root).expanduser().resolve()
    sys.path.insert(0, str(root))
    os.environ.setdefault("MADRONA_MWGPU_KERNEL_CACHE", str(root / "gpudrive_cache"))
    out = (REPO / args.out).resolve() if args.out else None
    res = bench(args, root) if args.bench else check(args, root)
    print(json.dumps(res, indent=1))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
