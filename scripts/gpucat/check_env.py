"""Checks gpucat/env.py before training (gpucat/PLAN.md, M2/M3) on CAT's
test scenes (400-499):

  replay       the SDC replayed exactly from its log (switched to an expert
               for the check; GPUDrive places it at the log one step late),
               no adversary: how often each test of the env fires (a logged
               drive should arrive or run out of time, neither crash nor
               leave the road); and how far the SDC drifts when driven by
               GPUDrive's inverse bicycle actions instead
  adversary    the exact replay with CAT's adversary (its plan against the
               logged route): collisions and their steps against the offline
               first overlap (export_adv_scenes.py's circle covers)
  --speed W    (a process of its own: Madrona keeps one simulator a process)
               world steps per second with random actions and every test

Run with GPUDrive's environment, from this repository's root:
    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.check_env --out logs/gpucat/check_env.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from gpucat.env import ARRIVED, CRASHED, EPISODE_STEPS, OUT_OF_ROAD, TIMEOUT, CatEnv, EnvSettings  # noqa: E402

NAMES = {ARRIVED: "arrived", OUT_OF_ROAD: "out_of_road", CRASHED: "crashed", TIMEOUT: "timeout"}


def replay(env, adversarial, exact=True):
    control = env.sim.controlled_state_tensor().to_torch()
    control[:, 0] = 0 if exact else 1  # the SDC replayed by the simulator itself
    obs = env.begin_episode(torch.full((env.W,), adversarial, device=env.device))
    acts = env.expert_actions()
    lag = 1 if exact else 0
    log = env.env.get_expert_actions()[1][:, 0] + env.mean[:, None]  # logged SDC positions [W, 91, 2]
    fired = {k: torch.zeros(env.W, dtype=torch.bool, device=env.device) for k in ("crash", "edge", "yellow", "off_route")}
    first = {k: torch.full((env.W,), -1, device=env.device) for k in fired}
    drift = []
    for t in range(EPISODE_STEPS):
        _, _, _, alive = env.step(acts[:, t], clip=False)
        for k in fired:
            new = env.flags[k] & alive & ~fired[k]
            first[k] = torch.where(new, t + 1, first[k])
            fired[k] |= env.flags[k] & alive
        drift.append(torch.where(alive, (env.flags["position"] - log[:, t + 1 - lag]).norm(dim=-1),
                                 torch.full_like(alive, float("nan"), dtype=torch.float32)))
    out = env.end_episode()
    control[:, 0] = 1
    drift = torch.stack(drift, 1)
    return out, fired, first, drift


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="logs/gpucat/scenes")
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--adv-index", default="adv_scenes/cat/index.json")
    p.add_argument("--speed", type=int, default=None, metavar="W", help="Only the speed test, with W worlds.")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    if args.speed:
        return speed(args)
    test = [str(i) for i in range(400, 500)]
    env = CatEnv(EnvSettings(worlds=len(test)), REPO / args.scenes, REPO / args.static, REPO / args.bank)
    env.load(test)
    res = {}
    for name, adversarial in (("replay", False), ("adversary", True)):
        out, fired, first, drift = replay(env, adversarial)
        outcome = out["outcome"]
        r = {"outcomes": {NAMES[c]: int((outcome == c).sum()) for c in NAMES},
             "fired": {k: int(v.sum()) for k, v in fired.items()},
             "drift_m": {"median": round(float(drift.nanmedian()), 3),
                         "q95": round(float(torch.nanquantile(drift.flatten(), 0.95)), 3),
                         "at_end_median": round(float(torch.nanmedian(drift[:, -1])), 3) if drift.shape[1] else None},
             "completion_mean": round(float(out["completion"].mean()), 3)}
        if name == "replay":
            r["fired_in"] = {k: [test[w] for w in torch.nonzero(v).flatten().tolist()] for k, v in fired.items()}
        else:
            idx = json.loads((REPO / args.adv_index).read_text())
            offline = [idx[s]["first_overlap_with_logged_ego"] for s in test]
            gpu = [int(x) if x > 0 else None for x in first["crash"].tolist()]
            both = [(g, o) for g, o in zip(gpu, offline) if g is not None and o is not None]
            r["collision_vs_offline"] = {"env": sum(g is not None for g in gpu),
                                        "offline_circles": sum(o is not None for o in offline), "both": len(both),
                                        "step_diff_median": (sorted(g - o for g, o in both)[len(both) // 2]
                                                             if both else None)}
        res[name] = r
        print(name, json.dumps(r), flush=True)
    out, _, _, drift = replay(env, False, exact=False)
    res["bicycle_inverse_drift_m"] = {"median": round(float(drift.nanmedian()), 3),
                                      "at_end_median": round(float(torch.nanmedian(drift[:, -1])), 3)}
    print("bicycle", res["bicycle_inverse_drift_m"])
    if args.out:
        (REPO / args.out).write_text(json.dumps(res, indent=1))


def speed(args):
    """Random actions with every test, at the training batch size."""
    env = CatEnv(EnvSettings(worlds=args.speed), REPO / args.scenes, REPO / args.static, REPO / args.bank)
    env.load([str(i) for i in range(args.speed)])
    times = []
    for episode in range(3):
        env.begin_episode(torch.rand(env.W, device=env.device) < 0.5)
        torch.cuda.synchronize()
        t0 = time.time()
        for t in range(EPISODE_STEPS):
            env.step(torch.rand(env.W, 2, device=env.device) * 2 - 1)
        env.end_episode()
        torch.cuda.synchronize()
        if episode:
            times.append(time.time() - t0)
    t0 = time.time()
    env.load([str(i) for i in range(100, 100 + args.speed)])
    torch.cuda.synchronize()
    res = {"worlds": env.W, "world_steps_per_s": round(env.W * EPISODE_STEPS / (sum(times) / len(times)), 1),
           "swap_s": round(time.time() - t0, 2)}
    print("speed", json.dumps(res))
    if args.out:
        with open(REPO / args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
