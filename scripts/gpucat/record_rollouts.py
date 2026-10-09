"""Records trained GPUDrive policies' test episodes step by step, for
analysing their crashes and early endings and for rebuilding the episodes
as Scenes (counterfactual attribution).

Each run's model.pt drives the SDC in the 100 test scenes (400-499) with
deterministic actions, in batches of --worlds scenes. Within a batch the
test adversaries run in the given order; "none" comes first, so that its
episode becomes each scene's evaluation history, as in evaluate.py.
Everything is recorded as the simulator saw it, so GPUDrive's one-step
replay lag and the injected adversary plan are included:

  pos [S, 91, 64, 2], yaw [S, 91, 64]
      every slot's absolute pose (raw scene coordinates) after reset (row
      0) and after each of the 90 steps. Slots beyond n_agents are padding.
  alive [S, 90]
      the episode was still running at that step.
  partner [S, 90, 64]
      the slots whose full-size box overlaps the SDC's, among the vehicles
      within the env's collision radius (as CatEnv.step).
  edge, yellow, off_route, crash [S, 90]
      CatEnv's per-step tests; lateral [S, 90] is the distance from the route.
  outcome, end_step, completion, adversarial, chosen [S]
  log_valid [S, 64, 91]
      validity of each slot's logged state, from the expert trajectory as
      loaded, before any plan is injected.
  sizes [S, 64, 2], types [S, 64], n_agents [S], gd_id [S, 64]
      gd_id is GPUDrive's id of each slot.

Writes OUT/<run>__<adversary>.npz.

    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.record_rollouts --runs replay_nav_s0 cat_nav_s0 --worlds 25
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpucat.env import EPISODE_STEPS, TRAJ_LEN, VALID, CatEnv, EnvSettings  # noqa: E402
from gpucat.geometry import boxes_overlap  # noqa: E402
from gpucat.ppo import ActorCritic  # noqa: E402

TEST = [str(i) for i in range(400, 500)]


def poses(env):
    g = env.sim.absolute_self_observation_tensor().to_torch()
    return g[..., :2] + env.mean[:, None], g[..., 7], g[..., 13]


def partners(env, pos, yaw):
    """[W, 64]: the slots whose box overlaps the SDC's (CatEnv.step's test)."""
    p, h = pos[:, 0], yaw[:, 0]
    others = torch.arange(64, device=env.device)[None] < env.n_agents[:, None]
    others &= (env.types == 1) & ((pos - p[:, None]).norm(dim=-1) < env.s.collision_radius)
    others[:, 0] = False
    return boxes_overlap(p, h, env.sizes[:, 0], pos, yaw, env.sizes) & others


def episode(env, net, adversary, args):
    W = env.W
    adversarial = torch.zeros(W, dtype=torch.bool, device=env.device) if adversary == "none" else \
        torch.ones(W, dtype=torch.bool, device=env.device)
    rule = "cat" if adversary == "none" else adversary
    obs = env.begin_episode(adversarial, rule, args.tau, args.rho, evaluation=True)
    rec = {k: [] for k in ("pos", "yaw", "alive", "partner", "edge", "yellow", "off_route", "crash", "lateral")}
    pos, yaw, gid = poses(env)
    rec["pos"].append(pos.clone())
    rec["yaw"].append(yaw.clone())
    end_step = torch.full((W,), EPISODE_STEPS, device=env.device)
    for t in range(EPISODE_STEPS):
        alive = ~env.done
        with torch.no_grad():
            a, _, _ = net.act(obs, True)
        obs, _, ended, _ = env.step(a)
        pos, yaw, _ = poses(env)
        rec["pos"].append(pos.clone())
        rec["yaw"].append(yaw.clone())
        rec["alive"].append(alive.clone())
        rec["partner"].append(partners(env, pos, yaw) & alive[:, None])
        for k in ("edge", "yellow", "off_route", "crash", "lateral"):
            rec[k].append(env.flags[k].clone())
        end_step = torch.where(ended & alive, torch.full_like(end_step, t + 1), end_step)
    out = env.end_episode(True)
    res = {k: torch.stack(v, 1).cpu().numpy() for k, v in rec.items()}
    res.update({k: v.cpu().numpy() for k, v in out.items()})
    res["end_step"] = end_step.cpu().numpy()
    res["gd_id"] = gid.cpu().numpy()
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--root", default="logs/gpucat/runs")
    p.add_argument("--adversaries", nargs="+", default=["none", "cat", "fair"])
    p.add_argument("--worlds", type=int, default=25, help="Scenes per batch (GPU memory).")
    p.add_argument("--scenes", default="logs/gpucat/scenes")
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--valid", default="logs/gpucat/valid_samples.npz")
    p.add_argument("--out", default="logs/gpucat/rollouts")
    args = p.parse_args(argv)
    assert args.adversaries[0] == "none", "the normal episode must come first (evaluation history)"
    assert len(TEST) % args.worlds == 0
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    cfgs = {r: json.loads((REPO / args.root / r / "config.json").read_text()) for r in args.runs}
    nav = {c["env"].get("navigation", False) for c in cfgs.values()}
    assert len(nav) == 1, "record runs with and without navigation observations separately"
    env = CatEnv(EnvSettings(worlds=args.worlds, navigation=nav.pop()), REPO / args.scenes, REPO / args.static,
                 REPO / args.bank, valid_path=REPO / args.valid)
    for run in args.runs:
        a = cfgs[run]["args"]
        args.tau, args.rho = a["tau"], a["rho"]
        ck = torch.load(REPO / args.root / run / "model.pt", map_location=env.device)
        net = ActorCritic(ck["obs_dim"]).to(env.device)
        net.load_state_dict(ck["net"])
        parts = {adv: [] for adv in args.adversaries}
        static = []
        for b in range(0, len(TEST), args.worlds):
            stems = TEST[b:b + args.worlds]
            env.load(stems)
            static.append({"log_valid": (env.traj[:, :, VALID] > 0.5).cpu().numpy().reshape(args.worlds, 64, TRAJ_LEN),
                           "sizes": env.sizes.cpu().numpy(), "types": env.types.cpu().numpy(),
                           "n_agents": env.n_agents.cpu().numpy()})
            for adv in args.adversaries:
                parts[adv].append(episode(env, net, adv, args))
        st = {k: np.concatenate([s[k] for s in static]) for k in static[0]}
        for adv, ps in parts.items():
            res = {k: np.concatenate([q[k] for q in ps]) for k in ps[0]}
            np.savez_compressed(out / f"{run}__{adv}.npz", stems=np.array(TEST), **res, **st)
            o = res["outcome"]
            print(f"{run} {adv}: arrive {(o == 1).mean():.2f} out_of_road {(o == 2).mean():.2f} crash {(o == 3).mean():.2f} "
                  f"completion {res['completion'].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
