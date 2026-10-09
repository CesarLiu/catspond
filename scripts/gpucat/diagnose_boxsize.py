"""Re-evaluates trained policies with MetaDrive's vehicle sizes instead of
the logged ones (gpucat/PLAN.md). CatEnv checks collisions, road edges and
yellow lines with the SDC's logged box (5.29 x 2.33 m) and every other
vehicle's logged box; CAT's MetaDrive uses DefaultVehicle for the ego
(4.515 x 1.852 m) and class boxes for traffic. Only CatEnv.sizes changes
here: no retraining, the same models, scenes, adversaries and actions.

The class box of a traffic vehicle is chosen by its logged length: up to
4.0 m 4.3 x 1.7 (S), up to 5.5 m 4.6 x 1.85 (M), above 5.74 x 2.3 (XL). An
approximation of MetaDrive's choice (which also cycles between sizes).

    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.diagnose_boxsize --runs replay_nav_s0 cat_nav_s0
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from gpucat.env import CatEnv, EnvSettings  # noqa: E402
from gpucat.ppo import ActorCritic  # noqa: E402
from scripts.gpucat.train import TEST, rollout  # noqa: E402

EGO = (4.515, 1.852)


class Policy:
    def __init__(self, net):
        self.net = net


def metadrive_sizes(sizes, types):
    out = sizes.clone()
    length = sizes[..., 0]
    cls = torch.where(length <= 4.0, 0, torch.where(length <= 5.5, 1, 2))
    table = torch.tensor([[4.3, 1.7], [4.6, 1.85], [5.74, 2.3]], device=sizes.device, dtype=sizes.dtype)
    veh = types == 1
    out[veh] = table[cls[veh]]
    out[:, 0] = torch.tensor(EGO, device=sizes.device, dtype=sizes.dtype)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--root", default="logs/gpucat/runs")
    p.add_argument("--adversaries", nargs="+", default=["cat", "fair"])
    args = p.parse_args(argv)
    cfgs = {r: json.loads((REPO / args.root / r / "config.json").read_text()) for r in args.runs}
    nav = {c["env"].get("navigation", False) for c in cfgs.values()}
    assert len(nav) == 1
    env = CatEnv(EnvSettings(worlds=len(TEST), navigation=nav.pop()), REPO / "logs/gpucat/scenes",
                 REPO / "logs/gpucat/static.npz", REPO / "logs/gpucat/candidates.npz")
    env.load(TEST)
    logged = env.sizes.clone()
    md = metadrive_sizes(logged, env.types)
    W = env.W
    for run in args.runs:
        a = cfgs[run]["args"]
        opts = argparse.Namespace(tau=a["tau"], rho=a["rho"])
        ck = torch.load(REPO / args.root / run / "model.pt", map_location=env.device)
        net = ActorCritic(ck["obs_dim"]).to(env.device)
        net.load_state_dict(ck["net"])
        pol = Policy(net)
        for name, sizes in (("logged", logged), ("metadrive", md)):
            env.sizes = sizes
            row = []
            for adv in ["none"] + args.adversaries:
                on = torch.full((W,), adv != "none", dtype=torch.bool, device=env.device)
                _, _, res = rollout(env, pol, on, "cat" if adv == "none" else adv, opts, evaluation=True,
                                    deterministic=True)
                o = res["outcome"]
                row.append(f"{adv}: arr {(o == 1).float().mean():.2f} oor {(o == 2).float().mean():.2f} "
                           f"crash {(o == 3).float().mean():.2f} rc {res['completion'].mean():.3f}")
            print(f"{run:18s} {name:9s} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
