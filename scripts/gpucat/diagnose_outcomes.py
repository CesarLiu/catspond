"""Breaks a trained policy's out-of-road endings on the 100 test scenes
(400-499, no adversary, deterministic actions) down by the test that ended
them: touching a road edge (MetaDrive's sidewalk), touching a solid yellow
line, or more than 10 m off the route. Prints the shares, the steps at which
they happen and the route completion reached, so a slow learner can be told
apart from one that cannot see what ends its episodes.

    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.diagnose_outcomes --run logs/gpucat/runs/replay_s0
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpucat.env import ARRIVED, CRASHED, EPISODE_STEPS, OUT_OF_ROAD, CatEnv, EnvSettings  # noqa: E402
from gpucat.ppo import ActorCritic  # noqa: E402

TEST = [str(i) for i in range(400, 500)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--scenes", default="logs/gpucat/scenes")
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    args = p.parse_args(argv)
    # the observations the run was trained with (runs before the navigation observations lack the key)
    navigation = json.loads((REPO / args.run / "config.json").read_text())["env"].get("navigation", False)
    env = CatEnv(EnvSettings(worlds=len(TEST), navigation=navigation), REPO / args.scenes, REPO / args.static,
                 REPO / args.bank)
    env.load(TEST)
    ck = torch.load(REPO / args.run / "model.pt", map_location=env.device)
    net = ActorCritic(ck["obs_dim"]).to(env.device)
    net.load_state_dict(ck["net"])
    W = env.W
    obs = env.begin_episode(torch.zeros(W, dtype=torch.bool, device=env.device), evaluation=True)
    cause = {k: torch.zeros(W, dtype=torch.bool, device=env.device) for k in ("edge", "yellow", "off_route")}
    end_t = torch.full((W,), EPISODE_STEPS, device=env.device)
    for t in range(EPISODE_STEPS):
        alive = ~env.done
        with torch.no_grad():
            a, _, _ = net.act(obs, True)
        obs, _, ended, _ = env.step(a)
        for k in cause:
            cause[k] |= ended & env.flags[k]
        end_t = torch.where(ended & alive, torch.full_like(end_t, t + 1), end_t)
        if bool(env.done.all()):
            break
    out = env.end_episode(True)
    o, c = out["outcome"].cpu().numpy(), out["completion"].cpu().numpy()
    t = end_t.cpu().numpy()
    print(f"obs_dim {ck['obs_dim']}, model at {ck['steps']} steps")
    for name, code in (("arrive", ARRIVED), ("out_of_road", OUT_OF_ROAD), ("crash", CRASHED)):
        m = o == code
        print(f"{name:12s} {m.mean():.2f}  median step {np.median(t[m]) if m.any() else float('nan'):5.0f}  "
              f"median completion {np.median(c[m]) if m.any() else float('nan'):.2f}")
    print(f"timeout      {(o == 4).mean():.2f}")
    oor = o == OUT_OF_ROAD
    for k, v in cause.items():
        v = v.cpu().numpy()
        print(f"  out of road by {k:9s} {(v & oor).sum():3d} of {oor.sum()}")
    only = {k: (v.cpu().numpy() & oor) for k, v in cause.items()}
    print(f"  edge only {(only['edge'] & ~only['yellow'] & ~only['off_route']).sum()}, yellow only "
          f"{(only['yellow'] & ~only['edge'] & ~only['off_route']).sum()}, off route only "
          f"{(only['off_route'] & ~only['edge'] & ~only['yellow']).sum()}")
    print("  out-of-road scenes:", " ".join(f"{TEST[i]}:{'E' if only['edge'][i] else ''}"
                                          f"{'Y' if only['yellow'][i] else ''}{'R' if only['off_route'][i] else ''}@{t[i]}"
                                          for i in np.nonzero(oor)[0]))


if __name__ == "__main__":
    main()
