"""Evaluates trained GPUDrive policies scene by scene (gpucat/PLAN.md, M3):
each run's model.pt on the 100 test scenes (400-499), deterministic actions,
as train.py's evaluation does -- a normal episode first (which becomes each
scene's evaluation history, as CAT's eval_policy), then one against each
test adversary chosen against it. Unlike train.py's eval.csv, it keeps every
scene's outcome, so runs can be compared scene by scene (paired).

Writes OUT/<run>.csv (scene, test_adversary, outcome, completion, adversary
present, chosen candidate) and prints the rates.

    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.evaluate --runs replay_nav_s0 cat_nav_s0 --out logs/gpucat/eval
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from gpucat.env import CatEnv, EnvSettings  # noqa: E402
from gpucat.ppo import ActorCritic  # noqa: E402
from scripts.gpucat.train import TEST, rollout  # noqa: E402

OUTCOMES = {0: "running", 1: "arrive", 2: "out_of_road", 3: "crash", 4: "timeout"}


class Policy:
    """train.rollout's view of a PPO object: .net."""

    def __init__(self, net):
        self.net = net


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="Run directories under --root.")
    p.add_argument("--root", default="logs/gpucat/runs")
    p.add_argument("--adversaries", nargs="+", default=["cat", "fair", "fair_valid"])
    p.add_argument("--scenes", default="logs/gpucat/scenes")
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--valid", default="logs/gpucat/valid_samples.npz")
    p.add_argument("--out", default="logs/gpucat/eval")
    args = p.parse_args(argv)
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    configs = {r: json.loads((REPO / args.root / r / "config.json").read_text()) for r in args.runs}
    navigation = {c["env"].get("navigation", False) for c in configs.values()}
    if len(navigation) != 1:
        raise ValueError("evaluate runs with and without navigation observations in separate calls")
    env = CatEnv(EnvSettings(worlds=len(TEST), navigation=navigation.pop()), REPO / args.scenes, REPO / args.static,
                 REPO / args.bank, valid_path=REPO / args.valid)
    env.load(TEST)
    W = env.W
    for run in args.runs:
        a = configs[run]["args"]
        opts = argparse.Namespace(tau=a["tau"], rho=a["rho"])
        ck = torch.load(REPO / args.root / run / "model.pt", map_location=env.device)
        net = ActorCritic(ck["obs_dim"]).to(env.device)
        net.load_state_dict(ck["net"])
        policy = Policy(net)
        rows = []
        tests = [("none", torch.zeros(W, dtype=torch.bool, device=env.device), "cat")]
        tests += [(rule, torch.ones(W, dtype=torch.bool, device=env.device), rule) for rule in args.adversaries]
        for name, adversarial, rule in tests:
            _, _, res = rollout(env, policy, adversarial, rule, opts, evaluation=True, deterministic=True)
            for i, stem in enumerate(TEST):
                rows.append({"scene": stem, "test_adversary": name, "outcome": OUTCOMES[int(res["outcome"][i])],
                             "completion": round(float(res["completion"][i]), 4),
                             "adversarial": int(res["adversarial"][i]), "chosen": int(res["chosen"][i])})
        with open(out / f"{run}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{run} ({ck['steps']} steps)")
        for name, *_ in tests:
            r = [x for x in rows if x["test_adversary"] == name]
            rate = lambda o: sum(x["outcome"] == o for x in r) / len(r)  # noqa: E731
            print(f"  {name:10s} arrive {rate('arrive'):.2f} crash {rate('crash'):.2f} out_of_road "
                  f"{rate('out_of_road'):.2f} completion {sum(x['completion'] for x in r) / len(r):.3f}")


if __name__ == "__main__":
    main()
