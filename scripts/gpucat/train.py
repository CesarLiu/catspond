"""Trains the SDC policy with PPO on CAT's training scenes in GPUDrive
(gpucat/PLAN.md, M2), in one of CAT's settings:

  replay   no adversary
  cat      CAT's adversary in a share of the episodes that grows with
           training as in cat_RLtrain.py: an episode is adversarial with
           probability 1 - max(1 - 2 t / T (1 - min_prob), min_prob)
  fair     the fair adversary (--tau, --rho), on the same schedule

Each iteration steps every world --horizon times (a world whose episode
ends starts the next at once, adversarial with the current probability)
and makes one PPO update; the worlds take a new batch of distinct training
scenes (0-399) every --swap-every iterations. Every --eval-every iterations the policy is
evaluated (deterministic actions) on the 100 test scenes (400-499): a
normal episode (which also becomes each scene's evaluation history, as
eval_policy's first round), then one against CAT's adversary and one
against the fair one chosen against it. Writes OUT/metrics.csv (training),
OUT/eval.csv, OUT/config.json and OUT/model.pt.

Run with GPUDrive's environment, from this repository's root:
    PYTHONPATH=~/gpudrive/build:~/gpudrive:. LD_LIBRARY_PATH=~/cuda-12.4/lib64 \\
        ~/gpudrive/.venv/bin/python -m scripts.gpucat.train --mode replay --seed 0 --out logs/gpucat/runs/replay_s0
"""

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpucat.env import ARRIVED, CRASHED, EPISODE_STEPS, OUT_OF_ROAD, CatEnv, EnvSettings  # noqa: E402
from gpucat.ppo import PPO, PPOSettings  # noqa: E402

TRAIN = [str(i) for i in range(400)]
TEST = [str(i) for i in range(400, 500)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["replay", "cat", "fair"], default="cat")
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--rho", type=float, default=0.1)
    p.add_argument("--min-prob", type=float, default=0.1, help="CAT's min_prob.")
    p.add_argument("--steps", type=float, default=1e7, help="Environment steps (SDC transitions) to train for.")
    p.add_argument("--worlds", type=int, default=192)
    p.add_argument("--horizon", type=int, default=64, help="Steps per world per PPO update.")
    p.add_argument("--swap-every", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--crash-penalty", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scenes", default="logs/gpucat/scenes")
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


def adversarial_probability(t, total, min_prob):
    """cat_RLtrain.py: random() > max(1 - (2 t / T)(1 - min_prob), min_prob)."""
    return 1.0 - max(1.0 - (2.0 * t / total) * (1.0 - min_prob), min_prob)


def summarize(out, mask=None):
    o = out["outcome"] if mask is None else out["outcome"][mask]
    c = out["completion"] if mask is None else out["completion"][mask]
    n = max(len(o), 1)
    return {"episodes": len(o), "arrive": float((o == ARRIVED).sum()) / n, "out_of_road": float((o == OUT_OF_ROAD).sum()) / n,
            "crash": float((o == CRASHED).sum()) / n, "completion": float(c.mean()) if len(c) else float("nan")}


def collect(env, ppo, obs, horizon):
    """horizon steps of every world with auto-reset; returns the batch, the
    bootstrap values and the next observations."""
    buf = {k: [] for k in ("obs", "act", "logp", "value", "reward", "ended", "alive")}
    for _ in range(horizon):
        with torch.no_grad():
            a, logp, v = ppo.net.act(obs)
        nxt, r, ended, alive = env.step(a)
        for k, x in zip(buf, (obs, a, logp, v, r, ended, alive)):
            buf[k].append(x)
        obs = nxt
    with torch.no_grad():
        last = ppo.net.value(obs)
    return {k: torch.stack(v) for k, v in buf.items()}, last, obs


def rollout(env, ppo, adversarial, rule, args, evaluation=False, deterministic=False):
    """One lockstep episode in every world (evaluation)."""
    obs = env.begin_episode(adversarial, rule, args.tau, args.rho, evaluation=evaluation)
    buf = {k: [] for k in ("obs", "act", "logp", "value", "reward", "ended", "alive")}
    for _ in range(EPISODE_STEPS):
        with torch.no_grad():
            a, logp, v = ppo.net.act(obs, deterministic)
        nxt, r, ended, alive = env.step(a)
        for k, x in zip(buf, (obs, a, logp, v, r, ended, alive)):
            buf[k].append(x)
        obs = nxt
        if bool(env.done.all()):
            break
    with torch.no_grad():
        last = ppo.net.value(obs)
    return {k: torch.stack(v) for k, v in buf.items()}, last, env.end_episode(evaluation)


def start_training_episodes(env, args, p_adv):
    rule = "fair" if args.mode == "fair" else "cat"
    adversarial = torch.rand(env.W, device=env.device) < p_adv
    return env.begin_episode(adversarial, rule, args.tau, args.rho, auto_reset=True, p_adv=p_adv)


def evaluate(env, ppo, args, train_batch):
    W = env.W
    stems = (TEST * ((W + len(TEST) - 1) // len(TEST)))[:W]
    env.load(stems)
    first = torch.zeros(W, dtype=torch.bool, device=env.device)
    first[: len(TEST)] = True
    none = torch.zeros(W, dtype=torch.bool, device=env.device)
    res = {}
    _, _, out = rollout(env, ppo, none, "cat", args, evaluation=True, deterministic=True)
    res["none"] = summarize(out, first)
    for rule in ("cat", "fair"):
        _, _, out = rollout(env, ppo, torch.ones(W, dtype=torch.bool, device=env.device), rule, args,
                            evaluation=True, deterministic=True)
        res[rule] = summarize(out, first)
    env.load(train_batch)
    return res


def main(argv=None):
    args = parse_args(argv)
    out = (REPO / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    es = EnvSettings(worlds=args.worlds, crash_penalty=args.crash_penalty)
    ps = PPOSettings()
    env = CatEnv(es, REPO / args.scenes, REPO / args.static, REPO / args.bank)
    rng = random.Random(args.seed)
    batch = rng.sample(TRAIN, args.worlds)
    env.load(batch)
    obs_dim = env.obs().shape[-1]
    ppo = PPO(obs_dim, ps)
    (out / "config.json").write_text(json.dumps({"args": vars(args), "env": asdict(es), "ppo": asdict(ps),
                                                 "obs_dim": obs_dim}, indent=1))
    mfile = open(out / "metrics.csv", "w", newline="")
    mw = csv.DictWriter(mfile, fieldnames=["iteration", "steps", "seconds", "p_adv", "adv_episodes", "arrive",
                                           "out_of_road", "crash", "completion", "return", "policy_loss",
                                           "value_loss", "entropy", "clip_frac"])
    mw.writeheader()
    efile = open(out / "eval.csv", "w", newline="")
    ew = csv.DictWriter(efile, fieldnames=["iteration", "steps", "test_adversary", "arrive", "out_of_road", "crash",
                                           "completion"])
    ew.writeheader()
    steps, it, t0 = 0, 0, time.time()
    p_adv_of = (lambda n: 0.0) if args.mode == "replay" else (
        lambda n: adversarial_probability(n, args.steps, args.min_prob))
    obs = start_training_episodes(env, args, p_adv_of(0))
    while steps < args.steps:
        if it and it % args.swap_every == 0:
            batch = rng.sample(TRAIN, args.worlds)
            env.load(batch)
            obs = start_training_episodes(env, args, p_adv_of(steps))
        p_adv = p_adv_of(steps)
        env.p_adv = p_adv
        buf, last, obs = collect(env, ppo, obs, args.horizon)
        stats = ppo.update(buf["obs"], buf["act"], buf["logp"], buf["value"], buf["reward"], buf["ended"],
                           buf["alive"], last)
        steps += stats["samples"]
        it += 1
        ep = env.take_finished()
        summ = summarize(ep)
        ret = float(buf["reward"].sum() / max(int(buf["ended"].sum()), 1))  # per finished episode
        mw.writerow({"iteration": it, "steps": steps, "seconds": round(time.time() - t0, 1), "p_adv": round(p_adv, 3),
                     "adv_episodes": int(ep["adversarial"].sum()), **{k: round(v, 4) for k, v in summ.items() if k != "episodes"},
                     "return": round(ret, 3), **{k: round(stats[k], 5) for k in ("policy_loss", "value_loss", "entropy",
                                                                                   "clip_frac")}})
        mfile.flush()
        if it % 10 == 0:
            print(f"it {it} steps {steps} ({time.time() - t0:.0f} s): p_adv {p_adv:.2f} arrive {summ['arrive']:.2f} "
                  f"oor {summ['out_of_road']:.2f} crash {summ['crash']:.2f} completion {summ['completion']:.2f} "
                  f"return {ret:.1f} log_std {stats['log_std']}", flush=True)
        if it % args.eval_every == 0 or steps >= args.steps:
            res = evaluate(env, ppo, args, batch)
            obs = start_training_episodes(env, args, p_adv_of(steps))
            for name, r in res.items():
                ew.writerow({"iteration": it, "steps": steps, "test_adversary": name,
                             **{k: round(v, 4) for k, v in r.items() if k != "episodes"}})
            efile.flush()
            print(f"  eval it {it}: " + "; ".join(f"{k}: arrive {v['arrive']:.2f} crash {v['crash']:.2f} oor "
                                                  f"{v['out_of_road']:.2f} rc {v['completion']:.2f}" for k, v in res.items()),
                  flush=True)
            torch.save({"net": ppo.net.state_dict(), "obs_dim": obs_dim, "steps": steps}, out / "model.pt")
    mfile.close()
    efile.close()


if __name__ == "__main__":
    main()
