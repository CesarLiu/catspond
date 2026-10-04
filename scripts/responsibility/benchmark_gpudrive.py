"""Measures GPUDrive's simulation speed on this machine, on GPUDrive's own
scenes (GPUDrive_mini) or on CAT's scenes exported by export_gpudrive.py.

One configuration per process (Madrona keeps one simulator per process):
NUM_WORLDS parallel worlds, each a different scene, every agent GPUDrive
controls stepped with random discrete actions, with observations, rewards
and dones read back every step as a policy would. Reports world steps/s
(one step of one scene, i.e. what one CAT environment step is) and
controlled-agent steps/s, after a warm-up, and the GPU memory used.

Run with GPUDrive's environment (built against CUDA 12.4):
    ~/gpudrive/.venv/bin/python -m scripts.responsibility.benchmark_gpudrive \\
        --data ~/gpudrive/data/processed/training --worlds 64
    ~/gpudrive/.venv/bin/python -m scripts.responsibility.benchmark_gpudrive \\
        --data logs/gpudrive/cat_scenes --prefix cat --worlds 64
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpudrive-root", default=str(Path.home() / "gpudrive"))
    p.add_argument("--data", required=True, help="Directory of GPUDrive JSON scenes.")
    p.add_argument("--prefix", default="tfrecord", help="File-name prefix of the scenes (GPUDrive's loader filters by it).")
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--steps", type=int, default=80, help="Timed steps per episode (an episode is 91).")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-agents", type=int, default=64, help="Controlled agents per world at most.")
    p.add_argument("--sim-only", action="store_true", help="Only step the simulator: no observations, rewards, dones.")
    p.add_argument("--out", default=None, help="Append the result as a JSON line.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data = str(Path(args.data).expanduser().resolve())
    out = Path(args.out).expanduser().resolve() if args.out else None
    root = Path(args.gpudrive_root).expanduser().resolve()
    os.chdir(root)  # GPUDrive resolves its assets relative to its root
    sys.path.insert(0, str(root))
    os.environ.setdefault("MADRONA_MWGPU_KERNEL_CACHE", str(root / "gpudrive_cache"))
    import subprocess

    import torch

    def gpu_used_mb():
        """This process's GPU memory by nvidia-smi: Madrona allocates outside torch."""
        rows = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.splitlines()
        return sum(int(used) for pid, used in (r.split(", ") for r in rows) if int(pid) == os.getpid())
    from gpudrive.env.config import EnvConfig
    from gpudrive.env.dataset import SceneDataLoader
    from gpudrive.env.env_torch import GPUDriveTorchEnv

    n_files = sum(1 for f in os.listdir(data) if f.startswith(args.prefix) and f.endswith(".json"))
    loader = SceneDataLoader(root=data, batch_size=args.worlds, dataset_size=n_files, file_prefix=args.prefix,
                             sample_with_replacement=args.worlds > n_files, shuffle=False, seed=0)
    t0 = time.time()
    env = GPUDriveTorchEnv(config=EnvConfig(), data_loader=loader, max_cont_agents=args.max_agents, device="cuda")
    init_s = time.time() - t0
    n_actions = env.action_space.n
    times, controlled = [], []
    for episode in range(args.episodes + 1):  # the first one warms up
        env.reset()
        controlled.append(int(env.cont_agent_mask.sum()))
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.steps):
            actions = torch.randint(0, n_actions, (args.worlds, args.max_agents), device="cuda")
            env.step_dynamics(actions)
            if not args.sim_only:
                env.get_obs()
                env.get_rewards()
                env.get_dones()
        torch.cuda.synchronize()
        if episode > 0:
            times.append(time.time() - t0)
    dt = sum(times) / len(times)
    result = {
        "data": data, "worlds": args.worlds, "sim_only": args.sim_only, "controlled_agents": controlled[-1], "steps": args.steps,
        "init_s": round(init_s, 1), "world_steps_per_s": round(args.worlds * args.steps / dt, 1),
        "agent_steps_per_s": round(controlled[-1] * args.steps / dt, 1),
        "gpu_mem_gb": round(gpu_used_mb() / 1024, 2),  # this process, by nvidia-smi (Madrona allocates outside torch)
    }
    print(json.dumps(result))
    if out:
        with open(out, "a") as f:
            f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
