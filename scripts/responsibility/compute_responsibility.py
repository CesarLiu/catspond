"""Safety and courtesy responsibility of the self-driving car (or any agent)
over CAT's scenes, with DenseTNT as the motion model.

For every scene and every context step k (10, 10 + stride, ..., up to 2 s
before the end of the log) it writes one row to OUT/windows.csv:

  safety     beta_s: > 0 when the agent kept less distance to a neighbour than
             most of its own alternatives would have (it gave up safety margin)
  courtesy   beta_c: how much its presence changed a vehicle neighbour's
             intended goal (KL, nats)
  safety_against / courtesy_toward   the neighbour each maximum came from

plus the full per-neighbour observations to OUT/obs/<scene>.pkl. Re-running
skips finished scenes; OUT/config.json pins the settings.

Example (from the repository root):
    python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 \\
        --out-dir logs/responsibility/sdc --n 50
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.densetnt import DenseTNT  # noqa: E402
from responsibility.interaction import InteractionConfig  # noqa: E402
from responsibility.metrics import ResponsibilityConfig, scene_responsibility  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402

ROW_FIELDS = ["scene", "scenario_id", "agent_id", "step", "time", "speed", "safety", "courtesy",
              "n_neighbours", "safety_against", "courtesy_toward"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--first", type=int, default=0, help="Index of the first scene file.")
    p.add_argument("--n", type=int, default=None, help="Number of scenes (default: all).")
    p.add_argument("--agent", default="sdc", help="sdc, adv (the other object of interest) or a track id.")
    d = ResponsibilityConfig()
    p.add_argument("--n-samples", type=int, default=d.n_safety_samples)
    p.add_argument("--cvar-alpha", type=float, default=d.cvar_alpha)
    p.add_argument("--d-sat", type=float, default=d.d_sat, help="<= 0 disables saturation.")
    p.add_argument("--horizon", type=int, default=d.metric_horizon, help="10 Hz steps scored.")
    p.add_argument("--stride", type=int, default=d.window_stride, help="Steps between context steps.")
    p.add_argument("--no-courtesy", action="store_true")
    i = InteractionConfig()
    p.add_argument("--max-neighbors", type=int, default=i.max_neighbors)
    p.add_argument("--gap-threshold", type=float, default=i.gap_threshold)
    p.add_argument("--pet-threshold", type=float, default=i.pet_threshold)
    p.add_argument("--ttc-threshold", type=float, default=i.ttc_threshold)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pick_agent(scene: Scene, which: str) -> int:
    if which == "sdc":
        return scene.sdc
    if which == "adv":
        others = [i for i in scene.objects_of_interest if i != scene.sdc]
        if not others:
            raise ValueError("no second object of interest")
        return others[0]
    return scene.index(which)


def main():
    args = parse_args()
    cfg = ResponsibilityConfig(
        n_safety_samples=args.n_samples, cvar_alpha=args.cvar_alpha,
        d_sat=args.d_sat if args.d_sat > 0 else None, metric_horizon=args.horizon,
        window_stride=args.stride, courtesy=not args.no_courtesy, seed=args.seed,
        interaction=InteractionConfig(max_neighbors=args.max_neighbors, gap_threshold=args.gap_threshold,
                                      pet_threshold=args.pet_threshold, ttc_threshold=args.ttc_threshold),
    )
    out = Path(args.out_dir)
    (out / "obs").mkdir(parents=True, exist_ok=True)
    settings = {"responsibility": asdict(cfg), "agent": args.agent, "scenes": str(Path(args.scenes).resolve())}
    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != json.loads(json.dumps(settings)):
            raise SystemExit(f"{out} holds results computed with other settings; use a new --out-dir")
    else:
        config_path.write_text(json.dumps(settings, indent=2))

    files = scene_files(args.scenes)[args.first:]
    if args.n is not None:
        files = files[: args.n]
    todo = [f for f in files if not (out / "obs" / f"{f.stem}.pkl").exists()]
    print(f"{len(files)} scenes, {len(files) - len(todo)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    model = DenseTNT(device=args.device)
    rows_path = out / "windows.csv"
    new_file = not rows_path.exists()
    with open(rows_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if new_file:
            writer.writeheader()
        for n, path in enumerate(todo, 1):
            t0 = time.time()
            scene = Scene.load(path)
            try:
                agent = pick_agent(scene, args.agent)
            except ValueError as e:
                print(f"{path.name}: skipped ({e})", flush=True)
                observations = []
            else:
                observations = scene_responsibility(model, scene, agent, cfg)
            for obs in observations:
                writer.writerow({"scene": path.stem, **obs.as_row()})
            handle.flush()
            tmp = out / "obs" / f"{path.stem}.pkl.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump([asdict(o) for o in observations], f)
            os.replace(tmp, out / "obs" / f"{path.stem}.pkl")
            s = [o.safety for o in observations]
            c = [o.courtesy for o in observations]
            print(f"[{n}/{len(todo)}] {path.name}: {len(observations)} windows, "
                  f"max safety {max(s, default=0):+.2f} m, max courtesy {max(c, default=0):.3f} nats "
                  f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
