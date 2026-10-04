"""Exports CAT's scenes (or the scenes rebuilt from policy rollouts) into
GPUDrive's JSON format (responsibility/gpudrive_export.py), for running
them in GPUDrive.

Run with GPUDrive's environment (it needs trimesh for GPUDrive's
expert marking). Writes OUT/cat_<scene>.json and OUT/../<OUT name>_index.json:
scene -> WOMD scenario id, and whether GPUDrive's own conversion would have
dropped the scene (traffic lights, 3D road structures). GPUDrive reads every
.json in its data directory, hence the index outside it.

Example (from this repository's root):
    ~/gpudrive/.venv/bin/python -m scripts.responsibility.export_gpudrive --out-dir logs/gpudrive/cat_scenes
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.gpudrive_export import scenario, scene_flags  # noqa: E402
from responsibility.rollouts import load_rollout, rollout_files, scene_from_rollout  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--rollouts", default=None, help="Export the scenes rebuilt from these rollouts instead.")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--no-experts", action="store_true", help="Skip GPUDrive's expert marking (trimesh).")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out.parent / f"{out.name}_index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    if args.rollouts:
        jobs = [(str(r["scene_file"]), Path(args.scenes) / f"{r['scene_file']}.pkl", r)
                for r in (load_rollout(f) for f in rollout_files(args.rollouts))]
    else:
        files = scene_files(args.scenes)[args.first:]
        files = files[:args.n] if args.n is not None else files
        jobs = [(f.stem, f, None) for f in files]
    done = 0
    for stem, path, rollout in jobs:
        target = out / f"cat_{stem}.json"
        if target.exists():
            continue
        with open(path, "rb") as f:
            description = pickle.load(f)
        scene = Scene.from_description(description)
        if rollout is not None:
            scene = scene_from_rollout(scene, rollout)
        data = scenario(scene, target.name, description["metadata"].get("tracks_to_predict", {}).values(),
                        experts=not args.no_experts)
        target.write_text(json.dumps(data))
        index[stem] = dict(scene_flags(scene), scenario_id=scene.scenario_id, objects=len(data["objects"]),
                           experts=sum(o["mark_as_expert"] for o in data["objects"]))
        done += 1
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
    lights = sum(v["traffic_lights"] for v in index.values())
    has_3d = sum(v["has_3d"] for v in index.values())
    print(f"exported {done} scene(s) to {out}; of {len(index)}: {lights} with traffic lights (signals dropped), "
          f"{has_3d} with 3D road structures (GPUDrive's conversion drops both)")


if __name__ == "__main__":
    main()
