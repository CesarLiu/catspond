"""Exports CAT's scenes (or the scenes rebuilt from policy rollouts) into
catk's cached-WOMD format, so CAT-K's SMART and catk's responsibility code
can measure them (responsibility/SMART_PLAN.md, S1).

Runs in catk's environment (Python 3.11; catk's install/setup_server.sh),
with catk's own second preprocessing stage: responsibility/catk_export.py
rebuilds what catk's protobuf decoders produce, and get_agent_features,
get_map_features / process_dynamic_map and preprocess_map from catk's
src/data_preprocess.py make the cached scenario of it, as for WOMD.

Writes OUT/cache/<scene>.pkl (the scenario id inside is the scene's file
stem, so results map back to the scenes here) and OUT/index.json (scene ->
WOMD scenario id, source files). catk lists every file in the cache
directory as a scenario, hence the separate index. Re-running skips scenes
already exported.

Example (from this repository's root):
    ~/venvs/catk/bin/python -m scripts.responsibility.export_catk --catk-root ~/catk \\
        --scenes raw_scenes_500 --out-dir logs/catk/scenes
    ~/venvs/catk/bin/python -m scripts.responsibility.export_catk --catk-root ~/catk \\
        --rollouts rollouts/replay/cat --out-dir logs/catk/replay_cat
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.catk_export import dynamic_map_infos, map_infos, track_infos  # noqa: E402
from responsibility.rollouts import load_rollout, rollout_files, scene_from_rollout  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catk-root", default=os.environ.get("CATK_ROOT", str(Path.home() / "catk")))
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--rollouts", default=None, help="Export the scenes rebuilt from these rollouts instead.")
    p.add_argument("--first", type=int, default=0, help="Without --rollouts: index of the first scene file.")
    p.add_argument("--n", type=int, default=None, help="Without --rollouts: number of scenes (default all).")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def catk_stage_two(catk_root):
    """catk's own second preprocessing stage."""
    sys.path.append(str(Path(catk_root).resolve()))
    from src.data_preprocess import get_agent_features, get_map_features, process_dynamic_map
    from src.smart.utils.preprocess import get_polylines_from_polygon, preprocess_map

    def build(scene: Scene, predict, stem: str, current: int = 10):
        lights = process_dynamic_map(dynamic_map_infos(scene.dynamic_map_states, scene.n_steps))
        maps = map_infos(scene.map_features, get_polylines_from_polygon)
        data = preprocess_map(get_map_features(maps, lights.loc[lights["time_step"] == current]))
        data["agent"] = get_agent_features(track_infos(scene, predict), split="validation",
                                           num_historical_steps=current + 1, num_steps=scene.n_steps)
        data["scenario_id"] = stem
        return data

    return build


def main(argv=None):
    args = parse_args(argv)
    build = catk_stage_two(args.catk_root)
    out = Path(args.out_dir)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}

    if args.rollouts:
        jobs = []
        for f in rollout_files(args.rollouts):
            rollout = load_rollout(f)
            jobs.append((str(rollout["scene_file"]), Path(args.scenes) / f"{rollout['scene_file']}.pkl", rollout, f))
    else:
        files = scene_files(args.scenes)[args.first:]
        files = files[:args.n] if args.n is not None else files
        jobs = [(f.stem, f, None, None) for f in files]

    done = 0
    for stem, scene_path, rollout, rollout_path in jobs:
        target = out / "cache" / f"{stem}.pkl"
        if target.exists():
            continue
        with open(scene_path, "rb") as f:
            description = pickle.load(f)
        scene = Scene.from_description(description)
        if rollout is not None:
            scene = scene_from_rollout(scene, rollout)
        meta = description["metadata"]
        data = build(scene, list(meta.get("tracks_to_predict", {})), stem, int(meta.get("current_time_index", 10)))
        tmp = target.with_name(f"{target.name}.tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp, target)
        index[stem] = {"scenario_id": scene.scenario_id, "scene": str(scene_path),
                       "rollout": None if rollout_path is None else str(rollout_path),
                       "agents": int(data["agent"]["num_nodes"])}
        done += 1
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
    print(f"exported {done} scene(s) to {out / 'cache'} ({len(index)} in {index_path})")


if __name__ == "__main__":
    main()
