"""Writes CAT's scenes as GPUDrive scenarios with only the self-driving car
controlled (gpucat/scenes.py): OUT/cat_<scene>.json and OUT/index.json
(scene -> SDC id, adversary id, objects kept / in the scene).

Example (from the repository root, this repository's environment):
    python -m scripts.gpucat.export_scenes --out-dir logs/gpucat/scenes
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gpucat.scenes import ego_only_scenario  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402

MAX_AGENTS = 64  # GPUDrive's kMaxAgentCount


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = scene_files(args.scenes)[args.first:]
    files = files[: args.n] if args.n is not None else files
    index = {}
    for f in files:
        with open(f, "rb") as h:
            scene = Scene.from_description(pickle.load(h))
        data = ego_only_scenario(scene, f"cat_{f.stem}.json")
        (out / f"cat_{f.stem}.json").write_text(json.dumps(data))
        index[f.stem] = {"sdc_id": data["objects"][0]["id"], "adversary_id": data["metadata"]["adversary_id"],
                         "objects": len(data["objects"]), "kept": min(len(data["objects"]), MAX_AGENTS)}
    (out / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    over = sum(v["objects"] > MAX_AGENTS for v in index.values())
    print(f"exported {len(index)} scenes to {out}; {over} have more than {MAX_AGENTS} objects (the farthest are dropped)")


if __name__ == "__main__":
    main()
