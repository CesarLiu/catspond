"""Writes CAT's scenes with the roles of the two objects of interest swapped:
the other object of interest becomes the self-driving car (the ego CAT
attacks and MetaDrive drives) and the logged self-driving car becomes the
adversary (responsibility/swap.py).

Scenes whose other object of interest is not a vehicle valid at every step
are skipped (41 of CAT's 500). The rest keep their file names, so the
folder can be read like raw_scenes_500: by MetaDrive (cat_advgen.py and
cat_RLtrain.py --scenes_dir, which splits it 0-399 / 400 on as CAT does),
export_adv_scenes.py, compute_responsibility.py and
visualize_responsibility.py (--scenes). OUT.index.json, next to the folder
(MetaDrive reads every file inside it), lists every scene, swapped (with
both ids) or skipped (with the reason).

Example (from the repository root):
    python -m scripts.responsibility.swap_roles --scenes raw_scenes_500 --out-dir raw_scenes_500_swapped
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.scene import scene_files  # noqa: E402
from responsibility.swap import index_path, scene_split, swap_problem, swapped  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    if out.resolve() == Path(args.scenes).resolve():
        raise SystemExit("--out-dir must differ from --scenes")
    out.mkdir(parents=True, exist_ok=True)
    index = {}
    for path in scene_files(args.scenes):
        with open(path, "rb") as f:
            description = pickle.load(f)
        problem = swap_problem(description)
        if problem is not None:
            index[path.stem] = {"swapped": False, "reason": problem}
            continue
        swap = swapped(description)
        with open(out / path.name, "wb") as f:
            pickle.dump(swap, f)
        index[path.stem] = {"swapped": True, **swap["metadata"]["roles_swapped"]}
    train, test = scene_split(out)
    n = sum(v["swapped"] for v in index.values())
    index_path(out).write_text(json.dumps({"roles_swapped": True, "source": str(Path(args.scenes).resolve()),
                                         "split": {"train": train, "test": test}, "scenes": index}, indent=1))
    print(f"{n} of {len(index)} scenes swapped into {out} ({train} train, {test} test); "
          f"{len(index) - n} skipped (see {index_path(out)})")


if __name__ == "__main__":
    main()
