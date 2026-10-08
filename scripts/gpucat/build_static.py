"""Writes gpucat/static.py's per-scene data for every scene to one .npz
(routes, road-edge and yellow-line segments, slot sizes and types).

Example (this repository's environment):
    python -m scripts.gpucat.build_static --out logs/gpucat/static.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from gpucat.static import pad, scene_static  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    files = scene_files(args.scenes)
    rows = [scene_static(Scene.load(f)) for f in files]
    out = {"stems": np.array([f.stem for f in files]),
           "route": pad([r["route"] for r in rows]), "route_n": np.array([len(r["route"]) for r in rows]),
           "route_len": np.array([r["route_len"] for r in rows]),
           "edges": pad([r["edges"] for r in rows]), "edges_n": np.array([len(r["edges"]) for r in rows]),
           "yellow": pad([r["yellow"] for r in rows]), "yellow_n": np.array([len(r["yellow"]) for r in rows]),
           "sizes": np.stack([r["sizes"] for r in rows]), "types": np.stack([r["types"] for r in rows]),
           "n_agents": np.array([r["n_agents"] for r in rows])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **out)
    print(f"{len(files)} scenes -> {args.out}: route points <= {out['route'].shape[1]}, road-edge segments "
          f"<= {out['edges'].shape[1]} (median {int(np.median(out['edges_n']))}), yellow segments <= "
          f"{out['yellow'].shape[1]} (median {int(np.median(out['yellow_n']))}); scenes without road edges "
          f"{int((out['edges_n'] == 0).sum())}; route length median {np.median(out['route_len']):.0f} m")


if __name__ == "__main__":
    main()
