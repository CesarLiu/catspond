"""Compares scenes exported by export_catk.py with catk's own cache of the
same WOMD scenarios (SMART_PLAN.md, S2): where a CAT scene is also in a
catk cache built from the original WOMD data, the two should hold the same
agents and map.

  agents  matched by track id: validity, role, type, size, and the largest
          position, heading and velocity differences over the steps both
          hold
  map     the token counts per map type (catk's pt_token type), and for every
          token of one side the distance from its middle point to the
          nearest token of the same type on the other side: a token with
          none within --tol metres is missing there

Runs in catk's environment (the caches hold torch tensors).

Example (from this repository's root):
    ~/venvs/catk/bin/python -m scripts.responsibility.verify_catk_export \\
        --export logs/catk/scenes --native ~/catk/data/womd_cache/validation
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np

TYPES = {0: "lane freeway", 1: "lane street", 2: "lane stop sign", 3: "bike lane / edge unknown",
         4: "edge boundary", 5: "edge median", 6: "line broken", 7: "line solid single", 8: "line double",
         9: "crosswalk/bump/driveway"}  # catk's point types (src/data_preprocess.py)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True, help="export_catk.py's output directory (cache/ and index.json).")
    p.add_argument("--native", required=True, help="catk's cache of the original WOMD scenarios.")
    p.add_argument("--tol", type=float, default=0.5, help="m: a token with no counterpart within this is missing.")
    return p.parse_args(argv)


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def compare_agents(a, b):
    ids_a, ids_b = _np(a["id"]).tolist(), _np(b["id"]).tolist()
    common = sorted(set(ids_a) & set(ids_b))
    out = {"agents": (len(ids_a), len(ids_b), len(common)), "valid_mismatch": 0, "role_mismatch": 0,
           "type_mismatch": 0, "pos": 0.0, "head": 0.0, "vel": 0.0, "shape": 0.0}
    for tid in common:
        i, j = ids_a.index(tid), ids_b.index(tid)
        va, vb = _np(a["valid_mask"][i]), _np(b["valid_mask"][j])
        out["valid_mismatch"] += int((va != vb).sum())
        out["role_mismatch"] += int((_np(a["role"][i]) != _np(b["role"][j])).any())
        out["type_mismatch"] += int(_np(a["type"][i]) != _np(b["type"][j]))
        both = va & vb
        if both.any():
            out["pos"] = max(out["pos"], float(np.abs(_np(a["position"][i])[both, :2] - _np(b["position"][j])[both, :2]).max()))
            dh = np.angle(np.exp(1j * (_np(a["heading"][i])[both] - _np(b["heading"][j])[both])))
            out["head"] = max(out["head"], float(np.abs(dh).max()))
            out["vel"] = max(out["vel"], float(np.abs(_np(a["velocity"][i])[both] - _np(b["velocity"][j])[both]).max()))
        out["shape"] = max(out["shape"], float(np.abs(_np(a["shape"][i]) - _np(b["shape"][j])).max()))
    return out


def unmatched(mid_a, type_a, mid_b, type_b, tol):
    """Per type: tokens of a with no token of b of that type within tol."""
    out = {}
    for t in np.unique(type_a):
        pa, pb = mid_a[type_a == t], mid_b[type_b == t]
        if len(pb) == 0:
            out[int(t)] = len(pa)
            continue
        missing = 0
        for k in range(0, len(pa), 2048):
            d = np.linalg.norm(pa[k:k + 2048, None] - pb[None], axis=-1).min(axis=1)
            missing += int((d > tol).sum())
        if missing:
            out[int(t)] = missing
    return out


def compare_map(a, b, tol):
    mid_a, mid_b = _np(a["map_save"]["traj_pos"])[:, 1], _np(b["map_save"]["traj_pos"])[:, 1]
    type_a, type_b = _np(a["pt_token"]["type"]), _np(b["pt_token"]["type"])
    counts = {int(t): (int((type_a == t).sum()), int((type_b == t).sum())) for t in np.union1d(type_a, type_b)}
    lights = {int(t): (int((_np(a["pt_token"]["light_type"]) == t).sum()), int((_np(b["pt_token"]["light_type"]) == t).sum()))
              for t in np.union1d(_np(a["pt_token"]["light_type"]), _np(b["pt_token"]["light_type"]))}
    return {"tokens": (len(type_a), len(type_b)), "per_type": counts, "lights": lights,
            "only_export": unmatched(mid_a, type_a, mid_b, type_b, tol),
            "only_native": unmatched(mid_b, type_b, mid_a, type_a, tol)}


def main(argv=None):
    args = parse_args(argv)
    export = Path(args.export)
    index = json.loads((export / "index.json").read_text())
    native = Path(args.native).expanduser()
    shared = [(stem, v["scenario_id"]) for stem, v in index.items() if (native / f"{v['scenario_id']}.pkl").exists()]
    print(f"{len(shared)} exported scenes are in the native cache: {[s for s, _ in shared]}")
    for stem, sid in sorted(shared, key=lambda x: int(x[0]) if x[0].isdigit() else math.inf):
        with open(export / "cache" / f"{stem}.pkl", "rb") as f:
            mine = pickle.load(f)
        with open(native / f"{sid}.pkl", "rb") as f:
            nat = pickle.load(f)
        ag = compare_agents(mine["agent"], nat["agent"])
        mp = compare_map(mine, nat, args.tol)
        print(f"\nscene {stem} ({sid})")
        print(f"  agents export/native/common {ag['agents']}; valid-step mismatches {ag['valid_mismatch']}, "
              f"role {ag['role_mismatch']}, type {ag['type_mismatch']}; max |diff| position {ag['pos']:.3f} m, "
              f"heading {ag['head']:.4f} rad, velocity {ag['vel']:.3f} m/s, size {ag['shape']:.3f} m")
        print(f"  map tokens export/native {mp['tokens']}; lights (light type: export, native) {mp['lights']}")
        for t, (n_a, n_b) in mp["per_type"].items():
            gone = mp["only_native"].get(t, 0)
            extra = mp["only_export"].get(t, 0)
            flag = "" if not gone and not extra else f"   <- {gone} native token(s) missing from the export, {extra} extra"
            print(f"    {TYPES.get(t, t):>26}: {n_a:5d} / {n_b:5d}{flag}")


if __name__ == "__main__":
    main()
