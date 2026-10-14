"""Locates, on the CPU, the likely partner of each vehicle collision a GPUDrive
evaluation recorded (logs/gpucat/eval/<run>.csv: outcome crash and the route
completion reached, no step or partner), from the logged scene alone.

The ego is assumed to sit on its logged route (static.npz) at the arc length
completion x route_len with the route's heading and the SDC's logged size
(the policies stay within 10 m of the route, so this is approximate
laterally). Every logged vehicle box (valid steps, full size) that overlaps
the ego box there at some log index k is a candidate; the log index at which
the logged SDC passed the same point is k_log, so k - k_log > 0 means the
ego came later than the log did there (slower), < 0 earlier. The candidate
with the smallest |k - k_log| is reported, classified in the route's frame
(same direction within 45 degrees: follower if its centre is behind, leader
if ahead; opposite beyond 135 degrees; crossing otherwise).

    ~/venvs/cat39/bin/python -m scripts.gpucat.locate_crashes --run replay_nav_s0
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


# The helpers below were first imported from an earlier GPUDrive version of
# scripts/gpucat/audit_background.py, which a MetaDrive analysis of the same
# name later replaced; they are copied here unchanged.

def load(scene_dir, stem):
    """The GPUDrive scenario logs/gpucat/scenes/cat_<stem>.json as arrays:
    pos [n, 91, 2], head [n, 91], valid [n, 91], size [n, 2] (length, width)."""
    d = json.loads((scene_dir / f"cat_{stem}.json").read_text())
    objs = d["objects"]
    n = len(objs)
    pos = np.array([[[p["x"], p["y"]] for p in o["position"]] for o in objs], dtype=float)  # [n, 91, 2]
    head = np.array([o["heading"] for o in objs], dtype=float)
    valid = np.array([o["valid"] for o in objs], dtype=bool)
    size = np.array([[o["length"], o["width"]] for o in objs], dtype=float)
    types = [o["type"] for o in objs]
    goal = np.array([[o["goalPosition"]["x"], o["goalPosition"]["y"]] for o in objs], dtype=float)
    ids = [o["id"] for o in objs]
    return dict(d=d, n=n, pos=pos, head=head, valid=valid, size=size, types=types, goal=goal, ids=ids)


def corners(p, h, size):
    """[..., 4, 2] corners of boxes at p [..., 2], heading h [...], size [..., 2]."""
    c, sn = np.cos(h)[..., None], np.sin(h)[..., None]
    fx, fy = c * size[..., :1] / 2, sn * size[..., :1] / 2
    lx, ly = -sn * size[..., 1:] / 2, c * size[..., 1:] / 2
    f = np.concatenate([fx, fy], -1)
    l = np.concatenate([lx, ly], -1)
    return np.stack([p + f + l, p + f - l, p - f - l, p - f + l], -2)


def overlap(p, h, size, P, H, S):
    """Bool [N]: the box (p, h, size) overlaps each of N boxes (SAT)."""
    a = corners(p, np.asarray(h), np.asarray(size))[None].repeat(len(P), 0)  # [N, 4, 2]
    b = corners(P, H, S)
    ok = np.ones(len(P), dtype=bool)
    for box_h in (np.full(len(P), h), H):
        for ang in (box_h, box_h + np.pi / 2):
            ax = np.stack([np.cos(ang), np.sin(ang)], -1)[:, None]  # [N, 1, 2]
            pa, pb = (a * ax).sum(-1), (b * ax).sum(-1)
            ok &= ~((pa.max(1) < pb.min(1)) | (pb.max(1) < pa.min(1)))
    return ok


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def classify(p, h, size_e, q, hq):
    """(kind, lon, lat, heading difference in degrees) of the partner at q, hq in the ego's frame."""
    d = q - p
    lon = d[0] * np.cos(h) + d[1] * np.sin(h)
    lat = -d[0] * np.sin(h) + d[1] * np.cos(h)
    dh = abs(wrap(hq - h))
    if dh < np.pi / 4:
        kind = "follower" if lon < 0 else "leader"
    elif dh > 3 * np.pi / 4:
        kind = "opposite"
    else:
        kind = "crossing"
    return kind, float(lon), float(lat), float(np.degrees(dh))


def point_on_route(route, n, s):
    seg = route[1:n] - route[:n - 1]
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0], np.cumsum(L)])
    k = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(L) - 1))
    u = (s - cum[k]) / max(L[k], 1e-9)
    return route[k] + np.clip(u, 0, 1) * seg[k], float(np.arctan2(seg[k, 1], seg[k, 0]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--adversary", default="none")
    ap.add_argument("--scenes", default="logs/gpucat/scenes")
    ap.add_argument("--static", default="logs/gpucat/static.npz")
    args = ap.parse_args(argv)
    z = np.load(REPO / args.static)
    zi = {str(s): i for i, s in enumerate(z["stems"])}
    rows = [r for r in csv.DictReader(open(REPO / "logs/gpucat/eval" / f"{args.run}.csv"))
            if r["test_adversary"] == args.adversary and r["outcome"] == "crash"]
    kinds, timing = Counter(), []
    for r in rows:
        stem, c = r["scene"], float(r["completion"])
        s = load(REPO / args.scenes, stem)
        row = zi[stem]
        route, n = z["route"][row], int(z["route_n"][row])
        p, h = point_on_route(route, n, c * float(z["route_len"][row]))
        sdc = s["d"]["metadata"]["sdc_track_index"]
        size_e = s["size"][sdc]
        # the log index at which the logged SDC was closest to p
        v = np.flatnonzero(s["valid"][sdc])
        k_log = int(v[np.argmin(np.linalg.norm(s["pos"][sdc, v] - p, axis=1))])
        best = None
        adv_id = s["d"]["metadata"].get("adversary_id")
        for i in range(s["n"]):
            if i == sdc or s["types"][i] != "vehicle":
                continue
            ks = np.flatnonzero(s["valid"][i])
            if len(ks) == 0:
                continue
            P, H = s["pos"][i, ks], s["head"][i, ks]
            S = np.repeat(s["size"][i][None], len(ks), 0)
            hit = overlap(p, h, size_e, P, H, S)
            for j in np.flatnonzero(hit):
                k = int(ks[j])
                if best is None or abs(k - k_log) < abs(best[1] - k_log):
                    best = (i, k, P[j], H[j])
        if best is None:
            print(f"{stem}: completion {c:.2f}: no logged vehicle overlaps the route point (lateral offset?)")
            kinds["none-on-route"] += 1
            continue
        i, k, q, hq = best
        kind, lon, lat, dh = classify(p, h, size_e, q, hq)
        kinds[kind] += 1
        timing.append(k - k_log)
        print(f"{stem}: completion {c:.2f}  partner {i} ({'adversary' if s['ids'][i] == adv_id else 'other'})  "
              f"k={k} k_log={k_log} (ego {k - k_log:+d} steps vs log)  {kind}  lon {lon:+.1f} lat {lat:+.1f} "
              f"dh {dh:.0f}")
    t = np.array(timing)
    print(f"\n{args.run} / {args.adversary}: {len(rows)} crashes  {dict(kinds)}")
    if len(t):
        print(f"ego later than the log (k - k_log > 0): {np.sum(t > 0)}, earlier: {np.sum(t < 0)}, same: {np.sum(t == 0)};"
              f" median {np.median(t):+.0f} steps")


if __name__ == "__main__":
    main()
