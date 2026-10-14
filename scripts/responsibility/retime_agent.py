"""Synthetic variants of one agent's logged drive for the aggression test
(responsibility/aggression_test.py): the same path, driven at a speed scaled
by --factor from step --start on, after a linear ramp of --ramp steps
(factor < 1: holding back; > 1: pressing on). Beyond the end of the logged
path it continues along the last heading. Velocities are the finite
differences of the new positions (0.1 s), headings follow the velocity above
0.5 m/s and are held below it; everything before --start is the log.

Scenes in which the agent is not valid from --start to the end are skipped,
since the path could not be re-timed through a gap. Each written scene keeps
the source's file name and records metadata.synthetic = {agent, factor,
start, ramp}.

    python -m scripts.responsibility.retime_agent --scenes raw_scenes_500 \\
        --out logs/responsibility/validity/scenes/sdc_speed0.5 --factor 0.5 --agent sdc --n 100
"""
import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.scene import Scene  # noqa: E402


def retime(sd: dict, agent: str, factor: float, start: int, ramp: int):
    """The scene description with the agent re-timed, or None when it has a gap after start."""
    scene = Scene.from_description(sd)
    ag = scene.sdc if agent == "sdc" else [j for j in scene.objects_of_interest if j != scene.sdc][0]
    tid = scene.track_ids[ag]
    st = sd["tracks"][tid]["state"]
    valid = np.asarray(st["valid"]).astype(bool)
    T = len(valid)
    if not valid[start:].all():
        return None
    pos = np.asarray(st["position"], dtype=np.float64)[:, :2]
    seg = np.linalg.norm(np.diff(pos[start:], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])  # arc length from the start step
    r = np.array([1 + (factor - 1) * min(1.0, (t + 1) / ramp) for t in range(len(seg))])
    s_new = np.concatenate([[0.0], np.cumsum(seg * r)])
    path = pos[start:]
    keep = np.concatenate([[True], seg > 1e-6])  # interpolate over non-zero segments only
    sk, pk = s[keep], path[keep]
    new = np.zeros_like(path)
    for t, sv in enumerate(s_new):
        if sv <= sk[-1] or len(sk) < 2:
            new[t, 0] = np.interp(sv, sk, pk[:, 0])
            new[t, 1] = np.interp(sv, sk, pk[:, 1])
        else:  # past the end of the logged path: straight on along its last heading
            d = pk[-1] - pk[-2]
            d = d / max(np.linalg.norm(d), 1e-9)
            new[t] = pk[-1] + d * (sv - sk[-1])
    full = pos.copy()
    full[start:] = new
    vel = np.zeros_like(full)
    vel[1:] = np.diff(full, axis=0) / 0.1
    vel[0] = vel[1]
    vel[:start + 1] = np.asarray(st["velocity"])[:start + 1, :2]
    heading = np.asarray(st["heading"], dtype=np.float64).copy()
    sp = np.linalg.norm(vel, axis=1)
    for t in range(start + 1, T):
        heading[t] = np.arctan2(vel[t, 1], vel[t, 0]) if sp[t] > 0.5 else heading[t - 1]
    out = copy.deepcopy(sd)
    st2 = out["tracks"][tid]["state"]
    st2["position"] = np.asarray(st2["position"]).copy()
    st2["position"][:, :2] = full
    st2["velocity"] = np.asarray(st2["velocity"]).copy()
    st2["velocity"][:, :2] = vel
    st2["heading"] = heading.astype(np.asarray(st["heading"]).dtype)
    out["metadata"]["synthetic"] = {"agent": tid, "factor": factor, "start": start, "ramp": ramp}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--out", required=True)
    p.add_argument("--factor", type=float, required=True, help="speed scale after the ramp")
    p.add_argument("--agent", default="sdc", choices=["sdc", "adv"], help="adv: the other object of interest")
    p.add_argument("--n", type=int, default=100, help="scenes 0 .. n-1")
    p.add_argument("--start", type=int, default=10, help="first re-timed step (CAT plans from step 10)")
    p.add_argument("--ramp", type=int, default=15, help="steps to reach the full factor")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    skipped = 0
    for i in range(a.n):
        src = Path(a.scenes) / f"{i}.pkl"
        with open(src, "rb") as f:
            sd = pickle.load(f)
        new = retime(sd, a.agent, a.factor, a.start, a.ramp)
        if new is None:
            skipped += 1
            continue
        with open(out / src.name, "wb") as f:
            pickle.dump(new, f)
    print(f"wrote {a.n - skipped}, skipped {skipped} (agent not valid throughout)")


if __name__ == "__main__":
    main()
