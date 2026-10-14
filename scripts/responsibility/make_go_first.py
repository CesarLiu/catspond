"""Ground truth for assertive driving that is not unsafe: logged pairs in
which the yielder kept the right of way, rewritten so that it takes it.

The source is validate_right_of_way.py's pairs.csv over a scene folder
(--pair ooi; it names the scene files): pairs whose paths cross or merge, with a holder of the right
of way (any rule but same-direction, which the right of way leaves to RSS),
where the holder passed P* first by at least --min-lead s. The pair is
then rewritten so that the yielder y leaves its conflict zone --pet s
before the holder h enters its own, for each requested PET
(post-encroachment time), in one of two ways (--mode):

  delay   h's whole logged track is shifted later in time, y keeps its
          log. Before its logged start h comes from further back, at its
          first logged speed along its first logged heading (it is seen
          from the same step as in its log), so that it is there when y
          enters. Only for rules whose priority does not depend on time
          (DELAY_CASES, neither agent at a traffic light): shifting h moves
          it against the light phases and changes the order of arrival at
          an all-way stop, so the rewritten pair would have another holder
          (on held-out WOMD 14 of 41 pairs at PET 2 s did before this
          restriction). y's own motion is the logged,
          lawful one; only its timing against h makes it go first, so a
          measure of y's kinematics alone cannot tell the two apart.
  retime  y is re-timed along its own logged path (same path and
          headings), h keeps its log: y follows its log up to
          t0 = max(10, its first step, target - --lead-time), target being
          the step it must be clear by, then accelerates at the smallest
          constant rate a <= --accel (m/s^2) that clears it in time, up to
          --max-speed (m/s), and keeps that speed; past the end of its
          logged path it drives on straight along its last heading.

On CAT's 500 scenes the holder enters the conflict about 4 s into the
9.1 s clip, so a yielder that waited can rarely get through earlier at a
plausible acceleration: retime keeps 2 of 88 pairs at PET 1 s, delay
most of them. Everyone else keeps their logs (open loop: nobody reacts, so
the holder's need to brake is what the measure must read). A rewrite is
dropped when it cannot be made (no a <= --accel; the delayed holder would
enter after the clip ends), or when the moved agent's footprint overlaps
any other agent's logged one (responsibility.motion_filter.collisions;
overlaps its logged track also had are not counted). A short PET is the
more assertive: the holder, had it reacted, had to brake harder. The
ground truth is set by construction (passing order and PET), not by the
measure it tests (responsibility/forced_yield.py).

Writes OUT/log/<scenario>.pkl (the source scene), OUT/<mode>_pet<P>/
<scenario>.pkl per mode and PET, and next to each folder
<folder>.index.json (scene.sidecar_index) with the judged agent (always
the yielder) and the other, the rule, and the rewrite (delay in steps; a,
t0 and the target step).
The judged agent is put first among the objects of interest, so that
compute_responsibility --use-ooi --agent adv judges it when it is not the
self-driving car ("agent_is_sdc" in the index marks the others). Scenes
whose yielder passed first in the log (lawfully or not; no label) go to
OUT/log_yielder_first, the natural counterpart.

Example:
    python -m scripts.responsibility.make_go_first \\
        --pairs logs/responsibility/right_of_way/validation_interactive/pairs.csv \\
        --out-dir logs/responsibility/forced_yield/scenes --mode delay retime --pet 1 2 4 --workers 4
"""

import argparse
import csv
import json
import multiprocessing
import pickle
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.motion_filter import collisions  # noqa: E402
from responsibility.forced_yield import pair_priority  # noqa: E402
from responsibility.right_of_way import RightOfWayParams, _Path, _signals, _window  # noqa: E402
from responsibility.scene import Scene, sidecar_index  # noqa: E402

DT = 0.1  # s per step
CONTEXT = 10  # steps of history every scene keeps logged (DenseTNT's current step)
FAR = 1e6  # m: where an agent no longer seen is put for the overlap check
DELAY_CASES = ("stop-sign", "driveway", "through-road", "left-turn")  # priorities that do not depend on time
UNTIMED = ("none", "stop")  # controls without a light phase


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", required=True, help="validate_right_of_way.py's pairs.csv (--pair ooi).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", nargs="+", choices=["delay", "retime"], default=["delay", "retime"])
    p.add_argument("--pet", type=float, nargs="+", default=[1.0, 2.0, 4.0], help="s: post-encroachment times.")
    p.add_argument("--accel", type=float, default=3.0, help="m/s^2: the most the re-timed yielder accelerates.")
    p.add_argument("--max-speed", type=float, default=15.0, help="m/s: the fastest it drives.")
    p.add_argument("--lead-time", type=float, default=5.0, help="s: how long before the target it leaves its log.")
    p.add_argument("--min-lead", type=float, default=0.5, help="s: passing orders closer than this are not used.")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def zones(scene: Scene, y: int, h: int, params: RightOfWayParams):
    """The pair's conflict as forced_yield.py judges it (pair_priority:
    the first crossing of their tracks, which re-timing along the same
    paths keeps): (y's arc length to clear its zone, the step h first
    enters its zone, the step y has cleared it, None if it never does), on
    the steps both are seen. None when h does not hold the right of way
    there."""
    prio = pair_priority(scene, y, h, params=params)
    c = None if prio is None else prio.conflict
    if c is None or prio.holder != h:
        return None
    steps = _window(scene, y, h, scene.n_steps - 1)
    path_y = _Path(scene, y, int(steps[-1]), params.extend)
    in_y, _, _ = _signals(scene, y, h, path_y, c.s[0], c.angle, steps, params)
    in_h, _, _ = _signals(scene, h, y, _Path(scene, h, int(steps[-1]), params.extend), c.s[1], c.angle, steps,
                          params)
    half = float(in_y[0] + abs(path_y.s_at(steps[:1])[0] - c.s[0]))  # h_y: in_zone = h_y - |s - s*|
    entered = np.flatnonzero(in_h > 0)
    if not entered.size or in_h[0] > 0:  # the holder never entered, or was inside when first seen
        return None
    s_clear = c.s[0] + half
    whole = _Path(scene, y, scene.n_steps - 1, 0.0)
    cleared = np.flatnonzero(whole.s[:whole.n_real] >= s_clear)
    return s_clear, int(steps[entered[0]]), int(whole.steps[cleared[0]]) if cleared.size else None


def delay(scene: Scene, h: int, shift: int):
    """h's track ``shift`` steps later: (position [T, 2], heading [T],
    velocity [T, 2], valid [T]). Before its first logged state it drives
    at that state's velocity, so it is seen from the same step as in its
    log."""
    seen = np.flatnonzero(scene.valid[h])
    first, n = int(seen[0]), scene.n_steps
    position = np.zeros((n, 2))
    heading, velocity = np.zeros(n), np.zeros((n, 2))
    valid = np.zeros(n, dtype=bool)
    position[shift:] = scene.position[h, :n - shift, :2]
    heading[shift:] = scene.heading[h, :n - shift]
    velocity[shift:] = scene.velocity[h, :n - shift]
    valid[shift:] = scene.valid[h, :n - shift]
    back = np.arange(first, min(first + shift, n))  # steps now before its first logged state
    start, v0 = scene.position[h, first, :2].astype(float), scene.velocity[h, first].astype(float)
    position[back] = start - (first + shift - back)[:, None] * DT * v0
    heading[back], velocity[back], valid[back] = scene.heading[h, first], v0, True
    return position, heading, velocity, valid


def retime(scene: Scene, y: int, s_clear: float, target: int, args):
    """y's track re-timed along its logged path to reach arc length s_clear
    by step ``target`` (module docstring): (position [T, 2], heading [T],
    velocity [T, 2], valid [T], a, t0), or None when it cannot."""
    seen = np.flatnonzero(scene.valid[y])
    pts = scene.position[y, seen, :2].astype(float)
    s_log = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if s_clear > s_log[-1]:
        return None
    t0 = max(CONTEXT, int(seen[0]), target - int(round(args.lead_time / DT)))
    if target <= t0 or t0 not in seen or not scene.valid[y, t0:target + 1].all():
        return None
    i0 = int(np.searchsorted(seen, t0))
    s0, v0 = s_log[i0], float(np.linalg.norm(scene.velocity[y, t0]))
    t = np.arange(t0, scene.n_steps)

    def arc(a):
        v = np.minimum(v0 + a * (t - t0) * DT, max(args.max_speed, v0))
        return s0 + np.concatenate([[0.0], np.cumsum((v[1:] + v[:-1]) / 2 * DT)]), v

    if arc(args.accel)[0][target - t0] < s_clear:
        return None
    lo, hi = 0.0, args.accel
    if arc(0.0)[0][target - t0] >= s_clear:
        hi = 0.0
    for _ in range(40):  # the smallest a, to 1e-12 m/s^2
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if arc(mid)[0][target - t0] >= s_clear else (mid, hi)
    s_new, v = arc(hi)
    moving = np.concatenate([[True], np.diff(s_log) > 1e-3])  # interpolate over points that advance
    s_m, x_m, y_m = s_log[moving], pts[moving, 0], pts[moving, 1]
    head_m = np.unwrap(scene.heading[y, seen].astype(float))[moving]
    position = scene.position[y, :, :2].astype(float).copy()
    heading = scene.heading[y].astype(float).copy()
    velocity = scene.velocity[y].astype(float).copy()
    valid = scene.valid[y].copy()
    position[t, 0], position[t, 1] = np.interp(s_new, s_m, x_m), np.interp(s_new, s_m, y_m)
    heading[t] = np.interp(s_new, s_m, head_m)
    beyond = s_new > s_log[-1]  # past the logged path: straight on along its last heading
    last = np.array([np.cos(head_m[-1]), np.sin(head_m[-1])])
    position[t[beyond]] = pts[-1] + (s_new[beyond] - s_log[-1])[:, None] * last
    velocity[t] = v[:, None] * np.stack([np.cos(heading[t]), np.sin(heading[t])], -1)
    valid[t] = scene.valid[y, t] | (t > seen[-1])  # seen as long as in its log, and on to the end
    return position, heading, velocity, valid, float(hi), t0


def overlaps(scene: Scene, y: int, position: np.ndarray, valid: np.ndarray, t0: int) -> bool:
    """Whether y's new footprint overlaps another agent's from t0 on (y:
    the agent moved, the yielder or the holder)."""
    others = [i for i in range(scene.n_agents) if i != y and scene.valid[i, t0 + 1:].any()]
    traj = np.where(valid[t0 + 1:, None], position[t0 + 1:], FAR)
    return bool(collisions(scene, y, t0, traj[None], scene.n_steps - 1 - t0, others).any())


def first_ooi(scene: Scene, agent: int) -> Scene:
    """The scene with ``agent`` first among the objects of interest."""
    return replace(scene, objects_of_interest=[agent] + [i for i in scene.objects_of_interest if i != agent])


def candidates(rows, min_lead: float):
    """(row, yielder, holder, group) of the usable pairs: "log" where the
    holder passed first, "log_yielder_first" where the yielder did."""
    out = []
    for r in rows:
        if (r["error"] or not r["holder"] or r["case"] == "same-direction" or r["kind"] not in ("crossing", "merge")
                or r["first"] not in ("a", "b") or r["lead"] == "" or float(r["lead"]) < min_lead):
            continue
        holder, yielder = (r["a"], r["b"]) if r["holder"] == "a" else (r["b"], r["a"])
        out.append((r, yielder, holder, "log" if r["first"] == r["holder"] else "log_yielder_first"))
    return out


def make(job):
    row, yielder, holder, group, args = job
    scene = Scene.load(row["file"])
    y, h = scene.index(yielder), scene.index(holder)
    meta = {"agent": yielder, "other": holder, "case": row["case"], "agent_is_sdc": y == scene.sdc}
    out = {group: (first_ooi(scene, y), meta)}
    if group != "log":
        return row["scenario_id"], out, {}
    z = zones(scene, y, h, RightOfWayParams())
    dropped = {}
    for mode in args.mode:
        for pet in args.pet:
            name, steps = f"{mode}_pet{pet:g}", int(round(pet / DT))
            if z is None:
                dropped[name] = "no-zone"
                continue
            s_clear, holder_in, yielder_clear = z
            if mode == "delay":
                if row["case"] not in DELAY_CASES or {row["control_a"], row["control_b"]} - set(UNTIMED):
                    dropped[name] = "timed-priority"
                    continue
                if yielder_clear is None:
                    dropped[name] = "not-cleared"
                    continue
                shift = yielder_clear + steps - holder_in
                if shift <= 0 or holder_in + shift > scene.n_steps - 2:
                    dropped[name] = "infeasible"
                    continue
                moved, (position, heading, velocity, valid), t0 = h, delay(scene, h, shift), 0
                rewrite = {"delay": shift}
            else:
                target = holder_in - steps
                new = retime(scene, y, s_clear, target, args)
                if new is None:
                    dropped[name] = "infeasible"
                    continue
                position, heading, velocity, valid, a, t0 = new
                moved, rewrite = y, {"accel": round(a, 3), "t0": t0, "target": target}
            if overlaps(scene, moved, position, valid, t0):
                dropped[name] = "overlap"
                continue
            rewritten = scene.with_track(moved, position, heading.astype(np.float32), velocity.astype(np.float32),
                                         valid)
            out[name] = (first_ooi(rewritten, y), {**meta, "mode": mode, "pet": pet, **rewrite})
    return row["scenario_id"], out, dropped


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    with open(args.pairs) as f:
        rows = list(csv.DictReader(f))
    jobs = [(*c, args) for c in candidates(rows, args.min_lead)]
    print(f"{len(rows)} pairs, {len(jobs)} usable "
          f"({sum(j[3] == 'log' for j in jobs)} holder first, {sum(j[3] != 'log' for j in jobs)} yielder first)",
          flush=True)
    results = (map(make, jobs) if args.workers <= 1
               else multiprocessing.Pool(args.workers).imap_unordered(make, jobs, chunksize=4))
    index, why = {}, {}
    for sid, written, dropped in results:
        for name, (scene, meta) in written.items():
            folder = out / name
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / f"{sid}.pkl", "wb") as f:
                pickle.dump(scene.to_description(), f)
            index.setdefault(name, {})[sid] = meta
        for name, reason in dropped.items():
            why.setdefault(name, {}).setdefault(reason, 0)
            why[name][reason] += 1
    for name, entries in sorted(index.items()):
        sidecar_index(out / name).write_text(json.dumps(entries, indent=1, sort_keys=True))
        print(f"{name}: {len(entries)} scenes" + (f"; dropped {why[name]}" if name in why else ""), flush=True)
    for name in sorted(set(why) - set(index)):
        print(f"{name}: 0 scenes; dropped {why[name]}")
    return index, why


if __name__ == "__main__":
    main()
