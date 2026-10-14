"""CPU audit of why GPUDrive PPO episodes end early (observation O1): it
checks the out-of-road tests, the ego's dynamics, the episode length and
when CAT's adversary attacks. No simulator is started.

  A  static     clearance (m) between the logged SDC footprint and the road
                edges / solid yellow lines that gpucat/env.py tests
                (logs/gpucat/static.npz), with the logged SDC size (what
                GPUDrive uses) and with MetaDrive's DefaultVehicle 4.515 x
                1.852 m; edges that cut the logged centre path; the segment
                counts within 1 m of the logged footprint
  B  crosscheck gpucat's out-of-road tests (gpucat/geometry.py, exactly as
                env.step calls them) applied to MetaDrive TD3 test rollouts
                (rollouts_eval/<policy>/none): the first step at which
                GPUDrive would end the episode, against MetaDrive's own
                verdict; MetaDrive's lateral (InterpolatingLine Option 1)
                against gpucat's polyline distance
  C  dynamics   the logged SDC's longitudinal acceleration and path
                curvature against the bicycle's limits ([-6, 4] m/s^2,
                +-0.3 1/m), and GPUDrive's forward bicycle model
                (src/dynamics.hpp) under braking at standstill
  D  timing     the step at which CAT's chosen candidate (chosen_cat,
                against the logged route) first collides with the logged
                SDC, and when MetaDrive TD3 episodes and GPUDrive PPO
                evaluation episodes end

Writes <dir>/static.csv, crosscheck.csv, timing.csv and report.json (CPU, about 7 minutes).

    cd /home/Cesar/catspond && ~/venvs/cat39/bin/python -m scripts.gpucat.audit_o1 --out <dir>
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpucat.geometry import boxes_overlap, route_coords, segments_touch_box  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

MD_SIZE = (4.515, 1.852)  # MetaDrive DefaultVehicle (component/vehicle/vehicle_type.py)
TEST = [str(i) for i in range(400, 500)]
POLICIES = ["td3_replay_s0", "td3_replay_s1", "td3_replay_s2", "td3_cat_s0", "td3_cat_s1", "td3_cat_s2",
            "td3_cat_fair2_0.1_s0", "td3_cat_fair2_0.1_s1", "td3_cat_fair2_0.1_s2", "replay"]


# ------------------------------------------------------------------ geometry helpers (numpy)

def box_segment_distance(c, yaw, length, width, segs):
    """[T, N] distance (m) between boxes (c [T, 2], yaw [T]) and segments
    [N, 4]; 0 when they touch."""
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]

    def local(x, y):
        dx, dy = x[None] - c[:, None, 0], y[None] - c[:, None, 1]
        return dx * cos + dy * sin, -dx * sin + dy * cos

    x0, y0 = local(segs[:, 0], segs[:, 1])
    x1, y1 = local(segs[:, 2], segs[:, 3])
    hx, hy = 0.5 * length, 0.5 * width

    def point_rect(px, py):
        return np.hypot(np.maximum(np.abs(px) - hx, 0), np.maximum(np.abs(py) - hy, 0))

    def point_seg(px, py):
        ex, ey = x1 - x0, y1 - y0
        t = np.clip(((px - x0) * ex + (py - y0) * ey) / np.maximum(ex * ex + ey * ey, 1e-12), 0, 1)
        return np.hypot(x0 + t * ex - px, y0 + t * ey - py)

    d = np.minimum(point_rect(x0, y0), point_rect(x1, y1))
    for px, py in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy)):
        d = np.minimum(d, point_seg(px, py))
    # touching (Liang-Barsky, as gpucat.geometry.segments_touch_box)
    dx, dy = x1 - x0, y1 - y0
    lo, hi = np.zeros_like(x0), np.ones_like(x0)
    ok = np.ones_like(x0, dtype=bool)
    for p, q in ((-dx, x0 + hx), (dx, hx - x0), (-dy, y0 + hy), (dy, hy - y0)):
        par = np.abs(p) < 1e-12
        ok &= ~(par & (q < 0))
        r = q / np.where(par, 1.0, p)
        lo = np.where(~par & (p < 0), np.maximum(lo, r), lo)
        hi = np.where(~par & (p > 0), np.minimum(hi, r), hi)
    return np.where(ok & (lo <= hi), 0.0, d)


def segments_cross(path, segs):
    """Indices of the segments [N, 4] the polyline path [P, 2] properly crosses."""
    if len(segs) == 0 or len(path) < 2:
        return np.zeros(0, dtype=int)
    a0, a1 = path[:-1][:, None], path[1:][:, None]
    b0, b1 = segs[None, :, :2], segs[None, :, 2:]

    def cross(o, p, q):
        return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])

    d1, d2 = cross(a0, a1, b0), cross(a0, a1, b1)
    d3, d4 = cross(b0, b1, a0), cross(b0, b1, a1)
    hit = (d1 * d2 < 0) & (d3 * d4 < 0)
    return np.flatnonzero(hit.any(0))


def metadrive_lateral(route_pts, p):
    """MetaDrive's |lateral| of p on PointLane(route_pts) (utils/interpolating_line.py,
    _get_properties + local_coordinates Option 1)."""
    pts = np.asarray(route_pts)[:, :2]
    segs = []
    i = 0
    while i < len(pts) - 1:
        for j in range(i + 1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) > 1:
                break
        s, e = pts[i], pts[j]
        if np.linalg.norm(s - e) >= 1e-6:
            segs.append((s, e))
        i = j
    if not segs:
        return float(np.linalg.norm(p - pts[0]))
    ret = []
    for s, e in segs:
        L = np.linalg.norm(e - s)
        d = (e - s) / L
        n = np.array([-d[1], d[0]])
        dx, dy = p - s
        lon = dx * d[0] + dy * d[1]
        lat = dx * n[0] + dy * n[1]
        if lon < 0.0 and np.hypot(dx, dy) < L * 2:
            return abs(lat)
        ret.append(abs(lat))
    return min(ret)


def path_yaw(xy):
    d = np.diff(xy, axis=0)
    y = np.arctan2(d[:, 1], d[:, 0])
    return np.concatenate([y, y[-1:]])


def pct(x, qs=(1, 5, 50, 95, 99)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {f"p{q}": round(float(np.percentile(x, q)), 3) for q in qs} if len(x) else {}


# ------------------------------------------------------------------ A: static geometry vs the logged path

def part_a(static, scenes, stems):
    idx = {s: i for i, s in enumerate(static["stems"])}
    rows = []
    for s in stems:
        k = idx[s]
        sc = scenes[s]
        sdc = sc.sdc
        pos = sc.position[sdc, :, :2].astype(float)
        yaw = sc.heading[sdc].astype(float)
        L, W = static["sizes"][k, 0]
        edges = static["edges"][k, : static["edges_n"][k]]
        yellow = static["yellow"][k, : static["yellow_n"][k]]
        row = {"scene": s, "length": L, "width": W, "route_len": float(static["route_len"][k])}
        for name, segs in (("edge", edges), ("yellow", yellow)):
            for tag, (l, w) in (("log", (L, W)), ("md", MD_SIZE)):
                if len(segs):
                    d = box_segment_distance(pos, yaw, l, w, segs)  # [91, N]
                    per_step = d.min(1)
                    row[f"{name}_clear_{tag}"] = float(per_step.min())
                    row[f"{name}_touch_steps_{tag}"] = int((per_step <= 0).sum())
                    row[f"{name}_first_touch_{tag}"] = int(np.argmax(per_step <= 0)) if (per_step <= 0).any() else -1
                    row[f"{name}_segs_within1m_{tag}"] = int((d.min(0) < 1.0).sum())
                    row[f"{name}_steps_within0.5m_{tag}"] = int((per_step < 0.5).sum())
                else:
                    row[f"{name}_clear_{tag}"] = float("inf")
                    row[f"{name}_touch_steps_{tag}"] = 0
                    row[f"{name}_first_touch_{tag}"] = -1
                    row[f"{name}_segs_within1m_{tag}"] = 0
                    row[f"{name}_steps_within0.5m_{tag}"] = 0
            row[f"{name}_cross_centre"] = len(segments_cross(pos, segs))
            row[f"{name}_n"] = len(segs)
        rows.append(row)
    return rows


def summarize_a(rows):
    out = {"scenes": len(rows)}
    w = np.array([r["width"] for r in rows])
    l = np.array([r["length"] for r in rows])
    out["sdc_size_log"] = {"length": pct(l, (5, 50, 95)), "width": pct(w, (5, 50, 95))}
    out["route_len_lt_1m"] = int(sum(r["route_len"] < 1.0 for r in rows))
    for name in ("edge", "yellow"):
        for tag in ("log", "md"):
            c = np.array([r[f"{name}_clear_{tag}"] for r in rows])
            out[f"{name}_{tag}"] = {
                "scenes_touch_on_log": int((c <= 0).sum()),
                "scenes_clear_lt_0.25m": int((c < 0.25).sum()),
                "scenes_clear_lt_0.5m": int((c < 0.5).sum()),
                "scenes_clear_lt_1.0m": int((c < 1.0).sum()),
                "clearance": pct(c[np.isfinite(c)], (5, 25, 50)),
                "median_segs_within1m": float(np.median([r[f"{name}_segs_within1m_{tag}"] for r in rows])),
                "scenes_any_seg_within1m": int(sum(r[f"{name}_segs_within1m_{tag}"] > 0 for r in rows)),
                "mean_steps_within0.5m": float(np.mean([r[f"{name}_steps_within0.5m_{tag}"] for r in rows])),
            }
        out[f"{name}_scenes_centre_path_crosses"] = int(sum(r[f"{name}_cross_centre"] > 0 for r in rows))
    return out


# ------------------------------------------------------------------ B: gpucat's tests on MetaDrive rollouts

def part_b(static, scenes, stems, root):
    idx = {s: i for i, s in enumerate(static["stems"])}
    rows = []
    for pol in POLICIES:
        d = root / pol / "none"
        if not d.exists():
            continue
        for s in stems:
            f = d / f"{s}.pkl"
            if not f.exists():
                continue
            r = pickle.load(open(f, "rb"))
            k = idx[s]
            pos = torch.tensor(np.asarray(r["ego"]["position"], dtype=np.float64))
            yaw = torch.tensor(np.asarray(r["ego"]["heading"], dtype=np.float64))
            T = len(pos)
            route = torch.tensor(static["route"][k], dtype=torch.float64)[None].expand(T, -1, -1)
            rn = torch.full((T,), int(static["route_n"][k]))
            lateral, sarc = route_coords(pos, route, rn)
            res = {"policy": pol, "scene": s, "md_reason": r["end"]["reason"], "md_step": int(r["end"]["step"]),
                   "md_rc": float(r["end"]["route_completion"])}
            for name in ("edges", "yellow"):
                segs = torch.tensor(static[name][k], dtype=torch.float64)[None].expand(T, -1, -1)
                n = torch.full((T,), int(static[f"{name}_n"][k]))
                for tag, (l, w) in (("log", tuple(static["sizes"][k, 0])), ("md", MD_SIZE)):
                    hit = segments_touch_box(pos, yaw, torch.full((T,), l, dtype=torch.float64),
                                             torch.full((T,), w, dtype=torch.float64), segs, n).numpy()
                    hit[0] = False  # env tests after the first step
                    res[f"{name}_{tag}_first"] = int(np.argmax(hit)) if hit.any() else -1
            off = (lateral > 10.0).numpy()
            res["offroute_gpu_first"] = int(np.argmax(off)) if off.any() else -1
            sc = scenes[s]
            mdroute = sc.position[sc.sdc, :, :2]
            md_lat = np.array([metadrive_lateral(mdroute, p) for p in pos.numpy()])
            res["offroute_md_first"] = int(np.argmax(md_lat > 10)) if (md_lat > 10).any() else -1
            res["lat_gpu_minus_md_max"] = float((lateral.numpy() - md_lat).max())
            res["comp_gpu_end"] = float(sarc[-1] / max(static["route_len"][k], 1.0))
            rows.append(res)
    return rows


def summarize_b(rows):
    out = {}
    for tag in ("log", "md"):
        first = []
        for r in rows:
            c = [r[f"edges_{tag}_first"], r[f"yellow_{tag}_first"], r["offroute_gpu_first"]]
            c = [x for x in c if x >= 0]
            first.append(min(c) if c else -1)
        first = np.array(first)
        reason = np.array([r["md_reason"] for r in rows])
        step = np.array([r["md_step"] for r in rows])
        earlier = (first >= 0) & (first < step)
        not_oor = reason != "out_of_road"
        out[f"gpu_size_{tag}"] = {
            "episodes": len(rows),
            "md_not_oor_but_gpu_oor_before_md_end": int((earlier & not_oor).sum()),
            "md_not_oor": int(not_oor.sum()),
            "by_md_reason": {k: int((earlier & (reason == k)).sum()) for k in np.unique(reason)},
            "edge_trigger_not_oor": int(sum((r[f"edges_{tag}_first"] >= 0) & (r[f"edges_{tag}_first"] < r["md_step"])
                                            & (r["md_reason"] != "out_of_road") for r in rows)),
            "yellow_trigger_not_oor": int(sum((r[f"yellow_{tag}_first"] >= 0) & (r[f"yellow_{tag}_first"] < r["md_step"])
                                              & (r["md_reason"] != "out_of_road") for r in rows)),
            "md_oor_gpu_also_by_md_end": int(((first >= 0) & (first <= step) & ~not_oor).sum()),
            "md_oor": int((~not_oor).sum()),
        }
    out["offroute_gpu_before_md"] = int(sum((r["offroute_gpu_first"] >= 0) and
                                            (r["offroute_md_first"] < 0 or r["offroute_gpu_first"] < r["offroute_md_first"])
                                            for r in rows))
    out["lat_gpu_minus_md_max"] = pct([r["lat_gpu_minus_md_max"] for r in rows], (50, 90, 99))
    reason = [r["md_reason"] for r in rows]
    out["md_reasons"] = {k: reason.count(k) for k in sorted(set(reason))}
    steps = np.array([r["md_step"] for r in rows])
    out["md_end_step"] = pct(steps, (10, 25, 50, 75, 90))
    out["md_episodes_longer_than_90"] = int((steps > 90).sum())
    out["md_arrive_after_90"] = int(sum((r["md_step"] > 90) and r["md_reason"] == "arrive_dest" for r in rows))
    for k in sorted(set(reason)):
        out[f"md_end_step_{k}"] = pct([r["md_step"] for r in rows if r["md_reason"] == k], (25, 50, 75))
    return out


# ------------------------------------------------------------------ C: dynamics

def part_c(scenes, stems, static):
    idx = {s: i for i, s in enumerate(static["stems"])}
    acc, kap, kap_w, inv_acc, inv_steer = [], [], [], [], []
    for s in stems:
        sc = scenes[s]
        v = sc.velocity[sc.sdc, :, :2].astype(float) if hasattr(sc, "velocity") else None
        p = sc.position[sc.sdc, :, :2].astype(float)
        h = sc.heading[sc.sdc].astype(float)
        speed = np.linalg.norm(v, axis=1)
        a = np.diff(speed) / 0.1
        acc.extend(a)
        # GPUDrive's inverse bicycle (src/dynamics.hpp, useEstimatedYaw): target yaw from the velocity
        tyaw = np.arctan2(v[1:, 1], v[1:, 0])
        den = speed[:-1] * 0.1 + 0.5 * a * 0.01
        dy = (tyaw - h[:-1] + np.pi) % (2 * np.pi) - np.pi
        st = np.where(np.abs(den) > 1e-6, dy / np.where(np.abs(den) > 1e-6, den, 1), 0.0)
        moving = speed[:-1] > 1.0
        inv_acc.extend(a[moving])
        inv_steer.extend(st[moving])
        # geometric path curvature over ~1 m chords
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        if cum[-1] < 3:
            continue
        g = np.arange(0, cum[-1], 1.0)
        q = np.stack([np.interp(g, cum, p[:, 0]), np.interp(g, cum, p[:, 1])], 1)
        th = np.unwrap(np.arctan2(np.diff(q[:, 1]), np.diff(q[:, 0])))
        k = np.diff(th) / 1.0
        kap.extend(k)
    acc, kap, inv_acc, inv_steer = map(np.asarray, (acc, kap, inv_acc, inv_steer))
    out = {"long_accel_mps2": pct(acc, (0.1, 1, 50, 99, 99.9)),
           "frac_accel_below_-6": float((acc < -6).mean()), "frac_accel_above_4": float((acc > 4).mean()),
           "path_curvature_1pm": pct(np.abs(kap), (50, 95, 99, 99.9)),
           "frac_path_metres_curv_gt_0.3": float((np.abs(kap) > 0.3).mean()),
           "inverse_bicycle_moving_accel": pct(inv_acc, (0.5, 50, 99.5)),
           "inverse_bicycle_moving_steer": pct(np.abs(inv_steer), (50, 99, 99.9)),
           "frac_inverse_steer_gt_0.3_moving": float((np.abs(inv_steer) > 0.3).mean()),
           "frac_inverse_accel_out_of_range_moving": float(((inv_acc < -6) | (inv_acc > 4)).mean())}
    # GPUDrive forwardBicycleModel at standstill (velocity sign is lost: speed = |v|)
    sims = {}
    for a_cmd in (-6.0, -2.0, -0.5, 0.0):
        x, vx = 0.0, 0.0
        for _ in range(20):
            speed = abs(vx)
            x = x + vx * 0.1 + 0.5 * a_cmd * 0.01
            vx = speed + a_cmd * 0.1
        sims[f"accel_{a_cmd}"] = {"displacement_after_2s_m": round(x, 3)}
    x, vx = 0.0, 10.0
    traj = []
    for _ in range(40):
        speed = abs(vx)
        x = x + vx * 0.1 + 0.5 * -6.0 * 0.01
        vx = speed - 0.6
        traj.append(x)
    sims["brake_-6_from_10mps"] = {"max_forward_m": round(max(traj), 3), "x_after_4s_m": round(traj[-1], 3)}
    sims["action_for_zero_accel"] = 2 * (0.0 + 6.0) / 10.0 - 1.0
    out["standstill_braking_sim"] = sims
    return out


# ------------------------------------------------------------------ D: attack timing

def yaw_of(xy):
    d = np.diff(xy, axis=0)
    y = np.arctan2(d[:, 1], d[:, 0])
    return np.concatenate([y, y[-1:]])


def part_d(cand_path, stems, root, eval_dir, static):
    z = np.load(cand_path)
    sidx = {str(s): i for i, s in enumerate(z["stems"])}
    stidx = {s: i for i, s in enumerate(static["stems"])}
    rows = []
    for s in stems:
        i = sidx[s]
        c = z["candidates"][i, z["chosen_cat"][i]].astype(float)  # [80, 2], scene steps 11..90
        e = z["ego_route"][i]
        ov, av = z["ov_size"][i], z["av_size"][i]
        cy, ey = yaw_of(c), yaw_of(e)
        hit = boxes_overlap(torch.tensor(e), torch.tensor(ey), torch.tensor(av)[None].expand(80, 2),
                            torch.tensor(c)[:, None], torch.tensor(cy)[:, None],
                            torch.tensor(ov)[None, None].expand(80, 1, 2))[:, 0].numpy()
        dist = np.linalg.norm(c - e, axis=1)
        first_box = int(np.argmax(hit)) + 11 if hit.any() else -1
        first_2m = int(np.argmax(dist < 2.0)) + 11 if (dist < 2.0).any() else -1
        # spatial: first step at which the candidate is within 2 m of any point of the ego's logged path
        dsp = np.linalg.norm(c[:, None] - e[None], axis=-1).min(1)
        first_path = int(np.argmax(dsp < 2.0)) + 11 if (dsp < 2.0).any() else -1
        # logged ego arc length -> step, for the completion proxy
        k = stidx[s]
        rows.append({"scene": s, "attack_box": first_box, "attack_2m": first_2m, "near_path_2m": first_path,
                     "min_gap_same_time": float(dist.min()), "chosen_cat": int(z["chosen_cat"][i])})
    att = {r["scene"]: r for r in rows}
    out = {}
    a = np.array([r["attack_box"] for r in rows])
    has = a >= 0
    out["scenes_with_box_collision_vs_log"] = int(has.sum())
    out["attack_box_step"] = pct(a[has], (10, 25, 50, 75, 90))
    for t in (20, 27, 40, 60):
        out[f"frac_attacks_after_step_{t}"] = round(float((a[has] > t).mean()), 3)
    a2 = np.array([r["attack_2m"] for r in rows])
    out["attack_2m_step"] = pct(a2[a2 >= 0], (10, 25, 50, 75, 90))
    # MetaDrive TD3 episodes (adv_mode none and cat): end step vs the attack step
    md = {}
    for pol in POLICIES[:-1]:
        for mode in ("none", "cat"):
            d = root / pol / mode
            if not d.exists():
                continue
            for s in stems:
                f = d / f"{s}.pkl"
                if f.exists():
                    r = pickle.load(open(f, "rb"))
                    md.setdefault(mode, []).append((s, int(r["end"]["step"]), r["end"]["reason"]))
    for mode, lst in md.items():
        ends = np.array([x[1] for x in lst])
        atk = np.array([att[x[0]]["attack_box"] for x in lst])
        ok = atk >= 0
        before = ends[ok] < atk[ok]
        out[f"md_{mode}_episodes"] = len(lst)
        out[f"md_{mode}_end_before_attack"] = round(float(before.mean()), 3)
        out[f"md_{mode}_end_before_attack_non_crash"] = round(float(np.mean(
            [e < t and r != "crash_vehicle" for (s, e, r), t in zip(lst, atk) if t >= 0])), 3)
    # GPUDrive PPO evaluation (per-scene outcome and completion): end step by the logged pace
    g = {}
    for f in sorted(eval_dir.glob("*.csv")):
        for r in csv.DictReader(open(f)):
            if r["test_adversary"] not in ("none", "cat"):
                continue
            g.setdefault((f.stem, r["test_adversary"]), []).append((r["scene"], r["outcome"], float(r["completion"])))
    out["gpudrive_eval"] = {}
    for (run, adv), lst in g.items():
        steps = []
        for s, o, comp in lst:
            sl = LOG_ARC[s]
            frac = sl / max(sl[-1], 1.0)
            # the step at which the logged SDC reached the same completion (arrival: 0.95)
            est = int(np.searchsorted(frac, comp if o != "arrive" else 0.95))
            steps.append((s, o, min(est, 90)))
        atk = np.array([att[s]["attack_box"] for s, _, _ in steps])
        est = np.array([e for _, _, e in steps])
        oc = np.array([o for _, o, _ in steps])
        ok = atk >= 0
        out["gpudrive_eval"][f"{run}/{adv}"] = {
            "outcomes": {k: int((oc == k).sum()) for k in np.unique(oc)},
            "est_end_before_attack_all": round(float((est[ok] < atk[ok]).mean()), 3),
            "est_end_before_attack_oor": int(((est < atk) & ok & (oc == "out_of_road")).sum()),
            "est_end_step_oor": pct(est[oc == "out_of_road"], (25, 50, 75)),
            "est_end_step_crash": pct(est[oc == "crash"], (25, 50, 75)),
        }
    return rows, out


LOG_ARC = {}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--rollouts", default="rollouts_eval")
    p.add_argument("--eval", default="logs/gpucat/eval")
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--all-scenes", action="store_true", help="Part A on all 500 scenes, not only 400-499.")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(REPO / args.static)
    static = {k: z[k] for k in z.files}
    static["stems"] = [str(s) for s in static["stems"]]
    stems_a = [str(i) for i in range(500)] if args.all_scenes else TEST
    scenes = {s: Scene.load(REPO / args.scenes / f"{s}.pkl") for s in sorted(set(stems_a) | set(TEST), key=int)}
    for s, sc in scenes.items():
        pth = sc.position[sc.sdc, :, :2].astype(float)
        LOG_ARC[s] = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pth, axis=0), axis=1))])
    report = {}
    rows_a = part_a(static, scenes, stems_a)
    report["A_static"] = summarize_a(rows_a)
    rows_b = part_b(static, scenes, TEST, REPO / args.rollouts)
    report["B_crosscheck"] = summarize_b(rows_b)
    report["C_dynamics"] = part_c(scenes, TEST, static)
    rows_d, report["D_timing"] = part_d(REPO / args.bank, TEST, REPO / args.rollouts, REPO / args.eval, static)
    for name, rows in (("static", rows_a), ("crosscheck", rows_b), ("timing", rows_d)):
        with open(out / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
