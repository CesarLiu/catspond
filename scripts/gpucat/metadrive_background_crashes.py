"""Where MetaDrive collisions without an adversary come from, by geometry and
rules only (no counterfactual verdict).

For every TD3 rollout in rollouts_eval/<policy>/none that ended in a vehicle
collision, the scene is rebuilt (rollouts.scene_from_rollout) and the
collision is described at the last step before contact at which both cars
exist (the step rear_end_rule reads):

  configuration   follower_hits_ego / ego_hits_leader / same_dir_side
                  (headings within 30 deg; end-to-end vs side-by-side as in
                  blame.rear_end_rule), opposite (> 150 deg), crossing (else)
  ego vs its log  speed at that step minus the logged ego's speed, and the
                  ego's lag along its logged path (arc length of the logged
                  ego at the same step minus the arc length of the simulated
                  ego's projection; > 0 means the ego is behind its log),
                  lateral offset from the logged path
  partner         object of interest or not, type, speed, steps since it
                  first appeared in the log, logged-ego gap at the crash step
  footprint gap   at the crash step, between the boxes (logged sizes, the
                  simulator's ego size): a sanity check that the boxes touch
  red light       the ego crossed a logged traffic-light stop line (from
                  behind to past the stop point along the controlled lane,
                  within 2.5 m laterally) while it was STOP / ARROW_STOP /
                  FLASHING_STOP, in the 3 s before contact (MetaDrive runs
                  CAT with no_light=True, so the policy cannot see lights
                  that logged traffic obeyed)
  reactive rule   would a partner that keeps a 1 s time gap have avoided it?
                  The partner "sees" the ego when the ego's box lies in the
                  corridor ahead of the partner (lateral overlap with the
                  partner's width + 0.5 m, length v * 1 s + 2 m) within 5 s
                  before contact. At the first such step t0 the deceleration
                  it needed to stop (crossing / opposite / side) or to match
                  the ego's along-track speed (same direction) before
                  reaching the ego is a_req = dv^2 / (2 gap). Avoidable when
                  a_req <= 4 m/s^2 (comfortable; RSS brake_min); also counted
                  at <= 8 m/s^2 (hard braking). A partner that never sees the
                  ego ahead (it is hit from behind or side-swiped by the ego)
                  cannot avoid it by braking.

Run from the repository root with the cat39 env (CPU only, about a minute):
  python -m scripts.gpucat.metadrive_background_crashes --out DIR
Writes DIR/crashes_geometry.csv (one row per crash);
metadrive_background_crashes_tables.py prints the tables. In the audit of
2026-10-09 (task A3), 43 of the 159 crashes of the replay, cat and fair2_0.1
groups (27%) were a logged follower rear-ending the ego, and a 1 s-gap
partner braking at most 4 m/s^2 would have avoided 80 of the 159.
"""

import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from shapely.geometry import Polygon  # noqa: E402

from responsibility.blame import crash_partner, rear_end_rule  # noqa: E402
from responsibility.rollouts import CRASH_REASONS, load_rollout, rollout_files, scene_from_rollout  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

SAME = np.deg2rad(30.0)
OPPOSITE = np.deg2rad(150.0)
TIME_GAP = 1.0  # s
STANDSTILL_GAP = 2.0  # m
COMFORT = 4.0  # m/s^2
HARD = 8.0  # m/s^2
SEE_WINDOW = 50  # steps before contact in which the partner may react
RED_STATES = ("LANE_STATE_STOP", "LANE_STATE_ARROW_STOP", "LANE_STATE_FLASHING_STOP")
DT = 0.1

GROUPS = {
    "replay": ["td3_replay_s0", "td3_replay_s1", "td3_replay_s2"],
    "cat": ["td3_cat_s0", "td3_cat_s1", "td3_cat_s2"],
    "fair2_0.1": ["td3_cat_fair2_0.1_s0", "td3_cat_fair2_0.1_s1", "td3_cat_fair2_0.1_s2"],
    "share": ["td3_cat_share_s0", "td3_cat_share_s1", "td3_cat_share_s2"],
    "rss": ["td3_cat_rss_s0", "td3_cat_rss_s1", "td3_cat_rss_s2"],
}


def wrap(a):
    return float(np.angle(np.exp(1j * a)))


def box(pos, head, length, width):
    c, s = np.cos(head), np.sin(head)
    pts = []
    for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
        x, y = dx * length / 2, dy * width / 2
        pts.append((pos[0] + c * x - s * y, pos[1] + s * x + c * y))
    return Polygon(pts)


def arc_lengths(points):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def project(point, poly, cum):
    """(arc length, lateral distance) of ``point`` on polyline ``poly``."""
    best = (0.0, np.inf)
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        d = b - a
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-9 else float(np.clip((point - a) @ d / L2, 0.0, 1.0))
        q = a + t * d
        dist = float(np.linalg.norm(point - q))
        if dist < best[1]:
            best = (cum[i] + t * np.sqrt(L2), dist)
    return best


def ego_vs_log(log: Scene, sim: Scene, k: int):
    ego = log.sdc
    v = log.valid[ego]
    pts = log.position[ego, v, :2]
    steps = np.nonzero(v)[0]
    cum = arc_lengths(pts)
    s_sim, lat = project(sim.position[ego, k, :2], pts, cum)
    # logged arc length at step k (nearest logged step)
    j = int(np.argmin(np.abs(steps - k)))
    s_log = cum[j]
    v_sim = float(np.linalg.norm(sim.velocity[ego, k]))
    v_log = float(np.linalg.norm(log.velocity[ego, k])) if log.valid[ego, k] else float("nan")
    return s_log - s_sim, lat, v_sim, v_log


def lane_dir_at(scene: Scene, lane_id, point):
    feat = scene.map_features.get(str(lane_id))
    if feat is None or "polyline" not in feat:
        return None
    poly = np.asarray(feat["polyline"])[:, :2]
    if len(poly) < 2:
        return None
    i = int(np.argmin(np.linalg.norm(poly - point, axis=1)))
    i = min(i, len(poly) - 2)
    d = poly[i + 1] - poly[i]
    return float(np.arctan2(d[1], d[0]))


def red_light_pass(scene: Scene, agent: int, first: int, last: int, half_width=2.5):
    """Steps in [first, last] at which ``agent``'s centre crossed a logged
    traffic-light stop line (from behind to past it along the controlled
    lane's direction, within ``half_width`` of the stop point laterally,
    heading within 45 deg of the lane) while that light was STOP /
    ARROW_STOP / FLASHING_STOP."""
    hits = []
    for _, light in scene.dynamic_map_states.items():
        sp = np.asarray(light.get("stop_point", [np.nan, np.nan]))[:2]
        states = light.get("state", {}).get("object_state", [])
        if not np.all(np.isfinite(sp)) or not states:
            continue
        lane_dir = lane_dir_at(scene, light.get("lane"), sp)
        if lane_dir is None:
            continue
        f = np.array([np.cos(lane_dir), np.sin(lane_dir)])
        n = np.array([-f[1], f[0]])
        for t in range(max(first, 1), min(last, scene.n_steps - 1) + 1):
            if t >= len(states) or not (scene.valid[agent, t] and scene.valid[agent, t - 1]):
                continue
            if states[t] not in RED_STATES:
                continue
            a0 = float((scene.position[agent, t - 1, :2] - sp) @ f)
            a1 = float((scene.position[agent, t, :2] - sp) @ f)
            lat = float((scene.position[agent, t, :2] - sp) @ n)
            if not (a0 < 0.0 <= a1) or abs(lat) > half_width:
                continue
            if abs(wrap(scene.heading[agent, t] - lane_dir)) > np.deg2rad(45):
                continue
            hits.append((t, states[t]))
    return hits


def reactive(sim: Scene, ego: int, other: int, crash: int, same_dir: bool, l_e, w_e, l_o, w_o):
    """(t0, a_req) for a partner keeping a 1 s gap; (None, None) if it never
    sees the ego in the corridor ahead."""
    for t in range(max(0, crash - SEE_WINDOW), crash):
        if not (sim.valid[ego, t] and sim.valid[other, t]):
            continue
        h = float(sim.heading[other, t])
        f = np.array([np.cos(h), np.sin(h)])
        n = np.array([-np.sin(h), np.cos(h)])
        vo = sim.velocity[other, t]
        v = float(vo @ f)
        if v < 0.5:
            continue
        reach = v * TIME_GAP + STANDSTILL_GAP
        corridor = box(sim.position[other, t, :2] + f * (l_o / 2 + reach / 2), h, reach, w_o + 0.5)
        ego_box = box(sim.position[ego, t, :2], float(sim.heading[ego, t]), l_e, w_e)
        if not corridor.intersects(ego_box):
            continue
        own = box(sim.position[other, t, :2], h, l_o, w_o)
        gap = max(own.distance(ego_box), 0.05)
        ve = float(sim.velocity[ego, t] @ f)
        dv = v - max(ve, 0.0) if same_dir else v
        if dv <= 0:
            return t, 0.0
        return t, dv * dv / (2 * gap)
    return None, None


def analyse(policy, rollout_dir, scenes_dir):
    rows = []
    for f in rollout_files(rollout_dir):
        r = load_rollout(f)
        if r["end"]["reason"] not in CRASH_REASONS:
            continue
        log = Scene.load(os.path.join(scenes_dir, f"{r['scene_file']}.pkl"))
        sim = scene_from_rollout(log, r)
        crash = r["end"]["step"]
        ego = sim.sdc
        cw = r["end"].get("crash_with")
        other = sim.index(cw) if cw is not None and str(cw) in sim.track_ids else None
        partner_from = "crash_with"
        if other is None:
            other = crash_partner(sim, ego, min(crash, sim.n_steps - 1))
            partner_from = "closest"
        row = dict(policy=policy, scene=r["scene_file"], crash_step=crash, reason=r["end"]["reason"],
                   partner_from=partner_from)
        if other is None or crash >= sim.n_steps:
            row["config"] = "no-partner"
            rows.append(row)
            continue
        steps = [k for k in range(crash, -1, -1) if sim.valid[ego, k] and sim.valid[other, k]]
        if not steps:
            row["config"] = "never-together"
            rows.append(row)
            continue
        k = next((k for k in steps if k < crash), steps[0])
        l_e, w_e = r["ego"]["size"] if r["ego"]["size"] else sim.shape_at(ego, k)
        l_o, w_o = sim.shape_at(other, k)
        dh = wrap(float(sim.heading[other, k]) - float(sim.heading[ego, k]))
        if abs(dh) <= SAME:
            hd = float(sim.heading[ego, k]) + dh / 2
            d = sim.position[other, k, :2] - sim.position[ego, k, :2]
            lon = float(d[0] * np.cos(hd) + d[1] * np.sin(hd))
            lat = float(-d[0] * np.sin(hd) + d[1] * np.cos(hd))
            if abs(lon) / ((l_e + l_o) / 2) <= abs(lat) / ((w_e + w_o) / 2):
                config = "same_dir_side"
            else:
                config = "ego_hits_leader" if lon > 0 else "follower_hits_ego"
        elif abs(dh) >= OPPOSITE:
            config = "opposite"
        else:
            config = "crossing"
        lag, lat_dev, v_sim, v_log = ego_vs_log(log, sim, k)
        first_valid = int(np.argmax(log.valid[other])) if log.valid[other].any() else -1
        # footprint gap at the crash step (and at k)
        def gap_at(t):
            if not (sim.valid[ego, t] and sim.valid[other, t]):
                return float("nan")
            a = box(sim.position[ego, t, :2], float(sim.heading[ego, t]), l_e, w_e)
            b = box(sim.position[other, t, :2], float(sim.heading[other, t]), *sim.shape_at(other, t))
            return float(a.distance(b)) if not a.intersects(b) else -float(a.intersection(b).area)
        # logged ego vs partner at the crash step: would the log have been in the way?
        log_gap = float("nan")
        if log.valid[ego, min(crash, 90)] and log.valid[other, min(crash, 90)]:
            t = min(crash, 90)
            a = box(log.position[ego, t, :2], float(log.heading[ego, t]), *log.shape_at(ego, t))
            b = box(log.position[other, t, :2], float(log.heading[other, t]), *log.shape_at(other, t))
            log_gap = float(a.distance(b))
        # ego lateral motion relative to the partner's heading in the last 2 s (cut-in)
        k2 = max(steps[-1], k - 20)
        h_o = float(sim.heading[other, k])
        n = np.array([-np.sin(h_o), np.cos(h_o)])
        rel = lambda t: float((sim.position[ego, t, :2] - sim.position[other, t, :2]) @ n) if (sim.valid[ego, t] and sim.valid[other, t]) else np.nan
        lat_shift = abs(rel(k) - rel(k2)) if k2 != k else 0.0
        t0, a_req = reactive(sim, ego, other, crash, config in ("follower_hits_ego", "same_dir_side", "ego_hits_leader"),
                             l_e, w_e, l_o, w_o)
        reds = red_light_pass(sim, ego, crash - 30, crash)
        log_reds = red_light_pass(log, log.sdc, 0, 90)
        v_other = float(np.linalg.norm(sim.velocity[other, k]))
        row.update(
            config=config, k=k, other_id=sim.track_ids[other], other_type=sim.types[other],
            other_is_ooi=int(other in log.objects_of_interest), heading_diff_deg=round(np.rad2deg(dh), 1),
            v_ego=round(v_sim, 2), v_ego_log=round(v_log, 2), dv_ego_vs_log=round(v_sim - v_log, 2),
            ego_lag_m=round(lag, 2), ego_lat_dev_m=round(lat_dev, 2), v_other=round(v_other, 2),
            other_first_valid=first_valid, other_age_steps=crash - first_valid,
            gap_crash=round(gap_at(crash), 3), gap_k=round(gap_at(k), 3), log_gap_crash=round(log_gap, 2),
            ego_lat_shift_vs_other_2s=round(lat_shift, 2),
            see_step=t0 if t0 is not None else "", a_req=round(a_req, 2) if a_req is not None else "",
            ego_red=int(bool(reds)), log_ego_red=int(bool(log_reds)),
            rule=rear_end_rule(sim, ego, other, crash),
        )
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="rollouts_eval")
    ap.add_argument("--scenes", default="raw_scenes_500")
    ap.add_argument("--eval", default="logs/responsibility/eval")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for group, policies in GROUPS.items():
        for p in policies:
            d = os.path.join(args.rollouts, p, "none")
            if not os.path.isdir(d):
                continue
            rs = analyse(p, d, args.scenes)
            # join crashes.csv (rule / RSS / counterfactual verdict) by scene
            cpath = os.path.join(args.eval, p, "none", "crashes.csv")
            extra = {}
            if os.path.exists(cpath):
                with open(cpath) as fh:
                    for c in csv.DictReader(fh):
                        extra[c["scene"]] = c
            for r in rs:
                r["group"] = group
                c = extra.get(str(r["scene"]), {})
                r["csv_rule"] = c.get("rule", "")
                r["csv_rss"] = c.get("rss", "")
                r["csv_rss_case"] = c.get("rss_case", "")
                r["csv_verdict"] = c.get("verdict", "")
                r["csv_other_id"] = c.get("other_id", "")
            rows.extend(rs)
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    out_csv = os.path.join(args.out, "crashes_geometry.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", out_csv, len(rows), "crashes")


if __name__ == "__main__":
    main()
