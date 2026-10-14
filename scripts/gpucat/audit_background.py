"""Where do MetaDrive crashes WITHOUT an adversary come from?  (task A3 of
the audit of 2026-10-09). An independent re-measurement, by a reviewer of that
audit, of scripts/gpucat/metadrive_background_crashes.py with different
thresholds (45/135 degrees instead of 30/150, a simpler reactive rule), on the
replay, cat and fair2_0.1 groups (3 seeds each). Of the 162 crashes, 28% are a
logged follower rear-ending the ego, and in 41 of those 45 the ego is more than
1 m/s slower than its log; cat policies are rear-ended in 49% of their crashes,
replay in 24% and fair2_0.1 in 10%.

For every no-adversary vehicle crash of the TD3 policies, classify the
geometry at the last step before contact (ego frame, headings from the
rebuilt scene), compare the ego with its own log, and estimate whether a
reactive partner would have avoided it.  Geometry and rules only; the
counterfactual verdict is read from crashes.csv but not used for classes.

Classes (heading difference d between partner and ego, lon/lat = partner
centre in the common-heading frame, scaled by half-extents as in
blame.rear_end_rule):
  rear_ended   |d|<=45deg, partner behind end-to-end   (logged follower hits ego)
  ego_rear     |d|<=45deg, partner ahead end-to-end    (ego hits a leader)
  side_same    |d|<=45deg, side by side
  oncoming     |d|>=135deg
  crossing     otherwise
Reactive-avoidability (simple rules, deceleration limit A in m/s^2):
  rear_ended: follower speed vf, ego speed ve along common heading. Look back up
    to 3 s; at the first step where the bumper gap / vf < 1 s (or the earliest
    step if already so) a follower keeping a 1 s gap brakes; avoidable if the
    needed decel (vf-ve)^2/(2*gap) <= A.  If vf<=ve at that moment: avoidable.
  crossing/oncoming/side: partner stops before the ego's crash point if, 1.5 s
    before the crash, v^2/(2*d) <= A where d = distance partner->ego's crash-time
    position minus half partner length (an optimistic yielding rule).
  ego_rear: not avoidable by reactive traffic (ego is the follower).

Run from the repository root with the cat39 env; CPU only, about a minute.
Writes one row per crash to OUT.csv and prints the class shares per group:
    python -m scripts.gpucat.audit_background OUT.csv
scripts/gpucat/audit_background_tables.py prints the summary tables.
"""
import sys, pickle, glob, os, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from responsibility.scene import Scene
from responsibility.rollouts import load_rollout, scene_from_rollout

POL = {"replay": ["td3_replay_s0", "td3_replay_s1", "td3_replay_s2"],
       "cat": ["td3_cat_s0", "td3_cat_s1", "td3_cat_s2"],
       "fair2_0.1": ["td3_cat_fair2_0.1_s0", "td3_cat_fair2_0.1_s1", "td3_cat_fair2_0.1_s2"]}
_scenes = {}
def scene_of(i):
    if i not in _scenes:
        _scenes[i] = Scene.load(ROOT / "raw_scenes_500" / f"{i}.pkl")
    return _scenes[i]

def wrap(a): return float(np.angle(np.exp(1j * a)))

def analyse(policy, row, A=6.0):
    i = int(row["scene"]); log = scene_of(i)
    ro = load_rollout(ROOT / "rollouts_eval" / policy / "none" / f"{i}.pkl")
    sc = scene_from_rollout(log, ro)
    e, o = sc.sdc, sc.index(row["other_id"]) if str(row["other_id"]) in sc.track_ids else None
    cs = ro["end"]["step"]
    out = dict(policy=policy, scene=i, crash_step=cs, other_id=row["other_id"], other_type=row["other_type"],
               rule=row["rule"], rss=row["rss"], verdict=row["verdict"])
    if o is None: out["cls"] = "unknown_partner"; return out
    ks = [k for k in range(min(cs, sc.n_steps - 1), -1, -1) if sc.valid[e, k] and sc.valid[o, k]]
    if not ks: out["cls"] = "unknown_partner"; return out
    k = next((k for k in ks if k < cs), ks[0])
    he, ho = float(sc.heading[e, k]), float(sc.heading[o, k]); d = wrap(ho - he)
    hc = he + d / 2
    dv = sc.position[o, k, :2] - sc.position[e, k, :2]
    lon = dv[0] * np.cos(hc) + dv[1] * np.sin(hc); lat = -dv[0] * np.sin(hc) + dv[1] * np.cos(hc)
    (le, we), (lo, wo) = sc.shape_at(e, k), sc.shape_at(o, k)
    endwise = abs(lon) / ((le + lo) / 2) > abs(lat) / ((we + wo) / 2)
    if abs(d) <= np.pi / 4: cls = ("ego_rear" if lon > 0 else "rear_ended") if endwise else "side_same"
    elif abs(d) >= 3 * np.pi / 4: cls = "oncoming"
    else: cls = "crossing"
    out.update(cls=cls, head_diff_deg=np.degrees(d), lon=lon, lat=lat)
    # speed relative to log (same step, scene is the logged one for the ego before replacement)
    ve = np.linalg.norm(sc.velocity[e, k]); vlog = np.linalg.norm(log.velocity[log.sdc, k]) if log.valid[log.sdc, k] else np.nan
    vo = np.linalg.norm(sc.velocity[o, k])
    lo_k = max(0, k - 20)
    seg = [j for j in range(lo_k, k + 1) if log.valid[log.sdc, j] and sc.valid[e, j]]
    sp_e = np.mean([np.linalg.norm(sc.velocity[e, j]) for j in seg]) if seg else np.nan
    sp_l = np.mean([np.linalg.norm(log.velocity[log.sdc, j]) for j in seg]) if seg else np.nan
    # along-route lag: sim ego position minus logged ego position, projected on logged heading
    lag = np.nan
    if log.valid[log.sdc, k]:
        dp = sc.position[e, k, :2] - log.position[log.sdc, k, :2]; h = log.heading[log.sdc, k]
        lag = dp[0] * np.cos(h) + dp[1] * np.sin(h)
    out.update(ve=ve, vlog=vlog, vo=vo, dv_log=ve - vlog, ratio=ve / vlog if vlog > 0.5 else np.nan,
               slower=bool(ve < vlog - 1.0) if np.isfinite(vlog) else False, faster=bool(ve > vlog + 1.0) if np.isfinite(vlog) else False,
               mean_dv_log20=sp_e - sp_l, lag_m=lag, ooi=bool(sc.track_ids[o] in [sc.track_ids[j] for j in sc.objects_of_interest]),
               o_state="parked" if vo < 0.5 else "moving")
    # reactive avoidability
    avoid = None
    if cls == "rear_ended":
        ucom = np.array([np.cos(hc), np.sin(hc)])
        avoid = False
        first = max(ks[-1], k - 30)
        for j in range(first, k + 1):
            if j not in ks: continue
            vf = float(sc.velocity[o, j] @ ucom); vee = float(sc.velocity[e, j] @ ucom)
            gap = float((sc.position[e, j, :2] - sc.position[o, j, :2]) @ ucom) - (le + lo) / 2
            if vf > 0.1 and gap / vf < 1.0:  # the 1 s gap is first violated: a reactive follower brakes now
                avoid = True if vf <= vee else ((vf - vee) ** 2 / (2 * max(gap, 0.1)) <= A)
                break
        else:
            avoid = None  # the gap never fell below 1 s before contact (fast approach / lateral entry)
        # stricter: the follower first notices 1 s (10 steps) before contact
        j = max(ks[-1], k - 10)
        if j in ks:
            vf = float(sc.velocity[o, j] @ ucom); vee = float(sc.velocity[e, j] @ ucom)
            gap = float((sc.position[e, j, :2] - sc.position[o, j, :2]) @ ucom) - (le + lo) / 2
            out["need_decel_1s"] = 0.0 if vf <= vee else (vf - vee) ** 2 / (2 * max(gap, 0.1))
            out["avoid_1s_notice"] = bool(out["need_decel_1s"] <= A)
    elif cls in ("crossing", "oncoming", "side_same"):
        j = max(ks[-1], k - 15)
        if j in ks:
            vp = float(np.linalg.norm(sc.velocity[o, j]))
            dist = float(np.linalg.norm(sc.position[e, k, :2] - sc.position[o, j, :2])) - lo / 2
            avoid = (vp <= 0.5) or (vp ** 2 / (2 * max(dist, 0.1)) <= A) if vp > 0.5 else True
            if vp <= 0.5: avoid = None  # a parked/slow partner: nothing for traffic to react to
    elif cls == "ego_rear":
        avoid = False
    out["avoid"] = avoid
    return out

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python -m scripts.gpucat.audit_background OUT.csv")
    rows = []
    for grp, pols in POL.items():
        for p in pols:
            with open(ROOT / f"logs/responsibility/eval/{p}/none/crashes.csv") as f:
                for r in csv.DictReader(f):
                    x = analyse(p, r); x["group"] = grp; rows.append(x)
    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(sys.argv[1])
    df.to_csv(out, index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print("n per group\n", df.group.value_counts())
    for g, d in list(df.groupby("group")) + [("ALL", df)]:
        print(f"\n=== {g} (n={len(d)}) ===")
        print((d.cls.value_counts() / len(d) * 100).round(1).to_string())
    return df
if __name__ == "__main__":
    main()
