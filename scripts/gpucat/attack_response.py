"""How MetaDrive policies respond to CAT's adversary at evaluation
(rollouts_eval/<policy>/{none,cat}/<scene>.pkl, test scenes 400-499).

CAT's eval_policy aims the adversary at the policy's own trajectory in the
normal episode (collect_rollouts.py: first the normal episode, then the
adversary generated against it). Until the adversary leaves its logged
track, both episodes see the same scene, so the ego repeats its normal
trajectory exactly. Any later difference is the policy's reaction to the
adversary. Per adversarial episode:

  planned hit   the planned adversary comes within HIT_DIST of the ego's
                normal trajectory at some step t* (the intended collision)
  onset         first step the adversary is more than 1 m from its log
  reaction      first step the ego is more than 0.5 m from its normal
                trajectory, and its delay after the onset
  at t*         how far the ego is behind (+) or ahead (-) of its normal
                trajectory along it, and how far to the side
  outcome       a crash with the adversary, another crash, out of road, ...

So general slowness does not evade the evaluation adversary, which is aimed
at the slow trajectory itself. This tells whether a policy evades by
reacting (braking, swerving), and how often.

    python -m scripts.gpucat.attack_response
"""

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from responsibility.rollouts import load_rollout  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

HIT_DIST = 3.0  # m between centres: the planned adversary meets the ego's normal trajectory
GROUPS = {"logged ego": ["replay"], "replay": [f"td3_replay_s{s}" for s in range(3)],
          "cat": [f"td3_cat_s{s}" for s in range(3)], "fair2_0.1": [f"td3_cat_fair2_0.1_s{s}" for s in range(3)],
          "cat_share": [f"td3_cat_share_s{s}" for s in range(3)], "cat_rss": [f"td3_cat_rss_s{s}" for s in range(3)]}


def arc_and_side(path, pts):
    seg = np.diff(path, axis=0)
    L = np.linalg.norm(seg, axis=1)
    keep = L > 1e-6
    a, seg, L = path[:-1][keep], seg[keep], L[keep]
    cum = np.r_[0, np.cumsum(L)]
    t = np.clip(((pts[:, None] - a) * seg).sum(-1) / L ** 2, 0, 1)
    q = a + t[..., None] * seg
    d = np.linalg.norm(pts[:, None] - q, axis=-1)
    k = d.argmin(1)
    return cum[k] + t[np.arange(len(pts)), k] * L[k], d[np.arange(len(pts)), k]


def analyse(run, scenes):
    out = []
    for i in range(400, 500):
        f0, f1 = REPO / f"rollouts_eval/{run}/none/{i}.pkl", REPO / f"rollouts_eval/{run}/cat/{i}.pkl"
        if not (f0.exists() and f1.exists()):
            continue
        r0, r1 = load_rollout(f0), load_rollout(f1)
        adv = r1["adversary"]
        if adv is None:
            continue
        e0, e1 = np.asarray(r0["ego"]["position"], float), np.asarray(r1["ego"]["position"], float)
        plan = np.asarray(adv["planned"], float)[:, :2]
        sc, aid = scenes[i]
        logged_adv = sc.position[sc.track_ids.index(adv["track_id"]), :, :2]
        av = np.asarray(adv["position"], float)
        n0 = len(e0)
        steps = np.arange(11, min(91, n0))
        d = np.linalg.norm(plan[steps] - e0[steps], axis=1)
        hit = np.flatnonzero(d < HIT_DIST)
        t_star = int(steps[hit[0]]) if len(hit) else None
        k = min(len(av), len(logged_adv))
        dev = np.linalg.norm(av[:k] - logged_adv[:k], axis=1)
        dev[np.isnan(dev)] = 0
        onset = int(np.argmax(dev > 1.0)) if (dev > 1.0).any() else None
        m = min(len(e0), len(e1))
        diff = np.linalg.norm(e1[:m] - e0[:m], axis=1)
        react = int(np.argmax(diff > 0.5)) if (diff > 0.5).any() else None
        end = r1["end"]
        crash_adv = bool(end.get("crash_vehicle")) and str(end.get("crash_with")) == str(adv["track_id"])
        rec = {"scene": i, "t_star": t_star, "onset": onset, "react": react, "end_step": end["step"],
               "crash_adv": crash_adv, "crash_other": bool(end.get("crash_vehicle")) and not crash_adv,
               "oor": bool(end.get("out_of_road")), "lag": np.nan, "side": np.nan}
        if t_star is not None and t_star < len(e1) and not np.isnan(e1[t_star, 0]):
            s0, _ = arc_and_side(e0[~np.isnan(e0[:, 0])], e0[[t_star]])
            s1, side = arc_and_side(e0[~np.isnan(e0[:, 0])], e1[[t_star]])
            rec["lag"], rec["side"] = float(s0[0] - s1[0]), float(side[0])
        out.append(rec)
    return out


def main():
    scenes = {}
    for i in range(400, 500):
        sc = Scene.from_description(pickle.load(open(REPO / f"raw_scenes_500/{i}.pkl", "rb")))
        scenes[i] = (sc, None)
    print("| training | episodes | planned hit | crash with adversary (of planned) | ended before t* | reacted before t* | reaction delay after onset (median steps) | at t*: >2 m behind / >2 m ahead / >1 m aside (of not ended) |")
    print("|---|---|---|---|---|---|---|---|")
    for g, runs in GROUPS.items():
        recs = [r for run in runs for r in analyse(run, scenes)]
        P = [r for r in recs if r["t_star"] is not None]
        ended = [r for r in P if r["end_step"] < r["t_star"] and not r["crash_adv"]]
        alive = [r for r in P if r["end_step"] >= r["t_star"] or r["crash_adv"]]
        reacted = [r for r in P if r["react"] is not None and r["react"] < r["t_star"]]
        delays = [r["react"] - r["onset"] for r in reacted if r["onset"] is not None]
        lagged = [r for r in alive if not np.isnan(r["lag"])]
        behind = sum(r["lag"] > 2 for r in lagged)
        ahead = sum(r["lag"] < -2 for r in lagged)
        aside = sum(r["side"] > 1 for r in lagged)
        print(f"| {g} | {len(recs)} | {len(P)} | {sum(r['crash_adv'] for r in P)}/{len(P)} "
              f"({100 * sum(r['crash_adv'] for r in P) / max(len(P), 1):.0f}%) | {len(ended)} | {len(reacted)} "
              f"({100 * len(reacted) / max(len(P), 1):.0f}%) | {np.median(delays) if delays else float('nan'):.0f} | "
              f"{behind} / {ahead} / {aside} of {len(lagged)} |")


if __name__ == "__main__":
    main()
