"""CPU follow-up to scripts/gpucat/audit_o1.py for observation O1 (the GPUDrive
PPO policy is weak and its episodes end early). No simulator is started.

  shift     sensitivity of gpucat's out-of-road tests (gpucat/env.py: the
            footprint touching a road edge or a solid yellow line, segments
            from logs/gpucat/static.npz) to a small lateral offset of the
            logged SDC path: the whole valid path is moved d metres along
            its left normal, d in SHIFTS, and a scene counts as out of road
            if any step's footprint touches a segment; with the logged SDC
            size (what GPUDrive uses, 5.286 x 2.332 m) and MetaDrive's
            DefaultVehicle (4.515 x 1.852 m)
  attack    the step at which CAT's chosen candidate (candidates.npz,
            chosen_cat, chosen against the logged route) first overlaps the
            logged SDC (boxes, every step; candidate index j is scene step
            11 + j), and its first approach within 2 m (centres)
  truncated what CAT chooses when the ego's stored trajectories end early:
            all AV_traj_num (5) histories set to the logged route cut at
            episode end step E (stored steps 11..E, as CatEnv._store keeps
            them); whether the chosen candidate collides within the stored
            steps by CAT's own criterion (gpucat.adversary.cat_scores, every
            5th step), and the step of that collision
  noise     the policy's exploration noise (state-independent log std from
            the training logs) in physical units against CAT's TD3
            (expl_noise 0.1 of the action range, MetaDrive's 40 degree
            steering on a 2.469 m wheelbase), and the open-loop lateral
            drift that i.i.d. curvature noise produces at the logged speed

    cd /home/Cesar/catspond && ~/venvs/cat39/bin/python -m scripts.gpucat.audit_o1_timing --out <dir>
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpucat import adversary as ga  # noqa: E402
from gpucat.geometry import boxes_overlap  # noqa: E402
from responsibility.scene import Scene  # noqa: E402
from scripts.gpucat.audit_o1 import MD_SIZE, box_segment_distance, pct, yaw_of  # noqa: E402

TEST = [str(i) for i in range(400, 500)]
SHIFTS = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
ENDS = (21, 27, 35, 45, 60, 75, 90)


def part_shift(static, scenes):
    idx = {s: i for i, s in enumerate(static["stems"])}
    res = {}
    per_scene = []
    for s in TEST:
        k, sc = idx[s], scenes[s]
        v = np.flatnonzero(sc.valid[sc.sdc])
        p = sc.position[sc.sdc, v, :2].astype(float)
        h = sc.heading[sc.sdc, v].astype(float)
        nrm = np.stack([-np.sin(h), np.cos(h)], 1)
        L, W = static["sizes"][k, 0]
        row = {"scene": s}
        for name in ("edges", "yellow"):
            segs = static[name][k, : static[f"{name}_n"][k]]
            for tag, (l, w) in (("log", (L, W)), ("md", MD_SIZE)):
                for d in SHIFTS:
                    if len(segs) == 0:
                        row[f"{name}_{tag}_{d}"] = -1
                        continue
                    dist = box_segment_distance(p + d * nrm, h, l, w, segs).min(1)
                    hit = dist <= 0
                    row[f"{name}_{tag}_{d}"] = int(v[np.argmax(hit)]) if hit.any() else -1
        per_scene.append(row)
    for tag in ("log", "md"):
        for d in SHIFTS:
            e = np.array([r[f"edges_{tag}_{d}"] for r in per_scene])
            y = np.array([r[f"yellow_{tag}_{d}"] for r in per_scene])
            anyhit = (e >= 0) | (y >= 0)
            first = np.where(anyhit, np.minimum(np.where(e >= 0, e, 999), np.where(y >= 0, y, 999)), -1)
            res[f"{tag}_shift_{d:+.2f}"] = {"scenes_out_of_road": int(anyhit.sum()), "edge": int((e >= 0).sum()),
                                            "yellow": int((y >= 0).sum()),
                                            "first_step": pct(first[anyhit], (25, 50, 75))}
        # either side: the path shifted by |d| to the left or to the right touches
        for d in (0.25, 0.5, 0.75, 1.0):
            both = np.array([(r[f"edges_{tag}_{d}"] >= 0 or r[f"yellow_{tag}_{d}"] >= 0) or
                             (r[f"edges_{tag}_{-d}"] >= 0 or r[f"yellow_{tag}_{-d}"] >= 0) for r in per_scene])
            res[f"{tag}_shift_either_side_{d:.2f}"] = int(both.sum())
    return res, per_scene


def first_overlap(c, e, ov, av):
    """[T] boxes of the candidate c [T, 2] against the ego e [T, 2] (path yaw)."""
    hit = boxes_overlap(torch.tensor(e), torch.tensor(yaw_of(e)), torch.tensor(av)[None].expand(len(e), 2),
                        torch.tensor(c)[:, None], torch.tensor(yaw_of(c))[:, None],
                        torch.tensor(ov)[None, None].expand(len(e), 1, 2))[:, 0].numpy()
    return hit


def part_attack(z, stems):
    sidx = {str(s): i for i, s in enumerate(z["stems"])}
    rows = []
    for s in stems:
        i = sidx[s]
        c = z["candidates"][i, z["chosen_cat"][i]].astype(float)
        e = z["ego_route"][i].astype(float)
        hit = first_overlap(c, e, z["ov_size"][i].astype(float), z["av_size"][i].astype(float))
        d = np.linalg.norm(c - e, axis=1)
        rows.append({"scene": s, "box": int(np.argmax(hit)) + 11 if hit.any() else -1,
                     "c2m": int(np.argmax(d < 2.0)) + 11 if (d < 2.0).any() else -1,
                     "score_pos": bool(z["score"][i, z["chosen_cat"][i]] > 0)})
    a = np.array([r["box"] for r in rows])
    c2 = np.array([r["c2m"] for r in rows])
    out = {"scenes": len(rows), "chosen_has_cat_score": int(sum(r["score_pos"] for r in rows)),
           "box_overlap_scenes": int((a >= 0).sum()), "box_step": pct(a[a >= 0], (10, 25, 50, 75, 90)),
           "centre_2m_scenes": int((c2 >= 0).sum()), "centre_2m_step": pct(c2[c2 >= 0], (10, 25, 50, 75, 90))}
    for t in (20, 27, 40, 50, 60, 75):
        out[f"box_attacks_after_step_{t}"] = round(float((a[a >= 0] > t).mean()), 3)
        out[f"box_attacks_at_or_before_step_{t}_of_100"] = int(((a >= 0) & (a <= t)).sum())
    return out, rows


def part_truncated(z, stems):
    sidx = {str(s): i for i, s in enumerate(z["stems"])}
    rows = [sidx[s] for s in stems]
    cand = torch.tensor(z["candidates"][rows], dtype=torch.float64)
    probs = ga.candidate_probs(torch.tensor(z["log_scores"][rows], dtype=torch.float64))
    route = torch.tensor(z["ego_route"][rows], dtype=torch.float64)
    ov = torch.tensor(z["ov_size"][rows], dtype=torch.float64)
    av = torch.tensor(z["av_size"][rows], dtype=torch.float64)
    B, M = len(rows), 5
    out = {}
    # the full route must reproduce the stored choice (sanity check of this call)
    for E in ENDS:
        n = E - 10  # stored steps 11..E
        hist = torch.zeros(B, M, 80, 2, dtype=torch.float64)
        hist[:, :, :n] = route[:, None, :n]
        hlen = torch.full((B, M), n, dtype=torch.long)
        score, md = ga.cat_scores(cand, probs, hist, hlen, torch.ones(B, M, dtype=torch.float64), ov, av)
        chosen = ga.select("cat", score, md)
        hit_in_window = (score[torch.arange(B), chosen] > 0).numpy()
        steps, full_steps = [], []
        for b in range(B):
            c = cand[b, chosen[b]].numpy()
            e = route[b].numpy()
            hit = first_overlap(c, e, ov[b].numpy(), av[b].numpy())
            full_steps.append(int(np.argmax(hit)) + 11 if hit.any() else -1)
            hw = hit[:n]
            steps.append(int(np.argmax(hw)) + 11 if hw.any() else -1)
        steps, full_steps = np.array(steps), np.array(full_steps)
        rec = {"stored_steps": n, "cat_score_hit_scenes": int(hit_in_window.sum()),
               "box_hit_within_stored_scenes": int((steps >= 0).sum()),
               "box_hit_step_within": pct(steps[steps >= 0], (25, 50, 75)),
               "box_hit_vs_full_log_scenes": int((full_steps >= 0).sum()),
               "box_hit_vs_full_log_step": pct(full_steps[full_steps >= 0], (25, 50, 75))}
        if E == 90:
            rec["matches_stored_chosen_cat"] = int((chosen.numpy() == z["chosen_cat"][rows]).sum())
        out[f"end_{E}"] = rec
    return out


def part_noise(scenes, log_dir):
    """Exploration noise in physical units and its open-loop lateral drift."""
    out = {"runs": {}}
    pat = re.compile(r"^it (\d+) steps (\d+).*log_std \[([-\d.e]+), ([-\d.e]+)\]")
    for f in sorted(log_dir.glob("*.log")):
        pts = []
        for line in open(f):
            m = pat.match(line)
            if m:
                pts.append((int(m.group(2)), float(m.group(3)), float(m.group(4))))
        if not pts:
            continue
        pick = {}
        for target in (0.1e6, 1e6, 2e6, 5e6):
            j = int(np.argmin([abs(p[0] - target) for p in pts]))
            if abs(pts[j][0] - target) < 0.15 * target:
                st, la, lk = pts[j]
                pick[f"{target / 1e6:g}M"] = {"steps": st, "sigma_accel_mps2": round(float(np.exp(la)) * 5.0, 2),
                                              "sigma_curv_1pm": round(float(np.exp(lk)) * 0.3, 4)}
        out["runs"][f.stem] = pick
    # CAT's TD3: N(0, 0.1) on the normalised steering, max_steering 40 deg, wheelbase 1.05234 + 1.4166 m
    td3_curv = float(np.tan(np.deg2rad(0.1 * 40.0)) / (1.05234 + 1.4166))
    out["td3_sigma_curv_1pm"] = round(td3_curv, 4)
    out["metadrive_max_curv_1pm"] = round(float(np.tan(np.deg2rad(40.0)) / (1.05234 + 1.4166)), 3)
    sp = np.concatenate([np.linalg.norm(scenes[s].velocity[scenes[s].sdc][scenes[s].valid[scenes[s].sdc]][:, :2], axis=1)
                         for s in TEST])
    v = float(np.median(sp[sp > 1.0]))
    out["median_logged_speed_moving_mps"] = round(v, 2)
    rng = np.random.default_rng(0)
    drift = {}
    for name, sig in (("ppo_init", 0.6065 * 0.3), ("ppo_2M", float(np.exp(-0.70)) * 0.3),
                      ("ppo_5M", float(np.exp(-1.336)) * 0.3), ("td3", td3_curv)):
        k = rng.normal(0, sig, size=(20000, 30))
        yaw = np.cumsum(k * v * 0.1, 1)
        lat = np.cumsum(np.sin(yaw) * v * 0.1, 1)
        drift[name] = {f"lat_std_m_after_{t}_steps": round(float(lat[:, t - 1].std()), 2) for t in (5, 10, 20, 30)}
        drift[name]["sigma_curv_1pm"] = round(sig, 4)
    out["open_loop_drift_at_median_speed"] = drift
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--static", default="logs/gpucat/static.npz")
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--logs", default="logs/gpucat")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zs = np.load(REPO / args.static)
    static = {k: zs[k] for k in zs.files}
    static["stems"] = [str(s) for s in static["stems"]]
    z = np.load(REPO / args.bank)
    z = {k: z[k] for k in z.files}
    scenes = {s: Scene.load(REPO / args.scenes / f"{s}.pkl") for s in TEST}
    report = {}
    report["noise"] = part_noise(scenes, REPO / args.logs)
    report["attack"], rows = part_attack(z, TEST)
    report["truncated"] = part_truncated(z, TEST)
    report["shift"], shift_rows = part_shift(static, scenes)
    (out / "report_timing.json").write_text(json.dumps(report, indent=1))
    (out / "attack_rows.json").write_text(json.dumps(rows))
    (out / "shift_rows.json").write_text(json.dumps(shift_rows))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
