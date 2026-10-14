"""Which pairs the MetaDrive evaluation's responsibility outputs are about:
the object of interest (OOI, = CAT's adversary in every CAT scene) or
background vehicles. CPU only, reads existing outputs; no motion model.

For every run logs/responsibility/eval/<policy>/<adv_mode>/ it reports

  crashes   vehicle-collision episodes in the rollouts, rows of crashes.csv,
            and the crash partner split into OOI / other vehicle / non-vehicle
            / no partner; the counterfactual verdict per partner class;
            and whether the recorded partner (recording.Recorder.nearest:
            closest *centre* among every logged object) is the agent whose
            footprint is closest at the crash step in the rebuilt scene
  windows   judged windows (speed >= 1 m/s), windows with no neighbour,
            with the OOI among the neighbours, with the OOI as the
            max-safety neighbour (safety_against), mean neighbour count;
            aggressive / timid rates as compare_policies computes them (max
            over all interacting neighbours) and restricted to the ego-OOI
            pair (OOI's per-neighbour beta_s; windows where the OOI was not
            selected count as beta_s = 0, an approximation of --use-ooi)

usage (repo root):
  ~/venvs/cat39/bin/python -m scripts.gpucat.audit_attribution_scope --out OUT_DIR
"""

import argparse
import csv
import glob
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.blame import crash_partner  # noqa: E402
from responsibility.interaction import InteractionConfig, interaction_scores  # noqa: E402
from responsibility.results import calibrate, calibrate_timid  # noqa: E402
from responsibility.rollouts import load_rollout, scene_from_rollout  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

EVAL = Path("logs/responsibility/eval")
ROLL = Path("rollouts_eval")
SCENES = Path("raw_scenes_500")


def ooi_of(scene_meta):
    sdc = str(scene_meta["sdc_id"])
    others = [str(i) for i in scene_meta["objects_of_interest"] if str(i) != sdc]
    return others[0] if others else None


def load_meta(stem, cache={}):
    if stem not in cache:
        with open(SCENES / f"{stem}.pkl", "rb") as f:
            d = pickle.load(f)
        cache[stem] = d["metadata"]
    return cache[stem]


def load_scene(stem, cache={}):
    if stem not in cache:
        cache[stem] = Scene.load(SCENES / f"{stem}.pkl")
    return cache[stem]


def thresholds():
    rows = list(csv.DictReader(open(EVAL / "replay/none/windows.csv")))
    s = [float(r["safety"]) for r in rows if float(r["speed"]) >= 1.0]
    c = [float(r["courtesy"]) for r in rows if float(r["speed"]) >= 1.0]
    return calibrate(s, 0.9, 0.1), calibrate(c, 0.9, 0.01), calibrate_timid(s, 0.1, 0.1)


def partner_check(stem, rollout):
    """(recorded partner, footprint-closest partner, gap of recorded, gap of closest) at the crash step."""
    scene = scene_from_rollout(load_scene(stem), rollout)
    step = rollout["end"]["step"]
    if step >= scene.n_steps:
        return None
    rec = rollout["end"].get("crash_with")
    closest = crash_partner(scene, scene.sdc, step)
    others = [i for i in range(scene.n_agents) if i != scene.sdc and scene.valid[i, step]]
    if not others:
        return rec, None, np.nan, np.nan
    gaps = interaction_scores(scene, scene.sdc, others, step, 0, InteractionConfig())["min_gap"]
    gap = dict(zip([scene.track_ids[i] for i in others], gaps))
    cid = scene.track_ids[closest] if closest is not None else None
    return rec, cid, float(gap.get(str(rec), np.nan)), float(gap.get(cid, np.nan))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--no-partner-check", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_s, t_c, t_t = thresholds()
    print(f"thresholds (replay/none): aggressive beta_s > {t_s:.3f} or beta_c > {t_c:.4f}; timid beta_s < {t_t:.3f}")

    crash_rows, window_rows, partner_rows = [], [], []
    for run in sorted(glob.glob(str(EVAL / "*/*/config.json"))):
        run = Path(run).parent
        policy, mode = run.parent.name, run.name
        rdir = ROLL / policy / mode
        rolls = {Path(f).stem: f for f in glob.glob(str(rdir / "*.pkl"))}
        n_crash_ep, n_ep = 0, 0
        crash_eps = []
        for stem, f in rolls.items():
            r = load_rollout(f)
            n_ep += 1
            if r["end"]["reason"] == "crash_vehicle":
                n_crash_ep += 1
                crash_eps.append((stem, r))
        crashes = list(csv.DictReader(open(run / "crashes.csv"))) if (run / "crashes.csv").exists() else []
        cls = Counter()
        verdict = defaultdict(Counter)
        adv_flag_vs_ooi = Counter()
        for c in crashes:
            ooi = ooi_of(load_meta(c["scene"]))
            oid = c.get("other_id") or ""
            if not oid:
                k = "no_partner"
            elif oid == ooi:
                k = "ooi"
            elif c.get("other_type") == "VEHICLE":
                k = "bg_vehicle"
            else:
                k = "bg_" + str(c.get("other_type"))
            cls[k] += 1
            verdict[k][c["verdict"]] += 1
            adv_flag_vs_ooi[(c["adversary"], k == "ooi")] += 1
        row = {"policy": policy, "adv_mode": mode, "episodes": n_ep, "crash_episodes": n_crash_ep,
               "crash_rows": len(crashes)}
        for k in ("ooi", "bg_vehicle", "no_partner"):
            row[f"n_{k}"] = cls[k]
        row["n_bg_nonvehicle"] = sum(v for k, v in cls.items() if k.startswith("bg_") and k != "bg_vehicle")
        attributed = {k: sum(verdict[k][v] for v in ("ego", "other", "shared", "ego-only")) for k in verdict}
        for k in ("ooi", "bg_vehicle"):
            a = attributed.get(k, 0)
            row[f"{k}_ego_fault"] = verdict[k]["ego"] / a if a else np.nan
            row[f"{k}_other_fault"] = verdict[k]["other"] / a if a else np.nan
        allattr = sum(attributed.values())
        row["all_ego_fault"] = sum(verdict[k]["ego"] for k in verdict) / allattr if allattr else np.nan
        row["unknown"] = sum(verdict[k]["unknown"] for k in verdict)
        row["adv_flag_mismatch"] = sum(v for (flag, is_ooi), v in adv_flag_vs_ooi.items()
                                       if (flag == "1") != is_ooi)
        crash_rows.append(row)

        if not args.no_partner_check:
            for stem, r in crash_eps:
                res = partner_check(stem, r)
                if res is None:
                    continue
                rec, cid, g_rec, g_c = res
                partner_rows.append({"policy": policy, "adv_mode": mode, "scene": stem, "step": r["end"]["step"],
                                     "recorded": rec, "footprint_closest": cid, "gap_recorded": g_rec,
                                     "gap_closest": g_c, "ooi": ooi_of(load_meta(stem))})

        # windows
        n = judged = none = with_ooi = ooi_worst = 0
        nb = []
        agg_all = tim_all = agg_ooi = tim_ooi = 0
        agg_ooi_present = tim_ooi_present = n_ooi_present_judged = 0
        bg_only_agg = 0
        for f in glob.glob(str(run / "obs/*.pkl")):
            stem = Path(f).stem
            ooi = ooi_of(load_meta(stem))
            for o in pickle.load(open(f, "rb")):
                n += 1
                if o["speed"] < 1.0:
                    continue
                judged += 1
                pn = o["per_neighbour"]
                nb.append(len(pn))
                none += not pn
                s_all, c_all = o["safety"], o["courtesy"]
                a_all = s_all > t_s or c_all > t_c
                agg_all += a_all
                tim_all += s_all < t_t
                if ooi in pn:
                    with_ooi += 1
                    e = pn[ooi]
                    s_o, c_o = e["safety"], (e["courtesy"] or 0.0)
                    worst = max(pn.items(), key=lambda kv: kv[1]["safety"])[0]
                    ooi_worst += worst == ooi
                    n_ooi_present_judged += 1
                    agg_ooi_present += s_o > t_s or c_o > t_c
                    tim_ooi_present += s_o < t_t
                else:
                    s_o, c_o = 0.0, 0.0
                    bg_only_agg += a_all
                agg_ooi += s_o > t_s or c_o > t_c
                tim_ooi += s_o < t_t
        window_rows.append({
            "policy": policy, "adv_mode": mode, "windows": n, "judged": judged,
            "no_neighbour": none / judged if judged else np.nan,
            "ooi_neighbour": with_ooi / judged if judged else np.nan,
            "ooi_is_worst_safety": ooi_worst / judged if judged else np.nan,
            "mean_neighbours": float(np.mean(nb)) if nb else np.nan,
            "aggressive_all": agg_all / judged if judged else np.nan,
            "timid_all": tim_all / judged if judged else np.nan,
            "aggressive_ooi_approx": agg_ooi / judged if judged else np.nan,
            "timid_ooi_approx": tim_ooi / judged if judged else np.nan,
            "aggressive_ooi_present": agg_ooi_present / n_ooi_present_judged if n_ooi_present_judged else np.nan,
            "timid_ooi_present": tim_ooi_present / n_ooi_present_judged if n_ooi_present_judged else np.nan,
            "aggressive_from_bg_only_windows": bg_only_agg / max(agg_all, 1),
        })

    def dump(rows, name):
        if not rows:
            return
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    dump(crash_rows, "crash_partners.csv")
    dump(window_rows, "window_pairs.csv")
    dump(partner_rows, "partner_check.csv")
    print(f"wrote {out}/crash_partners.csv, window_pairs.csv, partner_check.csv")


if __name__ == "__main__":
    main()
