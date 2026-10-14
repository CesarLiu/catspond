"""Behaviour profiles under different pair selections, from existing obs/*.pkl (task A6 of the attribution
audit of 2026-10-09). CPU only; writes DIR/window_profiles.csv.

For every run of logs/responsibility/eval, each judged window (speed >= 1 m/s) gives three beta_s values:
  all        the shipped one: max beta_s over the interacting neighbours (summarize/compare_policies)
  ooi        beta_s against the object of interest only; 0 where the OOI was not among the window's
             neighbours (what --use-ooi would give, as far as beta_s is ~0 for non-interacting agents)
  bg         max beta_s over the neighbours other than the OOI (0 if none)
Thresholds are recalibrated for every definition on replay/none (the compare_policies recipe: q0.9 of the
positive values for aggressive, q0.1 of the negative values for timid; floors 0.1 m), so the three tables
are each internally consistent. Reports per training setting x adversary mode (seeds pooled): aggressive
and timid shares, and their ratio to replay/none; also the Spearman rank agreement between the
'all' and 'ooi' orderings of the settings, per adversary mode; and the aggressive / timid share by
number of neighbours (the max over more neighbours is biased upwards).
  ~/venvs/cat39/bin/python -m scripts.gpucat.audit_attribution_windows --out DIR
"""
import argparse, glob, pickle, re, sys, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from responsibility.results import calibrate, calibrate_timid
EVAL = REPO / "logs/responsibility/eval"; SCENES = REPO / "raw_scenes_500"
_meta = {}
def ooi_of(stem):
    if stem not in _meta:
        md = pickle.load(open(SCENES / f"{stem}.pkl", "rb"))["metadata"]
        sdc = str(md["sdc_id"]); o = [str(i) for i in md["objects_of_interest"] if str(i) != sdc]
        _meta[stem] = (o[0] if o else None)
    return _meta[stem]

def load_run(run):
    out = []
    for f in glob.glob(str(run / "obs/*.pkl")):
        ooi = ooi_of(Path(f).stem)
        for o in pickle.load(open(f, "rb")):
            if o["speed"] < 1.0: continue
            pn = o["per_neighbour"]
            s_all = o["safety"]
            s_ooi = pn[ooi]["safety"] if ooi in pn else 0.0
            bgs = [v["safety"] for k, v in pn.items() if k != ooi]
            out.append((s_all, s_ooi, max(bgs, default=0.0), len(pn), ooi in pn))
    return np.array(out, dtype=float).reshape(-1, 5)

def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for c in sorted(glob.glob(str(EVAL / "*/*/config.json"))):
        d = Path(c).parent
        if d.parent.name in ("compare", "levels", "logs"): continue
        runs[(d.parent.name, d.name)] = load_run(d)
    ref = runs[("replay", "none")]
    thr = {}
    for j, name in enumerate(("all", "ooi", "bg")):
        v = ref[:, j]
        thr[name] = (calibrate(v, 0.9, 0.1), calibrate_timid(v, 0.1, 0.1))
        print(name, "thresholds aggressive >", round(thr[name][0], 3), "timid <", round(thr[name][1], 3))
    pooled = defaultdict(list)
    for (pol, mode), arr in runs.items():
        pooled[(re.sub(r"_s\d+$", "", pol), mode)].append(arr)
    rows = []
    for (setting, mode), arrs in pooled.items():
        x = np.concatenate(arrs)
        row = {"setting": setting, "adv_mode": mode, "seeds": len(arrs), "windows": len(x),
               "no_neighbour": float((x[:, 3] == 0).mean()), "ooi_present": float(x[:, 4].mean()),
               "mean_neigh": float(x[:, 3].mean())}
        for j, name in enumerate(("all", "ooi", "bg")):
            row[f"agg_{name}"] = float((x[:, j] > thr[name][0]).mean())
            row[f"timid_{name}"] = float((x[:, j] < thr[name][1]).mean())
        rows.append(row)
    base = {r["adv_mode"]: r for r in rows if r["setting"] == "replay"}
    for r in rows:
        ref_r = {r2["adv_mode"]: r2 for r2 in rows if r2["setting"] == "replay" and r2["adv_mode"] == "none"}["none"]
        for name in ("all", "ooi", "bg"):
            r[f"agg_{name}_x"] = r[f"agg_{name}"] / ref_r[f"agg_{name}"] if ref_r[f"agg_{name}"] else np.nan
            r[f"timid_{name}_x"] = r[f"timid_{name}"] / ref_r[f"timid_{name}"] if ref_r[f"timid_{name}"] else np.nan
    with open(out / "window_profiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    settings = [s for s in dict.fromkeys(r["setting"] for r in rows) if s != "replay"]
    for mode in ("none", "cat", "fair2_0.1", "fairinf_0.5"):
        sub = {r["setting"]: r for r in rows if r["adv_mode"] == mode and r["setting"] in settings}
        ss = [s for s in settings if s in sub]
        for k in ("agg", "timid"):
            print(f"mode {mode:12s} {k:5s} spearman(all, ooi) = {spearman([sub[s][k+'_all'] for s in ss], [sub[s][k+'_ooi'] for s in ss]):+.2f}"
                  f"  spearman(all, bg) = {spearman([sub[s][k+'_all'] for s in ss], [sub[s][k+'_bg'] for s in ss]):+.2f}")
    for r in rows:
        print(f"{r['setting']:26s} {r['adv_mode']:12s} n={r['windows']:5d} noN={r['no_neighbour']:.2f} ooi={r['ooi_present']:.2f} "
              f"agg all/ooi/bg {r['agg_all']:.3f}/{r['agg_ooi']:.3f}/{r['agg_bg']:.3f}  timid {r['timid_all']:.3f}/{r['timid_ooi']:.3f}/{r['timid_bg']:.3f}")
    # neighbour-count bias, all runs pooled
    allw = np.concatenate([v for v in runs.values()])
    print("\nby neighbour count (all runs pooled): n, share, aggressive(all), timid(all), mean beta_s(all)")
    for k in range(0, 6):
        m = allw[:, 3] == k if k < 5 else allw[:, 3] >= 5
        if m.sum(): print(k, int(m.sum()), round(m.mean(), 3), round(float((allw[m, 0] > thr['all'][0]).mean()), 3),
                          round(float((allw[m, 0] < thr['all'][1]).mean()), 3), round(float(allw[m, 0].mean()), 3))
    # replay/none only
    print("replay/none by neighbour count:")
    for k in range(0, 6):
        m = ref[:, 3] == k if k < 5 else ref[:, 3] >= 5
        if m.sum(): print(k, int(m.sum()), round(float((ref[m, 0] > thr['all'][0]).mean()), 3), round(float((ref[m, 0] < thr['all'][1]).mean()), 3))
if __name__ == "__main__": main()
