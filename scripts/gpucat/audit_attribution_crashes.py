"""Pooled crash-attribution tables by partner class (task A6 of the attribution audit of 2026-10-09:
how the attribution is applied). Reads crashes.csv of every run in logs/responsibility/eval,
seeds pooled per (training setting, adversary mode). Partner classes: ooi (the scene's object of interest =
CAT's adversary when one was injected, the plain logged vehicle in adv mode none) and bg (any other vehicle).
Prints n, share of ooi, and the ego/other/shared shares for all crashes, the ooi crashes and the bg crashes,
with a Wilson 95% interval for the ego share over all crashes. Pooling OOI and background crashes mixes two
populations: under CAT's adversary 78-83% of the TD3 crashes are with the OOI, without an adversary 39-59%.
  ~/venvs/cat39/bin/python -m scripts.gpucat.audit_attribution_crashes [--csv OUT.csv]
"""
import argparse, csv, glob, pickle, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "logs/responsibility/eval"; SCENES = REPO / "raw_scenes_500"
_m = {}
def ooi_of(stem):
    if stem not in _m:
        md = pickle.load(open(SCENES / f"{stem}.pkl", "rb"))["metadata"]
        sdc = str(md["sdc_id"]); o = [str(i) for i in md["objects_of_interest"] if str(i) != sdc]
        _m[stem] = o[0] if o else None
    return _m[stem]
def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv"); a = ap.parse_args()
    cells = defaultdict(lambda: {"ooi": Counter(), "bg": Counter()})
    for f in glob.glob(str(EVAL / "*/*/crashes.csv")):
        d = Path(f).parent; pol, mode = d.parent.name, d.name
        for r in csv.DictReader(open(f)):
            cls = "ooi" if r["other_id"] == ooi_of(r["scene"]) else "bg"
            if r["other_id"] == "": cls = "bg"
            cells[(re.sub(r"_s\d+$", "", pol), mode)][cls][r["verdict"]] += 1
    rows = []
    def tot(c, v): return c["ooi"][v] + c["bg"][v]
    for (s, m), c in cells.items():
        n_ooi, n_bg = sum(c["ooi"].values()), sum(c["bg"].values())
        att = lambda cc: sum(cc[v] for v in ("ego", "other", "shared", "ego-only"))
        a_all = att(c["ooi"]) + att(c["bg"])
        k = tot(c, "ego"); lo, hi = wilson(k, a_all)
        f = lambda cc: (cc["ego"] / att(cc) if att(cc) else np.nan)
        rows.append(dict(setting=s, adv_mode=m, n_crash=n_ooi + n_bg, ooi_frac=n_ooi / max(n_ooi + n_bg, 1),
                         ego_all=k / max(a_all, 1), ego_lo=lo, ego_hi=hi, ego_ooi=f(c["ooi"]), ego_bg=f(c["bg"]),
                         other_all=tot(c, "other") / max(a_all, 1), shared_all=tot(c, "shared") / max(a_all, 1),
                         n_ooi=n_ooi, n_bg=n_bg))
    rows.sort(key=lambda r: (r["adv_mode"], r["setting"]))
    for r in rows:
        print(f"{r['adv_mode']:12s} {r['setting']:26s} n={r['n_crash']:3d} ooi={r['ooi_frac']:.2f} ego_all={r['ego_all']:.2f} "
              f"[{r['ego_lo']:.2f},{r['ego_hi']:.2f}] ego_ooi={r['ego_ooi']:.2f}(n{r['n_ooi']}) ego_bg={r['ego_bg']:.2f}(n{r['n_bg']}) "
              f"other={r['other_all']:.2f} shared={r['shared_all']:.2f}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if __name__ == "__main__": main()
