"""Audit of which pairs the MetaDrive attribution and responsibility outputs involve (task A6 of the
attribution audit of 2026-10-09). CPU only; reads existing outputs and writes OUT/pairs_by_run.csv.

For every run in logs/responsibility/eval/<policy>/<adv_mode> it reports, from crashes.csv, the rollouts
and the scenes (read only):
  - the collision partner's class: the adversary (CAT's object of interest that was injected), the
    scene's other object of interest when no adversary was injected (adv mode none), other vehicles,
    non-vehicles (the partner is the closest centre among everything spawned, recording.Recorder.nearest);
  - the verdict mix per partner class;
  - from windows.csv / obs: which neighbour is the worst one (safety_against) and how many neighbours a window has,
    how often the OOI is among them, and how often it is the one that sets the window's beta_s.
Usage: ~/venvs/cat39/bin/python -m scripts.gpucat.audit_attribution_pairs --out DIR
"""
import argparse, csv, glob, json, os, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
EVAL = REPO / "logs/responsibility/eval"
ROLL = REPO / "rollouts_eval"
SCENES = REPO / "raw_scenes_500"
from responsibility.scene import Scene

_scene_cache = {}
def ooi_of(stem):
    if stem not in _scene_cache:
        s = Scene.load(SCENES / f"{stem}.pkl")
        _scene_cache[stem] = (s, [s.track_ids[i] for i in s.objects_of_interest if i != s.sdc],
                              s.track_ids[s.sdc])
    return _scene_cache[stem]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted(glob.glob(str(EVAL / "*/*/config.json"))):
        d = Path(run).parent; adv_mode = d.name; policy = d.parent.name
        if policy in ("compare", "levels", "logs"): continue
        # outcomes
        n_ep = len(list((ROLL / policy / adv_mode).glob("*.pkl")))
        crashes = list(csv.DictReader(open(d / "crashes.csv"))) if (d / "crashes.csv").exists() else []
        cls = Counter(); verdict = defaultdict(Counter); types = Counter()
        for r in crashes:
            s, ooi, sdc = ooi_of(r["scene"])
            oid = r["other_id"]
            if r["adversary"] == "1": c = "adversary"
            elif oid in ooi: c = "ooi_not_injected"  # adv mode none (or fair mode that placed none)
            elif r["other_type"] == "VEHICLE": c = "other_vehicle"
            else: c = "non_vehicle"
            cls[c] += 1; verdict[c][r["verdict"]] += 1; types[r["other_type"]] += 1
        # windows
        w = list(csv.DictReader(open(d / "windows.csv")))
        by_scene = defaultdict(list)
        for r in w: by_scene[r["scene"]].append(r)
        n_win = len(w); nn = np.array([int(r["n_neighbours"]) for r in w]) if w else np.zeros(1)
        ooi_worst = 0; worst_known = 0; ooi_in = 0
        for stem, rs in by_scene.items():
            s, ooi, sdc = ooi_of(stem)
            for r in rs:
                if r["safety_against"]:
                    worst_known += 1; ooi_worst += r["safety_against"] in ooi
        rows.append(dict(policy=policy, adv_mode=adv_mode, episodes=n_ep, crashes=len(crashes),
                         adversary=cls["adversary"], ooi_not_injected=cls["ooi_not_injected"],
                         other_vehicle=cls["other_vehicle"], non_vehicle=cls["non_vehicle"],
                         windows=n_win, windows_per_ep=round(n_win / max(n_ep, 1), 2),
                         mean_neigh=round(float(nn.mean()), 2), frac_win_ooi_worst=round(ooi_worst / max(worst_known, 1), 3),
                         verdict_by_class=json.dumps({k: dict(v) for k, v in verdict.items()})))
    with open(out / "pairs_by_run.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    for r in rows:
        print({k: v for k, v in r.items() if k != "verdict_by_class"})
    print("wrote", out / "pairs_by_run.csv")
if __name__ == "__main__": main()
