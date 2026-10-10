"""Collision attribution of the MetaDrive policies, stratified by partner
(logs/responsibility/eval/<policy>/<adv_mode>/crashes.csv, 100 test scenes
per run, the first crash of each episode).

The audit of 2026-10-09 (gpucat/PLAN.md) found that the counterfactual
verdict is reliable only in aggregate, that its partner mix differs between
adversary modes (78-83% of crashes under CAT's adversary are with the
object of interest, OOI), and that even the logged ego, which does not
react, is ruled 'ego' in 26.5% of its crashes with CAT's adversary. So
each policy group's crashes are split into OOI and background partners,
verdict shares are given with Wilson 95% intervals pooled over seeds, and
they are compared with the logged-ego reference (policy 'replay') in the
same mode.

    python -m scripts.gpucat.stratified_blame
"""

import csv
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from responsibility.adversarial import cat_adversary  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

EVAL = REPO / "logs/responsibility/eval"
GROUPS = {"logged ego": ["replay"], "replay": [f"td3_replay_s{s}" for s in range(3)],
          "cat": [f"td3_cat_s{s}" for s in range(3)], "fair2_0.1": [f"td3_cat_fair2_0.1_s{s}" for s in range(3)],
          "cat_share": [f"td3_cat_share_s{s}" for s in range(3)], "cat_rss": [f"td3_cat_rss_s{s}" for s in range(3)]}
MODES = ["none", "cat", "fair2_0.1"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return p, c - h, c + h


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return "–" if n == 0 else f"{100 * p:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] ({k}/{n})"


def main():
    ooi = {}
    for i in range(400, 500):
        sc = Scene.from_description(pickle.load(open(REPO / f"raw_scenes_500/{i}.pkl", "rb")))
        a = cat_adversary(sc)
        ooi[str(i)] = sc.track_ids[a] if a is not None else None
    for mode in MODES:
        print(f"\n### test adversary: {mode}\n")
        print("| training | episodes | crash rate | OOI share of crashes | ego verdict, OOI crashes | ego verdict, background crashes | other verdict, OOI crashes |")
        print("|---|---|---|---|---|---|---|")
        for g, runs in GROUPS.items():
            rows, eps = [], 0
            for r in runs:
                f = EVAL / r / mode / "crashes.csv"
                if not f.exists():
                    continue
                eps += 100
                rows += list(csv.DictReader(open(f)))
            if not eps:
                continue
            known = [x for x in rows if x["verdict"] in ("ego", "other", "shared")]
            o = [x for x in known if str(x["other_id"]) == str(ooi.get(x["scene"]))]
            b = [x for x in known if str(x["other_id"]) != str(ooi.get(x["scene"]))]
            print(f"| {g} | {eps} | {fmt(len(rows), eps)} | {fmt(len(o), len(known))} | "
                  f"{fmt(sum(x['verdict'] == 'ego' for x in o), len(o))} | {fmt(sum(x['verdict'] == 'ego' for x in b), len(b))} | "
                  f"{fmt(sum(x['verdict'] == 'other' for x in o), len(o))} |")


if __name__ == "__main__":
    main()
