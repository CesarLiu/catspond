"""Mechanism probe for the aggression test (responsibility/aggression_test.py):
does the history already commit the motion set? In the frames up to 4 s
before CAT's adversary first overlaps the logged ego, it counts how many of
the adversary's own alternatives (the 40 DenseTNT samples of a run's records,
T_f = 2 s) also come within 3 m (centre distance) of the ego, against the
actual trajectory, by time to the event. The victim (the logged ego) and the
logged adversary at the same steps are the comparisons.

beta_s near 0 while most alternatives also come within 3 m means the
alternatives inherit the attack from the 1.1 s of history: the motion model
cannot say what the attacker could have done instead.

Columns per condition and bin of time to the event (s): frames, the share in
which the actual trajectory comes within 3 m, the mean share of alternatives
that do, the mean margin min(D_alt, 10) - min(D_act, 10) the alternatives
would have kept in addition (m), and the alternatives' mean spread from the
actual position at 2 s (m).

    python -m scripts.responsibility.probe_inertia --cat-scenes adv_scenes/cat
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import first_overlaps  # noqa: E402
from responsibility.records import load_record, scene_of  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--config", default="sampled_ooi", help="runs/<config>/<condition>/records (40 samples)")
    p.add_argument("--cat-scenes", default="adv_scenes/cat")
    a = p.parse_args()
    overlap = first_overlaps(Path(a.cat_scenes))
    rows = []
    for cond in ("adv_cat", "sdc_vs_cat", "adv_logged"):
        recs = sorted((Path(a.val_dir) / "runs" / a.config / cond / "records").glob("*.pkl"), key=lambda q: int(q.stem))
        for rp in recs:
            e = overlap.get(int(rp.stem))
            if e is None:
                continue
            rec = load_record(rp)
            sc = scene_of(rec)
            agent = rec["agent"]
            other = [i for i in sc.objects_of_interest if i != agent][0]
            for fr in rec["frames"]:
                k = fr["step"]
                if not fr["metric_samples"] or not (e - 40 <= k < e):
                    continue
                fut = slice(k + 1, k + 21)
                b = sc.position[other, fut, :2]
                alt = np.linalg.norm(fr["samples"][:, :20, :2] - b, axis=-1).min(1)
                act = np.linalg.norm(sc.position[agent, fut, :2] - b, axis=-1).min()
                spread = np.linalg.norm(fr["samples"][:, 19, :2] - sc.position[agent, k + 20, :2], axis=-1).mean()
                rows.append({"cond": cond, "tte": (e - k) / 10, "alt_close": (alt < 3).mean(), "act_close": act < 3,
                             "alt_minus_act": np.mean(np.minimum(alt, 10) - min(act, 10)), "spread2s": spread})
    d = pd.DataFrame(rows)
    d["bin"] = pd.cut(d.tte, [0, 1, 2, 3, 4], right=True)
    pd.set_option("display.width", 200)
    print(d.groupby(["cond", "bin"]).agg(
        frames=("tte", "size"), actual_within_3m=("act_close", "mean"), alternatives_within_3m=("alt_close", "mean"),
        mean_margin_gain_m=("alt_minus_act", "mean"), spread_at_2s_m=("spread2s", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
