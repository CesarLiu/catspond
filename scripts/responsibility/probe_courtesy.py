"""Mechanism probe for the aggression test (responsibility/aggression_test.py):
does DenseTNT's prediction of the ego change when CAT's adversary bears down
on it? beta_c(adv -> ego) at frame k is KL(pi_ego(. | x_k) || pi_ego(. | x_k
without the adversary)). Both sides condition on the history up to k, so the
adversary's approach enters once it is history; the question is whether the
model responds to it.

Measured at 3, 2, 1 and 0.5 s before the adversary first overlaps the ego,
in CAT's scenes and at the same steps in the logged scenes (scenes whose
first overlap comes before step 40 are skipped):

  KL      beta_c toward the ego (nats)
  d_goal  distance between the expected goals with and without the adversary (m)
  d_prog  change in the ego's expected progress along its heading (m; < 0: predicted to hold back)
  gap     centre distance adversary-ego at k (m)

Writes one row per scene, source and lead time to --out (CSV) and prints the
medians and, paired by scene and step, how often the KL with CAT's adversary
exceeds the logged one.

    python -m scripts.responsibility.probe_courtesy --cat-scenes adv_scenes/cat \\
        --out logs/responsibility/validity/courtesy_probe.csv --device cuda
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import scene_index  # noqa: E402
from responsibility.densetnt import DenseTNT  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

LEADS = (30, 20, 10, 5)  # steps before the first overlap


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cat-scenes", default="adv_scenes/cat")
    p.add_argument("--raw-scenes", default="raw_scenes_500")
    p.add_argument("--n", type=int, default=100, help="scenes 0 .. n-1")
    p.add_argument("--out", required=True, help="CSV")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cat, raw = Path(a.cat_scenes), Path(a.raw_scenes)
    idx = scene_index(cat)
    model = DenseTNT(device=a.device)
    rows = []
    for s in range(a.n):
        e = idx[str(s)].get("first_overlap_with_logged_ego")
        if e is None or e - 30 < 10:
            continue
        for src, folder in (("CAT", cat), ("logged", raw)):
            sc = Scene.load(folder / f"{s}.pkl")
            ego = sc.sdc
            adv = [i for i in sc.objects_of_interest if i != ego][0]
            for lead in LEADS:
                k = e - lead
                if not (sc.valid[adv, k] and sc.valid[ego, k]):
                    continue
                pair = model.with_and_without(sc, k, ego, adv)
                if pair is None:
                    continue
                w, wo = pair
                lp, lq = w.log_prob, wo.log_prob
                kl = float((lp.exp() * (lp - lq)).sum())
                g = torch.as_tensor(w.goals_global, dtype=torch.float64, device=lp.device)
                mw = (lp.exp().double()[:, None] * g).sum(0).cpu().numpy()
                mwo = (lq.exp().double()[:, None] * g).sum(0).cpu().numpy()
                u = np.array([np.cos(sc.heading[ego, k]), np.sin(sc.heading[ego, k])])
                rows.append({"scene": s, "src": src, "lead_s": lead / 10, "KL": max(kl, 0.0),
                             "d_goal": float(np.linalg.norm(mw - mwo)), "d_prog": float((mw - mwo) @ u),
                             "gap": float(np.linalg.norm(sc.position[adv, k, :2] - sc.position[ego, k, :2]))})
    d = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["lead_s", "src"]).agg(
        n=("KL", "size"), gap_med=("gap", "median"), KL_med=("KL", "median"),
        KL_q90=("KL", lambda x: x.quantile(.9)), KL_gt_0p3=("KL", lambda x: (x > 0.3).mean()),
        d_goal_med=("d_goal", "median"), d_prog_med=("d_prog", "median")).round(3).to_string())
    pv = d.pivot_table(index=["scene", "lead_s"], columns="src", values="KL").dropna()
    print("\npaired, share of frames where KL(CAT) > KL(logged):",
          pv.groupby(level="lead_s").apply(lambda x: (x.CAT > x.logged).mean()).round(2).to_dict())


if __name__ == "__main__":
    main()
