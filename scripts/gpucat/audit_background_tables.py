"""Summary tables for scripts/gpucat/audit_background.py (task A3 of the
audit of 2026-10-09), recomputed from the rollouts: the collision classes per
policy group, the ego's speed and lag against its log in rear-end crashes, the
share of the object of interest per class, the share of crashes that a
reactive partner would have avoided for deceleration limits of 3, 4, 6 and
8 m/s^2, and the crash classes against the RSS verdict in crashes.csv. With
A = 6 m/s^2, 28% of the crashes are avoidable rear-ends and 73% are avoidable
with the optimistic rule for the other classes included.

    python -m scripts.gpucat.audit_background_tables   (repository root, cat39 env; CPU only)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from scripts.gpucat import audit_background as ab
import csv
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
res = {}
for A in (3.0, 4.0, 6.0, 8.0):
    rows = []
    for grp, pols in ab.POL.items():
        for p in pols:
            with open(ROOT / f"logs/responsibility/eval/{p}/none/crashes.csv") as f:
                for r in csv.DictReader(f):
                    x = ab.analyse(p, r, A=A); x["group"] = grp; rows.append(x)
    res[A] = pd.DataFrame(rows)
df = res[6.0]
n = df.groupby("group").size()
print("crashes per group", n.to_dict())
t = pd.crosstab(df.group, df.cls); print((t.div(n, axis=0) * 100).round(1)); print(t)
d = df[df.cls != "unknown_partner"]
rs = d[d.cls == "rear_ended"]
print("\nrear_ended: slower than log by >1 m/s, by group")
print(rs.groupby("group").agg(n=("scene", "size"), slower=("slower", "sum"), med_lag=("lag_m", "median"), med_dv=("dv_log", "median")))
print("\nmoving partner speed vs ego, rear_ended: median vo, ve", rs.vo.median(), rs.ve.median())
print("\nego faster than log (>1 m/s) among crossing/oncoming/ego_rear:", d[d.cls.isin(["crossing", "oncoming", "ego_rear"])].faster.mean().round(3))
print("\nsame-direction/side slower share:", d[d.cls == "side_same"].slower.mean())
print("\nOOI share by class\n", d.groupby("cls").ooi.mean().round(2))
print("\nAvoidable with reactive traffic by decel limit (share of ALL crashes in group; rear_ended only / + crossing,oncoming,side rule)")
for A, x in res.items():
    g = x.groupby("group")
    ra = g.apply(lambda y: ((y.cls == "rear_ended") & (y.avoid == True)).sum() / len(y) * 100)
    ot = g.apply(lambda y: ((y.cls.isin(["crossing", "oncoming", "side_same"])) & (y.avoid == True)).sum() / len(y) * 100)
    print(f"A={A}: rear-end only {ra.round(1).to_dict()} ; other-classes optimistic {ot.round(1).to_dict()}; all {(x[(x.avoid==True)].shape[0]/len(x)*100):.1f}% / rear only {((x.cls=='rear_ended')&(x.avoid==True)).sum()/len(x)*100:.1f}%")
print("\nrear_ended with ego slower: avoidable count at A=6:", ((rs.avoid == True) & rs.slower).sum(), "of", len(rs))
print("\nunknown partners\n", df[df.cls == "unknown_partner"][["policy", "scene", "crash_step"]])
print("\nrule/rss vs class (rule NaN = n/a):")
print(pd.crosstab(d.cls, d.rss.fillna("n/a")))
