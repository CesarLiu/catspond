"""Tables over crashes_geometry.csv (metadrive_background_crashes.py):
collision configurations per policy group and, within each, the share where
the ego was slower / behind its log, the partner was an object of interest,
the ego passed a red light, and a 1 s-gap reactive partner could have
braked (a_req <= 4 or 8 m/s^2). The thresholds: slower = more than 1 m/s
below the logged ego's speed, behind = more than 2 m behind its logged arc
position, stopped = below 0.5 m/s.

    python -m scripts.gpucat.metadrive_background_crashes_tables DIR/crashes_geometry.csv
"""
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
d = pd.read_csv(path)
d = d[d.config.notna() & ~d.config.isin(["no-partner", "never-together"])].copy()
d["slower"] = d.dv_ego_vs_log < -1.0          # m/s slower than the logged ego at the same step
d["behind"] = d.ego_lag_m > 2.0               # m behind the logged ego's arc position
d["stopped"] = d.v_ego < 0.5
d["seen"] = d.a_req.notna()
d["avoid4"] = d.a_req <= 4.0
d["avoid8"] = d.a_req <= 8.0
order = ["follower_hits_ego", "ego_hits_leader", "same_dir_side", "crossing", "opposite"]
groups = ["replay", "cat", "fair2_0.1", "share", "rss"]

def pct(x, n):
    return f"{x}/{n} ({100*x/max(n,1):.0f}%)"

print("## configuration per group (crashes without adversary)")
rows = []
for g in groups + ["ALL3", "ALL"]:
    sub = d[d.group == g] if g in groups else (d[d.group.isin(groups[:3])] if g == "ALL3" else d)
    n = len(sub)
    r = {"group": g, "n": n}
    for c in order:
        r[c] = pct(int((sub.config == c).sum()), n)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n## per configuration (replay+cat+fair2_0.1 pooled, then each group)")
for g in ["ALL3"] + groups[:3]:
    sub = d[d.group.isin(groups[:3])] if g == "ALL3" else d[d.group == g]
    rows = []
    for c in order:
        s = sub[sub.config == c]
        n = len(s)
        if n == 0:
            continue
        rows.append({"config": c, "n": n,
                     "ego slower than log (>1 m/s)": pct(int(s.slower.sum()), n),
                     "ego behind log (>2 m)": pct(int(s.behind.sum()), n),
                     "ego stopped (<0.5 m/s)": pct(int(s.stopped.sum()), n),
                     "median dv_ego-log m/s": round(s.dv_ego_vs_log.median(), 2),
                     "median lag m": round(s.ego_lag_m.median(), 1),
                     "partner OOI": pct(int(s.other_is_ooi.sum()), n),
                     "partner v<0.5": pct(int((s.v_other < 0.5).sum()), n),
                     "ego red light": pct(int(s.ego_red.sum()), n),
                     "reactive avoid a<=4": pct(int(s.avoid4.sum()), n),
                     "reactive avoid a<=8": pct(int(s.avoid8.sum()), n),
                     "rule other/ego/na": f"{(s.rule=='other').sum()}/{(s.rule=='ego').sum()}/{(~s.rule.isin(['other','ego'])).sum()}",
                     "rss other/ego/shared/na": f"{(s.csv_rss=='other').sum()}/{(s.csv_rss=='ego').sum()}/{(s.csv_rss=='shared').sum()}/{(~s.csv_rss.isin(['other','ego','shared'])).sum()}",
                     })
    print(f"\n### {g}")
    print(pd.DataFrame(rows).to_string(index=False))

print("\n## headline categories per group")
rows = []
for g in groups + ["ALL3"]:
    sub = d[d.group == g] if g in groups else d[d.group.isin(groups[:3])]
    n = len(sub)
    f = sub.config == "follower_hits_ego"
    rows.append({
        "group": g, "n": n,
        "rear-ended by logged follower": pct(int(f.sum()), n),
        "... while ego slower than log": pct(int((f & sub.slower).sum()), n),
        "... and 1s-gap follower could brake (<=4)": pct(int((f & sub.avoid4).sum()), n),
        "any partner could brake (<=4)": pct(int(sub.avoid4.sum()), n),
        "any partner could brake (<=8)": pct(int(sub.avoid8.sum()), n),
        "ego hits leader / side / cross / opp": f"{(sub.config=='ego_hits_leader').sum()}/{(sub.config=='same_dir_side').sum()}/{(sub.config=='crossing').sum()}/{(sub.config=='opposite').sum()}",
        "ego ran red": pct(int(sub.ego_red.sum()), n),
        "partner OOI": pct(int(sub.other_is_ooi.sum()), n),
    })
print(pd.DataFrame(rows).to_string(index=False))
