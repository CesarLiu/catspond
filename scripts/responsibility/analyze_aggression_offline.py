"""Scores the aggression test's conditions (responsibility/aggression_test.py)
from the offline beta_s tables of offline_safety_grid.py, over the grid of
horizons H, CVaR levels alpha and saturations d_sat. Three reports:

  grid    per (H, alpha, d_sat): AUROC of each condition against its control,
          scene score = max of beta_s before the first overlap (aggressive
          pairs) or min (timid pairs, reported as P(lower)); plus per-condition
          medians at H = 20 and 50
  paired  per variant (only one agent's trajectory changed: the "self"; the
          other is the "partner"): AUROC of the self against its own logged
          drive, of the partner against its logged drive, and of self against
          partner, for three scene scores: beta (max beta_s), rel (the two-sided
          score max [beta_s(self -> partner) - beta_s(partner -> self)]) and
          prox (closest approach); and the Spearman correlation of the change
          in score with how much the self closed in
  dose    the same changes split by how much the self closed in (> 1 m, within
          1 m, opened > 1 m): does the self's score move and the partner's not?

A score that measures aggression rises for the self when it closes in, and
not for the partner. Reads VAL/offline/<config>__<condition>.csv.gz.

    python -m scripts.responsibility.analyze_aggression_offline --config nms_ooi
    python -m scripts.responsibility.analyze_aggression_offline --config nms_ooi --report paired
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import at_setting, auroc, load_offline, pair_scores  # noqa: E402

GRID_CONDITIONS = ["adv_logged", "adv_mode", "adv_maxgap", "adv_near1", "adv_fairinf", "adv_cat",
                   "sdc_logged", "sdc_vs_cat", "sdc_slow0.5", "sdc_slow0.7", "sdc_fast1.3"]
AGG_PAIRS = [("adv_cat", "adv_logged"), ("adv_cat", "adv_mode"), ("adv_fairinf", "adv_mode"),
             ("adv_near1", "adv_mode"), ("adv_mode", "adv_logged"), ("sdc_fast1.3", "sdc_logged"),
             ("sdc_vs_cat", "sdc_logged"), ("adv_cat", "sdc_vs_cat")]
TIMID_PAIRS = [("adv_maxgap", "adv_mode"), ("adv_maxgap", "adv_logged"), ("sdc_slow0.5", "sdc_logged"),
               ("sdc_slow0.7", "sdc_logged")]
# variant: (self, partner, self logged, partner logged)
VARIANTS = {
    "cat": ("adv_cat", "sdc_vs_cat", "adv_logged", "sdc_logged"),
    "fairinf": ("adv_fairinf", "sdc_vs_fairinf", "adv_logged", "sdc_logged"),
    "near1": ("adv_near1", "sdc_vs_near1", "adv_logged", "sdc_logged"),
    "mode": ("adv_mode", "sdc_vs_mode", "adv_logged", "sdc_logged"),
    "maxgap": ("adv_maxgap", "sdc_vs_maxgap", "adv_logged", "sdc_logged"),
    "fast1.3": ("sdc_fast1.3", "adv_vs_fast1.3", "sdc_logged", "adv_logged"),
    "slow0.7": ("sdc_slow0.7", "adv_vs_slow0.7", "sdc_logged", "adv_logged"),
    "slow0.5": ("sdc_slow0.5", "adv_vs_slow0.5", "sdc_logged", "adv_logged"),
}


class Tables:
    def __init__(self, val_dir, config):
        self.val, self.config, self.cache = Path(val_dir), config, {}

    def get(self, condition):
        if condition not in self.cache:
            self.cache[condition] = load_offline(self.val, self.config, condition)
        return self.cache[condition]

    def at(self, condition, H, alpha, d_sat):
        t = self.get(condition)
        return None if t is None else at_setting(t, H, alpha, d_sat)


def scene_scores(t, how):
    g = t.groupby(level="scene").beta
    return g.max() if how == "max" else g.min()


def report_grid(T):
    for H in (20, 50):
        rows = []
        for c in GRID_CONDITIONS:
            x = T.at(c, H, 0.1, 10)
            if x is None:
                continue
            mx, mn = scene_scores(x, "max"), scene_scores(x, "min")
            rows.append({"cond": c, "scenes": len(mx), "windows": len(x), "max_med": round(mx.median(), 3),
                         "max_q90": round(mx.quantile(0.9), 3), "min_med": round(mn.median(), 3),
                         "win_pos>0.1": round((x.beta > 0.1).mean(), 3),
                         "win_neg<-0.1": round((x.beta < -0.1).mean(), 3),
                         "mingap_med": round(x.groupby(level="scene").gap_actual.min().median(), 2)})
        print(f"\n== {T.config} medians H={H} alpha=0.1 dsat=10")
        print(pd.DataFrame(rows).to_string(index=False))
    grid = [(H, al, ds) for H in (10, 20, 30, 40, 50) for al in (0.1, 0.5, 0.9) for ds in (10, 5, -1)
            if not (al != 0.1 and ds != 10)]
    rows = []
    for H, al, ds in grid:
        r = {"H": H, "alpha": al, "dsat": ds}
        for pairs, how in ((AGG_PAIRS, "max"), (TIMID_PAIRS, "min")):
            for pos, neg in pairs:
                xp, xn = T.at(pos, H, al, ds), T.at(neg, H, al, ds)
                if xp is None or xn is None:
                    continue
                v = auroc(scene_scores(xp, how), scene_scores(xn, how))
                r[f"{pos}>{neg}" if how == "max" else f"{pos}<{neg}"] = round(v if how == "max" else 1 - v, 3)
        rows.append(r)
    print(f"\n== {T.config} AUROC (scene max for aggressive pairs, scene min for timid pairs)")
    print(pd.DataFrame(rows).to_string(index=False))


def variant_scores(T, variant, H, al, ds):
    tabs = [T.at(c, H, al, ds) for c in VARIANTS[variant]]
    if any(t is None for t in tabs):
        return None
    S, P, S0, P0 = tabs
    ss, ps, ss0, ps0 = pair_scores(S, P), pair_scores(P, S), pair_scores(S0, P0), pair_scores(P0, S0)
    common = ss.index.intersection(ps.index).intersection(ss0.index).intersection(ps0.index)
    return ss.loc[common], ps.loc[common], ss0.loc[common], ps0.loc[common]


def report_paired(T):
    for H, al, ds in [(20, 0.1, 10), (30, 0.1, 10), (50, 0.1, 10), (50, 0.5, 10), (50, 0.1, -1), (80, 0.1, 10)]:
        rows = []
        for name in VARIANTS:
            v = variant_scores(T, name, H, al, ds)
            if v is None:
                continue
            ss, ps, ss0, ps0 = v
            dgap = ss.gap - ss0.gap  # < 0: the self closed in
            row = {"variant": name, "n": len(ss), "dgap_med": round(dgap.median(), 2)}
            for sc in ("beta", "rel", "prox"):
                row[f"{sc}:self"] = round(auroc(ss[sc], ss0[sc]), 3)
                row[f"{sc}:partner"] = round(auroc(ps[sc], ps0[sc]), 3)
                row[f"{sc}:self>partner"] = round(auroc(ss[sc], ps[sc]), 3)
                row[f"{sc}:rho"] = round(spearmanr(-dgap, ss[sc] - ss0[sc]).correlation, 2)
            for sc in ("beta_min", "rel_min"):
                row[f"{sc}:self<"] = round(1 - auroc(ss[sc], ss0[sc]), 3)
            rows.append(row)
        print(f"\n=== {T.config} H={H} alpha={al} dsat={ds}")
        print(pd.DataFrame(rows).to_string(index=False))


def report_dose(T):
    for H in (20, 50):
        rows = []
        for name in VARIANTS:
            tabs = [T.at(c, H, 0.1, 10) for c in VARIANTS[name]]
            if any(t is None for t in tabs):
                continue
            S, P, S0, P0 = tabs

            def scene_tab(a, b):
                j = a[["beta", "gap_actual"]].join(b[["beta"]], rsuffix="_o", how="inner")
                j["rel"] = j.beta - j.beta_o
                g = j.groupby(level="scene")
                return pd.DataFrame({"gap": g.gap_actual.min(), "bmax": g.beta.max(), "bmean": g.beta.mean(),
                                     "bmin": g.beta.min(), "rmean": g.rel.mean()})
            ss, ps, ss0, ps0 = scene_tab(S, P), scene_tab(P, S), scene_tab(S0, P0), scene_tab(P0, S0)
            idx = ss.index.intersection(ss0.index).intersection(ps.index).intersection(ps0.index)
            dg = (ss.gap - ss0.gap).loc[idx]
            for lab, m in [("closed in >1m", dg < -1), ("|dgap|<=1m", dg.abs() <= 1), ("opened >1m", dg > 1)]:
                if m.sum() < 3:
                    continue
                i = idx[m.values]
                rows.append({"H": H, "variant": name, "stratum": lab, "n": int(m.sum()), "dgap": round(dg[m].median(), 2),
                             "dBmax_self": round((ss.bmax - ss0.bmax).loc[i].mean(), 2),
                             "dBmax_partner": round((ps.bmax - ps0.bmax).loc[i].mean(), 2),
                             "dBmean_self": round((ss.bmean - ss0.bmean).loc[i].mean(), 3),
                             "dBmean_partner": round((ps.bmean - ps0.bmean).loc[i].mean(), 3),
                             "dBmin_self": round((ss.bmin - ss0.bmin).loc[i].mean(), 2),
                             "dRmean_self": round((ss.rmean - ss0.rmean).loc[i].mean(), 3)})
        print(pd.DataFrame(rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--config", default="nms_ooi", help="offline/<config>__<condition>.csv.gz")
    p.add_argument("--report", nargs="+", default=["grid", "paired", "dose"], choices=["grid", "paired", "dose"])
    a = p.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    T = Tables(a.val_dir, a.config)
    for r in a.report:
        {"grid": report_grid, "paired": report_paired, "dose": report_dose}[r](T)


if __name__ == "__main__":
    main()
