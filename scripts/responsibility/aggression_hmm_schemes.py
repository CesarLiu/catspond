"""Compares ways of turning per-window responsibility into a verdict on the
aggression test's conditions (responsibility/aggression_test.py): the HMM
levels of fit_levels.py as they are, and the alternatives tried against them.

Features per window: beta_s from the offline tables of offline_safety_grid.py
at horizon H (alpha 0.1, d_sat 10 m; optionally the two-sided score
rel = beta_s(agent -> partner) - beta_s(partner -> agent) at the same window)
and beta_c from the courtesy run's windows.csv (beta_c does not depend on H).
Every scheme is fitted or calibrated on logged driving (sdc and adv) and
applied to every condition. Schemes:

  A  H = 2 s, (beta_s, beta_c), BIC over 2-7 levels, elevated levels aggressive (fit_levels.py)
  B  as A with log(1 + beta_c) (--log-courtesy)
  C  thresholds at the q90 of the logged positive (and q10 of the negative) values, 2 s
  D  the same thresholds at 5 s, on beta_s and on rel
  E  5 s, (beta_s, log(1 + beta_c)), BIC, elevated levels, three EM starts
  F  5 s, signed-log rel alone, 3 levels: the one around 0 normal, above aggressive, below timid

For each condition: the share of aggressive (and timid) windows, of scenes
with any or a majority of such windows, and the AUROC of the scene's maximum
aggressive posterior against the same agent's logged drive.

    python -m scripts.responsibility.aggression_hmm_schemes            # all schemes
    python -m scripts.responsibility.aggression_hmm_schemes D F        # some
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import CONDITIONS, auroc, load_offline  # noqa: E402
from responsibility.hmm import fit_hmm_with_model_selection  # noqa: E402
from responsibility.levels import elevated_in, elevated_levels, relabel_by_aggressiveness  # noqa: E402
from responsibility.results import read_windows  # noqa: E402

ORDER = ["adv_logged", "sdc_logged", "adv_mode", "sdc_vs_mode", "adv_maxgap", "sdc_vs_maxgap", "adv_near1",
         "sdc_vs_near1", "adv_fairinf", "sdc_vs_fairinf", "adv_cat", "sdc_vs_cat", "sdc_fast1.3", "adv_vs_fast1.3",
         "sdc_slow0.7", "adv_vs_slow0.7", "sdc_slow0.5", "adv_vs_slow0.5"]


class Data:
    def __init__(self, val_dir, config, courtesy_config):
        self.val, self.config, self.court, self.cache = Path(val_dir), config, courtesy_config, {}

    def offline(self, c):
        if c not in self.cache:
            self.cache[c] = load_offline(self.val, self.config, c)
        return self.cache[c]

    def table(self, c, H, rel=False):
        """Windows of a condition: scene, step, speed, safety, courtesy (, partner, rel)."""
        d = self.offline(c)
        d = d[(d.H == H) & (d.alpha == 0.1) & (d.dsat == 10)][["scene", "step", "beta", "speed", "gap_actual"]]
        w = pd.DataFrame(read_windows(self.val / "runs" / self.court / c))[["scene", "step", "courtesy"]]
        w["scene"] = w.scene.astype(int)
        t = d.merge(w, on=["scene", "step"], how="inner").rename(columns={"beta": "safety"})
        if rel:
            p = self.offline(CONDITIONS[c].partner)
            p = p[(p.H == H) & (p.alpha == 0.1) & (p.dsat == 10)][["scene", "step", "beta"]]
            t = t.merge(p.rename(columns={"beta": "partner"}), on=["scene", "step"], how="inner")
            t["rel"] = t.safety - t.partner
        return t.sort_values(["scene", "step"])


def seqs(t, feat):
    return {s: feat(g) for s, g in t.groupby("scene")}


def scheme(name, feat, tables, grid=(2, 3, 4, 5, 6, 7), n_states=None, rule="elevated", scene_share=0.5,
           n_features=2, seeds=(0,)):
    fit_seqs = [x for c in ("sdc_logged", "adv_logged") for x in seqs(tables[c], feat).values() if len(x)]
    best = None
    for sd in seeds:  # several EM starts: keep the lowest BIC
        sel = fit_hmm_with_model_selection(fit_seqs, n_states_grid=[n_states] if n_states else list(grid),
                                           n_features=n_features, n_iter=200, seed=sd)
        if best is None or min(sel.bic_by_n_states) < min(best.bic_by_n_states):
            best = sel
    hmm = best.best_model
    scale = np.concatenate(fit_seqs).std(axis=0)
    timid = set()
    if rule == "elevated":
        relabel_by_aggressiveness(hmm, scale)
        agg = set(elevated_levels(hmm, scale))
    else:  # "signed": ordered by mean; the level nearest 0 is normal, above aggressive, below timid
        order = np.argsort(hmm.means_[:, 0], kind="stable")
        hmm.startprob_ = hmm.startprob_[order]
        hmm.transmat_ = hmm.transmat_[np.ix_(order, order)]
        hmm.means_, hmm.covars_ = hmm.means_[order], hmm.covars_[order]
        normal = int(np.argmin(np.abs(hmm.means_[:, 0])))
        agg, timid = set(range(normal + 1, hmm.n_states)), set(range(0, normal))
    print(f"\n##### {name}: BIC {dict(zip(best.n_states_grid, [round(b) for b in best.bic_by_n_states]))}, "
          f"H={hmm.n_states}")
    kinds = elevated_in(hmm, scale) if n_features == 2 else ["-"] * hmm.n_states
    for z in range(hmm.n_states):
        tag = " AGG" if z in agg else " TIMID" if z in timid else ""
        print(f"  level {z}: mean {np.round(hmm.means_[z], 3)} std {np.round(np.sqrt(hmm.covars_[z]), 3)} "
              f"stay {hmm.transmat_[z, z]:.2f} {kinds[z]}{tag}")
    ref, rows = {}, []
    for c in ORDER:
        if c not in tables:
            continue
        post = {s: hmm.forward_filter(x) for s, x in seqs(tables[c], feat).items()}
        lv = {s: p.argmax(1) for s, p in post.items()}
        a_win = np.concatenate([np.isin(v, list(agg)) for v in lv.values()])
        t_win = np.concatenate([np.isin(v, list(timid)) for v in lv.values()])
        share = pd.Series({s: np.isin(v, list(agg)).mean() for s, v in lv.items()})
        pmax = pd.Series({s: p[:, sorted(agg)].sum(1).max() if agg else 0.0 for s, p in post.items()})
        tmax = pd.Series({s: p[:, sorted(timid)].sum(1).max() if timid else 0.0 for s, p in post.items()})
        if CONDITIONS[c].reference is None:
            ref[c] = (pmax, tmax)
        base = ref.get(CONDITIONS[c].reference)
        rows.append({"cond": c, "label": CONDITIONS[c].label, "win": len(a_win), "agg_win": round(a_win.mean(), 3),
                     "timid_win": round(t_win.mean(), 3),
                     f"scenes>={scene_share:.0%}": round((share >= scene_share).mean(), 2),
                     "scenes_any": round((share > 0).mean(), 2),
                     "AUC_agg": None if base is None else round(auroc(pmax, base[0]), 3),
                     "AUC_timid": None if base is None or not timid else round(auroc(tmax, base[1]), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


def thresholds(name, tables, col="safety", q=0.9):
    ref = pd.concat([tables[c] for c in ("sdc_logged", "adv_logged")])
    pos, neg = ref[col][ref[col] > 0], ref[col][ref[col] < 0]
    hi, lo = max(pos.quantile(q), 0.05), min(neg.quantile(1 - q), -0.05)
    print(f"\n##### {name}: aggressive {col} > {hi:.3f}, timid < {lo:.3f}")
    rows = []
    for c in ORDER:
        if c not in tables:
            continue
        t = tables[c]
        rows.append({"cond": c, "label": CONDITIONS[c].label, "agg_win": round((t[col] > hi).mean(), 3),
                     "timid_win": round((t[col] < lo).mean(), 3),
                     "scenes_any_agg": round(t.groupby("scene")[col].max().gt(hi).mean(), 2),
                     "scenes_any_timid": round(t.groupby("scene")[col].min().lt(lo).mean(), 2)})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("schemes", nargs="*", default=list("ABCDEF"))
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--config", default="nms_ooi", help="offline beta_s tables")
    p.add_argument("--courtesy-config", default="nms_ooi_court", help="runs with beta_c")
    a = p.parse_args()
    pd.set_option("display.width", 250)
    D = Data(a.val_dir, a.config, a.courtesy_config)
    conds = [c for c in ORDER if D.offline(c) is not None and (D.val / "runs" / D.court / c).exists()]
    print("conditions:", conds)
    T20 = {c: D.table(c, 20) for c in conds}
    T50 = {c: D.table(c, 50, rel=True) for c in conds}
    raw = lambda g: g[["safety", "courtesy"]].to_numpy()  # noqa: E731
    logc = lambda g: np.stack([g.safety.to_numpy(), np.log1p(g.courtesy.to_numpy())], 1)  # noqa: E731
    slog = lambda v: np.sign(v) * np.log1p(np.abs(v) / 0.1)  # noqa: E731
    if "A" in a.schemes:
        scheme("A. current: H=20, (beta_s, beta_c), BIC 2-7, elevated levels", raw, T20)
    if "B" in a.schemes:
        scheme("B. current + --log-courtesy", logc, T20)
    if "C" in a.schemes:
        thresholds("C. thresholds on beta_s, H=20 (summarize_responsibility)", T20)
        thresholds("C'. thresholds on beta_c, H=20", T20, col="courtesy")
    if "D" in a.schemes:
        thresholds("D. thresholds on beta_s, H=50", T50)
        thresholds("D'. thresholds on rel = beta_s(a) - beta_s(b), H=50", T50, col="rel")
    if "E" in a.schemes:
        scheme("E. H=50, (beta_s, log1p beta_c), BIC, elevated", logc, T50, seeds=(0, 1, 2))
    if "F" in a.schemes:
        scheme("F. H=50, signed-log rel only, 3 levels, signed rule", lambda g: slog(g.rel.to_numpy())[:, None],
               T50, n_states=3, rule="signed", n_features=1, seeds=(0, 1, 2))
        scheme("F'. H=50, signed-log beta_s only, 3 levels, signed rule",
               lambda g: slog(g.safety.to_numpy())[:, None], T50, n_states=3, rule="signed", n_features=1,
               seeds=(0, 1, 2))


if __name__ == "__main__":
    main()
