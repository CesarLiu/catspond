"""Tests whether the framework of Hsu et al. (IROS 2023), as implemented
here, tells aggressive driving apart, on the labelled conditions of
responsibility/aggression_test.py (CAT's adversaries and their controls).

Framework under test, the paper's: open loop, T_f = 2 s, every frame: beta_s
(CVaR_0.1 over 40 samples, max over interacting neighbours, Eqs. 3 and 5) and
beta_c (KL, max over vehicle neighbours, Eqs. 4 and 5); a Gaussian HMM fitted
on logged driving (BIC over 3, 5, 7, 9, 11 levels, three EM starts); the
causal filter (Eq. 6); aggressive = the levels elevated in safety or courtesy
(levels.elevated_levels).

The event step e of a scene is the adversary's first footprint overlap with
the ego, or else its closest approach; the event phase is the 3 s before e.
Frames at and after an overlap are dropped. Each stage is scored by AUROC
against the same agent's logged drive (95% interval from 2000 scene
bootstraps):

  1 beta_s toward the target (the other object of interest), event phase
  2 beta_s, max over neighbours (the paper's beta_s), event phase
  3 beta_c toward the target, and 4 max over neighbours, event phase
  5 HMM posterior of the aggressive levels, event phase
  6 per sequence: longest stay in the aggressive levels, its share of frames,
    and the scene maxima of beta_s and beta_c

The HMM is cross-fitted: fitted on the logged drives of the even scenes and
applied to the odd ones, and the reverse; the operating point (minimum stay)
is set on the fitting half's logged drives at 10% false positives. Pre-set
criteria: aggressive vs logged >= 0.8 at that operating point; mode and the
victim vs logged within 0.4-0.6; attacker vs victim >= 0.8; monotone in the
grade (near 1 m <= fair-inf <= CAT).

Reads VAL/runs/<config>/<condition>/obs (compute_responsibility, or SMART's
via import_catk_obs.py) and the condition scenes (VAL/scenes, plus the logged
and CAT folders). Prints the tables; --json writes the sequence-level results
and the event timing for plot_aggression.py.

    python -m scripts.responsibility.evaluate_aggression --config paper                # DenseTNT, every frame
    python -m scripts.responsibility.evaluate_aggression --config paper --subsample 5 --frame-s 0.5
    python -m scripts.responsibility.evaluate_aggression --config smart --frame-s 0.5  # SMART's 0.5 s windows
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import (  # noqa: E402
    CONDITIONS, auroc, boot_auroc, fmt_ci, longest_run, scene_dir, scene_index)
from responsibility.hmm import fit_hmm_with_model_selection  # noqa: E402
from responsibility.levels import elevated_in, elevated_levels, relabel_by_aggressiveness  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

TESTED = ["adv_logged", "sdc_logged", "adv_mode", "adv_near1", "adv_fairinf", "adv_cat", "adv_maxgap", "sdc_vs_cat"]
DT, PHASE = 0.1, 30  # s per step; event phase in steps
TTE_BINS = [0, 0.5, 1, 2, 3, 4, 6, 9]


class Test:
    def __init__(self, a):
        self.a = a
        self.val = Path(a.val_dir)
        self.rng = np.random.default_rng(0)  # one stream for every bootstrap, so a run is reproducible
        self.folders = {c: scene_dir(CONDITIONS[c].scene_set, self.val, a.raw_scenes, a.cat_scenes) for c in TESTED}

    def ci(self, pos, neg, col):
        return fmt_ci(boot_auroc(pos, neg, col, rng=self.rng))

    def event_steps(self, c):
        """Per scene: (event step, whether it is an overlap) for the adversary vs the logged ego."""
        folder = self.folders[c]
        index = scene_index(folder)
        out = {}
        for s in range(self.a.n):
            ov = index.get(str(s), {}).get("first_overlap_with_logged_ego")
            if ov is not None:
                out[s] = (int(ov), True)
                continue
            sc = Scene.load(folder / f"{s}.pkl")
            adv = [i for i in sc.objects_of_interest if i != sc.sdc][0]
            t = np.arange(11, sc.n_steps)
            t = t[sc.valid[adv, t] & sc.valid[sc.sdc, t]]
            d = np.linalg.norm(sc.position[adv, t, :2] - sc.position[sc.sdc, t, :2], axis=1)
            out[s] = (int(t[np.argmin(d)]), False)
        return out

    def load(self, c):
        who = CONDITIONS[c].agent
        run = self.val / "runs" / self.a.config / c
        rows = []
        for s in range(self.a.n):
            p = run / "obs" / f"{s}.pkl"
            if not p.exists():
                continue
            sc = Scene.load(self.folders[c] / f"{s}.pkl")
            adv = [i for i in sc.objects_of_interest if i != sc.sdc][0]
            target = sc.track_ids[sc.sdc] if who == "adv" else sc.track_ids[adv]
            with open(p, "rb") as f:
                obs = pickle.load(f)
            for o in obs:
                if (o["step"] - 10) % self.a.subsample:
                    continue
                pn = o["per_neighbour"]
                t = pn.get(target, {})
                others = [v for k, v in pn.items() if k != target]
                rows.append({"scene": s, "step": o["step"], "speed": o["speed"], "safety": o["safety"],
                             "courtesy": o["courtesy"], "t_safety": t.get("safety", 0.0),
                             "t_courtesy": t.get("courtesy") or 0.0, "target_in": target in pn,
                             "o_safety": max((v["safety"] for v in others), default=0.0)})
        return pd.DataFrame(rows)


def fit_hmm(seqs):
    best = None
    for seed in (0, 1, 2):  # EM depends on its start: keep the lowest BIC
        sel = fit_hmm_with_model_selection(seqs, n_states_grid=[3, 5, 7, 9, 11], n_iter=200, seed=seed)
        if best is None or min(sel.bic_by_n_states) < min(best.bic_by_n_states):
            best = sel
    hmm = best.best_model
    scale = np.concatenate(seqs).std(axis=0)
    relabel_by_aggressiveness(hmm, scale)
    return hmm, scale, best


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--config", default="paper", help="runs/<config>/<condition>/obs")
    p.add_argument("--raw-scenes", default="raw_scenes_500")
    p.add_argument("--cat-scenes", default="adv_scenes/cat", help="export_adv_scenes --rule cat output")
    p.add_argument("--n", type=int, default=100, help="scenes 0 .. n-1")
    p.add_argument("--subsample", type=int, default=1, help="keep every n-th frame (k = 10, 10 + n, ...)")
    p.add_argument("--frame-s", type=float, default=0.1, help="seconds between kept frames (0.5 for SMART)")
    p.add_argument("--skip", nargs="*", default=[], help="conditions to leave out (e.g. one still running)")
    p.add_argument("--json", help="write the sequence-level results and the timing here")
    a = p.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    T = Test(a)
    frame_s = a.frame_s
    feats = lambda g: g[["safety", "courtesy"]].to_numpy(dtype=float)  # noqa: E731

    data = {c: T.load(c) for c in TESTED
            if c not in a.skip and (T.val / "runs" / a.config / c / "obs").exists()}
    print({c: (d.scene.nunique(), len(d)) for c, d in data.items()})
    common = sorted(set.intersection(*[set(d.scene) for d in data.values()]))
    print(f"scenes present in every condition: {len(common)}")
    events = {c: T.event_steps(c) for c in data if CONDITIONS[c].reference is not None}
    for c, d in data.items():  # drop frames at and after an overlap
        d.drop(d.index[~d.scene.isin(common)], inplace=True)
        if c in events:
            e = d.scene.map(lambda s: events[c][s][0])
            hit = d.scene.map(lambda s: events[c][s][1])
            d.drop(d.index[hit & (d.step >= e)], inplace=True)

    # HMM, cross-fitted on the logged drives
    folds = {0: [s for s in common if s % 2 == 0], 1: [s for s in common if s % 2 == 1]}
    post_agg = {c: pd.Series(np.nan, index=d.index) for c, d in data.items()}
    run_tab = {c: {} for c in data}
    op = {}
    for f, fit_scenes in folds.items():
        seqs = [feats(g) for c in ("sdc_logged", "adv_logged") for s, g in data[c].groupby("scene")
                if s in fit_scenes and len(g) > 1]
        hmm, scale, sel = fit_hmm(seqs)
        agg = elevated_levels(hmm, scale)
        kinds = elevated_in(hmm, scale)
        print(f"\n[fold {f}] fitted on {sum(len(x) for x in seqs)} logged frames of {len(fit_scenes)} scenes; "
              f"BIC {dict(zip(sel.n_states_grid, [round(b) for b in sel.bic_by_n_states]))} -> H={hmm.n_states}; "
              f"aggressive levels {agg}")
        for z in range(hmm.n_states):
            print(f"   level {z}: safety {hmm.means_[z, 0]:+.3f} +- {np.sqrt(hmm.covars_[z, 0]):.3f} m, "
                  f"courtesy {hmm.means_[z, 1]:.3f} +- {np.sqrt(hmm.covars_[z, 1]):.3f} nats, "
                  f"stay {hmm.transmat_[z, z]:.2f}, {kinds[z]}")
        fit_runs = [longest_run(np.isin(hmm.forward_filter(feats(g)).argmax(1), agg), frame_s)
                    for c in ("sdc_logged", "adv_logged") for s, g in data[c].groupby("scene") if s in fit_scenes]
        op[f] = float(np.quantile(fit_runs, 0.9)) + 1e-9  # 10% false positives on the fitting half
        for c, d in data.items():
            for s, g in d.groupby("scene"):
                if s in fit_scenes:
                    continue
                post = hmm.forward_filter(feats(g))
                post_agg[c].loc[g.index] = post[:, agg].sum(1)
                run_tab[c][s] = (g.step.to_numpy(), np.isin(post.argmax(1), agg), op[f])
    print(f"\noperating points (minimum stay in aggressive levels, s): {op}")
    for c, d in data.items():
        d["p_agg"] = post_agg[c]

    # frame level, event phase, stages 1-5
    rows = []
    for c in data:
        ref = CONDITIONS[c].reference
        if ref is None:
            continue
        ev = events[c]

        def phase(d):
            e = d.scene.map(lambda s: ev[s][0])
            return d[(d.step >= e - PHASE) & (d.step < e)]
        pc, pr = phase(data[c]), phase(data[ref])
        r = {"condition": c, "frames": len(pc)}
        for col, name in (("t_safety", "1 beta_s target"), ("safety", "2 beta_s max"),
                          ("t_courtesy", "3 beta_c target"), ("courtesy", "4 beta_c max"), ("p_agg", "5 HMM p(agg)")):
            r[name] = T.ci(pc, pr, col)
        rows.append(r)
    print("\nEvent phase (3 s before the event), frame AUROC vs the same agent's logged drive at the same steps:")
    print(pd.DataFrame(rows).to_string(index=False))

    # sequence level: each condition against the same agent's log cut at the same step
    ref_all = pd.concat([data["adv_logged"], data["sdc_logged"]])
    thr = float(ref_all.safety[ref_all.safety > 0].quantile(0.9))
    print(f"\nraw temporal baseline: beta_s threshold = q90 of the logged positive values = {thr:.3f} m")

    def seq_scores(c, cut=None):
        out = []
        d = data[c]
        for s, (steps, agg_mask, opf) in run_tab[c].items():
            keep = steps < cut[s] if cut is not None and s in cut else np.ones(len(steps), bool)
            g = d[d.scene == s]
            g = g[g.step.isin(steps[keep])]
            out.append({"scene": s, "run": longest_run(agg_mask[keep], frame_s),
                        "flag": longest_run(agg_mask[keep], frame_s) >= opf,
                        "share": agg_mask[keep].mean() if keep.any() else 0.0,
                        "max_bs": g.safety.max(), "max_bs_t": g.t_safety.max(), "max_bc": g.courtesy.max(),
                        "run_bs": longest_run((g.safety > thr).to_numpy(), frame_s)})
        return pd.DataFrame(out)

    rows, seq, result = [], {}, {"config": a.config, "frame_s": frame_s, "sequence": {}, "timing": {}}
    for c in data:
        ref = CONDITIONS[c].reference
        cut = {s: e for s, (e, hit) in events[c].items() if hit} if c in events else None
        t = seq_scores(c)
        seq[c] = t
        r = {"condition": c, "flagged (HMM)": round(t.flag.mean(), 2), "median stay (s)": t.run.median()}
        if ref is not None:
            tr = seq_scores(ref, cut)
            res = {}
            for col, name in (("run", "HMM stay"), ("share", "HMM share"), ("max_bs", "max beta_s (paper)"),
                              ("max_bs_t", "max beta_s target"), ("max_bc", "max beta_c"),
                              ("run_bs", "run beta_s>q90")):
                res[col] = boot_auroc(t, tr, col, rng=T.rng)
                r[name] = fmt_ci(res[col])
            for col, name in (("run", "TPR@FPR10 HMM stay"), ("max_bs", "TPR@FPR10 max beta_s")):
                cut_v = tr[col].quantile(0.9)
                tpr, fpr = (t[col] > cut_v).mean(), (tr[col] > cut_v).mean()
                r[name] = f"{tpr:.2f} (FPR {fpr:.2f})"
                res[f"tpr_{col}"] = [float(tpr), float(fpr)]
            result["sequence"][c] = res
        rows.append(r)
    print("\nSequence level, AUROC vs the same agent's logged drive cut at the same step (95% CI):")
    print(pd.DataFrame(rows).to_string(index=False))
    if "adv_cat" in seq and "sdc_vs_cat" in seq:
        result["attacker_vs_victim"] = {}
        for col in ("run", "share", "max_bs", "max_bs_t"):
            v = boot_auroc(seq["adv_cat"], seq["sdc_vs_cat"], col, rng=T.rng)
            result["attacker_vs_victim"][col] = v
            print(f"attacker vs victim (same scenes, same frames), AUROC of {col}: {fmt_ci(v)}")

    # timing: aggressive-level share and beta by time to the event (CAT)
    for c in ("adv_cat", "sdc_vs_cat"):
        if c not in data:
            continue
        frames = []
        for src in (c, CONDITIONS[c].reference):
            d = data[src].copy()
            d["tte"] = (d.scene.map(lambda s: events[c][s][0]) - d.step) * DT
            d["bin"] = pd.cut(d.tte, TTE_BINS, right=True)
            frames.append(d)
        d, ref = frames
        t = pd.DataFrame({"p_agg": d.groupby("bin").p_agg.mean(), "p_agg logged": ref.groupby("bin").p_agg.mean(),
                          "beta_s med": d.groupby("bin").safety.median(),
                          "beta_s med logged": ref.groupby("bin").safety.median(),
                          "beta_c med": d.groupby("bin").courtesy.median(),
                          "beta_c med logged": ref.groupby("bin").courtesy.median()})
        print(f"\n{c}: by time to the event (s)")
        print(t.round(3).to_string())
        result["timing"][c] = {"bins": [str(b) for b in t.index], **{k: t[k].tolist() for k in t.columns}}

    # beta_c sees only the history: at k = 10 CAT's adversary and the log share it
    if "adv_cat" in data:
        x = data["adv_cat"].query("step == 10").set_index("scene").courtesy
        y = data["adv_logged"].query("step == 10").set_index("scene").courtesy
        j = pd.concat([x, y], axis=1, join="inner")
        print(f"\nbeta_c at k = 10 (same history, different future): max |CAT - logged| = "
              f"{np.abs(j.iloc[:, 0] - j.iloc[:, 1]).max():.2e} nats over {len(j)} scenes "
              f"(neighbour selection uses the future; per pair the values are equal)")
        x = data["adv_cat"].query("step == 10").set_index("scene").safety
        y = data["adv_logged"].query("step == 10").set_index("scene").safety
        print(f"beta_s at k = 10: CAT median {x.median():+.3f} m vs logged {y.median():+.3f} m")

    # timid: only where the logged pair interacts
    if "adv_maxgap" in data:
        gap = {}
        for s in common:
            sc = Scene.load(Path(a.raw_scenes) / f"{s}.pkl")
            adv = [i for i in sc.objects_of_interest if i != sc.sdc][0]
            t = np.arange(11, sc.n_steps)
            t = t[sc.valid[adv, t] & sc.valid[sc.sdc, t]]
            gap[s] = np.linalg.norm(sc.position[adv, t, :2] - sc.position[sc.sdc, t, :2], axis=1).min() \
                if len(t) else np.inf
        inter = [s for s in common if gap[s] < 10]
        m, lg = data["adv_maxgap"], data["adv_logged"]
        mm = m[m.scene.isin(inter)].groupby("scene").t_safety.min()
        lm = lg[lg.scene.isin(inter)].groupby("scene").t_safety.min()
        print(f"\ntimid (max gap) in the {len(inter)} scenes where the logged pair comes within 10 m: "
              f"min beta_s toward the ego, median {mm.median():+.3f} vs logged {lm.median():+.3f} m; "
              f"AUROC (lower = more timid) {1 - auroc(mm, lm):.2f}; share of frames in aggressive levels "
              f"{m[m.scene.isin(inter)].p_agg.mean():.3f} vs logged {lg[lg.scene.isin(inter)].p_agg.mean():.3f}")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
