"""Figures of the aggression test (responsibility/aggression_test.py), drawn
from its outputs; each figure is skipped when its inputs are missing.

  roc.png                ROC of the scene max beta_s (target pair, 2 s) of each
                         condition against the same agent's logged drive
  attacker_vs_victim.png AUROC of attacker vs victim for beta_s alone and the
                         two-sided score, at 2 s and 5 s
  beta_before_event.png  mean beta_s by time before the first overlap: CAT's
                         attacker, its victim, the logged adversary
                         (these three from VAL/offline/<--offline-config>)
  stages.png             sequence AUROC of the paper's pipeline at three stages:
                         target pair, max over neighbours, HMM verdict (--paper-json)
  densetnt_vs_smart.png  the same measures under DenseTNT and SMART (--dtnt-json, --smart-json)
  smart_timing.png       SMART's median beta_s and beta_c by time to the event (--smart-json)
  reaction.png           the ego's predicted progress with the attacker minus without it
                         (--courtesy-probe, --smart-probe)
  filters.png            beta_s with and without valid-counterfactual filters (--filter-runs),
                         the mass each filter removes, and detection with the nmsvalid runs
  rules.png              holder-first rate per right-of-way rule and the forced-yield
                         flags (--rules: validate_right_of_way's and evaluate_forced_yield's summary.md)

The --*-json files are evaluate_aggression.py --json outputs.

    python -m scripts.responsibility.plot_aggression --out-dir logs/responsibility/validity/figures \\
        --cat-scenes adv_scenes/cat --paper-json V/eval_paper.json --dtnt-json V/eval_dtnt_s5.json \\
        --smart-json V/eval_smart.json --courtesy-probe V/courtesy_probe.csv --smart-probe V/smart_probe.csv
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import at_setting, auroc, first_overlaps, load_offline, pair_scores  # noqa: E402

NAVY, BLUE, ORANGE, GREY, LBLUE = "#1B2F4A", "#2E75B6", "#C8641E", "#8A94A3", "#8AB4DE"
plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 18, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#FAFAF7", "axes.facecolor": "#FAFAF7",
                     "savefig.facecolor": "#FAFAF7"})
# Share of probability mass each filter removes, measured with the path-based route on the
# first 30 scenes (README, Design decisions, "Valid counterfactuals", Results on the first 30 scenes).
FILTER_MASS = {"route\n(same intent)": 23, "drivable\narea": 12, "third-agent\ncollision": 1, "kinematics": 0}


def bar_labels(ax, bars, fmt="{:.2f}", dy=0.01, size=15):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", fontsize=size)


class Offline:
    def __init__(self, val, config):
        self.val, self.config = val, config

    def at(self, cond, H, config=None):
        t = load_offline(self.val, config or self.config, cond)
        return None if t is None else at_setting(t, H, 0.1, 10)

    def scores(self, cond, partner, H, config=None):
        a, b = self.at(cond, H, config), self.at(partner, H, config)
        return None if a is None or b is None else pair_scores(a, b)


def roc_curve(pos, neg):
    t = np.unique(np.concatenate([pos, neg]))[::-1]
    return (np.array([0] + [(neg >= x).mean() for x in t]), np.array([0] + [(pos >= x).mean() for x in t]))


def fig_roc(O, out):
    ref = O.scores("adv_logged", "sdc_logged", 20)
    ref_sdc = O.scores("sdc_logged", "adv_logged", 20)
    if ref is None or ref_sdc is None:
        return
    curves = [("CAT attacker", "adv_cat", "sdc_vs_cat", ref, NAVY, "-"),
              ("fair-inf attacker", "adv_fairinf", "sdc_vs_fairinf", ref, BLUE, "-"),
              ("near-miss 1 m", "adv_near1", "sdc_vs_near1", ref, LBLUE, "-"),
              ("victim (logged SDC hit by CAT)", "sdc_vs_cat", "adv_cat", ref_sdc, ORANGE, "--"),
              ("mode (negative control)", "adv_mode", "sdc_vs_mode", ref, GREY, ":")]
    fig, ax = plt.subplots(figsize=(9, 7.6))
    for lab, s, p, r, c, ls in curves:
        x = O.scores(s, p, 20)
        if x is None:
            continue
        com = x.index.intersection(r.index)
        pos, neg = x.loc[com].beta.values, r.loc[com].beta.values
        f, t = roc_curve(pos, neg)
        ax.plot(f, t, color=c, ls=ls, lw=3, label=f"{lab}  {auroc(pos, neg):.2f}")
    ax.plot([0, 1], [0, 1], color="#C9D0D8", lw=1.5)
    ax.set_xlabel("false-positive rate (logged drives)")
    ax.set_ylabel("true-positive rate")
    ax.set_title("scene max $\\beta_s$ vs the same agent's logged drive (AUROC)")
    ax.legend(loc="lower right", fontsize=15, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "roc.png", dpi=150)


def fig_attacker_victim(O, out):
    rows = []
    for H in (20, 50):
        a, v = O.scores("adv_cat", "sdc_vs_cat", H), O.scores("sdc_vs_cat", "adv_cat", H)
        if a is None or v is None:
            return
        com = a.index.intersection(v.index)
        rows.append((auroc(a.loc[com].beta, v.loc[com].beta), auroc(a.loc[com].rel, v.loc[com].rel)))
    fig, ax = plt.subplots(figsize=(9, 6.4))
    x, w = np.arange(2), 0.36
    bar_labels(ax, ax.bar(x - w / 2, [r[0] for r in rows], w, color=LBLUE, label="$\\beta_s$(a→b) only"), size=17)
    bar_labels(ax, ax.bar(x + w / 2, [r[1] for r in rows], w, color=NAVY,
                          label="two-sided: $\\beta_s$(a→b) − $\\beta_s$(b→a)"), size=17)
    ax.axhline(0.5, color=GREY, ls=":", lw=1.5)
    ax.axhline(0.8, color=ORANGE, ls="--", lw=1.5)
    ax.text(-0.45, 0.81, "target 0.8", color=ORANGE, fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(["horizon 2 s (paper)", "horizon 5 s"])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC, attacker vs victim")
    ax.legend(frameon=False, fontsize=15, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "attacker_vs_victim.png", dpi=150)


def fig_before_event(O, cat_scenes, out):
    ov = first_overlaps(cat_scenes)
    if not ov:
        return
    bins = np.arange(0, 6.5, 0.5)
    fig, ax = plt.subplots(figsize=(10, 6.4))
    for lab, cond, c, ls in [("CAT attacker", "adv_cat", NAVY, "-"), ("victim (logged SDC)", "sdc_vs_cat", ORANGE, "--"),
                             ("logged adversary, same steps", "adv_logged", GREY, ":")]:
        d = O.at(cond, 20)
        if d is None:
            return
        d = d.reset_index()
        d["ov"] = d.scene.map(ov)
        d = d[d.ov.notna() & (d.step < d.ov)]
        d["bin"] = pd.cut((d.ov - d.step) / 10, bins)
        g = d.groupby("bin").beta.mean()
        ax.plot([b.mid for b in g.index], g.values, color=c, ls=ls, lw=3, marker="o", label=lab)
    ax.invert_xaxis()
    ax.set_xlabel("time before the first overlap (s)")
    ax.set_ylabel("mean $\\beta_s$ toward the partner (m)")
    ax.axvspan(0, 2, color="#E3EDF7", zorder=0)
    ax.text(1.0, 0.12, "collision within\nthe 2 s horizon", ha="center", fontsize=15, color=BLUE)
    ax.legend(frameon=False, fontsize=15, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "beta_before_event.png", dpi=150)


def load_json(p):
    return json.loads(Path(p).read_text()) if p and Path(p).exists() else None


def fig_stages(paper, out):
    if paper is None:
        return
    seq = paper["sequence"]
    series = [("CAT attacker", "adv_cat", NAVY, "-"), ("fair-inf", "adv_fairinf", BLUE, "-"),
              ("near-miss 1 m", "adv_near1", LBLUE, "-"), ("victim", "sdc_vs_cat", ORANGE, "--"),
              ("mode (control)", "adv_mode", GREY, ":")]
    fig, ax = plt.subplots(figsize=(10, 6.6))
    for lab, c, col, ls in series:
        if c not in seq:
            continue
        v = [seq[c]["max_bs_t"][0], seq[c]["max_bs"][0], seq[c]["run"][0]]
        ax.plot(range(3), v, color=col, ls=ls, lw=3, marker="o", ms=9)
        ax.text(2.08, v[2], f"{lab}  {v[2]:.2f}", va="center", fontsize=15, color=col)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["target pair\nmax $\\beta_s$", "paper: max over\nneighbours", "HMM\nverdict"])
    ax.set_xlim(-0.2, 3.0)
    ax.set_ylim(0.45, 1)
    ax.axhline(0.5, color="#C9D0D8", lw=1.2)
    ax.set_ylabel("sequence-level AUROC vs logged drive")
    fig.tight_layout()
    fig.savefig(out / "stages.png", dpi=150)


def fig_models(dtnt, smart, out):
    if dtnt is None or smart is None:
        return

    def measures(r):
        s = r["sequence"]
        return [s["adv_cat"]["max_bs"][0], s["adv_cat"]["max_bc"][0], s["adv_cat"]["run"][0],
                s["sdc_vs_cat"]["max_bs"][0], r["attacker_vs_victim"]["max_bs"][0], s["adv_mode"]["max_bs"][0]]
    cats = ["attacker\nmax $\\beta_s$", "attacker\nmax $\\beta_c$", "attacker\nHMM verdict", "victim\nmax $\\beta_s$",
            "attacker vs\nvictim ($\\beta_s$)", "mode\n(control)"]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    x, w = np.arange(len(cats)), 0.38
    bar_labels(ax, ax.bar(x - w / 2, measures(dtnt), w, color=LBLUE, label="DenseTNT"))
    bar_labels(ax, ax.bar(x + w / 2, measures(smart), w, color=NAVY, label="SMART"))
    ax.axhline(0.5, color=GREY, ls=":", lw=1.5)
    ax.axvspan(2.5, 5.5, color="#FBEDE3", zorder=0)
    ax.text(4, 0.99, "attribution and controls", ha="center", fontsize=14, color=ORANGE)
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=16)
    ax.set_ylabel("AUROC vs logged drive")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "densetnt_vs_smart.png", dpi=150)


def fig_smart_timing(smart, out):
    if smart is None or "adv_cat" not in smart["timing"] or "sdc_vs_cat" not in smart["timing"]:
        return
    a, v = smart["timing"]["adv_cat"], smart["timing"]["sdc_vs_cat"]
    labels = [re.sub(r"[(\]]", "", b).replace(", ", "–") for b in a["bins"]]
    x = np.arange(len(labels))[::-1]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    axs[0].plot(x, a["beta_s med"], color=NAVY, lw=3, marker="o", label="CAT attacker")
    axs[0].plot(x, v["beta_s med"], color=ORANGE, ls="--", lw=3, marker="o", label="victim")
    axs[0].plot(x, a["beta_s med logged"], color=GREY, ls=":", lw=3, label="attacker's logged drive")
    axs[0].set_title("median $\\beta_s$ (m)")
    axs[1].plot(x, a["beta_c med"], color=NAVY, lw=3, marker="o", label="CAT attacker")
    axs[1].plot(x, a["beta_c med logged"], color=NAVY, ls=":", lw=2.5, label="attacker's logged drive")
    axs[1].plot(x, v["beta_c med"], color=ORANGE, ls="--", lw=3, marker="o", label="victim (SDC)")
    axs[1].plot(x, v["beta_c med logged"], color=ORANGE, ls=":", lw=2.5, label="victim's logged drive")
    axs[1].set_title("median $\\beta_c$ (nats)")
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_xlabel("time before the event (s)")
        ax.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "smart_timing.png", dpi=150)


def fig_reaction(dtnt_csv, smart_csv, out):
    if not (dtnt_csv and Path(dtnt_csv).exists() and smart_csv and Path(smart_csv).exists()):
        return
    leads = [2.0, 1.0, 0.5]
    med = lambda f: pd.read_csv(f).query("src == 'CAT'").groupby("lead_s").d_prog.median().reindex(leads)  # noqa: E731
    d, s = med(dtnt_csv), med(smart_csv)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    x, w = np.arange(3), 0.36
    bars = list(ax.bar(x - w / 2, d.values, w, color=LBLUE, label="DenseTNT")) + \
        list(ax.bar(x + w / 2, s.values, w, color=NAVY, label="SMART"))
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + (0.02 if h >= 0 else -0.06), f"{h:+.2f}", ha="center", fontsize=15)
    ax.axhline(0, color=GREY, lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g} s before\ncollision" for t in leads])
    ax.set_ylabel("change in ego's predicted progress (m)")
    ax.set_ylim(min(-0.85, s.min() - 0.1), 0.2)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("does the model expect the ego to react to the attacker?")
    fig.tight_layout()
    fig.savefig(out / "reaction.png", dpi=150)


def fig_filters(O, filter_runs, out):
    if not filter_runs:
        return

    def windows(run):
        fs = glob.glob(f"{run}/windows*.csv")
        return pd.concat([pd.read_csv(f) for f in fs]).set_index(["scene", "step"]) if fs else None
    a, b = windows(filter_runs[0]), windows(filter_runs[1])
    if a is None or b is None:
        return
    j = a[["safety"]].join(b[["safety"]], rsuffix="_f", how="inner").dropna()
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.3), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    ax = axs[0]
    ax.scatter(j.safety, j.safety_f, s=22, color=NAVY, alpha=0.55)
    lim = [min(j.safety.min(), j.safety_f.min()) - 0.1, max(j.safety.max(), j.safety_f.max()) + 0.1]
    ax.plot(lim, lim, color=GREY, ls="--", lw=1.5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("$\\beta_s$, unfiltered (m)")
    ax.set_ylabel("$\\beta_s$, valid alternatives only (m)")
    ax.set_title(f"{len(j)} windows: Spearman {spearmanr(j.safety, j.safety_f).correlation:.2f}")
    ax = axs[1]
    bars = ax.bar(range(len(FILTER_MASS)), list(FILTER_MASS.values()), color=[NAVY, BLUE, LBLUE, LBLUE])
    bar_labels(ax, bars, "{:.0f}%", 0.6, 16)
    ax.set_xticks(range(len(FILTER_MASS)))
    ax.set_xticklabels(list(FILTER_MASS), fontsize=15)
    ax.set_ylabel("probability mass removed (%)")
    ax.set_ylim(0, 28)
    ax.set_title("what each filter removes")
    ax = axs[2]
    conds = [("CAT", "adv_cat", "sdc_vs_cat"), ("fair-inf", "adv_fairinf", "sdc_vs_fairinf"),
             ("near 1 m", "adv_near1", "sdc_vs_near1"), ("mode", "adv_mode", "sdc_vs_mode")]
    res = {}
    for cfg in (O.config, O.config.replace("nms", "nmsvalid")):
        ref = O.scores("adv_logged", "sdc_logged", 20, cfg)
        for lab, s, p in conds:
            sa, pa = O.at(s, 20, cfg), O.at(p, 20, cfg)
            if pa is None:  # the partner's run is not needed for beta; take it from the unfiltered config
                pa = O.at(p, 20)
            if ref is None or sa is None or pa is None:
                continue
            x = pair_scores(sa, pa)
            com = x.index.intersection(ref.index)
            res[(cfg, lab)] = auroc(x.loc[com].beta, ref.loc[com].beta)
    labs = [c[0] for c in conds]
    xs, w = np.arange(len(labs)), 0.38
    for i, (cfg, col, name) in enumerate([(O.config, LBLUE, "all alternatives"),
                                          (O.config.replace("nms", "nmsvalid"), NAVY, "valid alternatives")]):
        bar_labels(ax, ax.bar(xs + (i - 0.5) * w, [res.get((cfg, lb), np.nan) for lb in labs], w, color=col,
                              label=name), size=14)
    ax.axhline(0.5, color=GREY, ls=":", lw=1.5)
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    ax.set_ylabel("AUROC vs logged drive")
    ax.legend(frameon=False, fontsize=14, loc="upper right")
    ax.set_title("aggression detection")
    fig.tight_layout()
    fig.savefig(out / "filters.png", dpi=150)


def md_table(path):
    """Rows of the first markdown table in a file, as lists of cell strings (header first)."""
    rows = [ln.strip().strip("|").split("|") for ln in Path(path).read_text().splitlines()
            if ln.startswith("|") and not set(ln) <= set("|-: ")]
    return [[c.strip() for c in r] for r in rows]


def fig_rules(rules, out):
    if not rules or not all(Path(p).exists() for p in rules):
        return
    row = md_table(rules[0])
    keep = ["stop-sign", "all-way-stop/first-in", "red-light", "through-road", "driveway", "left-turn",
            "same-direction rules: lane-change"]
    by = {r[0]: r for r in row[1:]}
    names, n, rate = [], [], []
    for k in keep:
        if k in by and by[k][3] not in ("–", "-"):
            names.append(k.replace("same-direction rules: ", "").replace("/", "\n") + ("\n(dropped)" if "lane" in k else ""))
            n.append(int(by[k][2]))
            rate.append(float(by[k][3].rstrip("%")))
    fy = md_table(rules[1])
    head = fy[0]
    sets = {r[0]: r for r in fy[1:]}
    pick = [("log", "logged,\nholder first"), ("log_yielder_first", "logged,\nyielder first"),
            ("delay_pet2", "rewrite\nPET 2 s"), ("delay_pet1", "rewrite\nPET 1 s")]
    i_thr, i_duty = head.index("> threshold"), head.index("> 4 (duty violated)")
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
    cols = [ORANGE if "dropped" in nm else NAVY for nm in names]
    bars = axs[0].barh(range(len(names))[::-1], rate, color=cols)
    for i, b in enumerate(bars):
        axs[0].text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{rate[i]:.0f}%  (n={n[i]})", va="center", fontsize=14)
    axs[0].set_yticks(range(len(names))[::-1])
    axs[0].set_yticklabels(names, fontsize=14)
    axs[0].set_xlim(0, 118)
    axs[0].set_xlabel("holder passed first (%)")
    axs[0].set_title("right of way, held-out WOMD")
    x, w = np.arange(len(pick)), 0.38
    vals = lambda i: [float(sets[k][i].rstrip("%")) if k in sets else np.nan for k, _ in pick]  # noqa: E731
    bar_labels(axs[1], axs[1].bar(x - w / 2, vals(i_thr), w, color=NAVY, label="$a_{req}$ above threshold"), "{:.0f}%", 1.5)
    bar_labels(axs[1], axs[1].bar(x + w / 2, vals(i_duty), w, color=LBLUE, label="yield duty violated (> 4 m/s²)"),
               "{:.0f}%", 1.5)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([lab for _, lab in pick], fontsize=14)
    axs[1].set_ylim(0, 105)
    axs[1].set_ylabel("pairs flagged (%)")
    axs[1].legend(frameon=False, fontsize=14, loc="upper left")
    axs[1].set_title("forced yields: assertive, not yet unsafe")
    fig.tight_layout()
    fig.savefig(out / "rules.png", dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-dir", default="logs/responsibility/validity")
    p.add_argument("--offline-config", default="nms_ooi")
    p.add_argument("--cat-scenes", default="adv_scenes/cat")
    p.add_argument("--paper-json", help="evaluate_aggression --config paper --json")
    p.add_argument("--dtnt-json", help="evaluate_aggression --config paper --subsample 5 --frame-s 0.5 --json")
    p.add_argument("--smart-json", help="evaluate_aggression --config smart --frame-s 0.5 --json")
    p.add_argument("--courtesy-probe", help="probe_courtesy.py's CSV")
    p.add_argument("--smart-probe", help="probe_smart.py's CSV")
    p.add_argument("--filter-runs", nargs=2, metavar=("UNFILTERED", "FILTERED"),
                   help="two compute_responsibility runs of the same scenes, without and with filters")
    p.add_argument("--rules", nargs=2, metavar=("RIGHT_OF_WAY_SUMMARY", "FORCED_YIELD_SUMMARY"))
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    O = Offline(Path(a.val_dir), a.offline_config)
    fig_roc(O, out)
    fig_attacker_victim(O, out)
    fig_before_event(O, Path(a.cat_scenes), out)
    fig_stages(load_json(a.paper_json), out)
    fig_models(load_json(a.dtnt_json), load_json(a.smart_json), out)
    fig_smart_timing(load_json(a.smart_json), out)
    fig_reaction(a.courtesy_probe, a.smart_probe, out)
    fig_filters(O, a.filter_runs, out)
    fig_rules(a.rules, out)
    print("wrote", sorted(f.name for f in out.glob("*.png")))


if __name__ == "__main__":
    main()
