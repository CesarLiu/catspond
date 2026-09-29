"""Responsibility levels: fits a Gaussian HMM over the (safety, courtesy)
sequences of one or more responsibility runs, choosing the number of levels
H by BIC (paper Sec. IV-A/V-B), orders the levels from calmest (0) to most
aggressive (H-1), and assigns every window its level with the causal Bayes
filter (Eq. 6).

  --runs      runs to label, e.g. the self-driving car and CAT's adversary
  --fit-runs  runs whose windows fit the HMM (default: all --runs); use
              logged driving here when labelling a policy's run

A window is aggressive when its level stands out from the calmest level by
more than half a spread (the feature's standard deviation over the fitted
windows) in safety or in courtesy responsibility, and a scene when at least
--scene-share of its windows are; --aggressive-levels K instead takes the
top K levels. Not just the top level: the levels found on logged driving
separate *kinds* of aggressiveness -- one elevated in safety responsibility
(giving up margin), another in courtesy (changing others' plans) -- and the
ordering alone does not say which kind is worse. Each level is reported with
the dimension it is elevated in. Levels are relative to the fitted
population, so what carries information is comparing runs -- e.g. a
policy's share of aggressive windows against logged driving's. Writes to
--out-dir:

  hmm.pkl            the fitted, relabelled HMM (+ feature settings)
  bic.json           BIC for every H tried
  levels.csv         every window with its level and the level's posterior
  scenes_levels.csv  per run and scene: windows per level, aggressive windows
  levels.png         safety vs courtesy coloured by level, and the BIC curve

Example:
    python -m scripts.responsibility.fit_levels \\
        --runs logs/responsibility/sdc logs/responsibility/adv --out-dir logs/responsibility/levels
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.hmm import fit_hmm_with_model_selection  # noqa: E402
from responsibility.levels import (  # noqa: E402
    assign_levels,
    elevated_levels,
    level_table,
    relabel_by_aggressiveness,
)
from responsibility.results import features, read_windows, sequences  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--fit-runs", nargs="+", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-states", type=int, nargs="+", default=[2, 3, 4, 5, 6, 7])
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-courtesy", action="store_true",
                   help="Fit on log(1 + courtesy): courtesy is heavy-tailed (a few windows of several nats).")
    p.add_argument("--aggressive-levels", type=int, default=None,
                   help="Count the top K levels as aggressive (default: the levels elevated in safety or courtesy).")
    p.add_argument("--scene-share", type=float, default=0.5,
                   help="Share of a scene's windows in aggressive levels that makes the scene aggressive.")
    return p.parse_args()


def run_name(path) -> str:
    return Path(path).name


def main():
    args = parse_args()
    fit_runs = args.fit_runs or args.runs
    by_run = {run_name(r): sequences(read_windows(r), run_name(r)) for r in dict.fromkeys(args.runs + fit_runs)}

    fit_seqs = [features(seq, args.log_courtesy) for r in fit_runs for seq in by_run[run_name(r)].values()]
    n_windows = sum(len(s) for s in fit_seqs)
    print(f"fitting on {len(fit_seqs)} sequences / {n_windows} windows from {[run_name(r) for r in fit_runs]}")
    grid = [h for h in args.n_states if h <= n_windows]
    selection = fit_hmm_with_model_selection(fit_seqs, n_states_grid=grid, n_iter=args.n_iter, seed=args.seed)
    hmm = selection.best_model
    scale = np.concatenate(fit_seqs).std(axis=0)
    relabel_by_aggressiveness(hmm, scale)
    if args.aggressive_levels is not None:
        top = set(range(hmm.n_states - args.aggressive_levels, hmm.n_states))
    else:
        top = set(elevated_levels(hmm, scale))
    for h, bic in zip(selection.n_states_grid, selection.bic_by_n_states):
        print(f"  H={h}: BIC {bic:.1f}{'  <- selected' if h == selection.best_n_states else ''}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "hmm.pkl", "wb") as f:
        pickle.dump({"hmm": hmm, "log_courtesy": args.log_courtesy, "feature_scale": scale,
                     "aggressive_levels": sorted(top)}, f)
    (out / "bic.json").write_text(json.dumps(
        {"n_states": selection.n_states_grid, "bic": selection.bic_by_n_states,
         "selected": selection.best_n_states}, indent=2))

    window_rows, scene_rows, counts = [], [], {}
    for r in args.runs:
        name = run_name(r)
        seqs = by_run[name]
        counts[name] = np.zeros(hmm.n_states, dtype=int)
        labelled = assign_levels(hmm, [features(s, args.log_courtesy) for s in seqs.values()])
        for (run, scene, agent), seq, (level, prob) in zip(seqs.keys(), seqs.values(), labelled):
            for w, z, p in zip(seq, level, prob):
                window_rows.append({"run": run, "scene": scene, "agent_id": agent, "step": w["step"],
                                    "time": w["time"], "speed": w["speed"], "safety": w["safety"],
                                    "courtesy": w["courtesy"], "level": int(z), "level_probability": round(float(p), 4),
                                    "aggressive": int(z in top)})
            counts[name] += np.bincount(level, minlength=hmm.n_states)
            scene_rows.append({
                "run": run, "scene": scene, "agent_id": agent, "windows": len(seq),
                **{f"level_{z}": int((level == z).sum()) for z in range(hmm.n_states)},
                "aggressive_windows": int(np.isin(level, list(top)).sum()),
                "max_level": int(level.max()),
                "first_aggressive_time": next((w["time"] for w, z in zip(seq, level) if z in top), ""),
                "aggressive_share": round(float(np.isin(level, list(top)).mean()), 3),
                "aggressive": int(np.isin(level, list(top)).mean() >= args.scene_share),
            })
    for path, rows in ((out / "levels.csv", window_rows), (out / "scenes_levels.csv", scene_rows)):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    table = level_table(hmm, counts, scale)
    print(f"\n{hmm.n_states} levels (0 = calmest; aggressive: {sorted(top)}), courtesy "
          f"{'as log(1 + nats)' if args.log_courtesy else 'in nats'}:")
    header = (f"{'level':>5} {'safety':>14} {'courtesy':>16} {'stay':>5} {'elevated in':>12}"
              + "".join(f" {n:>10}" for n in counts))
    print(header)
    for row in table:
        print(f"{row['level']:>5} {row['mean_safety']:+7.3f}+-{row['std_safety']:<5.3f} "
              f"{row['mean_courtesy']:8.4f}+-{row['std_courtesy']:<6.4f} {row['stay_probability']:5.2f}"
              f" {row['elevated_in']:>12}"
              + "".join(f" {100 * row[f'share_{n}']:9.1f}%" for n in counts))
    for name in counts:
        scenes = [s for s in scene_rows if s["run"] == name]
        share = counts[name][sorted(top)].sum() / max(counts[name].sum(), 1)
        print(f"{name}: {100 * share:.1f}% of windows in aggressive levels; "
              f"{sum(s['aggressive'] for s in scenes)}/{len(scenes)} scenes with >= {100 * args.scene_share:.0f}% of them")

    plot(window_rows, hmm, selection, top, args.log_courtesy, out / "levels.png")
    print(f"\nwrote {out}/hmm.pkl, bic.json, levels.csv, scenes_levels.csv, levels.png")


def plot(rows, hmm, selection, top, log_courtesy, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120, gridspec_kw={"width_ratios": [3, 2]})
    cmap = plt.get_cmap("viridis", hmm.n_states)
    s = np.array([r["safety"] for r in rows])
    c = np.array([r["courtesy"] for r in rows])
    c = np.log1p(c) if log_courtesy else c
    z = np.array([r["level"] for r in rows])
    for level in range(hmm.n_states):
        m = z == level
        ax.scatter(s[m], c[m], s=10, alpha=0.6, color=cmap(level),
                   label=f"level {level}{' (aggressive)' if level in top else ''}")
    ax.set_xlabel(r"safety responsibility $\beta_s$ [m]")
    ax.set_ylabel(r"courtesy $\log(1+\beta_c)$" if log_courtesy else r"courtesy responsibility $\beta_c$ [nats]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    bx.plot(selection.n_states_grid, selection.bic_by_n_states, marker="o")
    bx.axvline(selection.best_n_states, color="#d62728", ls="--", lw=0.8)
    bx.set_xlabel("number of levels H")
    bx.set_ylabel("BIC")
    bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
