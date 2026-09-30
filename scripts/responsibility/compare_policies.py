"""Compares driving policies by what they did in MetaDrive and how
responsibly they drove: one row per run of compute_responsibility.py over a
policy's rollouts (--rollouts), judged against a reference run -- the
replayed log (the logged driving in the scenes the policies saw).

Per run (policy x adversary mode):

  episodes, crash rate (vehicle collisions), route completion, arrival and
      out-of-road rates                    from the rollouts
  ego-fault share    of the attributed collisions, the share where the ego's
                     safety responsibility exceeded the other's
                     (crashes*.csv; responsibility/blame.py); other-fault
                     likewise
  rule agreement     of the collisions both the counterfactual verdict and
                     the rear-end rule ("the follower is at fault") decide
                     as ego or other, the share where they agree; rule
                     coverage: the share of attributed collisions the rule
                     decides at all (rear-end ones)
  stopped            share of windows slower than --min-speed (not judged)
  aggressive         share of judged windows with beta_s or beta_c above the
                     reference's thresholds (summarize_responsibility's
                     calibration); split into safety / courtesy
  timid              share of judged windows with beta_s below the
                     reference's timid threshold: more margin to everyone
                     than the alternatives, far beyond what logged drivers keep
  x ref              aggressive and timid shares relative to the reference's
  beta_s median, beta_c p90   of the judged windows
  levels             with --hmm (fit_levels on the reference, e.g.
                     --fit-runs <replay run>): share of windows per
                     responsibility level and in the aggressive levels

Writes OUT/comparison.csv, OUT/comparison.md and OUT/comparison.png (beta_s
per run; level shares or aggressive/timid shares), and prints the table.

Example:
    python -m scripts.responsibility.compare_policies \\
        --runs logs/responsibility/policies/replay/none logs/responsibility/policies/td3_cat/none \\
               logs/responsibility/policies/td3_cat/cat \\
        --hmm logs/responsibility/policies/levels/hmm.pkl --out-dir logs/responsibility/policies/compare
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.levels import assign_levels  # noqa: E402
from responsibility.results import (  # noqa: E402
    calibrate,
    calibrate_timid,
    features,
    read_config,
    read_crashes,
    read_windows,
    sequences,
)
from responsibility.rollouts import load_rollout, outcome, rollout_files  # noqa: E402

FAULT_VERDICTS = ("ego", "other", "shared", "ego-only")
SIDES = ("ego", "other")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="compute_responsibility.py outputs, one per policy/mode.")
    p.add_argument("--labels", nargs="+", default=None, help="Row names (default: policy/adv_mode of the rollouts).")
    p.add_argument("--reference", default=None, help="Run calibrating the thresholds (default: the first --runs).")
    p.add_argument("--hmm", default=None, help="hmm.pkl of fit_levels.py, for level shares.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--timid-quantile", type=float, default=0.1)
    p.add_argument("--safety-floor", type=float, default=0.1)
    p.add_argument("--courtesy-floor", type=float, default=0.01)
    p.add_argument("--min-speed", type=float, default=1.0, help="m/s; slower windows are counted as stopped.")
    return p.parse_args(argv)


def run_outcomes(run):
    """(label from the rollouts, their outcome rows) -- (None, []) if the run
    was not over rollouts or they are not at the recorded path."""
    config = read_config(run) or {}
    directory = config.get("rollouts")
    if not directory or not Path(directory).is_dir():
        if directory:
            print(f"warning: rollouts of {run} not found at {directory}; episode columns left empty")
        return None, []
    rows = [outcome(load_rollout(f)) for f in rollout_files(directory)]
    label = f"{rows[0]['policy']}/{rows[0]['adv_mode']}" if rows else None
    return label, rows


def _windows(run):
    try:
        return read_windows(run)
    except FileNotFoundError:
        return []


def _mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def thresholds(reference_windows, args):
    judged = [w for w in reference_windows if w["speed"] >= args.min_speed]
    safety = [w["safety"] for w in judged]
    return (calibrate(safety, args.quantile, args.safety_floor),
            calibrate([w["courtesy"] for w in judged], args.quantile, args.courtesy_floor),
            calibrate_timid(safety, args.timid_quantile, args.safety_floor))


def run_row(label, windows, outcomes, crashes, t_s, t_c, t_t, min_speed, levels=None):
    judged = [w for w in windows if w["speed"] >= min_speed]
    s = np.array([w["safety"] for w in judged])
    c = np.array([w["courtesy"] for w in judged])
    attributed = [r for r in crashes if r["verdict"] in FAULT_VERDICTS]
    decided = [r for r in attributed if r["verdict"] in SIDES and r.get("rule") in SIDES]
    row = {
        "run": label,
        "episodes": len(outcomes),
        "crash_rate": _mean([o["crash_vehicle"] for o in outcomes]),
        "ego_fault_share": _mean([r["verdict"] == "ego" for r in attributed]),
        "other_fault_share": _mean([r["verdict"] == "other" for r in attributed]),
        "rule_agreement": _mean([r["verdict"] == r["rule"] for r in decided]),
        "rule_coverage": _mean([r.get("rule") in SIDES for r in attributed]),
        "route_completion": _mean([o["route_completion"] for o in outcomes]),
        "arrive_rate": _mean([o["arrive_dest"] for o in outcomes]),
        "out_of_road_rate": _mean([o["out_of_road"] for o in outcomes]),
        "windows": len(windows),
        "stopped": _mean([w["speed"] < min_speed for w in windows]),
        "aggressive": _mean(((s > t_s) | (c > t_c)).tolist()),
        "aggressive_safety": _mean((s > t_s).tolist()),
        "aggressive_courtesy": _mean((c > t_c).tolist()),
        "timid": _mean((s < t_t).tolist()),
        "safety_median": float(np.median(s)) if s.size else float("nan"),
        "courtesy_p90": float(np.quantile(c, 0.9)) if c.size else float("nan"),
    }
    if levels is not None:
        hmm, log_courtesy, aggressive = levels
        seqs = list(sequences(windows).values())
        counts = np.zeros(hmm.n_states)
        for level, _ in assign_levels(hmm, [features(q, log_courtesy) for q in seqs]):
            counts += np.bincount(level, minlength=hmm.n_states)
        total = max(counts.sum(), 1)
        for z in range(hmm.n_states):
            row[f"level_{z}"] = counts[z] / total
        row["aggressive_levels"] = counts[list(aggressive)].sum() / total if aggressive else 0.0
    return row


def with_reference_ratios(rows, reference):
    for row in rows:
        for key in ("aggressive", "timid"):
            ref = reference[key]
            row[f"{key}_x_ref"] = row[key] / ref if ref and np.isfinite(ref) and ref > 0 else float("nan")
    return rows


PERCENT = ("crash_rate", "ego_fault_share", "other_fault_share", "rule_agreement", "rule_coverage",
           "route_completion", "arrive_rate",
           "out_of_road_rate", "stopped", "aggressive", "aggressive_safety", "aggressive_courtesy", "timid",
           "aggressive_levels")
COLUMNS = [("run", "run"), ("episodes", "episodes"), ("crash_rate", "crash"), ("ego_fault_share", "ego-fault"),
           ("rule_agreement", "rule agree"), ("route_completion", "route compl."), ("stopped", "stopped"), ("aggressive", "aggressive"),
           ("aggressive_x_ref", "x ref"), ("timid", "timid"), ("timid_x_ref", "x ref"),
           ("safety_median", "β_s median"), ("courtesy_p90", "β_c p90")]


def fmt(key, value):
    if isinstance(value, str) or value is None:
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(value)
    if not np.isfinite(value):
        return "–"
    if key in PERCENT or key.startswith("level_"):
        return f"{100 * value:.1f}%"
    if key.endswith("_x_ref"):
        return f"{value:.2f}×"
    return f"{value:+.3f}" if key == "safety_median" else f"{value:.3f}"


def markdown(rows):
    columns = list(COLUMNS)
    extra = [k for k in rows[0] if k.startswith("level_") or k == "aggressive_levels"]
    columns += [(k, k.replace("_", " ")) for k in extra]
    lines = ["| " + " | ".join(name for _, name in columns) + " |",
             "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(fmt(k, row.get(k)) for k, _ in columns) + " |")
    return "\n".join(lines)


def plot(labels, safety, rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    level_keys = [k for k in rows[0] if k.startswith("level_")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    data = [np.clip(v, -5, 5) if len(v) else np.zeros(1) for v in safety]
    ax.violinplot(data, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel(r"safety responsibility $\beta_s$ [m] (clipped to $\pm$5)")
    ax.grid(alpha=0.3)
    x = np.arange(len(labels))
    if level_keys:
        cmap = plt.get_cmap("viridis", len(level_keys))
        bottom = np.zeros(len(labels))
        for z, key in enumerate(level_keys):
            share = np.array([r[key] for r in rows])
            bx.bar(x, share, bottom=bottom, color=cmap(z), label=f"level {z}")
            bottom += share
        bx.set_ylabel("share of windows")
    else:
        width = 0.4
        bx.bar(x - width / 2, [r["aggressive"] for r in rows], width, color="#d62728", label="aggressive")
        bx.bar(x + width / 2, [r["timid"] for r in rows], width, color="#1f77b4", label="timid")
        bx.set_ylabel("share of judged windows")
    bx.set_xticks(x)
    bx.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    bx.legend(fontsize=7)
    bx.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    if args.labels is not None and len(args.labels) != len(args.runs):
        raise SystemExit("--labels needs one name per run")
    reference = args.reference or args.runs[0]
    t_s, t_c, t_t = thresholds(_windows(reference), args)
    levels = None
    if args.hmm:
        with open(args.hmm, "rb") as f:
            saved = pickle.load(f)
        levels = (saved["hmm"], saved["log_courtesy"], saved.get("aggressive_levels", []))

    rows, labels, safety = [], [], []
    for i, run in enumerate(args.runs):
        label, outcomes = run_outcomes(run)
        label = args.labels[i] if args.labels else (label or Path(run).name)
        windows = _windows(run)
        rows.append(run_row(label, windows, outcomes, read_crashes(run), t_s, t_c, t_t, args.min_speed, levels))
        labels.append(label)
        safety.append(np.array([w["safety"] for w in windows if w["speed"] >= args.min_speed]))
    ref_row = rows[args.runs.index(reference)] if reference in args.runs else run_row(
        "reference", _windows(reference), [], [], t_s, t_c, t_t, args.min_speed)
    with_reference_ratios(rows, ref_row)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = markdown(rows)
    header = (f"Thresholds from {reference}: aggressive beta_s > {t_s:.3f} m or beta_c > {t_c:.4f} nats, "
              f"timid beta_s < {t_t:.3f} m (windows at >= {args.min_speed} m/s).")
    (out / "comparison.md").write_text(header + "\n\n" + table + "\n", encoding="utf-8")
    plot(labels, safety, rows, out / "comparison.png")
    print(header + "\n")
    print(table)
    print(f"\nwrote {out / 'comparison.csv'}, {out / 'comparison.md'}, {out / 'comparison.png'}")
    return rows


if __name__ == "__main__":
    main()
