"""Turns per-window responsibility (compute_responsibility.py) into a verdict:
did the agent drive aggressively, where and toward whom?

A window is flagged
  safety-aggressive    when beta_s > safety threshold (m): the agent gave up
                       more safety margin than its own alternatives would have
  courtesy-aggressive  when beta_c > courtesy threshold (nats): its presence
                       changed a neighbour's intended goal unusually much
and a scene is aggressive when at least --min-windows of its windows are.

Thresholds are either given, or calibrated as the --quantile of a reference
population of windows (--reference, default: the run itself). Calibrating on
logged driving -- e.g. the self-driving car over all 500 scenes -- makes
"aggressive" mean "more than logged drivers in the top (1 - quantile)".
Windows where the agent is slower than --min-speed are left out: a car
standing still has no alternatives worth comparing.

Writes OUT/windows_flagged.csv, OUT/scenes.csv and OUT/responsibility.png
(safety vs courtesy of every window, thresholds drawn), and prints the most
aggressive scenes.

Example:
    python -m scripts.responsibility.summarize_responsibility \\
        --run logs/responsibility/sdc --out-dir logs/responsibility/sdc/summary
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.results import read_windows  # noqa: E402


def calibrate(values, quantile, floor):
    """The quantile of the positive values (non-interacting windows sit at 0
    and would drag a plain quantile to 0), never below ``floor``."""
    positive = np.asarray([v for v in values if v > 0])
    if positive.size == 0:
        return floor
    return max(float(np.quantile(positive, quantile)), floor)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="Output directory of compute_responsibility.py.")
    p.add_argument("--reference", default=None, help="Run whose windows calibrate the thresholds (default: --run).")
    p.add_argument("--out-dir", default=None, help="Default: RUN/summary.")
    p.add_argument("--safety-threshold", type=float, default=None, help="m; default: calibrated.")
    p.add_argument("--courtesy-threshold", type=float, default=None, help="nats; default: calibrated.")
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--safety-floor", type=float, default=0.1,
                   help="m; a calibrated safety threshold is never below this (sampling noise).")
    p.add_argument("--courtesy-floor", type=float, default=0.01, help="nats; likewise for courtesy.")
    p.add_argument("--min-speed", type=float, default=1.0, help="m/s; slower windows are not judged.")
    p.add_argument("--min-windows", type=int, default=1, help="Flagged windows that make a scene aggressive.")
    p.add_argument("--top", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    rows = [r for r in read_windows(args.run) if r["speed"] >= args.min_speed]
    ref = rows if args.reference is None else [r for r in read_windows(args.reference) if r["speed"] >= args.min_speed]
    if not rows:
        raise SystemExit("no windows to judge")
    t_s = args.safety_threshold if args.safety_threshold is not None else \
        calibrate([r["safety"] for r in ref], args.quantile, args.safety_floor)
    t_c = args.courtesy_threshold if args.courtesy_threshold is not None else \
        calibrate([r["courtesy"] for r in ref], args.quantile, args.courtesy_floor)

    out = Path(args.out_dir or Path(args.run) / "summary")
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        r["safety_aggressive"] = int(r["safety"] > t_s)
        r["courtesy_aggressive"] = int(r["courtesy"] > t_c)
    with open(out / "windows_flagged.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    scenes = {}
    for r in rows:
        scenes.setdefault(r["scene"], []).append(r)
    summary = []
    for scene, ws in scenes.items():
        flagged = [w for w in ws if w["safety_aggressive"] or w["courtesy_aggressive"]]
        worst = max(ws, key=lambda w: (w["safety"] / t_s) + (w["courtesy"] / t_c))
        summary.append({
            "scene": scene, "scenario_id": ws[0]["scenario_id"], "agent_id": ws[0]["agent_id"],
            "windows": len(ws), "flagged": len(flagged),
            "safety_flagged": sum(w["safety_aggressive"] for w in ws),
            "courtesy_flagged": sum(w["courtesy_aggressive"] for w in ws),
            "max_safety": round(max(w["safety"] for w in ws), 4),
            "max_courtesy": round(max(w["courtesy"] for w in ws), 5),
            "first_flag_time": min((w["time"] for w in flagged), default=""),
            "worst_time": worst["time"], "worst_safety_against": worst["safety_against"],
            "worst_courtesy_toward": worst["courtesy_toward"],
            "aggressive": int(len(flagged) >= args.min_windows),
        })
    summary.sort(key=lambda s: (-s["flagged"], -s["max_safety"]))
    with open(out / "scenes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    n_aggr = sum(s["aggressive"] for s in summary)
    print(f"{len(rows)} windows (speed >= {args.min_speed} m/s) in {len(summary)} scenes")
    print(f"thresholds: safety > {t_s:.3f} m, courtesy > {t_c:.4f} nats "
          f"({'given' if args.safety_threshold is not None else f'q{args.quantile:g} of positive reference values'})")
    print(f"flagged windows: safety {sum(r['safety_aggressive'] for r in rows)}, "
          f"courtesy {sum(r['courtesy_aggressive'] for r in rows)}; aggressive scenes: {n_aggr}/{len(summary)}")
    print(f"\n{'scene':>6} {'windows':>7} {'flagged':>7} {'max safety':>11} {'max courtesy':>13}  worst at / against")
    for s in summary[: args.top]:
        print(f"{s['scene']:>6} {s['windows']:>7} {s['flagged']:>7} {s['max_safety']:>+11.3f} {s['max_courtesy']:>13.4f}"
              f"  t={s['worst_time']:.1f}s safety vs {s['worst_safety_against']}, courtesy to {s['worst_courtesy_toward']}")

    plot(rows, t_s, t_c, out / "responsibility.png")
    print(f"\nwrote {out / 'windows_flagged.csv'}, {out / 'scenes.csv'}, {out / 'responsibility.png'}")


def plot(rows, t_s, t_c, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.array([r["safety"] for r in rows])
    c = np.array([r["courtesy"] for r in rows])
    flagged = (s > t_s) | (c > t_c)
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=120)
    ax.scatter(s[~flagged], c[~flagged], s=10, alpha=0.5, color="#7f7f7f", label="not flagged")
    ax.scatter(s[flagged], c[flagged], s=14, alpha=0.8, color="#d62728", label="aggressive")
    ax.axvline(t_s, color="#d62728", lw=0.8, ls="--")
    ax.axhline(t_c, color="#d62728", lw=0.8, ls="--")
    ax.set_xlabel(r"safety responsibility $\beta_s$ [m]")
    ax.set_ylabel(r"courtesy responsibility $\beta_c$ [nats]")
    ax.set_title(f"{len(rows)} windows; thresholds {t_s:.2f} m / {t_c:.3f} nats", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
