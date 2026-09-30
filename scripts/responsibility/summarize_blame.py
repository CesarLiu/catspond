"""Summarises the collision attributions logged during RL training with
cat_RLtrain.py --blame_weighting share (logs/blame/<run>_s<seed>.csv,
responsibility/blame_reward.py): per run, how many collisions were
attributed, the verdicts, the mean penalty weight the ego kept, how often
the full penalty was kept, the agreement with the rear-end rule, the time
an attribution took, and how the mean weight moved over training (--bins
equal spans of training steps). A falling weight means the policy's
collisions became more and more its own fault.

Example:
    python -m scripts.responsibility.summarize_blame --logs logs/blame/*.csv --out logs/blame/summary.md
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

VERDICTS = ("ego", "other", "shared", "ego-only", "unknown", "error")
SIDES = ("ego", "other")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", nargs="+", required=True, help="Attribution logs of cat_RLtrain.py (one per run).")
    p.add_argument("--bins", type=int, default=4, help="Spans of training steps for the weight trend.")
    p.add_argument("--out", default=None, help="Also write the table (markdown).")
    return p.parse_args(argv)


def read_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["weight"] = float(r["weight"])
        r["total_steps"] = int(r["total_steps"])
        r["seconds"] = float(r["seconds"])
    return rows


def run_summary(name, rows, bins):
    w = np.array([r["weight"] for r in rows])
    verdicts = [r["verdict"] for r in rows]
    decided = [r for r in rows if r["verdict"] in SIDES and r.get("rule") in SIDES]
    out = {"run": name, "collisions": len(rows),
           "mean_weight": float(w.mean()) if w.size else float("nan"),
           "full_penalty": float(np.mean(w == 1.0)) if w.size else float("nan"),
           "mostly_other": float(np.mean(w < 0.5)) if w.size else float("nan"),
           "rule_agreement": float(np.mean([r["verdict"] == r["rule"] for r in decided])) if decided else float("nan"),
           "seconds": float(np.mean([r["seconds"] for r in rows])) if rows else float("nan")}
    for v in VERDICTS:
        out[f"verdict_{v}"] = verdicts.count(v) / len(rows) if rows else float("nan")
    steps = np.array([r["total_steps"] for r in rows])
    if rows:
        edges = np.linspace(0, steps.max(), bins + 1)
        which = np.clip(np.searchsorted(edges, steps, side="left") - 1, 0, bins - 1)
        out["trend"] = [float(w[which == b].mean()) if np.any(which == b) else float("nan") for b in range(bins)]
    else:
        out["trend"] = [float("nan")] * bins
    return out


def _p(v):
    return "–" if not np.isfinite(v) else f"{100 * v:.0f}%"


def markdown(summaries, bins):
    head = ["run", "collisions", "mean w", "full penalty", "w < 0.5", "ego", "other", "shared", "ego-only",
            "unknown/error", "rule agree", "s / attribution", f"mean w over training ({bins} spans)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in summaries:
        trend = " → ".join("–" if not np.isfinite(t) else f"{t:.2f}" for t in s["trend"])
        lines.append("| " + " | ".join([
            s["run"], str(s["collisions"]), "–" if not np.isfinite(s["mean_weight"]) else f"{s['mean_weight']:.2f}",
            _p(s["full_penalty"]), _p(s["mostly_other"]), _p(s["verdict_ego"]), _p(s["verdict_other"]),
            _p(s["verdict_shared"]), _p(s["verdict_ego-only"]), _p(s["verdict_unknown"] + s["verdict_error"]),
            _p(s["rule_agreement"]), "–" if not np.isfinite(s["seconds"]) else f"{s['seconds']:.1f}", trend]) + " |")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    summaries = [run_summary(Path(p).stem, read_log(p), args.bins) for p in args.logs]
    table = markdown(summaries, args.bins)
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return summaries


if __name__ == "__main__":
    main()
