"""Compares evaluate.py's per-scene results across runs (gpucat/PLAN.md, M3).

For each test adversary: every run's arrival, crash and out-of-road rates and
mean route completion; then, for each pair of runs listed in --pairs, the
difference on the same 100 scenes with a paired bootstrap 95% interval
(10,000 resamples of the scenes) and, for the rates, the exact McNemar test
on the scenes where the two runs disagree. With one seed per run this covers
the variation over scenes, not over training seeds.

    python -m scripts.gpucat.compare_runs --eval logs/gpucat/eval \\
        --pairs cat_nav_s0:replay_nav_s0 fair_nav_s0:cat_nav_s0 --out logs/gpucat/eval/compare.md
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

METRICS = ("arrive", "crash", "out_of_road", "completion")


def load(path: Path):
    """{test_adversary: {metric: array over scenes (sorted by scene)}}."""
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault(r["test_adversary"], []).append(r)
    out = {}
    for adv, rows in by.items():
        rows.sort(key=lambda r: int(r["scene"]))
        out[adv] = {m: np.array([float(r["completion"]) if m == "completion" else float(r["outcome"] == m)
                                 for r in rows]) for m in METRICS}
        out[adv]["scenes"] = [r["scene"] for r in rows]
    return out


def paired(a: np.ndarray, b: np.ndarray, binary: bool, rng: np.random.Generator):
    d = a - b
    idx = rng.integers(0, len(d), (10000, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    p = float("nan")
    if binary:
        n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return d.mean(), lo, hi, p


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default="logs/gpucat/eval")
    p.add_argument("--runs", nargs="*", default=None, help="Default: every CSV in --eval, sorted.")
    p.add_argument("--pairs", nargs="*", default=[], help="a:b -- report a minus b.")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    root = Path(args.eval)
    runs = args.runs or sorted(f.stem for f in root.glob("*.csv"))
    data = {r: load(root / f"{r}.csv") for r in runs}
    advs = list(next(iter(data.values())))
    rng = np.random.default_rng(0)
    lines = []
    for adv in advs:
        lines += [f"### test adversary: {adv}", "", "| run | arrive | crash | out of road | completion |", "|---|---|---|---|---|"]
        for r in runs:
            d = data[r][adv]
            lines.append(f"| {r} | {d['arrive'].mean():.2f} | {d['crash'].mean():.2f} | {d['out_of_road'].mean():.2f} | "
                         f"{d['completion'].mean():.3f} |")
        if args.pairs:
            lines += ["", "| a − b | metric | difference | 95% CI | McNemar p |", "|---|---|---|---|---|"]
            for pair in args.pairs:
                a, b = pair.split(":")
                assert data[a][adv]["scenes"] == data[b][adv]["scenes"]
                for m in METRICS:
                    diff, lo, hi, pv = paired(data[a][adv][m], data[b][adv][m], m != "completion", rng)
                    lines.append(f"| {a} − {b} | {m} | {diff * 100:+.1f} pp | [{lo * 100:+.1f}, {hi * 100:+.1f}] | "
                                 f"{'–' if m == 'completion' else f'{pv:.3f}'} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
