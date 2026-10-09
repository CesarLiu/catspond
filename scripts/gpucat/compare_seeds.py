"""Compares training settings over several seeds (gpucat/PLAN.md, the
multi-seed study), from evaluate.py's per-scene CSVs named
<setting>_s<seed>.csv.

For each test adversary it reports every setting's mean ± std over seeds of
the arrival, crash and out-of-road rates and the route completion. Then, for
each pair in --pairs, it reports a minus b with two measures of uncertainty:

  Welch     a t-test on the per-seed means (seeds as the unit: the variation
            from training)
  CI        a 95% hierarchical bootstrap interval (10,000 resamples), which
            redraws the seeds of each setting independently and the test
            scenes jointly for both settings, so it covers the variation
            over seeds and over scenes

    python -m scripts.gpucat.compare_seeds --eval logs/gpucat/eval_2m \\
        --pairs cat:replay fair:cat fair_valid:cat fair_valid:fair --out logs/gpucat/eval_2m/compare_seeds.md
"""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

from scripts.gpucat.compare_runs import METRICS, load


def gather(root: Path):
    """{setting: {test_adversary: {metric: array [seeds, scenes]}}} and the seeds per setting."""
    runs = {}
    for f in sorted(root.glob("*.csv")):
        m = re.fullmatch(r"(.+)_s(\d+)", f.stem)
        if m:
            runs.setdefault(m.group(1), {})[int(m.group(2))] = load(f)
    data, seeds = {}, {}
    for setting, by_seed in runs.items():
        seeds[setting] = sorted(by_seed)
        advs = by_seed[seeds[setting][0]]
        data[setting] = {adv: {k: np.stack([by_seed[s][adv][k] for s in seeds[setting]]) for k in METRICS}
                         for adv in advs}
    return data, seeds


def hierarchical_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 10000):
    """95% interval of mean(a) - mean(b), a [Sa, N] and b [Sb, N] over the same N scenes."""
    scenes = rng.integers(0, a.shape[1], (n, a.shape[1]))
    sa = rng.integers(0, a.shape[0], (n, a.shape[0]))
    sb = rng.integers(0, b.shape[0], (n, b.shape[0]))
    ma = np.array([a[sa[i]][:, scenes[i]].mean() for i in range(n)])
    mb = np.array([b[sb[i]][:, scenes[i]].mean() for i in range(n)])
    return np.percentile(ma - mb, [2.5, 97.5])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default="logs/gpucat/eval_2m")
    p.add_argument("--settings", nargs="*", default=None)
    p.add_argument("--pairs", nargs="*", default=[], help="a:b -- report a minus b.")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    data, seeds = gather(Path(args.eval))
    settings = args.settings or sorted(data)
    rng = np.random.default_rng(0)
    lines = ["seeds: " + ", ".join(f"{s} {seeds[s]}" for s in settings), ""]
    for adv in data[settings[0]]:
        lines += [f"### test adversary: {adv}", "", "| setting | arrive | crash | out of road | completion |",
                  "|---|---|---|---|---|"]
        for s in settings:
            d = data[s][adv]
            cells = [f"{100 * d[k].mean(1).mean():.1f} ± {100 * d[k].mean(1).std(ddof=1) if len(seeds[s]) > 1 else 0:.1f}"
                     for k in METRICS]
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        if args.pairs:
            lines += ["", "| a − b | metric | difference | 95% CI (seeds × scenes) | Welch p |", "|---|---|---|---|---|"]
            for pair in args.pairs:
                a, b = pair.split(":")
                for k in METRICS:
                    x, y = data[a][adv][k], data[b][adv][k]
                    lo, hi = hierarchical_ci(x, y, rng)
                    pv = ttest_ind(x.mean(1), y.mean(1), equal_var=False).pvalue if min(len(x), len(y)) > 1 else float("nan")
                    lines.append(f"| {a} − {b} | {k} | {100 * (x.mean() - y.mean()):+.1f} pp | "
                                 f"[{100 * lo:+.1f}, {100 * hi:+.1f}] | {pv:.3f} |")
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
