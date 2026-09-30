"""Does the verdict depend on the motion model? Compares two responsibility
runs over the same scenes and agent made with different models (e.g.
DenseTNT and UniTraj's MTR; UNITRAJ_PLAN.md, U5).

Windows are matched by (scene, agent, step). Reported:

  windows    matched, and in one run only (e.g. an agent one model predicts
             and the other does not)
  rank corr  Spearman correlation of beta_s and of beta_c over the matched
             windows (all, and those at >= --min-speed)
  flags      each run judged against its own thresholds (as
             summarize_responsibility calibrates them: q90 of the positive
             values, timid q10 of the negative safety values): agreement and
             Cohen's kappa of the aggressive and the timid flags
  scenes     Spearman correlation of the scenes' shares of aggressive
             windows, and agreement of the scene verdicts (aggressive when at
             least --min-windows windows are)
  policies   with --policy-tables A.csv B.csv (compare_policies.py's
             comparison.csv under each model): per column, Spearman
             correlation over the policies and whether "more than the
             reference" (x ref > 1) agrees -- the conclusion a paper states

Writes OUT/model_comparison.md and OUT/model_comparison.png (beta_s and
beta_c of one run against the other).

Example:
    python -m scripts.responsibility.compare_models \\
        --runs logs/responsibility/sdc logs/responsibility_mtr/sdc --out-dir logs/responsibility/model_comparison
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.results import calibrate, calibrate_timid, read_config, read_windows  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs=2, required=True, metavar=("A", "B"))
    p.add_argument("--labels", nargs=2, default=None, help="Default: each run's model name.")
    p.add_argument("--policy-tables", nargs=2, default=None, metavar=("A_CSV", "B_CSV"))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--timid-quantile", type=float, default=0.1)
    p.add_argument("--safety-floor", type=float, default=0.1)
    p.add_argument("--courtesy-floor", type=float, default=0.01)
    p.add_argument("--min-speed", type=float, default=1.0)
    p.add_argument("--min-windows", type=int, default=1)
    return p.parse_args(argv)


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    return float(spearmanr(a, b).correlation)


def kappa(a, b) -> float:
    """Cohen's kappa of two binary labelings."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    po = float(np.mean(a == b))
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float("nan") if pe == 1.0 else (po - pe) / (1 - pe)


def label_of(run):
    model = (read_config(run) or {}).get("model", {"name": "densetnt"})
    return model.get("name", Path(run).name)


def flags(rows, args):
    judged = [r for r in rows if r["speed"] >= args.min_speed]
    t_s = calibrate([r["safety"] for r in judged], args.quantile, args.safety_floor)
    t_c = calibrate([r["courtesy"] for r in judged], args.quantile, args.courtesy_floor)
    t_t = calibrate_timid([r["safety"] for r in judged], args.timid_quantile, args.safety_floor)
    return (t_s, t_c, t_t), {
        (r["scene"], r["agent_id"], r["step"]): (r["safety"] > t_s or r["courtesy"] > t_c, r["safety"] < t_t)
        for r in judged}


def compare_windows(rows_a, rows_b, args):
    key = lambda r: (r["scene"], r["agent_id"], r["step"])  # noqa: E731
    a, b = {key(r): r for r in rows_a}, {key(r): r for r in rows_b}
    both = sorted(set(a) & set(b))
    judged = [k for k in both if a[k]["speed"] >= args.min_speed]
    out = {"matched": len(both), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a)),
           "judged": len(judged)}
    for name in ("safety", "courtesy"):
        out[f"spearman_{name}"] = spearman([a[k][name] for k in both], [b[k][name] for k in both])
        out[f"spearman_{name}_judged"] = spearman([a[k][name] for k in judged], [b[k][name] for k in judged])
    thr_a, fa = flags(rows_a, args)
    thr_b, fb = flags(rows_b, args)
    common = [k for k in judged if k in fa and k in fb]
    for i, name in ((0, "aggressive"), (1, "timid")):
        la, lb = [fa[k][i] for k in common], [fb[k][i] for k in common]
        out[f"{name}_share_a"], out[f"{name}_share_b"] = float(np.mean(la)), float(np.mean(lb))
        out[f"{name}_agreement"] = float(np.mean(np.equal(la, lb)))
        out[f"{name}_kappa"] = kappa(la, lb)
    out["thresholds_a"], out["thresholds_b"] = thr_a, thr_b

    scenes = sorted({k[0] for k in common})
    share_a = [np.mean([fa[k][0] for k in common if k[0] == s]) for s in scenes]
    share_b = [np.mean([fb[k][0] for k in common if k[0] == s]) for s in scenes]
    count_a = [sum(fa[k][0] for k in common if k[0] == s) for s in scenes]
    count_b = [sum(fb[k][0] for k in common if k[0] == s) for s in scenes]
    verdict_a = [c >= args.min_windows for c in count_a]
    verdict_b = [c >= args.min_windows for c in count_b]
    out.update(scenes=len(scenes), scene_spearman=spearman(share_a, share_b),
               scene_agreement=float(np.mean(np.equal(verdict_a, verdict_b))) if scenes else float("nan"),
               scene_kappa=kappa(verdict_a, verdict_b) if scenes else float("nan"),
               scenes_aggressive_a=int(sum(verdict_a)), scenes_aggressive_b=int(sum(verdict_b)))
    out["pairs"] = [(a[k]["safety"], b[k]["safety"], a[k]["courtesy"], b[k]["courtesy"]) for k in both]
    return out


def read_table(path):
    with open(path) as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def compare_policy_tables(path_a, path_b, columns=("crash_rate", "ego_fault_share", "aggressive", "timid",
                                                   "aggressive_x_ref", "timid_x_ref", "safety_median")):
    """Per column: Spearman over the policies in both tables, and for the
    "x ref" columns the agreement of "above the reference"."""
    ta, tb = read_table(path_a), read_table(path_b)
    runs = [r for r in ta if r in tb]
    rows = []
    for col in columns:
        va = np.array([float(ta[r].get(col) or "nan") for r in runs])
        vb = np.array([float(tb[r].get(col) or "nan") for r in runs])
        ok = np.isfinite(va) & np.isfinite(vb)
        row = {"column": col, "policies": int(ok.sum()), "spearman": spearman(va[ok], vb[ok])}
        if col.endswith("_x_ref"):
            row["above_ref_agreement"] = float(np.mean((va[ok] > 1) == (vb[ok] > 1))) if ok.any() else float("nan")
        rows.append(row)
    return runs, rows


def report(out, labels, policy=None) -> str:
    a, b = labels
    f = lambda v: "–" if v is None or not np.isfinite(v) else f"{v:.3f}"  # noqa: E731
    lines = [f"# {a} vs {b}", "",
             f"Windows: {out['matched']} matched ({out['judged']} judged), {out['only_a']} only in {a}, "
             f"{out['only_b']} only in {b}.", "",
             "| | all windows | judged windows |", "|---|---|---|",
             f"| Spearman β_s | {f(out['spearman_safety'])} | {f(out['spearman_safety_judged'])} |",
             f"| Spearman β_c | {f(out['spearman_courtesy'])} | {f(out['spearman_courtesy_judged'])} |", "",
             f"Thresholds, each run on itself (aggressive β_s, β_c; timid β_s): "
             f"{a} {', '.join(f'{t:.3f}' for t in out['thresholds_a'])}; "
             f"{b} {', '.join(f'{t:.3f}' for t in out['thresholds_b'])}", "",
             f"| flag | share {a} | share {b} | agreement | Cohen's κ |", "|---|---|---|---|---|"]
    for name in ("aggressive", "timid"):
        lines.append(f"| {name} | {f(out[f'{name}_share_a'])} | {f(out[f'{name}_share_b'])} | "
                     f"{f(out[f'{name}_agreement'])} | {f(out[f'{name}_kappa'])} |")
    lines += ["", f"Scenes ({out['scenes']}): Spearman of the aggressive share {f(out['scene_spearman'])}; "
                  f"verdicts agree in {f(out['scene_agreement'])} (κ {f(out['scene_kappa'])}); aggressive scenes "
                  f"{out['scenes_aggressive_a']} ({a}) vs {out['scenes_aggressive_b']} ({b})."]
    if policy is not None:
        runs, rows = policy
        lines += ["", f"Policy tables ({len(runs)} runs in both):", "",
                  "| column | policies | Spearman | agreement on x ref > 1 |", "|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['column']} | {r['policies']} | {f(r['spearman'])} | "
                         f"{f(r.get('above_ref_agreement', float('nan')))} |")
    return "\n".join(lines) + "\n"


def plot(pairs, labels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = np.asarray(pairs, dtype=float).reshape(-1, 4)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=120)
    for ax, (x, y), name in ((axes[0], (0, 1), r"$\beta_s$ [m]"), (axes[1], (2, 3), r"$\beta_c$ [nats]")):
        ax.scatter(p[:, x], p[:, y], s=6, alpha=0.4, color="#1f77b4")
        if len(p):
            lo, hi = np.nanmin(p[:, [x, y]]), np.nanmax(p[:, [x, y]])
            ax.plot([lo, hi], [lo, hi], color="#7f7f7f", lw=0.8, ls="--")
        ax.set_xlabel(f"{name}, {labels[0]}")
        ax.set_ylabel(f"{name}, {labels[1]}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    labels = args.labels or [label_of(r) for r in args.runs]
    if labels[0] == labels[1]:
        labels = [f"{labels[0]} (A)", f"{labels[1]} (B)"]
    out = compare_windows(read_windows(args.runs[0]), read_windows(args.runs[1]), args)
    policy = compare_policy_tables(*args.policy_tables) if args.policy_tables else None
    text = report(out, labels, policy)
    dest = Path(args.out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model_comparison.md").write_text(text, encoding="utf-8")
    plot(out["pairs"], labels, dest / "model_comparison.png")
    print(text)
    print(f"wrote {dest / 'model_comparison.md'}, {dest / 'model_comparison.png'}")
    return out, policy


if __name__ == "__main__":
    main()
