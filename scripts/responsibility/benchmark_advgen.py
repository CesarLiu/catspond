"""Compares CAT's adversary selection with the responsibility-constrained ones
on CAT's scenes, open-loop against the logged ego trajectory (no MetaDrive):
for every scene and rule, whether the chosen adversary trajectory is predicted
to collide with the ego, how responsible the adversary is for it (safety
responsibility toward the ego over 8 s), how avoidable it is for the ego
(the share of the ego's DenseTNT motion set that escapes it), and how close
it gets. The logged adversary's own responsibility and avoidability,
computed the same way, are the references that tau and rho can be
calibrated against.

It also checks that the "cat" rule reproduces AdvGenerator.generate: CAT's
own code is run on the same candidates and must choose the same trajectory;
and it runs the generator the way cat_advgen.py / cat_RLtrain.py call it
(before_episode, generate, adv_traj) against a stand-in for MetaDrive's env,
checking that adv_traj has CAT's format.

Example (from the repository root):
    python -m scripts.responsibility.benchmark_advgen --n 50 --thresholds 0.5 1 2 --avoid 0.1 0.3 0.5 \\
        --out logs/responsibility/advgen_benchmark.csv

Parts run in parallel (--first/--n, one --out each) are combined with
    python -m scripts.responsibility.benchmark_advgen --summarize part_*.csv --plot tradeoff.png

The summary ends with the fair (tau, rho) settings recommended for RL: those
keeping a predicted collision rate of --min-collision (0.3), with the fewest
unavoidable collisions, then the least responsible adversary. --plot draws
the trade-off: collision rate against the chosen adversaries' mean beta and
against the share of unavoidable collisions, one curve over tau per rule.
"""

import argparse
import copy
import csv
import pickle
import sys
import types
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.adversarial import (  # noqa: E402
    RULES,
    ResponsibleAdvGenerator,
    adversary_responsibility,
    ego_avoidability,
    select,
)
from responsibility.scene import Scene, scene_files  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                   help="m; the constrained and fair rules are evaluated at each.")
    p.add_argument("--avoid", type=float, nargs="+", default=[0.3],
                   help="The fair rule is evaluated at each of these minimum ego avoidabilities (rho).")
    p.add_argument("--penalty", type=float, default=1.0, help="m; the penalized rule's scale.")
    p.add_argument("--samples", type=int, default=40)
    p.add_argument("--horizon", type=int, default=80)
    p.add_argument("--out", default=None, help="CSV with one row per scene and rule.")
    p.add_argument("--no-cat-check", action="store_true")
    p.add_argument("--summarize", nargs="+", default=None, metavar="CSV",
                   help="Only print the summary of existing --out CSVs (e.g. parts run in parallel).")
    p.add_argument("--plot", default=None, metavar="PNG",
                   help="Also plot the trade-off: collision rate vs adversary beta and vs unavoidable collisions.")
    p.add_argument("--min-collision", type=float, default=MIN_COLLISION,
                   help="Fair settings are recommended for RL only if they keep this predicted collision rate.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def cat_storage(description, scene: Scene):
    """AdvGenerator's per-scene storage, built without a MetaDrive env: sizes
    from the log at the current step, the logged ego route as the ego's
    trajectory (as CAT starts before any policy rollout)."""
    from advgen.adv_generator import AdvGenerator

    fake = types.SimpleNamespace(env=types.SimpleNamespace(
        current_seed=0,
        engine=types.SimpleNamespace(data_manager=types.SimpleNamespace(_scenario={0: copy.deepcopy(description)}))))
    feat, adv_agent, route, adv_past = AdvGenerator._parse(fake)
    adv, ego = scene.index(adv_agent), scene.sdc

    def size(i):
        length, width = scene.shape_at(i, 10)
        return {"w": float(width), "l": float(length)}

    return {"traffic_motion_feat": feat, "adv_agent": adv_agent, "adv_past": adv_past,
            "adv_info": size(adv), "ego_info": size(ego),
            "AV_trajs": deque([route], maxlen=1), "AV_probs": deque([1.0], maxlen=1),
            "AV_trajs_eval": deque([route], maxlen=1)}


def cat_choice(storage, candidates, log_scores):
    """AdvGenerator.generate on the given candidates (its model call replaced),
    returning the adversary future it picks."""
    from advgen.adv_generator import AdvGenerator

    cat = object.__new__(AdvGenerator)
    cat.args = types.SimpleNamespace(AV_traj_num=1, other_params={})
    cat.env = types.SimpleNamespace(current_seed=0)
    cat.storage = {0: storage}
    cat.model = lambda batch, device: (np.stack([candidates, candidates]), np.stack([log_scores, log_scores]), None)
    import advgen.adv_generator as module

    original = module.process_data
    module.process_data = lambda feat, args: [None]  # the replaced model does not need inputs
    try:
        cat.generate()
    finally:
        module.process_data = original
    return np.array(cat.adv_traj)[11:, :2]


def fake_env(description, seed, scene):
    """The parts of MetaDrive's WaymoEnv that AdvGenerator touches."""
    ego_id = str(description["metadata"]["sdc_id"])

    def obj(i):
        length, width = scene.shape_at(i, 10)
        return types.SimpleNamespace(top_down_width=float(width), top_down_length=float(length),
                                     position=scene.position[i, 10, :2], velocity=scene.velocity[i, 10],
                                     heading_theta=float(scene.heading[i, 10]))

    def get_objects(names):
        return {n: obj(scene.sdc if n == "default_agent" else scene.index(n)) for n in names}

    engine = types.SimpleNamespace(data_manager=types.SimpleNamespace(_scenario={seed: copy.deepcopy(description)}),
                                   get_objects=get_objects, get_object=get_objects,
                                   traffic_manager=types.SimpleNamespace(set_adv_info=lambda name, plan: None))
    del ego_id
    return types.SimpleNamespace(current_seed=seed, engine=engine)


def check_drop_in(gen, description, scene, seed):
    gen.before_episode(fake_env(description, seed, scene))
    gen.generate()
    traj = np.array(gen.adv_traj)
    past = scene.position[scene.index(gen.adv_agent), :11, :2]
    ok = traj.shape == (91, 5) and np.allclose(traj[:11, :2], past) and len(gen.selections) >= 1
    print(f"drop-in generate(): adv_traj {traj.shape} (x, y, vx, vy, yaw), past kept: "
          f"{np.allclose(traj[:11, :2], past)}, rule {gen.selections[-1]['rule']} -> "
          f"candidate {gen.selections[-1]['chosen']} ({gen.selections[-1]['why']})")
    return ok


UNAVOIDABLE = 0.1  # ego avoidability below which a collision counts as unavoidable in the summary
MIN_COLLISION = 0.3  # predicted collision rate a fair setting must keep to be recommended for RL


def parse_rule(name):
    """(family, tau, rho) of a rule name: cat, penalized, constrained@1, fair@1,0.3."""
    family, _, params = name.partition("@")
    values = [float(v) for v in params.split(",")] if params else []
    return family, (values[0] if values else None), (values[1] if len(values) > 1 else None)


def rule_stats(rows):
    """One summary per rule, in the order the rules first appear."""
    out = []
    for name in dict.fromkeys(r["rule"] for r in rows):
        rs = [r for r in rows if r["rule"] == name]
        b = np.array([float(r["beta"]) for r in rs])
        hit = np.array([int(r["collision"]) for r in rs]).astype(bool)
        avoid = np.array([float(r["avoid"]) if r.get("avoid") not in (None, "") else np.nan for r in rs])
        has_avoid = bool(np.isfinite(avoid).any())
        family, tau, rho = parse_rule(name)
        out.append({"rule": name, "family": family, "tau": tau, "rho": rho, "scenes": len(rs),
                    "collision": float(hit.mean()), "beta_mean": float(b.mean()), "beta_median": float(np.median(b)),
                    "avoid_mean": float(np.nanmean(avoid)) if has_avoid else np.nan,
                    "unavoidable": float(np.mean(hit & (avoid < UNAVOIDABLE))) if has_avoid else np.nan})
    return out


def logged_references(rows):
    """The logged adversaries' beta and ego avoidability, one per scene."""
    beta = np.array([float(v) for _, v in sorted({(r["scene"], r["logged_adversary_beta"]) for r in rows})])
    avoid = np.array([float(v) for _, v in sorted({(r["scene"], r.get("logged_adversary_avoid", ""))
                                                    for r in rows}) if v not in (None, "")])
    return beta, avoid


def recommend(stats, min_collision=MIN_COLLISION, top=2):
    """The fair (tau, rho) settings worth training with: predicted collision
    at least ``min_collision`` (enough adversarial signal), then the fewest
    unavoidable collisions, then the least responsible adversary, then the
    most collisions."""
    fair = [s for s in stats if s["family"] == "fair" and s["collision"] >= min_collision]
    fair.sort(key=lambda s: (np.nan_to_num(s["unavoidable"], nan=1.0), s["beta_mean"], -s["collision"]))
    return fair[:top]


def summarize(rows, min_collision=MIN_COLLISION):
    stats = rule_stats(rows)
    print(f"\n{len({r['scene'] for r in rows})} scenes, open-loop against the logged ego")
    print(f"{'rule':>18} {'collision':>10} {'mean beta':>10} {'median beta':>12} {'mean avoid':>11} "
          f"{'unavoidable hits':>17}")
    for st in stats:
        print(f"{st['rule']:>18} {100 * st['collision']:9.0f}% {st['beta_mean']:+10.2f} {st['beta_median']:+12.2f} "
              f"{st['avoid_mean']:11.2f} {100 * st['unavoidable']:16.0f}%")
    logged, avoid = logged_references(rows)
    print(f"logged adversary beta: median {np.median(logged):+.2f}, q75 {np.quantile(logged, 0.75):+.2f}, "
          f"q90 {np.quantile(logged, 0.9):+.2f} m  (tau can be calibrated on these)")
    if avoid.size:
        print(f"logged adversary ego avoidability: median {np.median(avoid):.2f}, q10 {np.quantile(avoid, 0.1):.2f}, "
              f"min {avoid.min():.2f}  (rho can be calibrated on these; 'unavoidable' = avoid < {UNAVOIDABLE:g})")
    best = recommend(stats, min_collision)
    if best:
        print(f"\nfair settings for RL (predicted collision >= {100 * min_collision:.0f}%, fewest unavoidable "
              f"collisions, then least responsible adversary):")
        for st in best:
            print(f"  {st['rule']}: collision {100 * st['collision']:.0f}%, unavoidable {100 * st['unavoidable']:.0f}%, "
                  f"mean beta {st['beta_mean']:+.2f} m, mean avoid {st['avoid_mean']:.2f}"
                  f"   -> TAU={st['tau']:g} RHO={st['rho']:g}")
    elif any(st["family"] == "fair" for st in stats):
        print(f"\nno fair setting keeps a predicted collision rate of {100 * min_collision:.0f}%; "
              f"lower --min-collision or try larger tau / smaller rho")
    return stats


def plot_tradeoff(rows, path):
    """Collision rate against the chosen adversaries' responsibility and
    against the share of unavoidable collisions: one curve over tau for
    constrained and for fair at each rho, CAT's and the penalized rule as
    points, the logged adversaries' beta q90 as reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = rule_stats(rows)
    logged, _ = logged_references(rows)
    q90 = float(np.quantile(logged, 0.9))
    curves = {}
    for st in stats:
        if st["tau"] is not None:
            key = "constrained" if st["family"] == "constrained" else f"fair, rho {st['rho']:g}"
            curves.setdefault(key, []).append(st)
    cmap = plt.get_cmap("viridis", max(len(curves), 2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=120)
    panels = ((axes[0], "beta_mean", 1.0, "mean adversary safety responsibility beta [m]"),
              (axes[1], "unavoidable", 100.0, f"unavoidable collisions (ego avoidability < {UNAVOIDABLE:g}) [%]"))
    for ax, xkey, scale, xlabel in panels:
        for k, (key, sts) in enumerate(curves.items()):
            sts = sorted(sts, key=lambda s: s["tau"])
            x = [scale * s[xkey] for s in sts]
            y = [100 * s["collision"] for s in sts]
            style = dict(color="#d62728", ls="--") if key == "constrained" else dict(color=cmap(k))
            ax.plot(x, y, marker="o", label=key, **style)
            for xi, yi, s in zip(x, y, sts):
                ax.annotate(f"{s['tau']:g}", (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
        for st in stats:
            if st["tau"] is None and np.isfinite(st[xkey]):
                x, y = scale * st[xkey], 100 * st["collision"]
                ax.scatter([x], [y], marker="s", s=40, color="k" if st["family"] == "cat" else "grey", zorder=3)
                ax.annotate(st["rule"], (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8)
        if xkey == "beta_mean":
            ax.axvline(q90, color="k", lw=0.8, ls=":")
            ax.text(q90, 2, " logged q90", fontsize=7)
        ax.axhline(100 * MIN_COLLISION, color="grey", lw=0.6, ls=":")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("predicted collision [%]")
        ax.set_ylim(0, 105)
        ax.margins(x=0.1)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, title="labels: tau [m]", title_fontsize=7)
    axes[0].set_title("attack success vs adversary responsibility", fontsize=9)
    axes[1].set_title("attack success vs unavoidable collisions", fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    args = parse_args()
    if args.summarize:
        rows = []
        for path in args.summarize:
            with open(path) as f:
                rows.extend(csv.DictReader(f))
        summarize(rows, args.min_collision)
        if args.plot:
            plot_tradeoff(rows, args.plot)
        return
    cat_parser = argparse.ArgumentParser()  # as cat_advgen.py builds it
    cat_parser.add_argument("--OV_traj_num", type=int, default=32)
    cat_parser.add_argument("--AV_traj_num", type=int, default=1)
    gen = ResponsibleAdvGenerator(cat_parser, argv=[
        "--adv_selection", "constrained", "--resp_samples", str(args.samples),
        "--resp_horizon", str(args.horizon), "--resp_device", args.device])
    rules = ([("cat", None, None), ("penalized", None, None)] + [("constrained", t, None) for t in args.thresholds]
             + [("fair", t, r) for t in args.thresholds for r in args.avoid])
    rows, cat_agree = [], []
    drop_in_ok = True
    for path in scene_files(args.scenes)[args.first: args.first + args.n]:
        with open(path, "rb") as f:
            description = pickle.load(f)
        scene = Scene.from_description(copy.deepcopy(description))
        st = cat_storage(description, scene)
        ego_route = list(st["AV_trajs"])
        out = gen.choose(scene, ego_route, [1.0], st["adv_info"], st["ego_info"], seed=int(path.stem),
                         rules=["fair"])
        score, min_dist, beta, avoid = out["score"], out["min_dist"], out["beta"], out["avoid"]
        adv = scene.index(st["adv_agent"])
        logged = scene.position[adv, 11:91, :2]
        logged_beta = float(adversary_responsibility(out["samples"], logged[None], ego_route, [1.0], gen.cfg,
                                                     args.horizon)[0])
        logged_avoid = ""
        if out["ego_samples"] is not None:
            ego = scene.sdc
            logged_avoid = round(float(ego_avoidability(
                out["ego_samples"], logged[None], (scene.position[ego, 10, :2], float(scene.heading[ego, 10])),
                (scene.position[adv, 10, :2], float(scene.heading[adv, 10])), st["ego_info"], st["adv_info"],
                horizon=args.horizon)[0][0]), 4)
        if not args.no_cat_check:
            cat_future = cat_choice(st, out["candidates"], out["log_scores"])
            ours = out["candidates"][select("cat", score, min_dist, beta)[0]]
            cat_agree.append(bool(np.allclose(cat_future, ours, atol=1e-6)))
            if len(cat_agree) == 1:
                drop_in_ok = check_drop_in(gen, description, scene, int(path.stem))
        line = [f"{path.stem:>4}", f"logged adversary beta {logged_beta:+.2f} avoid {logged_avoid}"]
        for rule, t, r in rules:
            j, why = select(rule, score, min_dist, beta, threshold=t if t is not None else 1.0, penalty=args.penalty,
                            avoid=avoid, min_avoid=r if r is not None else 0.3)
            name = rule if t is None else f"{rule}@{t:g}" if r is None else f"{rule}@{t:g},{r:g}"
            rows.append({"scene": path.stem, "rule": name, "chosen": j, "why": why,
                         "collision": int(score[j] > 0), "score": round(float(score[j]), 5),
                         "beta": round(float(beta[j]), 4), "avoid": round(float(avoid[j]), 4),
                         "min_dist": int(min_dist[j]),
                         "logged_adversary_beta": round(logged_beta, 4), "logged_adversary_avoid": logged_avoid,
                         "candidates_colliding": int((score > 0).sum()),
                         "candidates_avoidable_colliding": int(((score > 0) & (avoid >= 0.3)).sum())})
            line.append(f"{name}: {'HIT' if score[j] > 0 else 'miss'} beta {beta[j]:+.2f} avoid {avoid[j]:.2f}")
        print("  ".join(line), flush=True)

    summarize(rows, args.min_collision)
    if args.plot:
        plot_tradeoff(rows, args.plot)
    if cat_agree:
        print(f"cat rule reproduces AdvGenerator.generate: {sum(cat_agree)}/{len(cat_agree)} scenes")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.out}")
    if (cat_agree and not all(cat_agree)) or not drop_in_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
