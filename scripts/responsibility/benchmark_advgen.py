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
    python -m scripts.responsibility.benchmark_advgen --summarize part_*.csv
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
                                   get_objects=get_objects, get_object=get_objects)
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


def summarize(rows):
    print(f"\n{len({r['scene'] for r in rows})} scenes, open-loop against the logged ego")
    print(f"{'rule':>18} {'collision':>10} {'mean beta':>10} {'median beta':>12} {'mean avoid':>11} "
          f"{'unavoidable hits':>17}")
    for name in dict.fromkeys(r["rule"] for r in rows):
        rs = [r for r in rows if r["rule"] == name]
        b = np.array([float(r["beta"]) for r in rs])
        hit = np.array([int(r["collision"]) for r in rs])
        avoid = np.array([float(r["avoid"]) if r.get("avoid") not in (None, "") else np.nan for r in rs])
        unavoidable = np.mean(hit.astype(bool) & (avoid < UNAVOIDABLE)) if np.isfinite(avoid).any() else np.nan
        print(f"{name:>18} {100 * hit.mean():9.0f}% {b.mean():+10.2f} {np.median(b):+12.2f} "
              f"{np.nanmean(avoid) if np.isfinite(avoid).any() else np.nan:11.2f} {100 * unavoidable:16.0f}%")
    logged = np.array([float(v) for _, v in sorted({(r["scene"], r["logged_adversary_beta"]) for r in rows})])
    print(f"logged adversary beta: median {np.median(logged):+.2f}, q75 {np.quantile(logged, 0.75):+.2f}, "
          f"q90 {np.quantile(logged, 0.9):+.2f} m  (tau can be calibrated on these)")
    avoid = np.array([float(v) for _, v in sorted({(r["scene"], r.get("logged_adversary_avoid", ""))
                                                    for r in rows}) if v not in (None, "")])
    if avoid.size:
        print(f"logged adversary ego avoidability: median {np.median(avoid):.2f}, q10 {np.quantile(avoid, 0.1):.2f}, "
              f"min {avoid.min():.2f}  (rho can be calibrated on these; 'unavoidable' = avoid < {UNAVOIDABLE:g})")


def main():
    args = parse_args()
    if args.summarize:
        rows = []
        for path in args.summarize:
            with open(path) as f:
                rows.extend(csv.DictReader(f))
        summarize(rows)
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

    summarize(rows)
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
