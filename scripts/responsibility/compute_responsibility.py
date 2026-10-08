"""Safety and courtesy responsibility of the self-driving car (or any agent)
over CAT's scenes, with DenseTNT (default) or UniTraj's MTR (--model mtr
--checkpoint ...) as the motion model.

For every scene and every context step k (10, 10 + stride, ..., up to 2 s
before the end of the log) it writes one row to OUT/windows.csv:

  safety     beta_s: > 0 when the agent kept less distance to a neighbour than
             most of its own alternatives would have (it gave up safety margin)
  courtesy   beta_c: how much its presence changed a vehicle neighbour's
             intended goal (KL, nats)
  safety_against / courtesy_toward   the neighbour each maximum came from

plus the full per-neighbour observations to OUT/obs/<scene>.pkl. With
--save-records it also writes OUT/records/<scene>.pkl: the scene, the agent's
motion set and goal distribution and every neighbour's goal distributions with
and without it at every step, and the values -- enough to inspect and
visualise the run offline (visualize_responsibility.py --record) without
DenseTNT; values are identical with or without records. Re-running
skips finished scenes; OUT/config.json pins the settings. With
--num-shards N --shard-index i, N processes split the scenes round-robin into
the same OUT, each writing windows.shard-<i>-of-<N>.csv (run_h200.sh does this).

With --rollouts DIR it measures a driving policy instead of the log: every
rollout in DIR (collect_rollouts.py) is played back into its scene -- the
simulated ego (and adversary) in place of the logged ones, objects the
simulator did not spawn removed (responsibility/rollouts.py) -- and the ego
is queried there. Windows stop where the episode did: at a collision, or a
full metric horizon before any other end. Every collision with another road
user is also attributed (responsibility/blame.py: the ego's and the other's
safety responsibility toward each other in the 2 s before it; "rule": the
rear-end rule's verdict; "rss": Responsibility-Sensitive Safety's,
responsibility/rss.py -- the baselines it is compared with) and written to
OUT/crashes.csv (crashes.shard-<i>-of-<N>.csv).

--valid-counterfactuals (or the separate --lane-route, --route-tolerance,
--drivable-half-width, --kinematics, --collision-filter) restricts beta_s to
the valid part of the agent's motion set: its own route lanes (lane changes
and every branch after the log included), on the road, kinematically
feasible, through no third agent (responsibility/motion_filter.py,
responsibility/lanes.py); the per-neighbour observations then also record
how much of the set was kept. It also takes beta_c over the neighbour's
reachable goals only (--courtesy-valid-goals). On a map without lane topology
(perception), --intent tells the same intent from the trajectories instead
(responsibility/intent.py), and --drivable-edges judges the drivable area
by road edges instead of lane centrelines (responsibility/edges.py). With DenseTNT, --motion-set weighted uses its whole goal
grid, probability-weighted, instead of 40 samples; --motion-set topk uses its --n-samples most probable
goals, probability-weighted; --motion-set nms uses --n-samples goals spread over the distribution by CAT's
goal non-maximum suppression, each weighted by the probability of the goals nearest to it.

Example (from the repository root):
    python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 \\
        --out-dir logs/responsibility/sdc --n 50
    python -m scripts.responsibility.compute_responsibility --scenes raw_scenes_500 \\
        --rollouts rollouts/td3_cat/none --out-dir logs/responsibility/policies/td3_cat/none
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.blame import rollout_blame  # noqa: E402
from responsibility.models import add_model_arguments, load_model, model_settings  # noqa: E402
from responsibility.motion_filter import MotionFilterConfig  # noqa: E402
from responsibility.interaction import InteractionConfig  # noqa: E402
from responsibility.metrics import MOTION_SETS, ResponsibilityConfig, scene_responsibility  # noqa: E402
from responsibility.records import run_scene, save_record  # noqa: E402
from responsibility.rollouts import (  # noqa: E402
    last_window_step,
    load_rollout,
    outcome,
    rollout_files,
    scene_from_rollout,
)
from responsibility.rss import rollout_rss  # noqa: E402
from responsibility.scene import Scene, scene_files  # noqa: E402

ROW_FIELDS = ["scene", "scenario_id", "agent_id", "step", "time", "speed", "safety", "courtesy",
              "n_neighbours", "safety_against", "courtesy_toward"]
CRASH_FIELDS = ["scene", "policy", "adv_mode", "crash_step", "window", "other_id", "other_type", "adversary",
                "beta_ego", "beta_other", "share", "verdict", "rule", "rss", "rss_case"]


def _m(value):
    return "n/a" if value is None else f"{value:+.2f} m"


def blame_row(model, scene, agent, rollout, cfg):
    """The attribution of the rollout's vehicle collision (verdict "unknown"
    when there is no window before it with both sides present)."""
    step = rollout["end"]["step"]
    adversary = rollout["adversary"]["track_id"] if rollout["adversary"] is not None else ""
    row = {"scene": rollout["scene_file"], "policy": rollout["policy"], "adv_mode": rollout["adv_mode"],
           "crash_step": step, "adversary": adversary, "verdict": "unknown", "rule": "n/a", "rss": "n/a",
           "rss_case": ""}
    blame = rollout_blame(model, scene, rollout, cfg) if agent == scene.sdc else None
    if blame is not None:
        row.update({k: v for k, v in blame.as_row().items() if k in CRASH_FIELDS})
    rss = rollout_rss(scene, rollout) if agent == scene.sdc else None  # also where the model cannot attribute
    if rss is not None:
        row.update(rss.as_row())
    row["adversary"] = int(bool(adversary) and row.get("other_id") == adversary)
    return row


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--rollouts", default=None,
                   help="Directory of a policy's rollouts (collect_rollouts.py): measure them instead of the log.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--first", type=int, default=0, help="Index of the first scene (or rollout) file.")
    p.add_argument("--n", type=int, default=None, help="Number of scenes (default: all).")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--agent", default="sdc", help="sdc, adv (the other object of interest) or a track id.")
    d = ResponsibilityConfig()
    p.add_argument("--n-samples", type=int, default=d.n_safety_samples)
    p.add_argument("--cvar-alpha", type=float, default=d.cvar_alpha)
    p.add_argument("--motion-set", default=d.motion_set, choices=MOTION_SETS,
                   help="weighted: the model's whole motion set, probability-weighted "
                        "(densetnt: its goal grid up to 0.999 of the mass; mtr: its 64 intentions); "
                        "topk: its --n-samples most probable members, probability-weighted; "
                        "nms: --n-samples members spread by CAT's goal NMS, each weighted by the "
                        "probability nearest to it (responsibility/modes.py).")
    f = MotionFilterConfig()
    g = p.add_argument_group("valid counterfactuals for beta_s (responsibility/motion_filter.py)")
    g.add_argument("--lane-route", action="store_true",
                   help="Same intent by lanes: goals (or, without goals, end points) on the agent's route lanes, "
                        "their side neighbours and what follows the log.")
    g.add_argument("--courtesy-valid-goals", action="store_true",
                   help="beta_c over the neighbour's goals on lanes it can reach only.")
    g.add_argument("--intent", action="store_true",
                   help="Same intent without lane topology (responsibility/intent.py), for perceived maps: "
                        "heading within --intent-heading of the logged path where the alternative is, "
                        "within --intent-lateral of that path, and with --intent-edges across no road edge.")
    g.add_argument("--intent-heading", type=float, default=f.intent_heading, help="deg.")
    g.add_argument("--intent-lateral", type=float, default=f.intent_lateral, help="m; <= 0 disables.")
    g.add_argument("--intent-edges", action="store_true", help="--intent also uses the map's road edges.")
    g.add_argument("--route-tolerance", type=float, default=f.route_tolerance,
                   help="m: keep only alternatives within this of the agent's logged route (same intent).")
    g.add_argument("--drivable-half-width", type=float, default=f.drivable_half_width,
                   help="m: keep only alternatives within this of a vehicle lane centreline.")
    g.add_argument("--drivable-edges", action="store_true",
                   help="Keep only alternatives whose path crosses no road edge (one the logged path crosses "
                        "does not count); for a map built by perception, instead of --drivable-half-width.")
    g.add_argument("--kinematics", action="store_true", help="Keep only kinematically feasible alternatives.")
    g.add_argument("--collision-filter", action="store_true",
                   help="Drop alternatives that drive through a third agent's logged future.")
    g.add_argument("--valid-counterfactuals", action="store_true",
                   help="--lane-route, drivable half-width 3 m, --kinematics, --collision-filter and "
                        "--courtesy-valid-goals (an explicit --drivable-half-width overrides 3 m).")
    p.add_argument("--d-sat", type=float, default=d.d_sat, help="<= 0 disables saturation.")
    p.add_argument("--horizon", type=int, default=d.metric_horizon, help="10 Hz steps scored.")
    p.add_argument("--stride", type=int, default=d.window_stride, help="Steps between context steps.")
    p.add_argument("--no-courtesy", action="store_true")
    p.add_argument("--save-records", action="store_true",
                   help="Also write OUT/records/<scene>.pkl for offline inspection and visualisation.")
    p.add_argument("--top-mass", type=float, default=0.99,
                   help="Goal distributions in records keep the most probable goals covering this mass.")
    i = InteractionConfig()
    p.add_argument("--max-neighbors", type=int, default=i.max_neighbors)
    p.add_argument("--gap-threshold", type=float, default=i.gap_threshold)
    p.add_argument("--pet-threshold", type=float, default=i.pet_threshold)
    p.add_argument("--ttc-threshold", type=float, default=i.ttc_threshold)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_model_arguments(p)
    return p.parse_args()


# settings added after runs were made, with the value those runs used
LATER_SETTINGS = {"model": {"name": "densetnt"}}
LATER_RESPONSIBILITY = {"motion_set": "sampled", "filter": asdict(MotionFilterConfig()), "courtesy_valid_goals": False}


def same_settings(stored, settings) -> bool:
    """Whether a run's stored config.json matches the current settings,
    reading settings it predates as their values at the time."""
    stored = dict(stored)
    for key, value in LATER_SETTINGS.items():
        stored.setdefault(key, value)
    stored["responsibility"] = {**LATER_RESPONSIBILITY, **stored.get("responsibility", {})}
    # filters added later were off in the runs that predate them
    stored["responsibility"]["filter"] = {**LATER_RESPONSIBILITY["filter"], **stored["responsibility"]["filter"]}
    return stored == json.loads(json.dumps(settings))


def pick_agent(scene: Scene, which: str) -> int:
    if which == "sdc":
        return scene.sdc
    if which == "adv":
        others = [i for i in scene.objects_of_interest if i != scene.sdc]
        if not others:
            raise ValueError("no second object of interest")
        return others[0]
    return scene.index(which)


def main():
    args = parse_args()
    valid = args.valid_counterfactuals
    motion_filter = MotionFilterConfig(
        lane_route=args.lane_route or valid, route_tolerance=args.route_tolerance,
        intent=args.intent, intent_heading=args.intent_heading,
        intent_lateral=args.intent_lateral if args.intent_lateral > 0 else None, intent_edges=args.intent_edges,
        drivable_half_width=(args.drivable_half_width if args.drivable_half_width is not None
                             else (3.0 if valid else None)),
        drivable_edges=args.drivable_edges,
        kinematics=args.kinematics or valid, collision=args.collision_filter or valid)
    cfg = ResponsibilityConfig(
        filter=motion_filter, courtesy_valid_goals=args.courtesy_valid_goals or valid,
        n_safety_samples=args.n_samples, cvar_alpha=args.cvar_alpha, motion_set=args.motion_set,
        d_sat=args.d_sat if args.d_sat > 0 else None, metric_horizon=args.horizon,
        window_stride=args.stride, courtesy=not args.no_courtesy, seed=args.seed,
        interaction=InteractionConfig(max_neighbors=args.max_neighbors, gap_threshold=args.gap_threshold,
                                      pet_threshold=args.pet_threshold, ttc_threshold=args.ttc_threshold),
    )
    out = Path(args.out_dir)
    (out / "obs").mkdir(parents=True, exist_ok=True)
    settings = {"responsibility": asdict(cfg), "agent": args.agent, "scenes": str(Path(args.scenes).resolve()),
                "model": model_settings(args)}
    if args.rollouts:
        settings["rollouts"] = str(Path(args.rollouts).resolve())
    config_path = out / "config.json"
    if config_path.exists():
        if not same_settings(json.loads(config_path.read_text()), settings):
            raise SystemExit(f"{out} holds results computed with other settings; use a new --out-dir")
    else:  # shards may race here; they write the same content
        tmp = config_path.with_name(f"config.json.tmp{os.getpid()}")
        tmp.write_text(json.dumps(settings, indent=2))
        os.replace(tmp, config_path)

    files = (rollout_files(args.rollouts) if args.rollouts else scene_files(args.scenes))[args.first:]
    if args.n is not None:
        files = files[: args.n]
    files = files[args.shard_index :: args.num_shards]
    # with --save-records, scenes finished without a record are redone too
    # (their repeated CSV rows are dropped when read: responsibility/results.py)
    todo = [f for f in files if not (out / "obs" / f"{f.stem}.pkl").exists()
            or (args.save_records and not (out / "records" / f"{f.stem}.pkl").exists())]
    print(f"{len(files)} scenes, {len(files) - len(todo)} done, {len(todo)} to go", flush=True)
    if not todo:
        return

    model = load_model(args, args.device)
    suffix = ".csv" if args.num_shards == 1 else f".shard-{args.shard_index}-of-{args.num_shards}.csv"
    rows_path = out / f"windows{suffix}"
    crashes_path = out / f"crashes{suffix}"
    new_file = not rows_path.exists()
    with open(rows_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if new_file:
            writer.writeheader()
        for n, path in enumerate(todo, 1):
            t0 = time.time()
            rollout, last_step = None, None
            if args.rollouts:
                rollout = load_rollout(path)
                scene = scene_from_rollout(Scene.load(Path(args.scenes) / f"{rollout['scene_file']}.pkl"), rollout)
                last_step = last_window_step(rollout, cfg.metric_horizon)
            else:
                scene = Scene.load(path)
            try:
                agent = pick_agent(scene, args.agent)
            except ValueError as e:
                print(f"{path.name}: skipped ({e})", flush=True)
                observations = []
            else:
                if args.save_records:
                    observations, record = run_scene(model, scene, agent, cfg, args.top_mass, last_step)
                    record["scene_file"] = path.stem
                    if rollout is not None:
                        record["rollout"] = outcome(rollout)
                    save_record(record, out / "records" / f"{path.stem}.pkl")
                else:
                    observations = scene_responsibility(model, scene, agent, cfg, last_step)
                if rollout is not None and rollout["end"]["reason"] == "crash_vehicle" and agent == scene.sdc:
                    row = blame_row(model, scene, agent, rollout, cfg)
                    new_crashes = not crashes_path.exists()
                    with open(crashes_path, "a", newline="") as f:
                        crash_writer = csv.DictWriter(f, fieldnames=CRASH_FIELDS)
                        if new_crashes:
                            crash_writer.writeheader()
                        crash_writer.writerow(row)
                    print(f"  collision at step {row['crash_step']} with {row.get('other_id', '?')}: {row['verdict']} "
                          f"(beta ego {_m(row.get('beta_ego'))}, other {_m(row.get('beta_other'))})", flush=True)
            for obs in observations:
                writer.writerow({"scene": path.stem, **obs.as_row()})
            handle.flush()
            tmp = out / "obs" / f"{path.stem}.pkl.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                pickle.dump([asdict(o) for o in observations], f)
            os.replace(tmp, out / "obs" / f"{path.stem}.pkl")
            s = [o.safety for o in observations]
            c = [o.courtesy for o in observations]
            print(f"[{n}/{len(todo)}] {path.name}: {len(observations)} windows, "
                  f"max safety {max(s, default=0):+.2f} m, max courtesy {max(c, default=0):.3f} nats "
                  f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
