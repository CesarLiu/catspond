"""Records what a driving policy does in MetaDrive (CAT's WaymoEnv, with
CAT's evaluation settings), one rollout per scene, for
compute_responsibility.py --rollouts (layout: responsibility/rollouts.py).

  --policy replay            the logged ego (ReplayEgoCarPolicy): the
                             reference every policy is compared against
  --policy models/<name>     a TD3 policy saved by cat_RLtrain.py --save_model
  --adversary                also run CAT's adversarial evaluation: after the
                             normal episode, an adversary is generated against
                             the ego's own trajectory in it (as eval_policy
                             does) and a second episode is played with it.
                             --adv_selection cat|constrained|penalized chooses
                             the generator (responsibility/adversarial.py)

Output: OUT/<policy_name>/none/<scene>.pkl, and with --adversary
OUT/<policy_name>/<mode>/<scene>.pkl (mode: cat, constrained<tau>,
penalized<p>, named as cat_RLtrain.py names its runs), each directory with a
config.json. Scenes are the MetaDrive scenario indices --first ... --first+n-1
(default: CAT's test split 400-499); --num_shards/--shard_index split them
over processes. Finished scenes are skipped when re-run.

Checks printed per episode (the M3.0 checks of RESEARCH_PLAN.md):
  replay     the recorded ego against the logged one (max error; should be
             < 0.1 m, which also confirms that state i is the log's step i)
  adversary  how far the adversary moved away from its logged track, and
             whether its recorded states follow the planned trajectory with
             a lag of 0 or 1 step (the traffic manager applies plan[i] after
             step i + 1)

Example (on the server, from the repository root):
    python -m scripts.responsibility.collect_rollouts --policy replay --out_dir rollouts
    python -m scripts.responsibility.collect_rollouts --policy models/cat --policy_name td3_cat \\
        --adversary --adv_selection cat --out_dir rollouts
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.rollouts import end_reason, make_rollout, save_rollout  # noqa: E402

VEHICLE_CONFIG = dict(lidar=dict(num_lasers=30, distance=50, num_others=3),
                      side_detector=dict(num_lasers=30), lane_line_detector=dict(num_lasers=12))


def env_config(scenes, replay: bool):
    """cat_RLtrain.py's config_test over all scenes (indices are chosen by
    force_seed), with the logged ego for --policy replay."""
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy

    config = dict(
        data_directory=str(Path(scenes).resolve()), start_scenario_index=0, num_scenarios=500,
        crash_vehicle_done=True, sequential_seed=True, force_reuse_object_name=True, horizon=50,
        no_light=True, no_static_vehicles=True, reactive_traffic=False, vehicle_config=VEHICLE_CONFIG,
    )
    if replay:
        config["agent_policy"] = ReplayEgoCarPolicy
    return config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                allow_abbrev=False)
    p.add_argument("--policy", default="replay", help="replay, or a TD3 model prefix (models/<name>).")
    p.add_argument("--policy_name", default=None, help="Output name (default: replay or the model's name).")
    p.add_argument("--adversary", action="store_true", help="Also run CAT's adversarial episode per scene.")
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--out_dir", default="rollouts")
    p.add_argument("--first", type=int, default=400, help="First MetaDrive scenario index.")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_index", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0, help="Stop episodes after this many steps (0: until done).")
    p.add_argument('--OV_traj_num', type=int, default=32)  # as cat_RLtrain.py; read by CAT's generator
    p.add_argument('--AV_traj_num', type=int, default=1)
    known, _ = p.parse_known_args()
    generator = None
    if known.adversary:
        from responsibility.adversarial import make_adv_generator

        generator = make_adv_generator(p)  # adds CAT's options and parses the whole command line
        args = generator.args
    else:
        args = p.parse_args()
    return args, generator


def adv_mode_name(args) -> str:
    rule = getattr(args, "adv_selection", "cat")
    if rule == "cat":
        return "cat"
    return f"{rule}{args.resp_threshold:g}" if rule == "constrained" else f"{rule}{args.resp_penalty:g}"


class Recorder:
    """The ego's (and the adversary's) state and which logged objects exist,
    at every step of an episode."""

    def __init__(self, env, adversary=None):
        self.env = env
        self.adversary = None if adversary is None else str(adversary)
        self.ego, self.adv, self.present = [], [], []

    def record(self):
        ego = self.env.vehicle
        self.ego.append((*ego.position[:2], ego.heading_theta, *ego.velocity[:2]))
        tm = self.env.engine.traffic_manager
        ids = {str(k): v for k, v in tm._scenario_id_to_obj_id.items()}
        self.present.append(set(ids))
        if self.adversary is not None:
            obj = self._object(ids.get(self.adversary))
            self.adv.append((*obj.position[:2], obj.heading_theta, *obj.velocity[:2]) if obj is not None
                            else (np.nan,) * 5)

    def _object(self, obj_id):
        if obj_id is None:
            return None
        return self.env.engine.get_objects([obj_id]).get(obj_id)

    def nearest(self):
        """Scenario id of the logged object closest to the ego now."""
        tm = self.env.engine.traffic_manager
        ego = np.asarray(self.env.vehicle.position[:2])
        best, best_d = None, np.inf
        for sid, obj_id in tm._scenario_id_to_obj_id.items():
            obj = self._object(obj_id)
            if obj is None:
                continue
            d = np.linalg.norm(np.asarray(obj.position[:2]) - ego)
            if d < best_d:
                best, best_d = str(sid), d
        return best

    @staticmethod
    def _track(rows):
        a = np.asarray(rows, dtype=np.float64)
        return {"position": a[:, :2], "heading": a[:, 2], "velocity": a[:, 3:5]}

    def rollout(self, scene_file, scenario_id, policy, adv_mode, info, planned=None):
        crash = bool(info.get("crash_vehicle"))
        ids = sorted(set().union(*self.present))
        mask = np.array([[sid in step for step in self.present] for sid in ids], dtype=bool).reshape(len(ids), -1)
        ego = self._track(self.ego)
        size = self.env.vehicle
        ego["size"] = (float(size.top_down_length), float(size.top_down_width))
        adversary = None
        if self.adversary is not None:
            adversary = dict(self._track(self.adv), track_id=self.adversary, planned=planned)
        end = {"step": len(self.ego) - 1, "reason": end_reason(info, crash),
               "route_completion": float(info.get("route_completion", np.nan)), "crash_vehicle": crash,
               "crash_object": bool(info.get("crash_object")), "out_of_road": bool(info.get("out_of_road")),
               "arrive_dest": bool(info.get("arrive_dest")), "crash_with": self.nearest() if crash else None}
        return make_rollout(scene_file, scenario_id, policy, adv_mode, ego=ego, end=end, adversary=adversary,
                            present={"track_ids": ids, "mask": mask}, no_static_vehicles=True)


def play(env, state, act, recorder, generator=None, max_steps=0):
    """One episode from the reset that returned ``state``; returns the last
    step's info."""
    recorder.record()
    info, steps = {}, 0
    while True:
        if generator is not None:
            generator.log_AV_history()
        state, _, done, info = env.step(act(state))
        recorder.record()
        steps += 1
        if done or (max_steps and steps >= max_steps):
            return info


def logged_track(env, track_id):
    data = env.engine.data_manager.current_scenario
    state = data["tracks"][str(track_id)]["state"]
    return np.asarray(state["position"])[:, :2], np.asarray(state["valid"], dtype=bool)


def replay_error(env, rollout):
    data = env.engine.data_manager.current_scenario
    pos, valid = logged_track(env, data["metadata"]["sdc_id"])
    ego = rollout["ego"]["position"]
    n = min(len(ego), len(pos))
    ok = valid[:n]
    return float(np.max(np.linalg.norm(ego[:n][ok] - pos[:n][ok], axis=-1))) if ok.any() else float("nan")


def adversary_check(env, rollout):
    """(mean distance of the adversary from its logged track after step 11,
    mean error against the plan at lag 0 and at lag 1)."""
    adv = rollout["adversary"]
    pos, plan = adv["position"], adv["planned"]
    logged, valid = logged_track(env, adv["track_id"])
    n = min(len(pos), len(logged), len(plan))
    t = np.arange(12, n)
    ok = t[np.isfinite(pos[t]).all(-1) & valid[t]]
    moved = float(np.mean(np.linalg.norm(pos[ok] - logged[ok], axis=-1))) if ok.size else float("nan")
    lag = []
    for d in (0, 1):
        s = np.arange(max(1, d), n)
        s = s[np.isfinite(pos[s]).all(-1)]
        lag.append(float(np.mean(np.linalg.norm(pos[s] - plan[s - d, :2], axis=-1))) if s.size else float("nan"))
    return moved, lag[0], lag[1]


class SceneIndex:
    """MetaDrive scenario index -> scene file stem, verified by scenario id."""

    def __init__(self, scenes):
        self.scenes = Path(scenes)
        self.by_id = None

    @staticmethod
    def _id(path):
        from responsibility.scene import Scene

        return Scene.load(path).scenario_id

    def stem(self, index, scenario_id):
        guess = self.scenes / f"{index}.pkl"
        if guess.exists() and self._id(guess) == scenario_id:
            return guess.stem
        if self.by_id is None:
            from responsibility.scene import scene_files

            self.by_id = {self._id(p): p.stem for p in scene_files(self.scenes)}
        return self.by_id[scenario_id]


def write_config(directory, settings):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(settings)):
            raise SystemExit(f"{directory} holds rollouts collected with other settings; use another --out_dir")
        return
    tmp = path.with_name(f"config.json.tmp{os.getpid()}")
    tmp.write_text(json.dumps(settings, indent=2))
    os.replace(tmp, path)


def main():
    args, generator = parse_args()
    from metadrive.envs.real_data_envs.waymo_env import WaymoEnv

    replay = args.policy == "replay"
    name = args.policy_name or ("replay" if replay else Path(args.policy).name)
    env = WaymoEnv(config=env_config(args.scenes, replay))
    if replay:
        def act(_):
            return [1.0, 0.0]
    else:
        import torch

        from saferl_algo import TD3

        max_action = float(env.action_space.high[0])
        policy = TD3.TD3(state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                         max_action=max_action)
        policy.load(args.policy)
        torch.set_grad_enabled(False)

        def act(state):
            return policy.select_action(np.array(state))

    modes = ["none"] + ([adv_mode_name(args)] if generator is not None else [])
    root = Path(args.out_dir) / name
    base = {"policy": args.policy, "policy_name": name, "scenes": str(Path(args.scenes).resolve()),
            "env": {k: v for k, v in env_config(args.scenes, False).items() if k != "data_directory"}}
    for mode in modes:
        settings = dict(base, adv_mode=mode)
        if mode != "none":
            settings["adversary"] = {k: getattr(args, k) for k in vars(args)
                                     if k.startswith("resp_") or k in ("adv_selection", "OV_traj_num", "AV_traj_num")}
        write_config(root / mode, settings)

    index = SceneIndex(args.scenes)
    seeds = list(range(args.first, args.first + args.n))[args.shard_index::args.num_shards]
    stems = {}
    checks = {"replay": [], "moved": [], "lag0": [], "lag1": []}
    for k, seed in enumerate(seeds, 1):
        t0 = time.time()
        state = env.reset(force_seed=seed)
        scenario_id = str(env.engine.data_manager.current_scenario["metadata"].get(
            "scenario_id", env.engine.data_manager.current_scenario.get("id")))
        stem = stems.setdefault(seed, index.stem(seed, scenario_id))
        if all((root / m / f"{stem}.pkl").exists() for m in modes):
            print(f"[{k}/{len(seeds)}] scene {stem}: done", flush=True)
            continue

        # the normal episode (CAT's eval_policy: first round)
        if generator is not None:
            generator.before_episode(env)
        recorder = Recorder(env)
        info = play(env, state, act, recorder, generator, args.max_steps)
        if generator is not None:
            generator.after_episode(update_AV_traj=True, mode="eval")
        rollout = recorder.rollout(stem, scenario_id, name, "none", info)
        save_rollout(rollout, root / "none" / f"{stem}.pkl")
        line = f"[{k}/{len(seeds)}] scene {stem}: none {rollout['end']['reason']} at {rollout['end']['step']}"
        if replay:
            err = replay_error(env, rollout)
            checks["replay"].append(err)
            line += f" (replay error {err:.3f} m)"

        # the adversarial episode (second round), with the ego's trajectory from the first
        if generator is not None:
            state = env.reset(force_seed=seed)
            env.vehicle.ego_crash_flag = False
            generator.before_episode(env)
            generator.generate(mode="eval")
            planned = np.array(generator.adv_traj, dtype=np.float64)  # the traffic manager consumes the list
            env.engine.traffic_manager.set_adv_info(generator.adv_agent, generator.adv_traj)
            recorder = Recorder(env, adversary=generator.adv_agent)
            info = play(env, state, act, recorder, None, args.max_steps)
            rollout = recorder.rollout(stem, scenario_id, name, modes[1], info, planned=planned)
            save_rollout(rollout, root / modes[1] / f"{stem}.pkl")
            moved, lag0, lag1 = adversary_check(env, rollout)
            for key, value in (("moved", moved), ("lag0", lag0), ("lag1", lag1)):
                checks[key].append(value)
            line += (f"; {modes[1]} {rollout['end']['reason']} at {rollout['end']['step']} "
                     f"(adversary {moved:.2f} m off its log; plan error lag0 {lag0:.2f} / lag1 {lag1:.2f} m)")
        print(line + f" ({time.time() - t0:.1f} s)", flush=True)

    if checks["replay"]:
        e = np.array(checks["replay"])
        print(f"\nreplay: max ego error {np.nanmax(e):.3f} m, median {np.nanmedian(e):.3f} m; "
              f"{np.mean(e < 0.1) * 100:.0f}% of episodes within 0.1 m")
    if checks["moved"]:
        print(f"\nadversary: {np.nanmean(checks['moved']):.2f} m off its log on average "
              f"(~0 would mean set_adv_info had no effect); plan error lag 0 {np.nanmean(checks['lag0']):.2f} m, "
              f"lag 1 {np.nanmean(checks['lag1']):.2f} m")
    if generator is not None and hasattr(generator, "report"):
        generator.report()
    env.close()


if __name__ == "__main__":
    main()
