"""Driving-policy rollouts in MetaDrive and the scenes they produced.

A rollout is what a policy did in one scene (scripts/responsibility/
collect_rollouts.py records it): the self-driving car's simulated states,
the adversary's if CAT placed one, which logged objects the simulator
actually spawned, and how the episode ended. ``scene_from_rollout`` turns it
back into a ``Scene`` -- the logged scene with the simulated tracks in
place of the logged ones -- so everything that measures logged driving
(responsibility, records, summaries, levels) measures the policy.

Rollout layout (``make_rollout`` builds and ``check_rollout`` validates it):

  version, scene_file, scenario_id, policy, adv_mode
  ego        {"position" [T, 2], "heading" [T], "velocity" [T, 2] or None,
              "size" (length, width) in the simulator or None}
  adversary  None, or the same plus "track_id" and "planned" (CAT's adv_traj,
             [91, 5] x, y, vx, vy, yaw)
  present    None, or {"track_ids": [M], "mask": bool [M, T]}: which logged
             objects existed in the simulation at each step
  no_static_vehicles   the simulator's setting (used when present is None)
  end        {"step", "reason", "route_completion", "crash_vehicle",
              "crash_object", "out_of_road", "arrive_dest", "crash_with"}

State i is the scene's step i: 0 right after reset, then one per env.step,
the last at end["step"] = T - 1 (MetaDrive's step is 0.1 s, the log's
rate). Steps where an object did not exist hold NaN.

What differs from the log in the rebuilt scene:
  - the ego (and the adversary) follow the simulated states and are
    invalid after the episode ended (and wherever they hold NaN);
  - objects the simulator never spawned are invalid throughout. With
    ``present`` that is read off the rollout; without it, MetaDrive's own
    rules are applied: with no_static_vehicles (CAT's setting) vehicles whose
    logged positions spread less than 3 m are left out, and cones/barriers
    logged for fewer than 20 steps always are;
  - other objects keep their logged states (CAT's traffic is not reactive),
    restricted to the steps they existed in the simulation.
Agent sizes stay the logged ones, so a replayed rollout rebuilds exactly the
logged scene minus the objects the simulator dropped.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from responsibility.scene import Scene

VERSION = 1

# ScenarioTrafficManager's spawning rules (metadrive/manager/scenario_traffic_manager.py)
STATIC_THRESHOLD = 3.0  # m: a vehicle whose logged x or y spreads (std) less than this is static
MIN_VALID_FRAME_LEN = 20  # steps: cones and barriers logged for fewer are noise
SPAWNED_TYPES = ("VEHICLE", "CYCLIST", "PEDESTRIAN", "TRAFFIC_CONE", "TRAFFIC_BARRIER")

CRASH_REASONS = ("crash_vehicle", "crash_object", "crash_building")
END_REASONS = CRASH_REASONS + ("out_of_road", "arrive_dest", "max_step")


def end_reason(info: Dict, crash_flag: bool = False) -> str:
    """The first applicable termination in MetaDrive's step info (the order
    CAT's evaluation cares about: collisions first)."""
    if crash_flag or info.get("crash_vehicle"):
        return "crash_vehicle"
    for key in ("crash_object", "crash_building", "out_of_road", "arrive_dest"):
        if info.get(key):
            return key
    return "max_step"


def _track(position, heading, velocity=None, size=None) -> Dict:
    position = np.asarray(position, dtype=np.float32)[:, :2]
    heading = np.asarray(heading, dtype=np.float32)
    if len(heading) != len(position):
        raise ValueError("position and heading cover different steps")
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=np.float32)[:, :2]
        if len(velocity) != len(position):
            raise ValueError("position and velocity cover different steps")
    return {"position": position, "heading": heading, "velocity": velocity,
            "size": None if size is None else tuple(float(s) for s in size)}


def make_rollout(scene_file: str, scenario_id: str, policy: str, adv_mode: str,
                 ego: Dict, end: Dict, adversary: Optional[Dict] = None,
                 present: Optional[Dict] = None, no_static_vehicles: bool = True) -> Dict:
    """A rollout record. ``ego``/``adversary`` hold position, heading and
    optionally velocity and size (the adversary also track_id and planned);
    ``end`` at least step and reason."""
    rollout = {
        "version": VERSION, "scene_file": str(scene_file), "scenario_id": str(scenario_id),
        "policy": str(policy), "adv_mode": str(adv_mode),
        "ego": _track(ego["position"], ego["heading"], ego.get("velocity"), ego.get("size")),
        "adversary": None, "present": None, "no_static_vehicles": bool(no_static_vehicles),
        "end": {"route_completion": float("nan"), "crash_vehicle": False, "crash_object": False,
                "out_of_road": False, "arrive_dest": False, "crash_with": None, **end},
    }
    if adversary is not None:
        adv = _track(adversary["position"], adversary["heading"], adversary.get("velocity"), adversary.get("size"))
        adv["track_id"] = str(adversary["track_id"])
        adv["planned"] = None if adversary.get("planned") is None else np.asarray(adversary["planned"], np.float32)
        rollout["adversary"] = adv
    if present is not None:
        rollout["present"] = {"track_ids": [str(t) for t in present["track_ids"]],
                              "mask": np.asarray(present["mask"], dtype=bool)}
    check_rollout(rollout)
    return rollout


def check_rollout(rollout: Dict) -> None:
    """Raises ValueError when a rollout does not have the documented layout."""
    if rollout.get("version") != VERSION:
        raise ValueError(f"rollout version {rollout.get('version')}, expected {VERSION}")
    for key in ("scene_file", "scenario_id", "policy", "adv_mode", "ego", "end", "adversary", "present"):
        if key not in rollout:
            raise ValueError(f"rollout has no {key!r}")
    n = len(rollout["ego"]["position"])
    if n == 0:
        raise ValueError("rollout has no states")
    end = rollout["end"]
    if end.get("step") != n - 1:
        raise ValueError(f"end step {end.get('step')} but {n} states (expected end step {n - 1})")
    if end.get("reason") not in END_REASONS:
        raise ValueError(f"unknown end reason {end.get('reason')!r}")
    adv = rollout["adversary"]
    if adv is not None:
        if len(adv["position"]) != n:
            raise ValueError("adversary and ego states cover different steps")
        if "track_id" not in adv:
            raise ValueError("adversary has no track_id")
    present = rollout["present"]
    if present is not None and present["mask"].shape != (len(present["track_ids"]), n):
        raise ValueError(f"present mask {present['mask'].shape}, expected ({len(present['track_ids'])}, {n})")


def not_spawned(scene: Scene, no_static_vehicles: bool = True) -> List[int]:
    """Agents MetaDrive never spawns for this scene (other than the ego):
    static vehicles (with ``no_static_vehicles``), noise cones/barriers and
    unsupported types."""
    out = []
    for i in range(scene.n_agents):
        if i == scene.sdc:
            continue
        kind = scene.types[i]
        valid = scene.valid[i]
        if kind not in SPAWNED_TYPES:
            out.append(i)
        elif kind == "VEHICLE" and no_static_vehicles:
            points = scene.position[i, valid]
            moving = points.shape[0] > 0 and np.max(np.std(points, axis=0)[:2]) > STATIC_THRESHOLD
            if not moving:
                out.append(i)
        elif kind in ("TRAFFIC_CONE", "TRAFFIC_BARRIER") and valid.sum() < MIN_VALID_FRAME_LEN:
            out.append(i)
    return out


def _velocity_from(position: np.ndarray, dt: float = 0.1) -> np.ndarray:
    vel = np.zeros_like(position)
    if len(position) > 1:
        vel[:-1] = (position[1:] - position[:-1]) / dt
        vel[-1] = vel[-2]
    return vel


def _simulated(scene: Scene, agent: int, track: Dict) -> tuple:
    """Whole-clip arrays for ``agent`` from its simulated states: the
    simulated steps, invalid beyond them and where they hold NaN."""
    n_t = scene.n_steps
    t = min(len(track["position"]), n_t)
    pos = scene.position[agent].copy()
    head = scene.heading[agent].copy()
    vel = scene.velocity[agent].copy()
    valid = np.zeros(n_t, dtype=bool)
    sim_pos = track["position"][:t]
    ok = np.isfinite(sim_pos).all(-1) & np.isfinite(track["heading"][:t])
    pos[:t, :2] = np.where(ok[:, None], sim_pos, 0.0)
    head[:t] = np.where(ok, track["heading"][:t], 0.0)
    sim_vel = track["velocity"][:t] if track["velocity"] is not None else _velocity_from(sim_pos)
    vel[:t] = np.where(ok[:, None] & np.isfinite(sim_vel), sim_vel, 0.0)
    valid[:t] = ok
    return pos, head, vel, valid


def scene_from_rollout(scene: Scene, rollout: Dict) -> Scene:
    """The scene as the rollout played it (see the module docstring)."""
    check_rollout(rollout)
    n_t = scene.n_steps
    t = min(len(rollout["ego"]["position"]), n_t)
    out = scene.with_track(scene.sdc, *_simulated(scene, scene.sdc, rollout["ego"]))
    adv = rollout["adversary"]
    adv_index = None
    if adv is not None:
        adv_index = scene.index(adv["track_id"])
        out = out.with_track(adv_index, *_simulated(scene, adv_index, adv))

    valid = out.valid.copy()
    present = rollout["present"]
    if present is None:
        valid[not_spawned(scene, rollout.get("no_static_vehicles", True))] = False
    else:
        existed = {tid: row[:t] for tid, row in zip(present["track_ids"], present["mask"])}
        for i, tid in enumerate(scene.track_ids):
            if i in (scene.sdc, adv_index):
                continue
            if tid not in existed:
                valid[i] = False  # never in the simulation
            else:
                valid[i, :t] &= existed[tid]
    return Scene(**{**out.__dict__, "valid": valid})


def last_window_step(rollout: Dict, horizon: int) -> int:
    """The last context step whose metric horizon the rollout covers: up to
    the end when it ended in a collision (the collision is what the window
    must see), otherwise a full horizon before it (what the ego would have
    done after the episode stopped is unknown)."""
    end = rollout["end"]
    return end["step"] if end["reason"] in CRASH_REASONS else end["step"] - horizon


def replay_rollout(scene: Scene, policy: str = "replay", adv_mode: str = "none",
                   end_step: Optional[int] = None, reason: str = "arrive_dest",
                   no_static_vehicles: bool = True, with_present: bool = True) -> Dict:
    """The rollout a perfect replay of the logged ego would record (what
    ReplayEgoCarPolicy should produce), for tests and offline checks."""
    last = scene.n_steps - 1 if end_step is None else end_step
    sdc = scene.sdc
    steps = slice(0, last + 1)
    present = None
    if with_present:
        dropped = set(not_spawned(scene, no_static_vehicles))
        ids = [scene.track_ids[i] for i in range(scene.n_agents) if i != sdc and i not in dropped]
        rows = [scene.valid[scene.index(tid), steps] for tid in ids]
        present = {"track_ids": ids, "mask": np.array(rows, dtype=bool).reshape(len(ids), last + 1)}
    return make_rollout(
        scene_file="", scenario_id=scene.scenario_id, policy=policy, adv_mode=adv_mode,
        ego={"position": scene.position[sdc, steps, :2], "heading": scene.heading[sdc, steps],
             "velocity": scene.velocity[sdc, steps]},
        end={"step": last, "reason": reason, "route_completion": 1.0, "arrive_dest": reason == "arrive_dest"},
        present=present, no_static_vehicles=no_static_vehicles,
    )


def rollout_files(directory) -> List[Path]:
    """Rollouts of one policy and adversary mode, in numeric scene order."""
    files = [p for p in Path(directory).glob("*.pkl")]
    return sorted(files, key=lambda p: (int(p.stem) if p.stem.isdigit() else float("inf"), p.stem))


def save_rollout(rollout: Dict, path) -> None:
    check_rollout(rollout)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:
        pickle.dump(rollout, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_rollout(path) -> Dict:
    with open(path, "rb") as f:
        rollout = pickle.load(f)
    check_rollout(rollout)
    return rollout


def outcome(rollout: Dict) -> Dict:
    """The episode's result as one flat row."""
    end = rollout["end"]
    return {
        "scene": rollout["scene_file"], "policy": rollout["policy"], "adv_mode": rollout["adv_mode"],
        "steps": end["step"], "reason": end["reason"],
        "crash": int(end["reason"] in CRASH_REASONS or bool(end.get("crash_vehicle"))),
        "crash_vehicle": int(bool(end.get("crash_vehicle")) or end["reason"] == "crash_vehicle"),
        "out_of_road": int(bool(end.get("out_of_road"))), "arrive_dest": int(bool(end.get("arrive_dest"))),
        "route_completion": float(end.get("route_completion", float("nan"))),
        "crash_with": end.get("crash_with"),
    }
