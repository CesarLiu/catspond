"""Recording what happens in a MetaDrive episode as a rollout
(responsibility/rollouts.py), for collect_rollouts.py (evaluation) and
blame_reward.py (RL training). Everything here only touches the env through
the attributes it reads (env.vehicle, env.engine.traffic_manager,
env.engine.get_objects, env.engine.data_manager), so it runs against
stand-ins without MetaDrive.
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from responsibility.rollouts import end_reason, make_rollout


class Recorder:
    """The ego's (and the adversary's) state and which logged objects exist,
    at every step of an episode: ``record`` right after the reset and after
    every env.step, so state i is the scene's step i."""

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

    def _make(self, n, scene_file, scenario_id, policy, adv_mode, end, planned=None):
        """The rollout of the first ``n`` recorded states."""
        present = self.present[:n]
        ids = sorted(set().union(*present))
        mask = np.array([[sid in step for step in present] for sid in ids], dtype=bool).reshape(len(ids), len(present))
        ego = self._track(self.ego[:n])
        size = self.env.vehicle
        ego["size"] = (float(size.top_down_length), float(size.top_down_width))
        adversary = None
        if self.adversary is not None:
            adversary = dict(self._track(self.adv[:n]), track_id=self.adversary, planned=planned)
        return make_rollout(scene_file, scenario_id, policy, adv_mode, ego=ego, end=dict(end, step=n - 1),
                            adversary=adversary, present={"track_ids": ids, "mask": mask}, no_static_vehicles=True)

    def rollout(self, scene_file, scenario_id, policy, adv_mode, info, planned=None):
        """The whole episode, ended as the last step's ``info`` says."""
        crash = bool(info.get("crash_vehicle"))
        end = {"reason": end_reason(info, crash),
               "route_completion": float(info.get("route_completion", np.nan)), "crash_vehicle": crash,
               "crash_object": bool(info.get("crash_object")), "out_of_road": bool(info.get("out_of_road")),
               "arrive_dest": bool(info.get("arrive_dest")), "crash_with": self.nearest() if crash else None}
        return self._make(len(self.ego), scene_file, scenario_id, policy, adv_mode, end, planned)

    def crash_rollout(self, step, crash_with, scene_file, scenario_id, policy="td3", adv_mode="train"):
        """The episode up to a vehicle collision at recorded state ``step``
        with the logged object ``crash_with``, as if it had ended there (as
        it does with crash_vehicle_done, CAT's evaluation setting; its
        training carries on through collisions)."""
        end = {"reason": "crash_vehicle", "crash_vehicle": True, "crash_with": crash_with}
        return self._make(step + 1, scene_file, scenario_id, policy, adv_mode, end)


def current_scenario_id(env) -> str:
    scenario = env.engine.data_manager.current_scenario
    return str(scenario["metadata"].get("scenario_id", scenario.get("id")))


class SceneIndex:
    """MetaDrive scenario index -> scene file stem, verified by scenario id."""

    def __init__(self, scenes):
        self.scenes = Path(scenes)
        self.by_id: Optional[Dict[str, str]] = None

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
