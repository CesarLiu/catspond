"""CAT's scenes as GPUDrive scenarios with only the self-driving car under
policy control (responsibility.gpudrive_export does the field-by-field
conversion; this module only arranges the scenario for CAT's setting).

  order     the SDC first, CAT's adversary (the other object of interest)
            second, every other object by its distance to the SDC at step
            10: GPUDrive keeps at most 64 agents a world (kMaxAgentCount), and
            CAT's scenes hold up to 129, so the ones it drops are the far ones
  control   every object but the SDC is marked as an expert, i.e. replayed
            from its log (CAT's MetaDrive setting: reactive_traffic False)
  adversary its goal is put GOAL_OFFSET metres away: GPUDrive takes any agent
            within dist_to_goal_threshold of its JSON goal off the road, and
            treats one whose start is within 0.2 m of it as static -- neither
            may happen to an adversary that follows a generated plan, written
            into the simulator's trajectory tensor at run time
"""

from typing import Dict, Optional

import numpy as np

from responsibility.gpudrive_export import scenario
from responsibility.scene import Scene

GOAL_OFFSET = 5000.0  # m; far from any scene, short of GPUDrive's invalid marker (-1e4)


def cat_adversary(scene: Scene) -> Optional[int]:
    others = [i for i in scene.objects_of_interest if i != scene.sdc]
    return others[0] if others else None


def ego_only_scenario(scene: Scene, name: str) -> Dict:
    """The GPUDrive scenario of a CAT scene, arranged as described above;
    metadata gains ``adversary_id`` (None without a second object of
    interest)."""
    data = scenario(scene, name, (), experts=False)
    adv = cat_adversary(scene)
    t0 = 10
    anchor = scene.position[scene.sdc, t0, :2]

    def dist(i):
        valid = np.flatnonzero(scene.valid[i])
        if len(valid) == 0:
            return np.inf
        t = valid[np.argmin(np.abs(valid - t0))]
        return float(np.linalg.norm(scene.position[i, t, :2] - anchor))

    rest = sorted((i for i in range(scene.n_agents) if i not in (scene.sdc, adv)), key=dist)
    order = [scene.sdc] + ([adv] if adv is not None else []) + rest
    objects = [data["objects"][i] for i in order]
    for k, obj in enumerate(objects):
        obj["mark_as_expert"] = k != 0
    if adv is not None:
        goal = objects[1]["goalPosition"]
        objects[1]["goalPosition"] = {"x": goal["x"] + GOAL_OFFSET, "y": goal["y"] + GOAL_OFFSET, "z": goal["z"]}
    data["objects"] = objects
    data["metadata"]["sdc_track_index"] = 0
    data["metadata"]["adversary_id"] = None if adv is None else objects[1]["id"]
    return data
