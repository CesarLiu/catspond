"""Swapping the roles of CAT's two objects of interest: the self-driving car
becomes the adversary and the other object of interest the ego.

Nothing in CAT names the ego apart from ``metadata.sdc_id``. advgen takes
the ego from it and the adversary from the other object of interest, and
MetaDrive spawns its ego vehicle, its route and the replay policy from that
track. The swap is therefore a relabelling of the scene: ``sdc_id`` and
``sdc_track_index`` move to the other object of interest, and every
consumer follows (advgen, responsibility.adversarial, export_adv_scenes,
compute_responsibility, cat_advgen.py and cat_RLtrain.py with --scenes_dir).

A scene is swapped only if the other object of interest is a vehicle that
is valid at every step: MetaDrive spawns the ego at step 0 and drives it
along its whole logged route. Of CAT's 500 scenes, 459 qualify. In 24 the
other object is missing at step 0, and in 17 its track has gaps.

A swapped directory keeps the original file names (17.pkl is still scene
17) and holds nothing else: MetaDrive asserts that every file in it is a
scene. Its index (the scenes swapped and skipped, and why) is the file
<directory>.index.json next to it (index_path).
MetaDrive orders the files by number, so the training scenes (0-399) still
come first; scene_split counts them, and cat_RLtrain.py --scenes_dir uses it.
"""

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

TRAIN_SCENES = 400  # CAT's split of raw_scenes_500: 0-399 train, 400-499 test


def other_object(description: Dict) -> Optional[str]:
    """The object of interest that is not the self-driving car (track id), or None."""
    meta = description["metadata"]
    sdc = str(meta["sdc_id"])
    others = [str(t) for t in meta.get("objects_of_interest", []) if str(t) != sdc]
    return others[0] if others else None


def swap_problem(description: Dict) -> Optional[str]:
    """Why the roles of this scene cannot be swapped, or None if they can."""
    other = other_object(description)
    if other is None:
        return "no second object of interest"
    track = description["tracks"].get(other)
    if track is None:
        return f"object of interest {other} has no track"
    if track["type"] != "VEHICLE":
        return f"object of interest {other} is a {track['type']}, not a vehicle"
    valid = np.asarray(track["state"]["valid"], dtype=bool)
    if not valid[0]:
        return f"object of interest {other} is not present at step 0"
    if not valid.all():
        return f"object of interest {other} is missing at {int((~valid).sum())} steps"
    return None


def swapped(description: Dict) -> Dict:
    """A copy of the scene description with the roles swapped: the other
    object of interest is the self-driving car (``sdc_id``,
    ``sdc_track_index``), and ``metadata.roles_swapped`` records the
    original self-driving car."""
    problem = swap_problem(description)
    if problem is not None:
        raise ValueError(f"cannot swap the roles: {problem}")
    out = copy.deepcopy(description)
    meta = out["metadata"]
    old, new = str(meta["sdc_id"]), other_object(description)
    meta["sdc_id"] = new
    predict = meta.get("tracks_to_predict", {})
    if new in predict and "track_index" in predict[new]:
        meta["sdc_track_index"] = predict[new]["track_index"]
    elif "sdc_track_index" in meta:
        meta["sdc_track_index"] = list(out["tracks"]).index(new)
    meta["roles_swapped"] = {"original_sdc_id": old, "sdc_id": new}
    return out


def scene_split(directory, first_test: int = TRAIN_SCENES) -> Tuple[int, int]:
    """(training scenes, test scenes) of a scene directory, in MetaDrive's
    order (files by number): CAT's split by scene number, 0-399 train and
    400 on test, counted over the files present -- 400 / 100 for
    raw_scenes_500, fewer for a swapped or otherwise reduced copy."""
    from responsibility.scene import scene_files

    numbers = [int(p.stem) for p in scene_files(directory) if p.stem.isdigit()]
    train = sum(n < first_test for n in numbers)
    return train, len(numbers) - train


def index_path(directory) -> Path:
    """Where swap_roles.py writes a swapped directory's index (scene.sidecar_index)."""
    from responsibility.scene import sidecar_index

    return sidecar_index(directory)


def is_swapped(directory) -> bool:
    """Whether a scene directory holds swapped scenes (its first scene's
    ``metadata.roles_swapped``; also true for scenes derived from them, e.g.
    by export_adv_scenes.py, which keeps the metadata)."""
    import pickle

    from responsibility.scene import scene_files

    files = scene_files(directory)
    if not files:
        return False
    with open(files[0], "rb") as f:
        return "roles_swapped" in pickle.load(f).get("metadata", {})
