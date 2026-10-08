"""Swapping the roles of the two objects of interest (responsibility/swap.py, swap_roles.py)."""

import json
import pickle

import numpy as np
import pytest

from responsibility.scene import Scene, cat_agent_order
from responsibility.swap import index_path, is_swapped, scene_split, swap_problem, swapped
from scripts.responsibility import swap_roles
from tests.responsibility.conftest import make_scene, track


def description(**other):
    scene = make_scene({"0": track((0.0, 0.0), (10.0, 0.0)), "1": track((30.0, 3.5), (8.0, 0.0), **other),
                        "2": track((-20.0, 0.0), (5.0, 0.0))})
    return scene.to_description()


def test_swapping_makes_the_other_object_of_interest_the_self_driving_car():
    d = description()
    s = swapped(d)
    assert s["metadata"]["sdc_id"] == "1" and d["metadata"]["sdc_id"] == "0"  # a copy
    assert s["metadata"]["roles_swapped"] == {"original_sdc_id": "0", "sdc_id": "1"}
    scene = Scene.from_description(s)
    assert scene.track_ids[scene.sdc] == "1"
    assert [scene.track_ids[i] for i in cat_agent_order(scene)[:2]] == ["1", "0"]  # CAT's ego, then its adversary
    for tid in d["tracks"]:  # the tracks themselves are untouched
        np.testing.assert_array_equal(s["tracks"][tid]["state"]["position"], d["tracks"][tid]["state"]["position"])


def test_only_a_vehicle_present_at_every_step_can_become_the_ego():
    assert swap_problem(description()) is None
    late = np.ones(91, dtype=bool)
    late[0] = False
    assert "not present at step 0" in swap_problem(description(valid=late))
    gap = np.ones(91, dtype=bool)
    gap[40:45] = False
    assert "missing at 5 steps" in swap_problem(description(valid=gap))
    assert "not a vehicle" in swap_problem(description(kind="PEDESTRIAN"))
    with pytest.raises(ValueError):
        swapped(description(valid=late))


def test_the_split_counts_cat_s_numbers_over_the_files_present(tmp_path):
    for n in (0, 3, 399, 400, 450):
        (tmp_path / f"{n}.pkl").write_bytes(pickle.dumps(description()))
    assert scene_split(tmp_path) == (3, 2)
    assert not is_swapped(tmp_path)


def test_swap_roles_writes_scenes_only_and_its_index_beside_them(tmp_path):
    src, out = tmp_path / "scenes", tmp_path / "swapped"
    src.mkdir()
    late = np.ones(91, dtype=bool)
    late[0] = False
    for n, d in ((0, description()), (1, description(valid=late)), (400, description())):
        (src / f"{n}.pkl").write_bytes(pickle.dumps(d))
    swap_roles.main(["--scenes", str(src), "--out-dir", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["0.pkl", "400.pkl"]  # MetaDrive reads every file in it
    index = json.loads(index_path(out).read_text())
    assert index["split"] == {"train": 1, "test": 1}
    assert index["scenes"]["1"]["swapped"] is False and index["scenes"]["0"]["sdc_id"] == "1"
    assert is_swapped(out) and scene_split(out) == (1, 1)
