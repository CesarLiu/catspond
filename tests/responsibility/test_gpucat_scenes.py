import numpy as np

from gpucat.scenes import GOAL_OFFSET, ego_only_scenario
from tests.responsibility.conftest import make_scene, track


def test_sdc_first_adversary_second_others_by_distance_and_only_the_sdc_controlled():
    # 9 and 5 move (3 m/s), 7 is parked (MetaDrive's no_static_vehicles leaves it out)
    s = make_scene({"9": track((50, 0), (3, 0)), "0": track((0, 0), (10, 0)), "1": track((30, 5), (5, 0)),
                    "5": track((10, 0), (3, 0)), "7": track((5, 4), (0, 0))}, sdc="0", ooi=("0", "1"))
    data = ego_only_scenario(s, "cat_x.json")
    assert [o["id"] for o in data["objects"]] == [0, 1, 5, 9]  # SDC, adversary, then 10 m before 50 m away
    assert [o["mark_as_expert"] for o in data["objects"]] == [False, True, True, True]
    assert data["metadata"]["sdc_track_index"] == 0 and data["metadata"]["adversary_id"] == 1
    adv_last = data["objects"][1]["position"][-1]
    goal = data["objects"][1]["goalPosition"]
    assert np.isclose(goal["x"], adv_last["x"] + GOAL_OFFSET) and np.isclose(goal["y"], adv_last["y"] + GOAL_OFFSET)
    sdc_last = data["objects"][0]["position"][-1]
    assert data["objects"][0]["goalPosition"] == sdc_last  # the SDC keeps its logged goal
