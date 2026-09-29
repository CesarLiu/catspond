import numpy as np
import pytest

from responsibility.scene import cat_agent_order, womd_features
from tests.responsibility.conftest import make_scene, track


def _scene():
    return make_scene({
        "0": track((0, 0), (10, 0)),
        "1": track((30, 4), (8, 0)),
        "2": track((-20, -4), (5, 0), kind="PEDESTRIAN"),
    })


def test_scene_arrays_and_indices():
    scene = _scene()
    assert scene.n_agents == 3 and scene.n_steps == 91
    assert scene.sdc == 0 and scene.objects_of_interest == [0, 1]
    assert scene.index("2") == 2
    assert np.allclose(scene.shape_at(1, 10), [4.8, 2.0])


def test_features_window_follows_the_context_step():
    scene = _scene()
    f = womd_features(scene, 30, [0, 1, 2])
    x = scene.position[0, :, 0]
    assert np.allclose(f["state/past/x"][0], x[20:30])
    assert np.allclose(f["state/current/x"][0, 0], x[30])
    assert np.allclose(f["state/future/x"][0, :60], x[31:91])
    # beyond the clip the future is invalid and filled
    assert f["state/future/valid"][0, :60].all() and not f["state/future/valid"][0, 60:].any()
    assert (f["state/future/x"][0, 60:] == -1).all()


def test_features_mark_the_pair_and_leave_out_unlisted_agents():
    scene = _scene()
    f = womd_features(scene, 10, [1, 0])  # agent 2 left out
    assert list(f["state/id"][:3]) == [1, 0, -1]
    assert list(f["state/tracks_to_predict"][:3]) == [1, 1, 0]
    assert list(f["state/objects_of_interest"][:3]) == [1, 1, 0]
    assert list(f["state/is_sdc"][:3]) == [0, 1, 0]
    assert f["state/current/valid"][2, 0] == 0


def test_features_map_and_traffic_lights():
    scene = _scene()
    f = womd_features(scene, 45, [0, 1])
    types = f["roadgraph_samples/type"][:, 0]
    assert (types == 2).sum() == 3  # the lane's three points
    assert (types == 17).sum() == 3  # a stop sign point is written as three rows, as in CAT
    # the light turns red at step 50: past of step 45 covers 35..44 (green), current 45 (green)
    assert (f["traffic_light_state/past/state"][:, 0] == 6).all()
    later = womd_features(scene, 55, [0, 1])
    assert later["traffic_light_state/current/state"][0, 0] == 4
    assert list(later["traffic_light_state/past/state"][:, 0]) == [6] * 5 + [4] * 5


def test_features_need_a_pair():
    with pytest.raises(ValueError):
        womd_features(_scene(), 10, [0])


def test_cat_order_starts_with_the_sdc_and_the_adversary():
    scene = make_scene({"0": track((0, 0), (1, 0)), "1": track((5, 0), (1, 0)), "2": track((9, 0), (1, 0))},
                       sdc="1", ooi=("2", "1"))
    assert cat_agent_order(scene) == [1, 2, 0]


def test_with_track_replaces_one_agent():
    scene = _scene()
    pos = np.zeros((91, 2))
    other = scene.with_track(0, pos, np.zeros(91), np.zeros((91, 2)))
    assert np.allclose(other.position[0, :, :2], 0) and np.allclose(other.position[0, :, 2], scene.position[0, :, 2])
    assert np.allclose(scene.position[0, 20, 0], 10.0)  # the original is untouched (10 m/s for 1 s)
