import numpy as np
import pytest

pytest.importorskip("tensorflow")  # export_adv_scenes imports CAT's generator

from responsibility.scene import Scene  # noqa: E402
from scripts.responsibility.export_adv_scenes import adversarial_description, first_overlap  # noqa: E402
from tests.responsibility.conftest import make_scene, track  # noqa: E402


def test_only_the_adversary_future_changes():
    original = make_scene({"0": track((0, 0), (10, 0)), "1": track((0, 20), (10, 0))}).to_description()
    plan = np.zeros((91, 5))
    plan[:, 0] = np.arange(91)  # x
    plan[:, 1] = 5.0  # y
    plan[:, 2] = 10.0  # vx
    plan[:, 4] = 0.3  # yaw
    out = adversarial_description(original, "1", plan)
    before, after = original["tracks"]["1"]["state"], out["tracks"]["1"]["state"]
    assert np.array_equal(after["position"][:11], before["position"][:11])  # the history CAT plans from
    assert np.allclose(after["position"][11:, :2], plan[11:, :2]) and np.allclose(after["heading"][11:], 0.3)
    assert np.allclose(after["velocity"][11:], [10.0, 0.0]) and after["valid"][11:].all()
    assert after["position"][11:, 2].tolist() == before["position"][11:, 2].tolist()  # z kept
    assert np.array_equal(out["tracks"]["0"]["state"]["position"], original["tracks"]["0"]["state"]["position"])
    assert original["tracks"]["1"]["state"]["position"][20, 1] == 20.0  # the original is left as it was


def test_first_overlap_with_the_logged_ego():
    # the ego drives east along y = 0 and the adversary south down x = 30: both reach (30, 0) 2 s after step 10
    s = make_scene({"0": track((10, 0), (10, 0)), "1": track((30, 20), (0, -10), heading=-np.pi / 2)})
    step = first_overlap(s, 1, 0)
    assert step is not None and 25 <= step <= 32
    far = make_scene({"0": track((10, 0), (10, 0)), "1": track((30, 300), (0, -10), heading=-np.pi / 2)})
    assert first_overlap(far, 1, 0) is None


def test_the_index_is_kept_next_to_the_rule_folder(tmp_path):
    import json

    from scripts.responsibility.export_adv_scenes import open_index

    out = tmp_path / "cat"
    out.mkdir()
    (out / "index.json").write_text(json.dumps({"0": {"rule": "cat"}}))  # an older export's layout
    path, index = open_index(out)
    assert path == tmp_path / "cat.index.json" and index == {"0": {"rule": "cat"}}
    assert list(out.iterdir()) == []  # only scenes stay inside: MetaDrive asserts every file is one
    assert open_index(tmp_path / "fair1_0.3")[1] == {}
