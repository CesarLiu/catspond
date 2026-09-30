import csv
import json

import numpy as np
import pytest

from scripts.responsibility import compare_models as cm

FIELDS = ["scene", "scenario_id", "agent_id", "step", "time", "speed", "safety", "courtesy",
          "n_neighbours", "safety_against", "courtesy_toward"]


def _run(path, model, values):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model": {"name": model}}))
    with open(path / "windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for (scene, step, speed), (s, c) in values.items():
            w.writerow(dict(scene=scene, scenario_id="x", agent_id="0", step=step, time=step / 10, speed=speed,
                            safety=s, courtesy=c, n_neighbours=1, safety_against="1", courtesy_toward="1"))
    return path


def _values(rng, n_scenes=20):
    out = {}
    for scene in range(n_scenes):
        for step in range(10, 71, 10):
            out[(str(scene), step, 5.0)] = (float(rng.normal(0, 1)), float(abs(rng.normal(0, 0.2))))
    return out


def test_kappa():
    assert cm.kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cm.kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert np.isnan(cm.kappa([1, 1], [1, 1]))


def test_models_that_rank_alike_agree(tmp_path):
    rng = np.random.default_rng(0)
    a = _values(rng)
    # a sign-preserving monotone transform: same ranks, same flags on each run's own thresholds
    b = {k: (2.0 * s, 3.0 * c) for k, (s, c) in a.items()}
    b[("99", 10, 5.0)] = (0.0, 0.0)  # a window only B has
    ra, rb = _run(tmp_path / "a", "densetnt", a), _run(tmp_path / "b", "mtr", b)
    out, _ = cm.main(["--runs", str(ra), str(rb), "--out-dir", str(tmp_path / "out"), "--safety-floor", "0.0",
                      "--courtesy-floor", "0.0"])
    assert out["matched"] == len(a) and out["only_b"] == 1 and out["only_a"] == 0
    assert out["spearman_safety"] == pytest.approx(1.0) and out["spearman_courtesy"] == pytest.approx(1.0)
    assert out["aggressive_kappa"] == pytest.approx(1.0) and out["timid_kappa"] == pytest.approx(1.0)
    assert out["scene_agreement"] == pytest.approx(1.0)
    text = (tmp_path / "out" / "model_comparison.md").read_text(encoding="utf-8")
    assert text.startswith("# densetnt vs mtr")
    assert (tmp_path / "out" / "model_comparison.png").exists()


def test_unrelated_models_do_not(tmp_path):
    rng = np.random.default_rng(1)
    a, b = _values(rng, 40), _values(rng, 40)  # independent values
    out, _ = cm.main(["--runs", str(_run(tmp_path / "a", "densetnt", a)), str(_run(tmp_path / "b", "mtr", b)),
                      "--out-dir", str(tmp_path / "out")])
    assert abs(out["spearman_safety"]) < 0.2 and abs(out["aggressive_kappa"]) < 0.3


def test_policy_tables(tmp_path):
    cols = ["run", "crash_rate", "aggressive", "timid", "aggressive_x_ref", "timid_x_ref"]

    def table(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)
        return path

    ta = table(tmp_path / "a.csv", [["replay/none", 0, 0.1, 0.1, 1.0, 1.0], ["td3/none", 0.1, 0.05, 0.4, 0.5, 4.0],
                                    ["td3/cat", 0.2, 0.3, 0.2, 3.0, 2.0]])
    tb = table(tmp_path / "b.csv", [["replay/none", 0, 0.1, 0.1, 1.0, 1.0], ["td3/none", 0.1, 0.06, 0.3, 0.6, 3.0],
                                    ["td3/cat", 0.2, 0.2, 0.25, 2.0, 2.5]])
    runs, rows = cm.compare_policy_tables(ta, tb)
    by = {r["column"]: r for r in rows}
    assert runs == ["replay/none", "td3/none", "td3/cat"]
    assert by["timid_x_ref"]["above_ref_agreement"] == pytest.approx(1.0)
    assert by["aggressive"]["spearman"] == pytest.approx(1.0)
