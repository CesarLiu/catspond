import csv
import sys

import pytest

from scripts.responsibility import summarize_responsibility as summ

FIELDS = ["scene", "scenario_id", "agent_id", "step", "time", "speed", "safety", "courtesy",
          "n_neighbours", "safety_against", "courtesy_toward"]


def test_calibration_ignores_non_interacting_windows_and_respects_the_floor():
    values = [0.0] * 50 + [0.2, 0.4, 0.6, 0.8, 1.0]
    assert summ.calibrate(values, 0.5, floor=0.1) == pytest.approx(0.6)
    assert summ.calibrate([0.0, 0.01], 0.9, floor=0.1) == 0.1
    assert summ.calibrate([0.0, 0.0], 0.9, floor=0.05) == 0.05


def _write(run, rows):
    run.mkdir(parents=True)
    with open(run / "windows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for scene, step, speed, safety, courtesy in rows:
            writer.writerow(dict(scene=scene, scenario_id=f"id{scene}", agent_id="7", step=step, time=step / 10,
                                 speed=speed, safety=safety, courtesy=courtesy, n_neighbours=1,
                                 safety_against="3", courtesy_toward="3"))


def test_scenes_are_flagged_by_threshold(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    _write(run, [
        ("0", 10, 5.0, 0.05, 0.001), ("0", 20, 5.0, 0.02, 0.002),  # calm
        ("1", 10, 5.0, 0.90, 0.001),  # gave up margin
        ("2", 10, 5.0, 0.01, 0.300),  # imposed on a neighbour
        ("3", 10, 0.2, 5.00, 5.000),  # standing still: not judged
    ])
    monkeypatch.setattr(sys, "argv", ["x", "--run", str(run), "--safety-threshold", "0.5",
                                      "--courtesy-threshold", "0.1"])
    summ.main()
    with open(run / "summary" / "scenes.csv") as f:
        verdict = {r["scene"]: (int(r["aggressive"]), int(r["safety_flagged"]), int(r["courtesy_flagged"]))
                   for r in csv.DictReader(f)}
    assert verdict == {"0": (0, 0, 0), "1": (1, 1, 0), "2": (1, 0, 1)}
    assert (run / "summary" / "responsibility.png").exists()
    assert "aggressive scenes: 2/3" in capsys.readouterr().out
