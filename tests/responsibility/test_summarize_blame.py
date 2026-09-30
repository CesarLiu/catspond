import csv

import pytest

from responsibility.blame_reward import LOG_FIELDS
from scripts.responsibility import summarize_blame as sb


def _log(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for steps, verdict, rule, weight in rows:
            writer.writerow({k: "" for k in LOG_FIELDS} | {"total_steps": steps, "verdict": verdict, "rule": rule,
                                                           "weight": weight, "seconds": 1.5})


def test_training_attributions_are_summarised(tmp_path):
    # early on the collisions are mostly the other's fault, later the ego's
    _log(tmp_path / "cat_share_s0.csv", [(100, "other", "other", 0.0), (200, "other", "ego", 0.1),
                                         (300, "shared", "n/a", 1.0), (900, "ego", "ego", 0.9),
                                         (1000, "ego", "n/a", 1.0)])
    _log(tmp_path / "empty_s0.csv", [])
    out = tmp_path / "summary.md"
    summaries = sb.main(["--logs", str(tmp_path / "cat_share_s0.csv"), str(tmp_path / "empty_s0.csv"),
                         "--bins", "2", "--out", str(out)])
    s = summaries[0]
    assert s["run"] == "cat_share_s0" and s["collisions"] == 5
    assert s["mean_weight"] == pytest.approx(0.6) and s["full_penalty"] == pytest.approx(0.4)
    assert s["mostly_other"] == pytest.approx(0.4) and s["verdict_other"] == pytest.approx(0.4)
    assert s["rule_agreement"] == pytest.approx(2 / 3)  # of the three both decide
    assert s["trend"] == pytest.approx([(0.0 + 0.1 + 1.0) / 3, (0.9 + 1.0) / 2])
    assert summaries[1]["collisions"] == 0
    assert "| cat_share_s0 | 5 | 0.60 |" in out.read_text(encoding="utf-8")
