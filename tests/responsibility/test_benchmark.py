"""benchmark_advgen's trade-off summary, recommendation and plot, on
synthetic rows (no DenseTNT)."""

import pytest

from scripts.responsibility import benchmark_advgen as bench


def _rows():
    """10 scenes; per rule, how many chosen trajectories collide, how many of
    those are unavoidable (avoid < 0.1), and the adversary's beta."""
    table = {  # rule: (collisions, unavoidable, beta)
        "cat": (10, 3, 4.0),
        "constrained@1": (4, 1, 0.5),
        "fair@0.5,0.3": (2, 0, 0.1),  # too few collisions to train with
        "fair@1,0.3": (4, 0, 0.4),
        "fair@2,0.3": (5, 1, 0.8),
        "fair@1,0.5": (3, 0, 0.3),
        "fair@2,0.1": (6, 0, 0.9),
    }
    rows = []
    for rule, (hits, unavoidable, beta) in table.items():
        for scene in range(10):
            hit = scene < hits
            avoid = 0.05 if scene < unavoidable else 0.8
            rows.append({"scene": str(scene), "rule": rule, "collision": int(hit), "beta": beta, "avoid": avoid,
                         "logged_adversary_beta": 0.1 * scene, "logged_adversary_avoid": 0.9})
    return rows


def test_parse_rule():
    assert bench.parse_rule("cat") == ("cat", None, None)
    assert bench.parse_rule("constrained@0.5") == ("constrained", 0.5, None)
    assert bench.parse_rule("fair@1,0.3") == ("fair", 1.0, 0.3)


def test_rule_stats_and_recommendation(capsys):
    stats = {s["rule"]: s for s in bench.rule_stats(_rows())}
    assert stats["cat"]["collision"] == 1.0 and stats["cat"]["unavoidable"] == pytest.approx(0.3)
    assert stats["fair@2,0.3"]["tau"] == 2.0 and stats["fair@2,0.3"]["rho"] == 0.3
    best = bench.recommend(list(stats.values()), min_collision=0.3)
    # at least 30% collisions, no unavoidable ones, least responsible adversary first
    assert [s["rule"] for s in best] == ["fair@1,0.5", "fair@1,0.3"]
    bench.summarize(_rows(), 0.3)
    out = capsys.readouterr().out
    assert "TAU=1 RHO=0.5" in out and "logged adversary beta" in out
    bench.summarize(_rows(), 0.9)
    assert "no fair setting keeps" in capsys.readouterr().out


def test_the_tradeoff_plot_is_written(tmp_path):
    bench.plot_tradeoff(_rows(), tmp_path / "tradeoff.png")
    assert (tmp_path / "tradeoff.png").stat().st_size > 1000
    # older benchmark CSVs have no avoidability: the plot still works
    rows = [{k: v for k, v in r.items() if k not in ("avoid", "logged_adversary_avoid")} for r in _rows()]
    bench.plot_tradeoff(rows, tmp_path / "old.png")
    assert (tmp_path / "old.png").exists()
