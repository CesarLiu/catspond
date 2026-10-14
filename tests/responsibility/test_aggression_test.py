"""Tests for responsibility/aggression_test.py: the condition table is
consistent, AUROC and the scene scores compute what the scripts report."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from responsibility.aggression_test import (
    CONDITIONS, CONFIGS, SCENE_SETS, at_setting, auroc, boot_auroc, first_overlaps, load_offline, longest_run,
    pair_scores, scene_dir)


def test_conditions_are_consistent():
    for name, c in CONDITIONS.items():
        assert c.scene_set in SCENE_SETS
        assert c.agent in ("sdc", "adv")
        partner = CONDITIONS[c.partner]
        assert partner.partner == name, "partners pair up"
        assert partner.scene_set == c.scene_set and partner.agent != c.agent
        if c.reference is None:
            assert c.scene_set == "logged"
        else:
            ref = CONDITIONS[c.reference]
            assert ref.reference is None and ref.agent == c.agent
    assert {"paper", "nms_ooi"} <= set(CONFIGS)


def test_scene_dir(tmp_path):
    assert str(scene_dir("logged", tmp_path, "raw", "cat")) == "raw"
    assert str(scene_dir("cat", tmp_path, "raw", "advs/cat")) == "advs/cat"
    assert scene_dir("near1_0.5", tmp_path, "raw", "cat") == tmp_path / "scenes" / "near1_0.5"


def test_auroc_matches_mann_whitney_with_ties():
    rng = np.random.default_rng(0)
    pos, neg = rng.integers(0, 5, 40).astype(float), rng.integers(0, 4, 30).astype(float)
    u = mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg))
    assert auroc(pos, neg) == pytest.approx(u, abs=1e-12)
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([1, 1], [1, 1]) == 0.5
    assert np.isnan(auroc([], [1.0]))


def test_boot_auroc_interval_brackets_the_point():
    rng = np.random.default_rng(1)
    pos = pd.DataFrame({"scene": np.repeat(np.arange(20), 3), "x": rng.normal(1, 1, 60)})
    neg = pd.DataFrame({"scene": np.repeat(np.arange(20), 3), "x": rng.normal(0, 1, 60)})
    p, lo, hi = boot_auroc(pos, neg, "x", n=200)
    assert lo <= p <= hi and p > 0.6
    # one shared generator makes a series reproducible
    g1, g2 = np.random.default_rng(5), np.random.default_rng(5)
    assert boot_auroc(pos, neg, "x", n=50, rng=g1) == boot_auroc(pos, neg, "x", n=50, rng=g2)


def test_longest_run():
    assert longest_run([0, 1, 1, 0, 1, 1, 1, 0], 0.1) == pytest.approx(0.3)
    assert longest_run([], 0.5) == 0.0


def _table(rows):
    return pd.DataFrame(rows, columns=["scene", "step", "other", "H", "alpha", "dsat", "beta", "gap_actual",
                                       "gap_alt_mean", "overlap", "speed"])


def test_pair_scores_two_sided():
    # one scene, two windows; the agent's beta exceeds its partner's by 1.0 at step 15
    a = at_setting(_table([(0, 10, "b", 20, .1, 10, 0.2, 5.0, 6, -1, 9), (0, 15, "b", 20, .1, 10, 1.5, 2.0, 4, -1, 9)]),
                   20, .1, 10)
    b = at_setting(_table([(0, 10, "a", 20, .1, 10, 0.1, 5.0, 6, -1, 9), (0, 15, "a", 20, .1, 10, 0.5, 2.0, 4, -1, 9)]),
                   20, .1, 10)
    s = pair_scores(a, b).loc[0]
    assert s.beta == 1.5 and s.rel == pytest.approx(1.0) and s.gap == 2.0 and s.prox == -2.0
    assert s.beta_min == 0.2 and s.rel_min == pytest.approx(0.1)


def test_load_offline_drops_windows_from_the_overlap(tmp_path):
    (tmp_path / "offline").mkdir()
    _table([(0, 10, "b", 20, .1, 10, 0.1, 5, 6, 15, 9), (0, 15, "b", 20, .1, 10, 0.2, 0, 6, 15, 9),
            (1, 10, "b", 20, .1, 10, 0.3, 5, 6, -1, 9)]).to_csv(tmp_path / "offline" / "cfg__adv_cat.csv.gz",
                                                               index=False)
    d = load_offline(tmp_path, "cfg", "adv_cat")
    assert list(zip(d.scene, d.step)) == [(0, 10), (1, 10)]
    assert load_offline(tmp_path, "cfg", "missing") is None


def test_first_overlaps_reads_the_sidecar_index(tmp_path):
    folder = tmp_path / "near1_0.5"
    folder.mkdir()
    (tmp_path / "near1_0.5.index.json").write_text(json.dumps({
        "0": {"first_overlap_with_logged_ego": 53}, "1": {"first_overlap_with_logged_ego": None}}))
    assert first_overlaps(folder) == {0: 53}
