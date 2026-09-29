import csv
import pickle
import sys

import numpy as np
import pytest

from responsibility.hmm import GaussianHMM
from responsibility.levels import aggressiveness, assign_levels, relabel_by_aggressiveness
from responsibility.results import features, read_windows, sequences
from scripts.responsibility import fit_levels
from tests.responsibility.test_summarize import FIELDS

CALM, PUSHY = np.array([0.0, 0.01]), np.array([1.2, 0.8])


def _sequences(n_seq=40, seq_len=10, seed=0):
    """Sequences that switch between a calm and an aggressive regime."""
    rng = np.random.default_rng(seed)
    out, truth = [], []
    for _ in range(n_seq):
        z = np.zeros(seq_len, dtype=int)
        start = rng.integers(0, seq_len)
        z[start: start + rng.integers(1, 4)] = 1
        obs = np.where(z[:, None] == 1, PUSHY, CALM) + rng.normal(scale=0.05, size=(seq_len, 2))
        out.append(obs)
        truth.append(z)
    return out, truth


def test_relabelling_orders_levels_and_keeps_the_model():
    seqs, _ = _sequences()
    hmm = GaussianHMM(n_states=2, n_features=2, seed=3)
    hmm.fit(seqs, n_iter=100)
    before = sum(hmm.score(s) for s in seqs)
    scale = np.concatenate(seqs).std(axis=0)
    relabel_by_aggressiveness(hmm, scale)
    assert np.all(np.diff(aggressiveness(hmm, scale)) > 0)
    assert np.allclose(hmm.means_[0], CALM, atol=0.1) and np.allclose(hmm.means_[1], PUSHY, atol=0.1)
    assert sum(hmm.score(s) for s in seqs) == pytest.approx(before)
    assert np.allclose(hmm.transmat_.sum(-1), 1.0)


def test_assigned_levels_follow_the_regime():
    seqs, truth = _sequences()
    hmm = GaussianHMM(n_states=2, n_features=2, seed=1)
    hmm.fit(seqs, n_iter=100)
    relabel_by_aggressiveness(hmm, np.concatenate(seqs).std(axis=0))
    labelled = assign_levels(hmm, seqs)
    accuracy = np.mean(np.concatenate([lv == z for (lv, _), z in zip(labelled, truth)]))
    assert accuracy > 0.95
    assert all(((p > 0) & (p <= 1)).all() for _, p in labelled)


def _write_run(path, name, rows):
    path.mkdir(parents=True, exist_ok=True)
    with open(path / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for scene, step, safety, courtesy in rows:
            writer.writerow(dict(scene=scene, scenario_id=f"id{scene}", agent_id="7", step=step, time=step / 10,
                                 speed=5.0, safety=safety, courtesy=courtesy, n_neighbours=1,
                                 safety_against="3", courtesy_toward="3"))


def test_shards_are_merged_and_repeated_windows_kept_once(tmp_path):
    _write_run(tmp_path, "windows.shard-0-of-2.csv", [("0", 10, 0.1, 0.0), ("2", 10, 0.3, 0.0), ("0", 20, 0.2, 0.0)])
    _write_run(tmp_path, "windows.shard-1-of-2.csv", [("1", 10, 0.5, 0.0), ("10", 10, 0.6, 0.0)])
    # an interrupted scene written twice: the later copy wins
    _write_run(tmp_path, "windows.shard-9-of-9.csv", [("2", 10, 0.9, 0.0)])
    rows = read_windows(tmp_path)
    assert [(r["scene"], r["step"]) for r in rows] == [("0", 10), ("0", 20), ("1", 10), ("2", 10), ("10", 10)]
    assert [r["safety"] for r in rows if r["scene"] == "2"] == [0.9]
    seqs = sequences(rows, "run")
    assert [r["step"] for r in seqs[("run", "0", "7")]] == [10, 20]
    assert features(seqs[("run", "0", "7")], log_courtesy=True).shape == (2, 2)


def test_fit_levels_end_to_end(tmp_path, monkeypatch, capsys):
    seqs, _ = _sequences(n_seq=30, seq_len=7, seed=2)
    rows = [(str(i), 10 + 10 * t, float(s[t, 0]), float(max(s[t, 1], 0.0))) for i, s in enumerate(seqs) for t in range(7)]
    _write_run(tmp_path / "sdc", "windows.csv", rows)
    _write_run(tmp_path / "policy", "windows.csv", [("0", 10, 0.0, 0.0), ("0", 20, 1.3, 0.9), ("1", 10, 0.0, 0.01)])
    out = tmp_path / "levels"
    monkeypatch.setattr(sys, "argv", ["x", "--runs", str(tmp_path / "sdc"), str(tmp_path / "policy"),
                                      "--fit-runs", str(tmp_path / "sdc"), "--out-dir", str(out),
                                      "--n-states", "1", "2", "3"])
    fit_levels.main()
    saved = pickle.load(open(out / "hmm.pkl", "rb"))
    h = saved["hmm"].n_states
    assert h >= 2 and 0 not in saved["aggressive_levels"] and h - 1 in saved["aggressive_levels"]
    with open(out / "scenes_levels.csv") as f:
        policy = {r["scene"]: int(r["aggressive"]) for r in csv.DictReader(f) if r["run"] == "policy"}
    assert policy == {"0": 1, "1": 0}
    assert (out / "levels.png").exists() and "of windows in aggressive levels" in capsys.readouterr().out


def test_levels_report_what_they_are_elevated_in():
    from responsibility.levels import elevated_in

    hmm = GaussianHMM(n_states=4, n_features=2)
    hmm.means_ = np.array([[0.0, 0.0], [0.02, 0.01], [1.0, 0.05], [0.9, 1.1]])
    assert elevated_in(hmm, np.array([0.5, 0.5])) == ["-", "-", "safety", "both"]
    from responsibility.levels import elevated_levels

    assert elevated_levels(hmm, np.array([0.5, 0.5])) == [2, 3]
