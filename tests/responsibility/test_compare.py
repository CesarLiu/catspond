import csv
import json
import pickle

import numpy as np
import pytest

from responsibility.hmm import GaussianHMM
from responsibility.rollouts import make_rollout, save_rollout
from scripts.responsibility import compare_policies as cmp

WINDOW_FIELDS = ["scene", "scenario_id", "agent_id", "step", "time", "speed", "safety", "courtesy",
                 "n_neighbours", "safety_against", "courtesy_toward"]
CRASH_FIELDS = ["scene", "policy", "adv_mode", "crash_step", "window", "other_id", "other_type", "adversary",
                "beta_ego", "beta_other", "share", "verdict", "rule"]


def _rollouts(directory, policy, mode, ends):
    for scene, (reason, completion) in enumerate(ends):
        n = 30
        rollout = make_rollout(str(scene), f"s{scene}", policy, mode,
                               ego={"position": np.zeros((n, 2)), "heading": np.zeros(n)},
                               end={"step": n - 1, "reason": reason, "route_completion": completion,
                                    "crash_vehicle": reason == "crash_vehicle", "arrive_dest": reason == "arrive_dest"})
        save_rollout(rollout, directory / f"{scene}.pkl")


def _run(tmp_path, name, policy, mode, ends, windows, crashes=()):
    rollouts = tmp_path / "rollouts" / name
    _rollouts(rollouts, policy, mode, ends)
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"rollouts": str(rollouts)}))
    with open(run / "windows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
        writer.writeheader()
        for scene, step, speed, safety, courtesy in windows:
            writer.writerow(dict(scene=scene, scenario_id="x", agent_id="0", step=step, time=step / 10, speed=speed,
                                 safety=safety, courtesy=courtesy, n_neighbours=1, safety_against="1",
                                 courtesy_toward="1"))
    if crashes:
        with open(run / "crashes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CRASH_FIELDS)
            writer.writeheader()
            for scene, verdict, rule in crashes:
                writer.writerow(dict(scene=scene, policy=policy, adv_mode=mode, crash_step=29, window=10,
                                     other_id="1", other_type="VEHICLE", adversary=0, beta_ego=1.0,
                                     beta_other=0.0, share=1.0, verdict=verdict, rule=rule))
    return run


def test_policies_are_compared_against_the_replayed_log(tmp_path, capsys):
    # the replayed log: 10 judged windows, one beyond each threshold
    logged = [("0", 10 + i, 5.0, s, 0.0) for i, s in enumerate([-1.0, -0.2, -0.1, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 1.0])]
    replay = _run(tmp_path, "replay", "replay", "none", [("arrive_dest", 1.0), ("arrive_dest", 1.0)], logged)
    # a timid policy: keeps huge margins, stops half the time, never crashes
    timid = [("0", 10 + i, 5.0 if i % 2 else 0.0, -3.0, 0.0) for i in range(10)]
    slow = _run(tmp_path, "slow", "td3", "none", [("max_step", 0.4), ("max_step", 0.6)], timid)
    # an aggressive policy: gives up margin and crashes, once at fault; the
    # rear-end rule agrees on one collision, disagrees on one, is silent on one
    rash = _run(tmp_path, "rash", "td3", "cat", [("crash_vehicle", 0.3), ("crash_vehicle", 0.5),
                                                  ("crash_vehicle", 0.5)],
                [("0", 10 + i, 8.0, 2.0, 0.0) for i in range(10)],
                crashes=[("0", "ego", "ego"), ("1", "other", "ego"), ("2", "other", "n/a")])

    rows = cmp.main(["--runs", str(replay), str(slow), str(rash), "--out-dir", str(tmp_path / "cmp"),
                     "--quantile", "0.8", "--timid-quantile", "0.5"])
    by = {r["run"]: r for r in rows}
    assert list(by) == ["replay/none", "td3/none", "td3/cat"]
    assert by["replay/none"]["crash_rate"] == 0 and by["replay/none"]["route_completion"] == 1.0
    assert by["td3/none"]["stopped"] == pytest.approx(0.5)
    assert by["td3/none"]["timid"] == 1.0 and by["td3/none"]["aggressive"] == 0.0
    assert by["td3/cat"]["crash_rate"] == 1.0 and by["td3/cat"]["ego_fault_share"] == pytest.approx(1 / 3)
    assert by["td3/cat"]["rule_agreement"] == 0.5 and by["td3/cat"]["rule_coverage"] == pytest.approx(2 / 3)
    assert by["td3/cat"]["aggressive"] == 1.0
    ref_timid = by["replay/none"]["timid"]
    assert 0 < ref_timid < 1 and by["td3/none"]["timid_x_ref"] == pytest.approx(1.0 / ref_timid)
    assert (tmp_path / "cmp" / "comparison.png").exists()
    assert "| td3/cat |" in (tmp_path / "cmp" / "comparison.md").read_text(encoding="utf-8")


def test_level_shares_come_from_the_given_hmm(tmp_path):
    windows = [("0", 10 + i, 5.0, s, 0.0) for i, s in enumerate([0.0] * 5 + [2.0] * 5)]
    run = _run(tmp_path, "replay", "replay", "none", [("arrive_dest", 1.0)], windows)
    hmm = GaussianHMM(n_states=2, n_features=2)
    hmm.startprob_ = np.array([0.5, 0.5])
    hmm.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    hmm.means_ = np.array([[0.0, 0.0], [2.0, 0.0]])
    hmm.covars_ = np.array([[0.1, 0.1], [0.1, 0.1]])
    with open(tmp_path / "hmm.pkl", "wb") as f:
        pickle.dump({"hmm": hmm, "log_courtesy": False, "aggressive_levels": [1]}, f)
    rows = cmp.main(["--runs", str(run), "--hmm", str(tmp_path / "hmm.pkl"), "--out-dir", str(tmp_path / "cmp")])
    assert rows[0]["level_0"] == pytest.approx(0.5) and rows[0]["aggressive_levels"] == pytest.approx(0.5)
