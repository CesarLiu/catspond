import numpy as np
import pytest
import torch

pytest.importorskip("tensorflow")  # the numpy references import CAT's advgen

from advgen.adv_generator import get_polyline_vel, get_polyline_yaw  # noqa: E402
from gpucat import adversary as ga  # noqa: E402
from responsibility.adversarial import cat_candidate_probs, cat_collision_scores, select  # noqa: E402

f = torch.float64


def paths(rng, n, steps, turn=0.15):
    """n wiggly paths [n, steps, 2] heading roughly east from near the origin."""
    heading = np.cumsum(rng.normal(0, turn, (n, steps)), axis=1) + rng.uniform(-np.pi, np.pi, (n, 1))
    speed = rng.uniform(0.2, 1.5, (n, 1))
    step = np.stack([np.cos(heading), np.sin(heading)], -1) * speed[..., None]
    return np.cumsum(step, axis=1) + rng.normal(0, 3, (n, 1, 2))


def test_polyline_yaw_matches_cat_across_the_pi_wrap():
    rng = np.random.default_rng(0)
    trajs = paths(rng, 20, 80, turn=0.4)  # headings wander across +-pi
    mine = ga.polyline_yaw(torch.as_tensor(trajs, dtype=f)).numpy()
    for t, m in zip(trajs, mine):
        assert np.allclose(get_polyline_yaw(t), m, atol=1e-12)
    assert np.allclose(ga.polyline_vel(torch.as_tensor(trajs[0], dtype=f)).numpy(), get_polyline_vel(trajs[0]))


def test_scores_match_cat_for_ego_trajectories_of_any_length():
    rng = np.random.default_rng(1)
    for _ in range(5):
        cand = paths(rng, 32, 80) * 0.5
        lengths = [80, 61, 33, 12, 47]  # a 2-D boolean mask once broke the yaw of the shorter ones
        trajs = [paths(rng, 1, n)[0] * 0.5 for n in lengths]
        probs_av = rng.uniform(0.2, 1.0, 5)
        log_scores = np.sort(rng.normal(0, 1, 32))[::-1].copy()
        ov, av = {"l": 4.6, "w": 2.0}, {"l": 5.2, "w": 2.2}
        np_score, np_md = cat_collision_scores(cand, cat_candidate_probs(log_scores), trajs, probs_av, ov, av)
        pad = np.zeros((5, 80, 2))
        for i, (t, n) in enumerate(zip(trajs, lengths)):
            pad[i, :n] = t
        score, md = ga.cat_scores(torch.as_tensor(cand, dtype=f)[None],
                                  ga.candidate_probs(torch.as_tensor(log_scores, dtype=f))[None],
                                  torch.as_tensor(pad, dtype=f)[None], torch.as_tensor(lengths)[None],
                                  torch.as_tensor(probs_av, dtype=f)[None], torch.tensor([[4.6, 2.0]], dtype=f),
                                  torch.tensor([[5.2, 2.2]], dtype=f))
        assert np.allclose(score[0].numpy(), np_score, atol=1e-12) and np.array_equal(md[0].numpy(), np_md)
        assert int(ga.select("cat", score, md)[0]) == select("cat", np_score, np_md, None)[0]


def test_fair_rule_matches_numpy():
    rng = np.random.default_rng(2)
    for _ in range(200):
        score = np.where(rng.random(32) < 0.3, rng.random(32), 0.0)
        md = rng.integers(0, 20, 32).astype(float)
        beta = rng.normal(1.5, 2.0, 32)
        avoid = rng.choice([0.0, 0.05, 0.2, 0.6, 1.0], 32)
        expect = select("fair", score, md, beta, 2.0, avoid=avoid, min_avoid=0.1)[0]
        t = lambda x: torch.as_tensor(x, dtype=f)[None]  # noqa: E731
        assert int(ga.select("fair", t(score), t(md), t(beta), t(avoid), 2.0, 0.1)[0]) == expect


def test_plan_matches_cat():
    rng = np.random.default_rng(3)
    past, future = paths(rng, 1, 11)[0], paths(rng, 1, 80)[0]
    pos = np.concatenate([past, future])
    ref = np.concatenate([pos, get_polyline_vel(pos), get_polyline_yaw(pos).reshape(-1, 1)], 1)
    mine = ga.plan(torch.as_tensor(past, dtype=f)[None], torch.as_tensor(future, dtype=f)[None])[0].numpy()
    assert np.allclose(mine, ref, atol=1e-12)
