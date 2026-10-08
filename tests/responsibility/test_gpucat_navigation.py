"""CatEnv.navigation_obs on a stand-in world (no GPUDrive): the route ahead
in the SDC's frame and the ray distances to the edges and yellow lines."""

import math

import torch

from gpucat.env import RAY_RANGE, RAYS, ROUTE_AHEAD, CatEnv, EnvSettings


def test_navigation_obs_in_the_sdc_frame():
    env = object.__new__(CatEnv)
    env.s, env.device = EnvSettings(worlds=1), "cpu"
    env.route = torch.tensor([[[0.0, 0.0], [100.0, 0.0]]])
    env.route_n, env.route_len = torch.tensor([2]), torch.tensor([100.0])
    env.edges, env.edges_n = torch.tensor([[[0.0, 6.0, 100.0, 6.0]]]), torch.tensor([1])  # 4 m to the left
    env.yellow, env.yellow_n = torch.tensor([[[0.0, -3.0, 100.0, -3.0]]]), torch.tensor([1])  # 5 m to the right
    env._ego = lambda: (torch.tensor([[10.0, 2.0]]), torch.tensor([math.pi / 2]))  # facing left, 2 m off the route
    o = env.navigation_obs()
    k = len(ROUTE_AHEAD)
    assert o.shape == (1, 2 * k + 2 + 2 * RAYS) and o.dtype == torch.float32
    local = o[0, :2 * k].view(k, 2) * RAY_RANGE
    assert torch.allclose(local[0], torch.tensor([-2.0, -5.0]), atol=1e-5)  # 5 m along the route: behind and right
    assert math.isclose(float(o[0, 2 * k]), 0.2, abs_tol=1e-6) and math.isclose(float(o[0, 2 * k + 1]), 0.1, abs_tol=1e-6)
    edge, yellow = o[0, 2 * k + 2:2 * k + 2 + RAYS] * RAY_RANGE, o[0, 2 * k + 2 + RAYS:] * RAY_RANGE
    assert math.isclose(float(edge[0]), 4.0, abs_tol=1e-4) and math.isclose(float(edge[RAYS // 2]), RAY_RANGE)
    assert math.isclose(float(yellow[RAYS // 2]), 5.0, abs_tol=1e-4)
