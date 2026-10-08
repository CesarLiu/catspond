import torch

from gpucat.ppo import gae


def test_gae_does_not_bootstrap_across_an_episode_end():
    # one world, three steps: the episode ends at step 1, a new one runs at step 2
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.5], [0.5], [0.5]])
    ended = torch.tensor([[False], [True], [False]])
    alive = torch.ones(3, 1, dtype=torch.bool)
    adv, ret = gae(rewards, values, ended, alive, last_value=torch.tensor([10.0]), gamma=0.9, lam=1.0)
    # step 2 bootstraps from the last value; step 1 ends (no future); step 0 sees step 1 only
    assert torch.isclose(adv[2, 0], torch.tensor(3.0 + 0.9 * 10.0 - 0.5))
    assert torch.isclose(adv[1, 0], torch.tensor(2.0 - 0.5))
    assert torch.isclose(adv[0, 0], torch.tensor(1.0 + 0.9 * 0.5 - 0.5 + 0.9 * (2.0 - 0.5)))
    assert torch.allclose(ret, adv + values)
