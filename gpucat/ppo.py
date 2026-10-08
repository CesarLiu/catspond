"""PPO for the SDC policy (gpucat/PLAN.md, M2): a Gaussian policy over the
two actions in [-1, 1] (state-independent log std, actions clipped by the
env) and a value function, separate two-layer MLPs; GAE over each world's
episode, the transitions after a world's episode ended masked out.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PPOSettings:
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 4096
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    init_log_std: float = -0.5


def mlp(i, h, o, out_scale=1.0):
    net = nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 2 ** 0.5)
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, out_scale)
    return net


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int = 2, hidden: int = 256, init_log_std: float = -0.5):
        super().__init__()
        self.actor = mlp(obs_dim, hidden, act_dim, 0.01)
        self.critic = mlp(obs_dim, hidden, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))

    def dist(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())

    def act(self, obs, deterministic: bool = False):
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a, d.log_prob(a).sum(-1), self.critic(obs).squeeze(-1)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def gae(rewards, values, ended, alive, last_value, gamma, lam):
    """Advantages and returns [T, W] for lockstep episodes: alive[t] marks the
    transitions to learn from, ended[t] the ones that end an episode (no
    bootstrap past them; a world still alive after the last step bootstraps
    from last_value)."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    nxt_adv = torch.zeros_like(last_value)
    nxt_value = last_value
    for t in reversed(range(T)):
        cont = (~ended[t]).float()
        delta = rewards[t] + gamma * nxt_value * cont - values[t]
        nxt_adv = delta + gamma * lam * cont * nxt_adv
        nxt_adv = torch.where(alive[t], nxt_adv, torch.zeros_like(nxt_adv))
        adv[t] = nxt_adv
        nxt_value = torch.where(alive[t], values[t], nxt_value)
    return adv, adv + values


class PPO:
    def __init__(self, obs_dim: int, settings: PPOSettings, device: str = "cuda"):
        self.s = settings
        self.net = ActorCritic(obs_dim, 2, settings.hidden, settings.init_log_std).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=settings.lr, eps=1e-5)

    def update(self, obs, actions, logp, values, rewards, ended, alive, last_value):
        """One PPO update from a batch [T, W, ...] of lockstep episodes."""
        s = self.s
        adv, ret = gae(rewards, values, ended, alive, last_value, s.gamma, s.lam)
        m = alive.flatten()
        o, a, lp, ad, rt = obs.flatten(0, 1)[m], actions.flatten(0, 1)[m], logp.flatten()[m], adv.flatten()[m], ret.flatten()[m]
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        n = len(o)
        stats = {"samples": n}
        for _ in range(s.epochs):
            for idx in torch.randperm(n, device=o.device).split(s.minibatch):
                d = self.net.dist(o[idx])
                new_lp = d.log_prob(a[idx]).sum(-1)
                ratio = (new_lp - lp[idx]).exp()
                pg = -torch.min(ratio * ad[idx], ratio.clamp(1 - s.clip, 1 + s.clip) * ad[idx]).mean()
                v = self.net.value(o[idx])
                vf = 0.5 * (v - rt[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + s.value_coef * vf - s.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), s.max_grad_norm)
                self.opt.step()
        with torch.no_grad():
            stats.update(policy_loss=float(pg), value_loss=float(vf), entropy=float(ent),
                         clip_frac=float(((ratio - 1).abs() > s.clip).float().mean()),
                         log_std=self.net.log_std.detach().cpu().tolist())
        return stats
