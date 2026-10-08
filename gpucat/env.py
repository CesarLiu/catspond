"""CAT's training environment on GPUDrive: W worlds stepped together, each
the SDC under policy control in one of CAT's scenes, everyone else replayed
from the log, and -- in an adversarial episode -- CAT's adversary driving the
plan CAT (or the fair rule) chooses against the SDC's recent behaviour in
that scene (gpucat/adversary.py), written into GPUDrive's trajectory tensor.

Training resets each world on its own: a world whose episode ends starts
the next at once, in the same scene; evaluation runs in lockstep, one
episode a world (begin_episode). Scene batches are swapped by the caller
(load); CAT samples a scene per episode, here a batch of W distinct
training scenes stays for many episodes.

What follows MetaDrive's ScenarioEnv as CAT configures it, measured with
full-size boxes (GPUDrive's own collision boxes are 0.7 of the size):

  reward      driving_reward x the progress along the SDC's route (arc
              length, gpucat.geometry.route_coords); replaced by
              +success_reward on arrival, -out_of_road_penalty out of road,
              -crash_penalty on a collision with a vehicle (that precedence)
  arrival     route completion > 0.95 (or GPUDrive's goal reached)
  out of road farther than 10 m from the route, or the footprint touching a
              road-edge boundary (MetaDrive's sidewalk) or a solid yellow
              line (gpucat/static.py)
  collision   the footprint overlapping a vehicle's (static vehicles are
              not in the scenes, as with no_static_vehicles); here it ends
              the episode (crash_done, CAT's MetaDrive training does not)
  dynamics    GPUDrive's invertible bicycle: acceleration in accel_range
              (m/s^2) and curvature within +-curvature (1/m) from the two
              policy outputs in [-1, 1]
  observation GPUDrive's (ego state, 63 partners, 200 road points) and, with
              ``navigation``, what MetaDrive tells CAT's policy and GPUDrive's
              does not: the route ahead (points 5-50 m along it, in the SDC's
              frame, /50 m), the distance from the route (/10 m), the route
              completion, and the distances along 24 rays (every 15 degrees,
              /50 m) to the road edges and, separately, to the solid yellow
              lines. GPUDrive's road points type every line as a RoadLine, so
              without them the policy cannot tell the yellow lines that end
              an episode from the lane lines it may cross, and only knows the
              route's end point (replay_s0, 4.5M steps without them: 21%
              arrival, 51% out of road on the test scenes)

CAT's adversary bookkeeping: per scene, the SDC's last ``history``
trajectories (steps 11 on, as AdvGenerator.after_episode stores them;
episodes shorter than 10 steps after step 11 are ignored) start as the
logged route; evaluation keeps one of its own (AV_trajs_eval).

Needs GPUDrive's environment (Python 3.11, CUDA 12.4).
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from gpucat import adversary as ga
from gpucat.geometry import boxes_overlap, points_along, ray_distances, route_coords, segments_touch_box
from gpucat.static import StaticBank

TRAJ_LEN = 91
EPISODE_STEPS = TRAJ_LEN - 1
FIRST_STORED = 11  # AdvGenerator stores ego_traj[11:91]
MIN_STORED = 10  # and ignores shorter ones ("Ignore traj less than 1s")
POS, VEL, YAW, VALID = slice(0, 182), slice(182, 364), slice(364, 455), slice(455, 546)

# outcome codes per episode
RUNNING, ARRIVED, OUT_OF_ROAD, CRASHED, TIMEOUT = 0, 1, 2, 3, 4

# navigation observations
ROUTE_AHEAD = (5.0, 10.0, 20.0, 30.0, 50.0)  # m along the route
RAYS = 24
RAY_RANGE = 50.0  # m


@dataclass
class EnvSettings:
    worlds: int = 256
    accel_low: float = -6.0
    accel_high: float = 4.0
    curvature: float = 0.3
    driving_reward: float = 1.0
    success_reward: float = 10.0
    out_of_road_penalty: float = 10.0
    crash_penalty: float = 1.0  # MetaDrive's crash_vehicle_penalty
    crash_done: bool = True
    out_of_route: float = 10.0  # m
    arrive_completion: float = 0.95
    history: int = 5  # CAT's AV_traj_num
    collision_radius: float = 30.0  # m: vehicles farther from the SDC are not checked
    navigation: bool = True  # add the route and the rays to the observations


class CatEnv:
    def __init__(self, settings: EnvSettings, scene_dir: str, static_path: str, bank_path: str,
                 gpudrive_root: str = str(Path.home() / "gpudrive"), device: str = "cuda"):
        self.s, self.device, self.W = settings, device, settings.worlds
        self.scene_dir = str(Path(scene_dir).resolve())
        self.static = StaticBank(str(Path(static_path).resolve()), device)
        self.bank = ga.CandidateBank(str(Path(bank_path).resolve()), device)
        root = str(Path(gpudrive_root).expanduser().resolve())
        sys.path.insert(0, root)
        os.environ.setdefault("MADRONA_MWGPU_KERNEL_CACHE", os.path.join(root, "gpudrive_cache"))
        cwd = os.getcwd()
        os.chdir(root)  # GPUDrive resolves its assets relative to its root
        try:
            from gpudrive.env.config import EnvConfig
            from gpudrive.env.dataset import SceneDataLoader
            from gpudrive.env.env_torch import GPUDriveTorchEnv

            loader = SceneDataLoader(root=self.scene_dir, batch_size=self.W, dataset_size=self.W, file_prefix="cat",
                                     shuffle=False)
            config = EnvConfig(dynamics_model="bicycle", collision_behavior="ignore", remove_non_vehicles=False,
                               init_mode="all_objects")
            self.env = GPUDriveTorchEnv(config=config, data_loader=loader, max_cont_agents=64, device=device)
        finally:
            os.chdir(cwd)
        self.sim = self.env.sim
        # CAT's per-scene ego histories (rows of the candidate bank)
        S, M = len(self.bank.stems), settings.history
        route = self.bank.ego_route
        self.hist = route[:, None].repeat(1, M, 1, 1).clone()
        self.hist_len = torch.full((S, M), route.shape[1], device=device, dtype=torch.long)
        self.hist_prob = torch.ones(S, M, device=device)
        self.hist_yaw = ga.subsampled_yaw(self.hist.double(), self.hist_len).float()
        self.eval_hist, self.eval_len = route[:, None].clone(), torch.full((S, 1), route.shape[1], device=device)
        self.eval_yaw = ga.subsampled_yaw(self.eval_hist.double(), self.eval_len).float()
        self.load([Path(p).stem.split("_", 1)[1] for p in self.env.data_batch], swap=False)

    # ------------------------------------------------------------------ scenes

    def load(self, stems: Sequence[str], swap: bool = True) -> None:
        """Puts these W scenes into the worlds (GPUDrive re-reads them, which
        also discards any injected plans) and caches what the episodes need."""
        assert len(stems) == self.W
        if swap:
            self.env.swap_data_batch([os.path.join(self.scene_dir, f"cat_{s}.json") for s in stems])
        self.stems = list(stems)
        dev = self.device
        self.srow = torch.tensor([self.static.index[s] for s in stems], device=dev)
        self.brow = torch.tensor([self.bank.index[s] for s in stems], device=dev)
        self.mean = self.sim.world_means_tensor().to_torch()[:, :2].clone().to(dev)
        mask = self.env.cont_agent_mask
        if not (bool(mask[:, 0].all()) and int(mask.sum()) == self.W):
            raise RuntimeError("expected exactly the SDC (slot 0) under control in every world")
        self.traj = self.sim.expert_trajectory_tensor().to_torch()  # zero-copy [W, 64, 1456]
        self.orig_adv = self.traj[:, 1].clone()
        st = self.static
        self.route, self.route_n = st.route[self.srow], st.route_n[self.srow]
        self.route_len = st.route_len[self.srow]
        self.edges, self.edges_n = st.edges[self.srow], st.edges_n[self.srow]
        self.yellow, self.yellow_n = st.yellow[self.srow], st.yellow_n[self.srow]
        self.sizes, self.types, self.n_agents = st.sizes[self.srow], st.types[self.srow], st.n_agents[self.srow]

    # ------------------------------------------------------------------ adversary

    def choose_plans(self, worlds: torch.Tensor, rule: str, tau: float = 2.0, rho: float = 0.1,
                     evaluation: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """CAT's (or the fair rule's) plan [n, 91, 5] for these worlds against
        their scenes' stored ego trajectories, and the chosen candidates [n]."""
        b, bank = self.brow[worlds], self.bank
        if evaluation:
            hist, hlen, hyaw = self.eval_hist[b], self.eval_len[b], self.eval_yaw[b]
            hprob = torch.ones(len(b), 1, device=self.device)
        else:
            hist, hlen, hyaw, hprob = self.hist[b], self.hist_len[b], self.hist_yaw[b], self.hist_prob[b]
        score, md = ga.cat_scores(bank.cand[b], bank.probs_ov[b], hist, hlen, hprob, bank.ov_size[b],
                                  bank.av_size[b], yaw_ov=bank.yaw_ov[b], yaw_av=hyaw)
        if rule == "cat":
            chosen = ga.select("cat", score, md)
        else:
            beta = ga.adversary_beta(bank.samples[b], bank.cand[b], hist, hlen, hprob)
            chosen = ga.select("fair", score, md, beta, bank.avoid[b], tau, rho)
        return ga.plan(bank.adv_past[b], bank.cand[b, chosen]), chosen

    def inject(self, worlds: torch.Tensor, plans: torch.Tensor) -> None:
        """Writes the plans (raw coordinates) as the adversary's (slot 1)
        trajectory; steps before the adversary appears in the log stay as
        GPUDrive loaded them, invalid."""
        row = self.orig_adv[worlds]  # [n, 1456]
        n = len(worlds)
        appeared = row[:, VALID] > 0.5
        keep = ~appeared & (torch.arange(TRAJ_LEN, device=self.device) <= 10)
        pos = plans[..., :2] - self.mean[worlds][:, None]
        new = row.clone()
        new[:, POS] = torch.where(keep[..., None], row[:, POS].view(n, TRAJ_LEN, 2), pos).flatten(1)
        new[:, VEL] = torch.where(keep[..., None], row[:, VEL].view(n, TRAJ_LEN, 2), plans[..., 2:4]).flatten(1)
        new[:, YAW] = torch.where(keep, row[:, YAW], plans[..., 4])
        new[:, VALID] = (~keep).float()
        self.traj[worlds, 1] = new

    # ------------------------------------------------------------------ episodes

    def begin_episode(self, adversarial: torch.Tensor, rule: str = "cat", tau: float = 2.0, rho: float = 0.1,
                      evaluation: bool = False, auto_reset: bool = False, p_adv: float = 0.0) -> torch.Tensor:
        """Starts an episode in every world: the adversary restored to its
        log, or given a plan where ``adversarial`` [W] holds. Lockstep (for
        evaluation): a world whose episode ends stays done (masked) until the
        next begin_episode, and end_episode reports and stores every world's.
        With ``auto_reset`` (training): a world whose episode ends has it
        reported in ``self.finished`` and its SDC trajectory stored, and starts
        the next one at once, adversarial with probability ``self.p_adv``.
        Returns the SDC's observations [W, D]."""
        self.rule, self.tau, self.rho = rule, tau, rho
        self.evaluation, self.auto_reset, self.p_adv = evaluation, auto_reset, p_adv
        dev, W = self.device, self.W
        self.done = torch.zeros(W, dtype=torch.bool, device=dev)
        self.t = torch.zeros(W, dtype=torch.long, device=dev)
        self.outcome = torch.zeros(W, dtype=torch.long, device=dev)
        self.completion = torch.zeros(W, device=dev)
        self.adversarial = torch.zeros(W, dtype=torch.bool, device=dev)
        self.chosen = torch.full((W,), -1, dtype=torch.long, device=dev)
        self.prev_s = torch.zeros(W, device=dev)
        self.ego_buf = torch.full((W, TRAJ_LEN, 2), float("nan"), device=dev)
        self.finished = []
        self._restart(torch.arange(W, device=dev), adversarial)
        return self.obs()

    def _restart(self, worlds: torch.Tensor, adversarial: torch.Tensor) -> None:
        """New episodes in these worlds (adversarial [n] for each)."""
        self.traj[worlds, 1] = self.orig_adv[worlds]
        adversarial = adversarial & self.has_adversary()[worlds]
        self.chosen[worlds] = -1
        adv = worlds[adversarial]
        if len(adv):
            plans, chosen = self.choose_plans(adv, self.rule, self.tau, self.rho, self.evaluation)
            self.inject(adv, plans)
            self.chosen[adv] = chosen
        self.adversarial[worlds] = adversarial
        self.sim.reset(worlds.tolist())
        p, _ = self._ego()
        _, s = route_coords(p, self.route, self.route_n)
        self.t[worlds] = 0
        self.prev_s[worlds] = s[worlds]
        self.outcome[worlds] = RUNNING
        self.completion[worlds] = 0.0
        self.ego_buf[worlds] = float("nan")
        self.ego_buf[worlds, 0] = p[worlds]

    def has_adversary(self) -> torch.Tensor:
        return self.n_agents >= 2

    def obs(self) -> torch.Tensor:
        """The SDC's observations [W, D]. Asked for with the controlled-agent
        mask: a view into get_obs()'s [W, 64, D] would keep all of it alive
        in every stored step (6 GB over a 64-step rollout of 128 worlds)."""
        obs = self.env.get_obs(self.env.cont_agent_mask)
        if not self.s.navigation:
            return obs
        return torch.cat([obs, self.navigation_obs()], -1)

    def navigation_obs(self) -> torch.Tensor:
        """[W, 2 len(ROUTE_AHEAD) + 2 + 2 RAYS] (module docstring)."""
        p, h = self._ego()
        lateral, s = route_coords(p, self.route, self.route_n)
        ahead = points_along(self.route, self.route_n, s, torch.tensor(ROUTE_AHEAD, device=self.device)) - p[:, None]
        cos, sin = torch.cos(h)[:, None], torch.sin(h)[:, None]
        local = torch.stack([ahead[..., 0] * cos + ahead[..., 1] * sin, -ahead[..., 0] * sin + ahead[..., 1] * cos], -1)
        angles = torch.arange(RAYS, device=self.device) * (2.0 * torch.pi / RAYS)
        edge = ray_distances(p, h, self.edges, self.edges_n, angles, RAY_RANGE)
        yellow = ray_distances(p, h, self.yellow, self.yellow_n, angles, RAY_RANGE)
        completion = (s / self.route_len.clamp(min=1.0)).clamp(0.0, 1.0)
        return torch.cat([local.flatten(1) / RAY_RANGE, (lateral / self.s.out_of_route)[:, None], completion[:, None],
                          edge / RAY_RANGE, yellow / RAY_RANGE], -1).to(torch.float32)

    def _ego(self):
        g = self.sim.absolute_self_observation_tensor().to_torch()
        return g[:, 0, :2] + self.mean, g[:, 0, 7]

    def step(self, action: torch.Tensor, clip: bool = True):
        """action [W, 2] in [-1, 1] (not clipped with clip=False, for replaying
        the log). Returns (obs [W, D] -- with auto_reset, the next episode's
        first for worlds that just ended --, reward [W], ended [W] -- the
        episode ended at this step --, alive [W] -- the world's transition
        counts: always, with auto_reset); ``self.flags`` holds this step's
        tests per world."""
        s = self.s
        alive = ~self.done
        a = action.clamp(-1.0, 1.0) if clip else action
        acts = torch.zeros(self.W, 64, 3, device=self.device)
        acts[:, 0, 0] = s.accel_low + 0.5 * (a[:, 0] + 1.0) * (s.accel_high - s.accel_low)
        acts[:, 0, 1] = a[:, 1] * s.curvature
        self.env.step_dynamics(acts)
        self.t += alive.long()
        g = self.sim.absolute_self_observation_tensor().to_torch()
        pos, yaw = g[..., :2] + self.mean[:, None], g[..., 7]
        p, h = pos[:, 0], yaw[:, 0]
        size = self.sizes[:, 0]
        # collision with a vehicle
        others = torch.arange(64, device=self.device)[None] < self.n_agents[:, None]
        others &= (self.types == 1) & ((pos - p[:, None]).norm(dim=-1) < s.collision_radius)
        others[:, 0] = False
        crash = (boxes_overlap(p, h, size, pos, yaw, self.sizes) & others).any(-1)
        # route, road edges, yellow lines
        lateral, sarc = route_coords(p, self.route, self.route_n)
        edge = segments_touch_box(p, h, size[:, 0], size[:, 1], self.edges, self.edges_n)
        yellow = segments_touch_box(p, h, size[:, 0], size[:, 1], self.yellow, self.yellow_n)
        out_of_road = (lateral > s.out_of_route) | edge | yellow
        completion = sarc / self.route_len.clamp(min=1.0)
        goal = self.sim.done_tensor().to_torch()[:, 0].flatten().bool()
        arrive = (completion > s.arrive_completion) | (self.route_len < 1.0) | goal
        self.flags = {"crash": crash, "edge": edge, "yellow": yellow, "off_route": lateral > s.out_of_route,
                      "arrive": arrive, "lateral": lateral, "position": p}
        reward = s.driving_reward * (sarc - self.prev_s)
        reward = torch.where(crash, torch.full_like(reward, -s.crash_penalty), reward)
        reward = torch.where(out_of_road, torch.full_like(reward, -s.out_of_road_penalty), reward)
        reward = torch.where(arrive, torch.full_like(reward, s.success_reward), reward)
        timeout = self.t >= EPISODE_STEPS
        ended = alive & (arrive | out_of_road | (crash & s.crash_done) | timeout)
        reason = torch.where(arrive, ARRIVED, torch.where(out_of_road, OUT_OF_ROAD,
                             torch.where(crash & s.crash_done, CRASHED, TIMEOUT)))
        self.outcome = torch.where(ended, reason, self.outcome)
        self.completion = torch.where(alive, completion.clamp(0.0, 1.0), self.completion)
        reward = torch.where(alive, reward, torch.zeros_like(reward))
        self.prev_s = sarc
        w = torch.nonzero(alive).flatten()
        self.ego_buf[w, self.t[w].clamp(max=TRAJ_LEN - 1)] = p[w]
        if self.auto_reset:
            idx = torch.nonzero(ended).flatten()
            if len(idx):
                self.finished.append({"outcome": self.outcome[idx].clone(), "completion": self.completion[idx].clone(),
                                      "adversarial": self.adversarial[idx].clone()})
                self._store(idx)
                self._restart(idx, torch.rand(len(idx), device=self.device) < self.p_adv)
        else:
            self.done |= ended
        return self.obs(), reward, ended, alive

    def _store(self, worlds: torch.Tensor) -> None:
        """CAT's after_episode: the SDC's positions from step 11 on (at most
        80; episodes with fewer than 10 are ignored) become the newest of its
        scene's history (evaluation: its one evaluation trajectory)."""
        stored = self.ego_buf[worlds, FIRST_STORED:FIRST_STORED + 80]
        length = (~torch.isnan(stored[..., 0])).sum(1)
        ok = length >= MIN_STORED
        if not ok.any():
            return
        w, new, n = worlds[ok], torch.nan_to_num(stored[ok]), length[ok]
        b = self.brow[w]
        yaw = ga.subsampled_yaw(new.double(), n).float()
        if self.evaluation:
            self.eval_hist[b, 0], self.eval_len[b, 0], self.eval_yaw[b, 0] = new, n, yaw
        else:  # a deque: drop the oldest
            self.hist[b] = torch.cat([self.hist[b, 1:], new[:, None]], 1)
            self.hist_len[b] = torch.cat([self.hist_len[b, 1:], n[:, None]], 1)
            self.hist_yaw[b] = torch.cat([self.hist_yaw[b, 1:], yaw[:, None]], 1)

    def end_episode(self, evaluation: bool = False) -> Dict[str, torch.Tensor]:
        """Lockstep: stores every world's SDC trajectory and returns the
        episode's outcomes."""
        self._store(torch.arange(self.W, device=self.device))
        return {"outcome": self.outcome.clone(), "completion": self.completion.clone(),
                "adversarial": self.adversarial.clone(), "chosen": self.chosen.clone()}

    def take_finished(self) -> Dict[str, torch.Tensor]:
        """With auto_reset: the episodes finished since the last call."""
        if not self.finished:
            return {"outcome": torch.zeros(0, dtype=torch.long), "completion": torch.zeros(0),
                    "adversarial": torch.zeros(0, dtype=torch.bool)}
        out = {k: torch.cat([f[k] for f in self.finished]) for k in self.finished[0]}
        self.finished = []
        return out

    # ------------------------------------------------------------------ replay (checks)

    def expert_actions(self) -> torch.Tensor:
        """The SDC's logged motion as bicycle actions [W, 90, 2] (GPUDrive's
        inverse model), in the policy's [-1, 1] scale (outside it where the
        log needs more)."""
        a = self.env.get_expert_actions()[0][:, 0, :EPISODE_STEPS, :2]
        s = self.s
        accel = (a[..., 0] - s.accel_low) / (s.accel_high - s.accel_low) * 2.0 - 1.0
        return torch.stack([accel, a[..., 1] / s.curvature], -1)
