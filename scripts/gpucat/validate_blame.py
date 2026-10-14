"""Validate the counterfactual collision attribution (responsibility/blame.py)
on collisions whose responsible party is known by construction.

Real test scenes (raw_scenes_500/400-499.pkl) are edited so that two logged
vehicles collide in a way whose at-fault party is unambiguous. Only the
edited agent's states change (positions along its own logged path, heading
and velocity kept consistent); everyone else stays as logged. The scene is
then attributed with the real DenseTNT, exactly as compute_responsibility
attributes a rollout crash: the ego is invalid after the crash step (as in
rollouts.scene_from_rollout), and crash_blame is called at the first step
the two footprints overlap.

Cases (expected verdict for the listed ego):

  C1 rear-end, leader brakes: the leader L brakes at 3 m/s^2 to a stop from
     onset t0, the follower F keeps its logged motion and runs into it.
     The follower could have avoided it (log until t0 + 1 s, then 6 m/s^2
     braking, does not collide). ego = F -> "ego"; ego = L -> "other".
  C2 ego runs into a slower car ahead: F keeps its speed at t0 (or
     accelerates at 1.5 m/s^2) along its logged path, the logged leader is
     slower (and never brakes harder than 4 m/s^2). ego = F -> "ego";
     ego = L -> "other".
  C3 cut-in: B, in the lane next to A and alongside or ahead (0-10 m), moves
     laterally onto A's logged path over 2.5 s (smoothstep, peak lateral
     speed about 2 m/s) and A, logged, hits it or is hit side-on.
     ego = A -> "other"; ego = B -> "ego".
  C4 symmetric side-swipe: two cars side by side in adjacent lanes both
     drift toward each other by the same lateral distance over 2 s.
     Either ego -> "shared".

Every instance also records the RSS verdict, the rear-end rule, the
weighted motion set (motion_set="weighted"), LOOKBACK 10 and 30 (the
latter with a 30-step metric horizon, so the window still reaches the
crash), D_g saturated at 20 m instead of 10 m, DenseTNT's second
object-of-interest slot fixed to the crash partner (pair_*: as in CAT's
(sdc, adversary) instances, instead of the vehicle nearest to the
predicted agent; slot_* say whether the default already was the partner), and
per-side diagnostics of the motion set (the distribution of
C = D_g(alternative) - D_g(actual), the alternatives' travelled distance
relative to the actual one), and where the edited manoeuvre began relative
to the window (onset_vs_window = window - t0 > 0: DenseTNT saw it in its
history).

Run (repository root; one DenseTNT on the GPU, about 1-2 GB):
    ~/venvs/cat39/bin/python -m scripts.gpucat.validate_blame --device cuda --out <dir>
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path as FsPath

import numpy as np
import torch

sys.path.insert(0, str(FsPath(__file__).resolve().parents[2]))

from responsibility.blame import LOOKBACK, crash_blame, crash_partner, rear_end_rule, split  # noqa: E402
from responsibility.geometry import pairwise_min_distance_over_time  # noqa: E402
from responsibility.metrics import FIRST_STEP, ResponsibilityConfig, effective_horizon, motion_set  # noqa: E402
from responsibility.risk import cvar  # noqa: E402
from responsibility.rss import rss_blame  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

DT = 0.1
T = 91


# ----------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------

class LoggedPath:
    """An agent's logged path as a polyline over arc-length, with its logged
    heading; linear extrapolation beyond both ends."""

    def __init__(self, scene: Scene, agent: int):
        valid = np.nonzero(scene.valid[agent])[0]
        pts = scene.position[agent, valid, :2].astype(np.float64)
        hd = np.unwrap(scene.heading[agent, valid].astype(np.float64))
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
        s_all = np.concatenate([[0.0], np.cumsum(seg)])
        self.step_s = np.full(scene.n_steps, np.nan)
        self.step_s[valid] = s_all
        keep = np.concatenate([[True], seg > 0.05])
        self.s, self.pts, self.hd = s_all[keep], pts[keep], hd[keep]
        if len(self.s) < 2:
            raise ValueError("path too short")
        # extrapolation directions
        self.d0 = (self.pts[1] - self.pts[0]) / max(self.s[1] - self.s[0], 1e-6)
        self.d1 = (self.pts[-1] - self.pts[-2]) / max(self.s[-1] - self.s[-2], 1e-6)

    @property
    def length(self):
        return self.s[-1]

    def point(self, s):
        s = np.asarray(s, dtype=np.float64)
        x = np.interp(s, self.s, self.pts[:, 0])
        y = np.interp(s, self.s, self.pts[:, 1])
        out = np.stack([x, y], -1)
        lo, hi = s < self.s[0], s > self.s[-1]
        out[lo] = self.pts[0] + (s[lo] - self.s[0])[:, None] * self.d0
        out[hi] = self.pts[-1] + (s[hi] - self.s[-1])[:, None] * self.d1
        return out

    def heading(self, s):
        return np.interp(np.asarray(s, dtype=np.float64), self.s, self.hd)

    def project(self, p):
        """Points [M, 2] -> arc-length [M], signed lateral offset [M] (left +)."""
        p = np.asarray(p, dtype=np.float64)
        a, b = self.pts[:-1], self.pts[1:]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(-1), 1e-9)
        t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
        q = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(p[:, None] - q, axis=-1)
        j = np.argmin(d, axis=1)
        r = np.arange(len(p))
        s = self.s[j] + t[r, j] * np.sqrt(L2[j])
        cross = ab[j, 0] * (p[:, 1] - a[j, 1]) - ab[j, 1] * (p[:, 0] - a[j, 0])
        lat = np.sign(cross) * d[r, j]
        # beyond the ends: extend along the end directions
        before = (j == 0) & (t[r, 0] == 0)
        after = (j == len(ab) - 1) & (t[r, -1] == 1)
        s = np.where(before, self.s[0] + ((p - self.pts[0]) * self.d0).sum(-1), s)
        s = np.where(after, self.s[-1] + ((p - self.pts[-1]) * self.d1).sum(-1), s)
        return s, lat


def obb_overlap(p1, h1, l1, w1, p2, h2, l2, w2):
    """Separating-axis test of two oriented boxes, vectorised over steps."""
    p1, p2 = np.atleast_2d(p1), np.atleast_2d(p2)
    h1, h2 = np.atleast_1d(h1), np.atleast_1d(h2)
    d = p2 - p1
    axes = [np.stack([np.cos(h1), np.sin(h1)], -1), np.stack([-np.sin(h1), np.cos(h1)], -1),
            np.stack([np.cos(h2), np.sin(h2)], -1), np.stack([-np.sin(h2), np.cos(h2)], -1)]
    u1, v1, u2, v2 = axes
    sep = np.zeros(len(d), dtype=bool)
    for ax in axes:
        r1 = l1 / 2 * np.abs((u1 * ax).sum(-1)) + w1 / 2 * np.abs((v1 * ax).sum(-1))
        r2 = l2 / 2 * np.abs((u2 * ax).sum(-1)) + w2 / 2 * np.abs((v2 * ax).sum(-1))
        sep |= np.abs((d * ax).sum(-1)) > r1 + r2
    return ~sep


def overlap_steps(scene: Scene, a: int, b: int, steps) -> np.ndarray:
    steps = np.asarray(steps)
    la, wa = scene.shape_at(a, int(steps[0]))
    lb, wb = scene.shape_at(b, int(steps[0]))
    both = scene.valid[a, steps] & scene.valid[b, steps]
    ov = obb_overlap(scene.position[a, steps, :2], scene.heading[a, steps], la, wa,
                     scene.position[b, steps, :2], scene.heading[b, steps], lb, wb)
    return ov & both


def first_overlap(scene: Scene, a: int, b: int, start: int):
    steps = np.arange(start, scene.n_steps)
    ov = overlap_steps(scene, a, b, steps)
    return int(steps[np.argmax(ov)]) if ov.any() else None


def third_party_contact(scene: Scene, agents, partner_pair, until: int):
    """(agent, third, step) if any of ``agents`` overlaps a third agent up to ``until``."""
    for a in agents:
        for o in range(scene.n_agents):
            if o in partner_pair:
                continue
            steps = np.arange(0, until + 1)
            ov = overlap_steps(scene, a, o, steps)
            if ov.any():
                return a, o, int(steps[np.argmax(ov)])
    return None


def speed(scene: Scene, a: int, t: int) -> float:
    return float(np.linalg.norm(scene.velocity[a, t]))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------
# edits
# ----------------------------------------------------------------------

def along_path_track(scene: Scene, agent: int, path: LoggedPath, t0: int, v0: float, accel: float,
                     v_cap: float = 40.0):
    """The agent's track with, from t0 on, speed v0 + accel * t (clipped to
    [0, v_cap]) along its own logged path; valid from its first logged step on."""
    pos = scene.position[agent].copy()
    head = scene.heading[agent].copy()
    vel = scene.velocity[agent].copy()
    valid = scene.valid[agent].copy()
    tt = np.arange(t0, scene.n_steps)
    v = np.clip(v0 + accel * (tt - t0) * DT, 0.0, v_cap)
    s = path.step_s[t0] + np.concatenate([[0.0], np.cumsum((v[:-1] + v[1:]) / 2 * DT)])
    xy = path.point(s)
    hd = path.heading(s)
    pos[t0:, :2] = xy
    head[t0:] = hd
    vel[t0:] = v[:, None] * np.stack([np.cos(hd), np.sin(hd)], -1)
    valid[t0:] = True
    return pos, head, vel, valid


def lateral_shift_track(scene: Scene, agent: int, offsets: np.ndarray):
    """The agent's logged track shifted by ``offsets`` [T, 2] (zero before
    the edit), heading turned by the angle the shift adds to its velocity."""
    pos = scene.position[agent].copy()
    head = scene.heading[agent].copy().astype(np.float64)
    vel = scene.velocity[agent].copy().astype(np.float64)
    pos[:, :2] += offsets
    d_off = np.gradient(offsets, DT, axis=0)
    new_vel = vel + d_off
    turn = np.arctan2(new_vel[:, 1], new_vel[:, 0]) - np.arctan2(vel[:, 1], vel[:, 0])
    turn = np.angle(np.exp(1j * turn))
    moving = np.linalg.norm(vel, axis=-1) > 1.0
    head = head + np.where(moving & (np.linalg.norm(d_off, axis=-1) > 0), turn, 0.0)
    return pos, head.astype(np.float32), new_vel.astype(np.float32), scene.valid[agent].copy()


# ----------------------------------------------------------------------
# pair search
# ----------------------------------------------------------------------

def vehicles(scene: Scene):
    out = []
    for i in range(scene.n_agents):
        if scene.types[i] != "VEHICLE" or scene.valid[i].sum() < 40:
            continue
        sp = np.linalg.norm(scene.velocity[i], axis=-1)[scene.valid[i]]
        if sp.max() > 3.0:
            out.append(i)
    return out


def relations(scene: Scene, a: int, b: int, path_a: LoggedPath):
    """Per step where both are valid: (step, lon = s_b - s_a, lat of b from
    a's path, heading difference)."""
    both = np.nonzero(scene.valid[a] & scene.valid[b])[0]
    if len(both) < 30:
        return None
    if np.min(np.linalg.norm(scene.position[a, both, :2] - scene.position[b, both, :2], axis=-1)) > 50:
        return None
    s_b, lat = path_a.project(scene.position[b, both, :2])
    s_a = path_a.step_s[both]
    dh = np.abs(np.angle(np.exp(1j * (scene.heading[b, both] - scene.heading[a, both]))))
    return both, s_b - s_a, lat, dh


# ----------------------------------------------------------------------
# instance builders: return (scene, edited agents, role dict, t0, info) or a skip reason
# ----------------------------------------------------------------------

def finish(scene: Scene, a: int, b: int, t0: int, edited):
    c = first_overlap(scene, a, b, t0)
    if c is None:
        return "no-contact"
    if c < 30:
        return "contact-before-30"  # window would be clamped at FIRST_STEP; keep the window exactly c - 20
    if overlap_steps(scene, a, b, np.arange(0, t0 + 1)).any():
        return "overlap-before-edit"
    third = third_party_contact(scene, edited, (a, b), c)
    if third is not None:
        return "third-agent-contact"
    if not (scene.valid[a, c - LOOKBACK - 10:c + 1].all() and scene.valid[b, c - LOOKBACK - 10:c + 1].all()):
        return "not-valid-through-window"
    return c


def build_c1(scene, f, l, path_f, path_l, t0):
    """Leader l brakes at 3 m/s^2 from t0; follower f logged."""
    v0 = speed(scene, l, t0)
    if v0 < 4.0:
        return "leader-slow"
    sc = scene.with_track(l, *along_path_track(scene, l, path_l, t0, v0, -3.0))
    c = finish(sc, f, l, t0, [l])
    if isinstance(c, str):
        return c
    # avoidable for the follower: logged until t0 + 1 s, then 6 m/s^2 braking
    react = t0 + 10
    if react >= c:
        return "unavoidable-follower"
    vf = speed(sc, f, react)
    alt = sc.with_track(f, *along_path_track(sc, f, path_f, react, vf, -6.0))
    if first_overlap(alt, f, l, t0) is not None:
        return "unavoidable-follower"
    return sc, c, {"at_fault": f, "roles": {"follower": f, "leader": l}, "edited": [l], "t0": t0,
                   "edit": f"leader brakes 3 m/s^2 from {v0:.1f} m/s"}


def build_c2(scene, f, l, path_f, path_l, t0):
    """Follower f keeps its speed (or accelerates) from t0; leader logged."""
    v0 = speed(scene, f, t0)
    if v0 < 4.0:
        return "follower-slow"
    reason = None
    for accel in (0.0, 1.5):
        sc = scene.with_track(f, *along_path_track(scene, f, path_f, t0, v0, accel, v_cap=v0 + 8.0))
        c = finish(sc, f, l, t0, [f])
        if isinstance(c, str):
            reason = c
            continue
        # the leader behaves normally: no braking harder than 4 m/s^2 in [t0, c]
        vl = np.linalg.norm(sc.velocity[l, t0:c + 1], axis=-1)
        if len(vl) > 3 and np.min(np.diff(vl)) / DT < -4.0:
            return "leader-brakes-hard"
        if rear_end_rule(sc, f, l, c) != "ego":
            return "not-rear-end"
        return sc, c, {"at_fault": f, "roles": {"follower": f, "leader": l}, "edited": [f], "t0": t0,
                       "edit": f"follower holds {v0:.1f} m/s + {accel} m/s^2"}
    return reason


def build_c3(scene, a, b, path_a, path_b, t0, duration=25):
    """b (adjacent lane) moves onto a's path over ``duration`` steps from t0."""
    if speed(scene, a, t0) < 4.0 or speed(scene, b, t0) < 4.0:
        return "slow"
    steps = np.arange(scene.n_steps)
    valid_b = scene.valid[b]
    s_on_a, _ = path_a.project(scene.position[b, :, :2])
    target = path_a.point(s_on_a)
    w = smoothstep((steps - t0) / duration)[:, None]
    off = np.where(valid_b[:, None], (target - scene.position[b, :, :2]) * w, 0.0)
    sc = scene.with_track(b, *lateral_shift_track(scene, b, off))
    c = finish(sc, a, b, t0, [b])
    if isinstance(c, str):
        return c
    if c > t0 + duration + 20:
        return "contact-late"  # A had time to see it settled in its lane
    return sc, c, {"at_fault": b, "roles": {"victim": a, "cutter": b}, "edited": [b], "t0": t0,
                   "edit": f"cut-in over {duration * DT:.1f} s"}


def build_c4(scene, a, b, path_a, path_b, t0, duration=20, overlap=0.4):
    """a and b drift toward each other by the same lateral distance."""
    if speed(scene, a, t0) < 4.0 or speed(scene, b, t0) < 4.0:
        return "slow"
    te = min(t0 + duration, scene.n_steps - 1)
    d = scene.position[b, te, :2] - scene.position[a, te, :2]
    h = scene.heading[a, te]
    n = np.array([-np.sin(h), np.cos(h)])
    lat = float(d @ n)
    lon = float(d @ np.array([np.cos(h), np.sin(h)]))
    la, wa = scene.shape_at(a, te)
    lb, wb = scene.shape_at(b, te)
    if abs(lon) > (la + lb) / 2 - 1.0:
        return "not-side-by-side"
    need = abs(lat) - (wa + wb) / 2 + overlap
    if need <= 0:
        return "already-touching"
    steps = np.arange(scene.n_steps)
    w = smoothstep((steps - t0) / duration)[:, None]
    # each moves half of the way, perpendicular to its own heading, toward the other
    side = np.sign(lat)
    na = np.stack([-np.sin(scene.heading[a]), np.cos(scene.heading[a])], -1) * side
    nb = -np.stack([-np.sin(scene.heading[b]), np.cos(scene.heading[b])], -1) * side
    off_a = np.where(scene.valid[a][:, None], w * need / 2 * na, 0.0)
    off_b = np.where(scene.valid[b][:, None], w * need / 2 * nb, 0.0)
    sc = scene.with_track(a, *lateral_shift_track(scene, a, off_a))
    sc = sc.with_track(b, *lateral_shift_track(sc, b, off_b))
    c = finish(sc, a, b, t0, [a, b])
    if isinstance(c, str):
        return c
    return sc, c, {"at_fault": None, "roles": {"left_or_right_a": a, "b": b}, "edited": [a, b], "t0": t0,
                   "edit": f"both drift {need / 2:.2f} m over {duration * DT:.1f} s"}


def candidates(scene: Scene, case: str):
    """Ordered pairs (first, second, t0 list) whose geometry fits the case."""
    vs = vehicles(scene)
    paths = {}
    for i in vs:
        try:
            paths[i] = LoggedPath(scene, i)
        except ValueError:
            pass
    out = []
    for a in paths:
        for b in paths:
            if a == b:
                continue
            rel = relations(scene, a, b, paths[a])
            if rel is None:
                continue
            steps, lon, lat, dh = rel
            sp_a = np.linalg.norm(scene.velocity[a, steps], axis=-1)
            sp_b = np.linalg.norm(scene.velocity[b, steps], axis=-1)
            ok_t = (steps >= 12) & (steps <= 70) & (sp_a > 4) & (sp_b > 4)
            if case in ("C1", "C2"):  # a follower, b leader in the same lane
                m = ok_t & (np.abs(lat) < 1.2) & (dh < np.deg2rad(20)) & (lon > 7) & (lon < 35)
            elif case == "C3":  # b in the adjacent lane, alongside or ahead
                m = ok_t & (np.abs(lat) > 2.6) & (np.abs(lat) < 4.6) & (dh < np.deg2rad(12)) & (lon > 0) & (lon < 10)
            else:  # C4: side by side
                m = ok_t & (np.abs(lat) > 2.6) & (np.abs(lat) < 4.6) & (dh < np.deg2rad(10)) & (np.abs(lon) < 2.0)
                if a > b:
                    m &= False  # unordered pair
                # the drift (20 steps) must end where they are side by side
                if m.any():
                    t0s = [int(t) - 20 for t in steps[m] if t - 20 >= 12]
                    if t0s:
                        out.append((a, b, t0s, paths[a], paths[b]))
                continue
            if m.any():
                out.append((a, b, [int(t) for t in steps[m]], paths[a], paths[b]))
    return out


BUILDERS = {"C1": build_c1, "C2": build_c2, "C3": build_c3, "C4": build_c4}


# ----------------------------------------------------------------------
# attribution
# ----------------------------------------------------------------------

def _circles(xy, heading, length, width, n=3):
    """Circle centres [..., T, n, 2] and radius covering a box (geometry.circle_decomposition)."""
    frac = (2.0 * np.arange(n) + 1.0) / (2.0 * n) - 0.5
    d = np.stack([np.cos(heading), np.sin(heading)], -1)
    return xy[..., None, :] + d[..., None, :] * (length * frac)[:, None], np.sqrt((length / (2 * n)) ** 2 + (width / 2) ** 2)


def _gap(ca, ra, cb, rb):
    """Footprint gap [..., T] between circle sets [..., T, n, 2] and [T, n, 2], >= 0."""
    d = np.linalg.norm(ca[..., :, None, :] - cb[..., None, :, :], axis=-1).min(axis=(-2, -1))
    return np.maximum(d - ra - rb, 0.0)


def footprint_beta(scene, a, b, window, horizon, alts, cfg, valid):
    fut = slice(window + 1, window + 1 + horizon)
    la, wa = scene.shape_at(a, window)
    lb, wb = scene.shape_at(b, window)
    prev = np.concatenate([np.broadcast_to(scene.position[a, window, :2], (len(alts), 1, 2)), alts[:, :-1]], 1)
    step = alts - prev
    h_alt = np.arctan2(step[..., 1], step[..., 0])
    slow = np.linalg.norm(step, axis=-1) < 0.05  # standing: keep the current heading
    h_alt = np.where(slow, scene.heading[a, window], h_alt)
    cb, rb = _circles(scene.position[b, fut, :2], scene.heading[b, fut], lb, wb)
    ca_alt, ra = _circles(alts, h_alt, la, wa)
    ca_act, _ = _circles(scene.position[a, fut, :2], scene.heading[a, fut], la, wa)
    g_alt = np.where(valid[None], _gap(ca_alt, ra, cb, rb), np.inf).min(-1)
    g_act = np.where(valid, _gap(ca_act, ra, cb, rb), np.inf).min(-1)
    sat = cfg.d_sat or np.inf
    g_alt, g_act = np.minimum(g_alt, sat), min(float(g_act), sat)
    return float(cvar(torch.as_tensor(g_alt - g_act), cfg.cvar_alpha))


def side_diagnostics(model, scene, a, b, window, cfg, generator):
    """beta_s(a, b) as blame.pair_safety computes it, plus the distribution
    of C over a's motion set and how far the alternatives travel."""
    horizon = effective_horizon(scene, window, cfg)
    dist = model.distribution(scene, window, a)
    if dist is None:
        return None
    trajs, _, weights = motion_set(model, dist, cfg, generator)
    fut = slice(window + 1, window + 1 + horizon)
    s = torch.as_tensor(trajs[:, :horizon, :2], dtype=torch.float32)
    act = torch.as_tensor(scene.position[a, fut, :2], dtype=torch.float32)
    nb = torch.as_tensor(scene.position[b, fut, :2], dtype=torch.float32)
    vb = torch.as_tensor(scene.valid[b, fut]) & torch.as_tensor(scene.valid[a, fut])
    d_s = pairwise_min_distance_over_time(s, nb, valid_b=vb, d_sat=cfg.d_sat)
    d_a = pairwise_min_distance_over_time(act.unsqueeze(0), nb, valid_b=vb, d_sat=cfg.d_sat).squeeze(0)
    C = (d_s - d_a).numpy()
    # the same with D_g as the footprint gap (3 circles per box, headings of
    # the alternatives from their own displacement), not centre distance
    fp = footprint_beta(scene, a, b, window, horizon, np.asarray(trajs)[:, :horizon, :2], cfg, vb.numpy())
    start = scene.position[a, window, :2]
    alt_travel = np.linalg.norm(np.asarray(trajs)[:, horizon - 1, :2] - start, axis=-1)
    act_travel = float(np.linalg.norm(scene.position[a, window + horizon, :2] - start))
    alt_lat = None
    return {
        "beta": float(cvar(torch.as_tensor(C), cfg.cvar_alpha)), "fp_beta": fp,
        "d_actual": float(d_a), "C_min": float(C.min()), "C_med": float(np.median(C)), "C_max": float(C.max()),
        "frac_safer": float((C > 0.1).mean()),
        "travel_actual": act_travel, "travel_alt_med": float(np.median(alt_travel)),
        "travel_alt_p10": float(np.percentile(alt_travel, 10)), "travel_alt_p90": float(np.percentile(alt_travel, 90)),
        "speed_at_window": float(np.linalg.norm(scene.velocity[a, window])),
    }


class PairedDenseTNT:
    """DenseTNT with its second object-of-interest slot fixed per predicted
    agent (``pair[agent]``), as in CAT's own (sdc, adversary) instances;
    blame.pair_safety otherwise fills it with the vehicle nearest to the
    predicted agent at the window step (DenseTNT._partner)."""

    def __init__(self, model, pair):
        self.model, self.pair = model, pair

    def distribution(self, scene, step, agent, excluded=(), avoid_partner=()):
        partner = self.pair.get(agent)
        m = self.model
        if partner is None or not scene.valid[partner, step] or not scene.valid[agent, step] \
                or scene.types[agent] != "VEHICLE":
            return m.distribution(scene, step, agent, excluded, avoid_partner)
        rest = [i for i in range(scene.n_agents) if i not in (agent, partner) and i not in excluded]
        anchor = scene.position[agent, step, :2]
        from responsibility.densetnt import _last_valid_xy
        last_seen = np.where(scene.valid[rest][:, : step + 1].any(-1),
                             np.linalg.norm(_last_valid_xy(scene, rest, step) - anchor, axis=-1), np.inf)
        order = [agent, partner] + [rest[j] for j in np.argsort(last_seen, kind="stable")]
        mapping = m.instance(scene, step, agent, excluded, order=order, select=0)
        return None if mapping is None else m.goal_distributions([mapping])[0]

    def sample(self, dist, n, generator=None):
        return self.model.sample(dist, n, generator)

    def motion_set(self, dist, *a, **k):
        return self.model.motion_set(dist, *a, **k)


def expected_verdict(at_fault, ego):
    if at_fault is None:
        return "shared"
    return "ego" if ego == at_fault else "other"


def attribute(model, scene, case, c, info, sid, file_idx):
    rows = []
    pair = list(info["roles"].values())
    a, b = pair[0], pair[1]
    for ego, other in ((a, b), (b, a)):
        sc = replace(scene, sdc=ego)
        val = sc.valid.copy()
        val[ego, c + 1:] = False  # the episode ends with the collision (as scene_from_rollout)
        sc = replace(sc, valid=val)
        cfg = ResponsibilityConfig()
        blame = crash_blame(model, sc, ego, c, other, cfg)
        if blame is None:
            rows.append({"case": case, "scene": file_idx, "skip": "no-blame"})
            continue
        gen = torch.Generator().manual_seed(cfg.seed * 1000003 + c)
        de = side_diagnostics(model, sc, ego, other, blame.window, cfg, gen)
        do = side_diagnostics(model, sc, other, ego, blame.window, cfg, gen)
        weighted = crash_blame(model, sc, ego, c, other, ResponsibilityConfig(motion_set="weighted"))
        lb10 = crash_blame(model, sc, ego, c, other, cfg, lookback=10)
        # a longer window needs a longer horizon, or the horizon would end before the crash
        lb30 = crash_blame(model, sc, ego, c, other, ResponsibilityConfig(metric_horizon=30), lookback=30)
        sat20 = crash_blame(model, sc, ego, c, other, ResponsibilityConfig(d_sat=20.0))
        rss = rss_blame(sc, ego, other, c)
        paired = crash_blame(PairedDenseTNT(model, {ego: other, other: ego}), sc, ego, c, other, cfg)
        slot_e = model._partner(sc, blame.window, ego, ())
        slot_o = model._partner(sc, blame.window, other, ())
        role_of = {v: k for k, v in info["roles"].items()}
        row = {
            "case": case, "scene": file_idx, "scenario_id": sid, "ego_role": role_of[ego],
            "ego_id": sc.track_ids[ego], "other_id": sc.track_ids[other],
            "edit": info["edit"], "t0": info["t0"], "crash_step": c, "window": blame.window,
            "onset_vs_window": blame.window - info["t0"],
            "expected": expected_verdict(info["at_fault"], ego),
            "verdict": blame.verdict, "beta_ego": blame.beta_ego, "beta_other": blame.beta_other,
            "share": blame.share, "rule": blame.rule, "rss": rss.verdict, "rss_case": rss.case,
            "partner_ok": crash_partner(sc, ego, c) == other,
            "slot_ego_is_other": slot_e == other, "slot_other_is_ego": slot_o == ego,
            "pair_verdict": paired.verdict if paired else None,
            "pair_beta_ego": paired.beta_ego if paired else None,
            "pair_beta_other": paired.beta_other if paired else None,
            "w_verdict": weighted.verdict if weighted else None,
            "w_beta_ego": weighted.beta_ego if weighted else None,
            "w_beta_other": weighted.beta_other if weighted else None,
            "lb10_verdict": lb10.verdict if lb10 else None,
            "lb10_beta_ego": lb10.beta_ego if lb10 else None, "lb10_beta_other": lb10.beta_other if lb10 else None,
            "lb30_verdict": lb30.verdict if lb30 else None,
            "lb30_beta_ego": lb30.beta_ego if lb30 else None, "lb30_beta_other": lb30.beta_other if lb30 else None,
            "sat20_verdict": sat20.verdict if sat20 else None,
            "sat20_beta_ego": sat20.beta_ego if sat20 else None,
            "sat20_beta_other": sat20.beta_other if sat20 else None,
            "speed_ego_c": speed(sc, ego, c), "speed_other_c": speed(sc, other, c),
        }
        for prefix, d in (("e_", de), ("o_", do)):
            for k, v in (d or {}).items():
                row[prefix + k] = v
        if de is not None and do is not None:
            row["fp_verdict"] = split(de["fp_beta"], do["fp_beta"])[1]
        if de is not None and abs(de["beta"] - blame.beta_ego) > 1e-4:
            row["replicate_mismatch"] = de["beta"] - blame.beta_ego
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", default="raw_scenes_500")
    parser.add_argument("--first", type=int, default=400)
    parser.add_argument("--last", type=int, default=499)
    parser.add_argument("--cases", default="C1,C2,C3,C4")
    parser.add_argument("--per-case", type=int, default=30, help="instances per case at most")
    parser.add_argument("--per-scene", type=int, default=1, help="instances per case and scene at most")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    parser.add_argument("--time-limit", type=float, default=35 * 60, help="s")
    args = parser.parse_args()

    from responsibility.densetnt import DenseTNT
    model = DenseTNT(device=args.device)
    out = FsPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = args.cases.split(",")
    counts = Counter()
    skips = defaultdict(Counter)
    rows = []
    start = time.time()
    rng = np.random.default_rng(0)
    for idx in range(args.first, args.last + 1):
        if time.time() - start > args.time_limit or all(counts[c] >= args.per_case for c in cases):
            break
        scene = Scene.load(FsPath(args.scenes) / f"{idx}.pkl")
        for case in cases:
            if counts[case] >= args.per_case:
                continue
            built = 0
            cands = candidates(scene, case)
            rng.shuffle(cands)
            for a, b, t0s, pa, pb in cands:
                if built >= args.per_scene:
                    break
                # onsets spread over the admissible steps: first, middle, last
                n_try = 6 if case == "C4" else 3
                tries = sorted(set(t0s[j] for j in np.linspace(0, len(t0s) - 1, n_try).astype(int)))
                rng.shuffle(tries)
                for t0 in tries:
                    res = BUILDERS[case](scene, a, b, pa, pb, t0)
                    if isinstance(res, str):
                        skips[case][res] += 1
                        continue
                    sc, c, info = res
                    new = attribute(model, sc, case, c, info, scene.scenario_id, idx)
                    rows.extend(new)
                    built += 1
                    counts[case] += 1
                    break
        print(f"scene {idx}: {dict(counts)}  {time.time() - start:.0f} s", flush=True)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "instances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out / "skips.json", "w") as f:
        json.dump({k: dict(v) for k, v in skips.items()}, f, indent=1)
    print("skips", json.dumps({k: dict(v) for k, v in skips.items()}))
    print(f"done: {len(rows)} rows in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
