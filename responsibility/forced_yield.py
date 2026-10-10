"""How hard an agent's entry into a conflict forced another road user who
had the right of way to brake: an open-loop measure of assertive driving
that is not (yet) unsafe.

Safety responsibility (metrics.py, beta_s) and the yield duty of
right_of_way.py both look at the unsafe end: a gap the other could not have
kept, or a holder that could no longer stop. A driver who takes the right
of way from someone still far enough away to stop is assertive without
being dangerous, and neither registers it. This module measures it on the
two agents' tracks as they are (logged, or the agent's planned or
simulated track against the others' logs, which do not react).

Conflict and right of way come from right_of_way.py: the conflict point P*
of the two paths at the step judged (``at``; by default their closest
approach), each agent's conflict zone around it, and the holder of the
right of way (``priority``). At every step the agent enters its zone, the
braking it demands of the other is

    a_req = v_o^2 / (2 (d_o - v_o rho))                              [m/s^2]

where d_o is the other's distance to its own zone (m) and v_o its speed
(m/s): the constant deceleration that stops it before the zone after the
response time rho (RightOfWayParams.rho, RSS's 1 s). It is 0 when the other
has passed its zone or stands still before it, and infinite when it is
inside its zone or within one response time of it (no braking stops it in
time). The demand of the agent's worst entry is ``demand``.

a_req above RightOfWayParams.brake (RSS's brake_min, 4 m/s^2) is exactly a
violated yield duty (can_stop < 0), so the duty is the threshold of this
scale at its unsafe end. Below it, the scale grades the assertive band: a
holder made to brake at 2-4 m/s^2 was forced to yield but could. The
demand counts as forcing only when the other held the right of way
(``forced``): demanding braking of a yielder is that yielder's duty.
Thresholds are calibrated on logged yielders (scripts/responsibility/
evaluate_forced_yield.py), not fixed here.

Same-direction conflicts (cut-ins) have no holder by default
(right_of_way.py) and get no ``forced`` value; their zone geometry assumes
crossing paths.
"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from responsibility.right_of_way import RightOfWayParams, _crossings, _enters, _Path, _signals, _window, priority
from responsibility.scene import Scene

STANDSTILL = 0.1  # m/s: below this the other stands and needs no braking
CROSSING_NEAR = 100.0  # m: RightOfWayParams.near when the conflict is judged at the later arrival


def required_braking(distance, speed, rho: float) -> np.ndarray:
    """m/s^2 that stops a vehicle at ``speed`` (m/s) within ``distance`` (m)
    after ``rho`` s at constant speed; inf when it cannot stop in time."""
    distance, speed = np.asarray(distance, dtype=float), np.asarray(speed, dtype=float)
    room = distance - speed * rho
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(room > 0, speed ** 2 / (2 * np.where(room > 0, room, 1.0)), np.inf)
    return np.where(speed < STANDSTILL, np.where(distance >= 0, 0.0, np.inf), out)


@dataclass
class ForcedYield:
    case: str  # right_of_way.py's case (rule that named the holder, or why there is none)
    kind: Optional[str]  # conflict kind: "crossing", "merge", "closest"; None without a conflict
    holder: str  # "agent", "other" or "none"
    demand: Optional[float] = None  # m/s^2 the agent's worst entry demanded of the other (inf: could not stop)
    entry: Optional[int] = None  # step of that entry; None: the agent was not seen entering its zone
    demand_on_agent: Optional[float] = None  # the same, of the other's entries on the agent

    @property
    def forced(self) -> Optional[float]:
        """``demand`` when the other held the right of way, else None."""
        return self.demand if self.holder == "other" else None

    def as_row(self):
        def num(x):
            return "" if x is None else "inf" if np.isinf(x) else round(float(x), 3)
        return {"case": self.case, "kind": self.kind or "", "holder": self.holder, "demand": num(self.demand),
                "entry": "" if self.entry is None else self.entry, "forced": num(self.forced),
                "demand_on_agent": num(self.demand_on_agent)}


def crossing_step(scene: Scene, a: int, b: int) -> Optional[int]:
    """The step the later of a and b reaches the crossing of their whole
    tracks that they reach closest in time (None: the tracks do not cross,
    or one is not seen then). Paths up to that step reach P*: the later
    one stands at it, the earlier one has passed it."""
    last = scene.n_steps - 1
    pa, pb = _Path(scene, a, last, 0.0), _Path(scene, b, last, 0.0)
    best = None
    for s_a, s_b, _ in _crossings(pa, pb):
        t = []
        for path, s in ((pa, s_a), (pb, s_b)):
            reached = np.flatnonzero(path.s[:path.n_real] >= s)
            t.append(int(path.steps[reached[0]]) if reached.size else None)
        if None in t:
            continue
        if best is None or abs(t[0] - t[1]) < best[0]:
            best = (abs(t[0] - t[1]), max(t))
    if best is None or not (scene.valid[a, best[1]] and scene.valid[b, best[1]]):
        return None
    return best[1]


def _demand(in_x, in_o, cleared_o, speed_o, rho):
    """(worst a_req of x's entries on o, its index into the steps) or (None, None)."""
    entries = np.flatnonzero(_enters(in_x) > 0)
    if not entries.size:
        return None, None
    a = required_braking(-in_o, speed_o, rho)  # -in_zone: the distance to the zone while before it
    a = np.where(cleared_o > 0, 0.0, np.where(in_o > 0, np.inf, a))
    k = entries[int(np.argmax(a[entries]))]
    return float(a[k]), int(k)


def forced_yield(scene: Scene, agent: int, other: int, at: Optional[int] = None,
                 params: Optional[RightOfWayParams] = None) -> ForcedYield:
    """The braking ``agent``'s entries into the conflict with ``other``
    demanded of it (module docstring). ``at``: the step whose paths define
    the conflict; by default the pair's closest approach, or, when the
    paths up to it neither cross nor merge, the step the later of the two
    reaches the crossing of their whole tracks (crossing_step)."""
    p = params or RightOfWayParams()
    both = scene.valid[agent] & scene.valid[other]
    if not both.any():
        return ForcedYield("not-seen-together", None, "none")
    if at is None:
        gap = np.linalg.norm(scene.position[agent, :, :2] - scene.position[other, :, :2], axis=-1)
        prio = priority(scene, agent, other, int(np.argmin(np.where(both, gap, np.inf))), p)
        if prio.conflict is None or prio.conflict.kind == "closest":
            # paths up to the closest approach (plus EXTEND m) stop short of P* when one passed
            # seconds before the other; judge the crossing of their whole tracks instead
            at = crossing_step(scene, agent, other)
            if at is not None:
                late = priority(scene, agent, other, at, replace(p, near=max(p.near, CROSSING_NEAR)))
                if late.conflict is not None and late.conflict.kind != "closest":
                    prio = late
    else:
        prio = priority(scene, agent, other, at, p)
    holder = "none" if prio.holder is None else "agent" if prio.holder == agent else "other"
    c = prio.conflict
    out = ForcedYield(prio.case, None if c is None else c.kind, holder)
    if c is None or prio.approaches is None or (prio.case == "same-direction" and not p.same_direction_rules):
        return out
    steps = _window(scene, agent, other, scene.n_steps - 1)
    if steps.size < 2:
        return out
    sig = {}
    for x, o, s_star in ((agent, other, c.s[0]), (other, agent, c.s[1])):
        in_zone, cleared, _ = _signals(scene, x, o, _Path(scene, x, int(steps[-1]), p.extend), s_star, c.angle,
                                       steps, p)
        sig[x] = (in_zone, cleared, np.linalg.norm(scene.velocity[x, steps], axis=-1).astype(float))
    out.demand, k = _demand(sig[agent][0], sig[other][0], sig[other][1], sig[other][2], p.rho)
    out.entry = None if k is None else int(steps[k])
    out.demand_on_agent, _ = _demand(sig[other][0], sig[agent][0], sig[agent][1], sig[agent][2], p.rho)
    return out
