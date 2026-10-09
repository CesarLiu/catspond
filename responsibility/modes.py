"""A small, spread-out motion set by goal non-maximum suppression
(``--motion-set nms``).

The weighted motion set covers DenseTNT's whole goal grid (a median of 944
goals per window on the first 30 scenes), and its top 40 goals sit next to
one another around the mode. This set instead takes N goals (N =
n_safety_samples, 40) spread over the distribution, with the rule CAT uses
to pick its predicted modes (advgen.utils.select_goals_by_NMS):

1. Sort the goals by probability. Pick them greedily, skipping any goal
   closer than ``threshold`` (m) to one already picked, where the threshold
   is CAT's nms_threshold (7.2 m) times the speed scale factor (0.5 below
   1.4 m/s, 1.0 above 11 m/s, linear between).
2. If fewer than N survive, fill up with the next most probable goals not
   yet picked (CAT fills with random goals instead; here the set must be
   reproducible).
3. Weight each picked goal by the total probability of the goals whose
   nearest picked goal it is, so the weights sum to 1 and the set
   represents the whole distribution, not only its N picked points.

Only goals of positive probability are candidates, so a distribution
restricted to the route lanes (motion_filter.restrict_to_route) never picks
an off-route goal.
"""

from typing import Tuple

import numpy as np

NMS_THRESHOLD = 7.2  # m, CAT's args.nms_threshold (advgen/utils.py)
SPEED_LOWER, SPEED_UPPER = 1.4, 11.0  # m/s, advgen/utils_cython.pyx
SCALE_LOWER, SCALE_UPPER = 0.5, 1.0


def speed_scale_factor(speed: float) -> float:
    """advgen.utils_cython.speed_scale_factor, in numpy (for MTR, which does
    not load advgen)."""
    fraction = (float(speed) - SPEED_LOWER) / (SPEED_UPPER - SPEED_LOWER)
    return SCALE_LOWER + (SCALE_UPPER - SCALE_LOWER) * min(max(fraction, 0.0), 1.0)


def nms_select(points: np.ndarray, probs: np.ndarray, n: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """Indices [M] of the picked goals (M = min(n, goals of positive
    probability)), in the order picked, and their weights [M] (summing to 1):
    ``points`` [G, 2] in metres, ``probs`` [G], ``threshold`` in metres
    (already scaled by speed)."""
    points = np.asarray(points, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)
    order = np.argsort(-probs, kind="stable")
    order = order[probs[order] > 0]
    picked = []
    for i in order:
        if picked and np.min(np.linalg.norm(points[picked] - points[i], axis=-1)) < threshold:
            continue
        picked.append(i)
        if len(picked) == n:
            break
    if len(picked) < n:  # too few distinct goals: the next most probable ones
        chosen = set(picked)
        picked += [i for i in order if i not in chosen][: n - len(picked)]
    picked = np.asarray(picked, dtype=np.int64)
    if len(picked) == 0:
        return picked, np.zeros(0)
    mass = order  # goals of zero probability add nothing to a weight
    nearest = np.argmin(np.linalg.norm(points[mass][:, None, :] - points[picked][None], axis=-1), axis=1)
    weights = np.bincount(nearest, weights=probs[mass], minlength=len(picked))
    return picked, weights / weights.sum()
