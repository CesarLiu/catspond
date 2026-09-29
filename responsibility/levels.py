"""Responsibility levels (paper Sec. IV-A) as a data-driven scale of how
aggressively an agent drives.

A Gaussian HMM over the per-window (safety, courtesy) sequences of logged
driving discovers H levels (H chosen by BIC); the Bayes filter (Eq. 6) then
assigns every window the level most probable given the windows so far.
HMM states come out in arbitrary order, so they are relabelled by how
aggressive their mean is -- level 0 the calmest, level H-1 the most
aggressive -- which is what makes "the top level(s)" a verdict.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from responsibility.hmm import GaussianHMM


def aggressiveness(hmm: GaussianHMM, scale: np.ndarray) -> np.ndarray:
    """Per state: its mean safety and courtesy, each in units of ``scale``
    (the features' spread over the fitted data), summed."""
    return (hmm.means_ / np.maximum(scale, 1e-9)).sum(axis=-1)


def relabel_by_aggressiveness(hmm: GaussianHMM, scale: np.ndarray) -> np.ndarray:
    """Reorders the states in place so their aggressiveness increases with the
    level index; returns the permutation applied (new level -> old state)."""
    order = np.argsort(aggressiveness(hmm, scale), kind="stable")
    hmm.startprob_ = hmm.startprob_[order]
    hmm.transmat_ = hmm.transmat_[np.ix_(order, order)]
    hmm.means_ = hmm.means_[order]
    hmm.covars_ = hmm.covars_[order]
    return order


def assign_levels(hmm: GaussianHMM, sequences: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """For each sequence: the filtered level of every window [T] and the
    posterior probability of that level [T] (causal: window k only uses
    windows <= k)."""
    out = []
    for obs in sequences:
        post = hmm.forward_filter(obs)
        level = post.argmax(axis=-1)
        out.append((level, post[np.arange(len(level)), level]))
    return out


def elevated_in(hmm: GaussianHMM, scale: np.ndarray, calm: float = 0.25) -> List[str]:
    """Per level, which responsibility it stands out in, relative to the
    calmest level and in units of ``scale``: "safety", "courtesy", "both", or
    "-" when neither exceeds ``calm`` spreads."""
    base = hmm.means_[0]
    out = []
    for mean in hmm.means_:
        z = (mean - base) / np.maximum(scale, 1e-9)
        high = z > calm
        out.append("both" if high.all() else "safety" if high[0] else "courtesy" if high[1] else "-")
    return out


def level_table(hmm: GaussianHMM, level_counts: Dict[str, np.ndarray], scale: np.ndarray) -> List[Dict]:
    """Per level: its mean (safety, courtesy), spread, the dimension it is
    elevated in, and each run's share of windows in it."""
    kinds = elevated_in(hmm, scale)
    rows = []
    for z in range(hmm.n_states):
        row = {
            "level": z,
            "mean_safety": float(hmm.means_[z, 0]), "mean_courtesy": float(hmm.means_[z, 1]),
            "std_safety": float(np.sqrt(hmm.covars_[z, 0])), "std_courtesy": float(np.sqrt(hmm.covars_[z, 1])),
            "stay_probability": float(hmm.transmat_[z, z]),
            "elevated_in": kinds[z],
        }
        for run, counts in level_counts.items():
            row[f"share_{run}"] = float(counts[z] / max(counts.sum(), 1))
        rows.append(row)
    return rows
