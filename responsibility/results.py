"""Reading compute_responsibility.py outputs: windows.csv, or one
windows.shard-<i>-of-<n>.csv per shard when a run was split over processes."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

INT_FIELDS = ("step", "n_neighbours")
FLOAT_FIELDS = ("time", "speed", "safety", "courtesy")


def _scene_key(scene):
    """Numeric scene names (CAT's 0..499) in numeric order, others after."""
    return (0, int(scene), "") if str(scene).isdigit() else (1, 0, str(scene))


def window_files(run_dir) -> List[Path]:
    files = sorted(Path(run_dir).glob("windows*.csv"))
    if not files:
        raise FileNotFoundError(f"no windows*.csv in {run_dir}")
    return files


def read_windows(run_dir) -> List[Dict]:
    """Every window row of a run (all shards), typed, in (scene, step) order.
    A scene recomputed after an interrupted run appears twice in the files;
    the last copy of each (scene, agent, step) is kept."""
    unique = {}
    for path in window_files(run_dir):
        with open(path) as f:
            for r in csv.DictReader(f):
                unique[(r["scene"], r["agent_id"], r["step"])] = r
    rows = list(unique.values())
    for r in rows:
        for key in INT_FIELDS:
            r[key] = int(r[key])
        for key in FLOAT_FIELDS:
            r[key] = float(r[key])
    rows.sort(key=lambda r: (_scene_key(r["scene"]), r["agent_id"], r["step"]))
    return rows


def sequences(rows: List[Dict], run: str = "") -> Dict[Tuple[str, str, str], List[Dict]]:
    """Windows grouped into one time-ordered sequence per (run, scene, agent)."""
    out: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in rows:
        out.setdefault((run, r["scene"], r["agent_id"]), []).append(r)
    for seq in out.values():
        seq.sort(key=lambda r: r["step"])
    return out


def features(seq: List[Dict], log_courtesy: bool = False) -> np.ndarray:
    """[T, 2] (safety, courtesy) of a sequence, as the HMM sees them."""
    courtesy = np.array([r["courtesy"] for r in seq])
    if log_courtesy:
        courtesy = np.log1p(courtesy)
    return np.stack([np.array([r["safety"] for r in seq]), courtesy], axis=-1)
