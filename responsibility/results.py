"""Reading compute_responsibility.py outputs: windows.csv, or one
windows.shard-<i>-of-<n>.csv per shard when a run was split over processes
(likewise crashes*.csv for runs over rollouts), and the run's config.json."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def read_crashes(run_dir) -> List[Dict]:
    """The collision attributions of a run over rollouts (one per scene; the
    last copy if a scene was recomputed), in scene order; [] if none."""
    unique = {}
    for path in sorted(Path(run_dir).glob("crashes*.csv")):
        with open(path) as f:
            for r in csv.DictReader(f):
                unique[r["scene"]] = r
    rows = [unique[k] for k in sorted(unique, key=_scene_key)]
    for r in rows:
        for key in ("beta_ego", "beta_other", "share"):
            r[key] = float(r[key]) if r.get(key) not in (None, "") else None
    return rows


def read_config(run_dir) -> Optional[Dict]:
    path = Path(run_dir) / "config.json"
    return json.loads(path.read_text()) if path.exists() else None


def calibrate(values, quantile: float, floor: float) -> float:
    """Aggressiveness threshold: the quantile of the positive values
    (non-interacting windows sit at 0 and would drag a plain quantile to 0),
    never below ``floor``."""
    positive = np.asarray([v for v in values if v > 0])
    if positive.size == 0:
        return floor
    return max(float(np.quantile(positive, quantile)), floor)


def calibrate_timid(values, quantile: float, floor: float) -> float:
    """Timidity threshold on safety responsibility: the quantile of the
    negative values (q0.1 = the most cautious tenth of the windows that kept
    more margin than their alternatives), never above ``-floor``."""
    negative = np.asarray([v for v in values if v < 0])
    if negative.size == 0:
        return -floor
    return min(float(np.quantile(negative, quantile)), -floor)


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
