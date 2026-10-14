"""The labelled test of whether counterfactual responsibility tells aggressive
driving apart, shared by the scripts that build, score and plot it.

CAT's adversaries collide with the ego by construction, so they are labelled
aggressive without annotation. The test intervenes on one vehicle at a time:
in every condition exactly one agent's trajectory differs from the log, and
everything else in the scene stays as logged, so a change in its score can
only come from that agent. The adversaries of all conditions are DenseTNT's
own candidates for the scenario's other object of interest; they differ only
in how they were chosen (`export_adv_scenes.py`):

    condition       agent  scene set     how the trajectory was chosen          label
    adv_logged      adv    logged        the log                                normal (reference)
    sdc_logged      sdc    logged        the log                                normal (reference)
    adv_mode        adv    near0_1000    most probable non-colliding candidate  normal (control)
    adv_near1       adv    near1_0.5     closest approach 1 +- 0.5 m            aggressive, weak
    adv_fairinf     adv    fairinf_0.5   avoidable collision (avoidability 0.5) aggressive
    adv_cat         adv    cat           CAT's rule: most likely collision      aggressive, strong
    adv_maxgap      adv    near30_0.5    the largest gap to the ego             timid
    sdc_vs_<x>      sdc    as for adv_x  the logged ego facing that adversary   unchanged; sdc_vs_cat = victim

Retimed variants of the ego (`retime_agent.py`) add sdc_slow0.5, sdc_slow0.7
and sdc_fast1.3, with the logged adversary measured against them as
adv_vs_<x>. A condition's reference is the same agent's logged drive; its
partner is the other agent of the same scenes.

Scores are compared by AUROC, P(score of the condition > score of the
reference), computed from ranks with ties counted half.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Condition:
    scene_set: str  # key of SCENE_SETS
    agent: str  # "sdc" or "adv": the object of interest measured
    partner: str  # the condition of the other agent in the same scenes
    reference: str | None  # the same agent's logged drive (None for the references)
    label: str


SCENE_SETS = {
    # name: how the folder is made (run_aggression_test.sh); "logged" and "cat" are given as paths
    "logged": "the logged scenes (raw_scenes_500)",
    "cat": "export_adv_scenes --rule cat",
    "fairinf_0.5": "export_adv_scenes --rule fair --resp_threshold inf --resp_avoid 0.5",
    "near1_0.5": "export_adv_scenes --rule near --near_gap 1 --near_tol 0.5",
    "near0_1000": "export_adv_scenes --rule near --near_gap 0 --near_tol 1000",
    "near30_0.5": "export_adv_scenes --rule near --near_gap 30 --near_tol 0.5",
    "sdc_speed0.5": "retime_agent --agent sdc --factor 0.5",
    "sdc_speed0.7": "retime_agent --agent sdc --factor 0.7",
    "sdc_speed1.3": "retime_agent --agent sdc --factor 1.3",
}

CONDITIONS = {
    "adv_logged": Condition("logged", "adv", "sdc_logged", None, "normal"),
    "sdc_logged": Condition("logged", "sdc", "adv_logged", None, "normal"),
    "adv_mode": Condition("near0_1000", "adv", "sdc_vs_mode", "adv_logged", "normal (control)"),
    "adv_near1": Condition("near1_0.5", "adv", "sdc_vs_near1", "adv_logged", "aggressive (near miss)"),
    "adv_fairinf": Condition("fairinf_0.5", "adv", "sdc_vs_fairinf", "adv_logged", "aggressive (avoidable)"),
    "adv_cat": Condition("cat", "adv", "sdc_vs_cat", "adv_logged", "aggressive (CAT)"),
    "adv_maxgap": Condition("near30_0.5", "adv", "sdc_vs_maxgap", "adv_logged", "timid"),
    "sdc_vs_mode": Condition("near0_1000", "sdc", "adv_mode", "sdc_logged", "normal"),
    "sdc_vs_near1": Condition("near1_0.5", "sdc", "adv_near1", "sdc_logged", "victim (near miss)"),
    "sdc_vs_fairinf": Condition("fairinf_0.5", "sdc", "adv_fairinf", "sdc_logged", "victim (avoidable)"),
    "sdc_vs_cat": Condition("cat", "sdc", "adv_cat", "sdc_logged", "victim (CAT)"),
    "sdc_vs_maxgap": Condition("near30_0.5", "sdc", "adv_maxgap", "sdc_logged", "normal"),
    "sdc_slow0.5": Condition("sdc_speed0.5", "sdc", "adv_vs_slow0.5", "sdc_logged", "speed x0.5"),
    "sdc_slow0.7": Condition("sdc_speed0.7", "sdc", "adv_vs_slow0.7", "sdc_logged", "speed x0.7"),
    "sdc_fast1.3": Condition("sdc_speed1.3", "sdc", "adv_vs_fast1.3", "sdc_logged", "speed x1.3"),
    "adv_vs_slow0.5": Condition("sdc_speed0.5", "adv", "sdc_slow0.5", "adv_logged", "partner of x0.5"),
    "adv_vs_slow0.7": Condition("sdc_speed0.7", "adv", "sdc_slow0.7", "adv_logged", "partner of x0.7"),
    "adv_vs_fast1.3": Condition("sdc_speed1.3", "adv", "sdc_fast1.3", "adv_logged", "partner of x1.3"),
}

# compute_responsibility settings of the runs (runs/<config>/<condition>)
CONFIGS = {
    "paper": "--stride 1",  # the paper's setting: every frame, beta_s max over neighbours, beta_c
    "sampled_ooi": "--use-ooi --no-courtesy --save-records",
    "nms_ooi": "--use-ooi --motion-set nms --no-courtesy --save-records",
    "nmsvalid_ooi": "--use-ooi --motion-set nms --valid-counterfactuals --no-courtesy --save-records",
    "nms_ooi_court": "--use-ooi --motion-set nms",
    "nms_int": "--motion-set nms --no-courtesy",
}


def scene_dir(scene_set: str, val_dir: Path, raw_scenes: Path, cat_scenes: Path) -> Path:
    """The folder of a scene set: the logged scenes, CAT's export, or val_dir/scenes/<set>."""
    if scene_set == "logged":
        return Path(raw_scenes)
    if scene_set == "cat":
        return Path(cat_scenes)
    return Path(val_dir) / "scenes" / scene_set


def scene_index(folder: Path) -> dict:
    """export_adv_scenes' index of a scene folder (next to it, or inside it), or {}."""
    folder = Path(folder)
    for p in (folder.parent / f"{folder.name}.index.json", folder / "index.json"):
        if p.exists():
            return json.loads(p.read_text())
    return {}


def first_overlaps(folder: Path) -> dict[int, int]:
    """Scene -> first step the adversary's footprint overlaps the logged ego (scenes without one left out)."""
    out = {}
    for k, v in scene_index(folder).items():
        ov = v.get("first_overlap_with_logged_ego")
        if ov is not None:
            out[int(k)] = int(ov)
    return out


def auroc(pos, neg) -> float:
    """P(pos > neg) + P(pos = neg) / 2 from ranks; nan when either side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_auroc(pos: pd.DataFrame, neg: pd.DataFrame, col: str, n: int = 2000, rng=None):
    """AUROC of column col with a 95% interval from resampling scenes (both frames have a 'scene' column).

    Returns (point, low, high) over the scenes present in both. Pass one
    numpy Generator to a series of calls to make the whole series reproducible."""
    rng = np.random.default_rng(0) if rng is None else rng
    scenes = np.intersect1d(pos.scene.unique(), neg.scene.unique())
    gp = {s: g[col].to_numpy() for s, g in pos[pos.scene.isin(scenes)].groupby("scene")}
    gn = {s: g[col].to_numpy() for s, g in neg[neg.scene.isin(scenes)].groupby("scene")}
    if len(scenes) == 0:
        return float("nan"), float("nan"), float("nan")
    point = auroc(np.concatenate(list(gp.values())), np.concatenate(list(gn.values())))
    boots = []
    for _ in range(n):
        pick = rng.choice(scenes, len(scenes))
        boots.append(auroc(np.concatenate([gp[s] for s in pick]), np.concatenate([gn[s] for s in pick])))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def fmt_ci(point_lo_hi) -> str:
    p, lo, hi = point_lo_hi
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"


def longest_run(mask, frame_s: float) -> float:
    """Length in seconds of the longest stretch of consecutive True frames."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best * frame_s


def load_offline(val_dir: Path, config: str, condition: str) -> pd.DataFrame | None:
    """offline_safety_grid's table of a run, without the windows at or after the first overlap."""
    p = Path(val_dir) / "offline" / f"{config}__{condition}.csv.gz"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[(d.overlap < 0) | (d.step < d.overlap)]


def at_setting(table: pd.DataFrame, horizon: int, alpha: float, d_sat: float) -> pd.DataFrame:
    """Rows of an offline table at one (H, alpha, d_sat), indexed by (scene, step); d_sat -1 = none."""
    return table[(table.H == horizon) & (table.alpha == alpha) & (table.dsat == d_sat)].set_index(["scene", "step"])


def pair_scores(agent: pd.DataFrame, partner: pd.DataFrame) -> pd.DataFrame:
    """Per-scene scores of an agent against its partner at the same windows (tables from at_setting).

      beta      max_k beta_s(agent -> partner)
      rel       max_k [beta_s(agent -> partner) - beta_s(partner -> agent)]: the two-sided score
      prox      -min_k actual gap (a proximity baseline)
      beta_min, rel_min   the minima (timid direction)
      gap       min_k actual gap (m)"""
    j = agent[["beta", "gap_actual"]].join(partner[["beta"]], rsuffix="_o", how="inner")
    j["rel"] = j.beta - j.beta_o
    g = j.groupby(level="scene")
    return pd.DataFrame({"beta": g.beta.max(), "rel": g.rel.max(), "prox": -g.gap_actual.min(),
                         "beta_min": g.beta.min(), "rel_min": g.rel.min(), "gap": g.gap_actual.min()})
