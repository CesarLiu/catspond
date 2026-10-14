"""Recomputes beta_s offline from a run's records (compute_responsibility
--save-records) for a grid of horizons H (steps of 0.1 s), CVaR levels alpha
and saturations d_sat, without the motion model: the records hold every
window's motion set, so only the distances change. This is how the aggression
test varies T_f (2 s in the paper) up to 8 s.

For each window k and each neighbour b of the record, with the alternatives
the run kept for b (its filters), D_g is the closest approach to b's logged
track over the next H steps, both for every alternative and for the agent's
actual track, each clipped at d_sat; beta_s is the CVaR_alpha of the
difference (probability-weighted for weighted, top-k and NMS motion sets).

One row per (scene, step, other, H, alpha, d_sat) with columns beta,
gap_actual (the actual closest approach, a proximity baseline),
gap_alt_mean, overlap (export_adv_scenes' first overlap with the logged ego,
-1 when none or for logged scenes) and speed. d_sat -1 means no saturation.

    python -m scripts.responsibility.offline_safety_grid \\
        logs/responsibility/validity/runs/nms_ooi/adv_cat logs/responsibility/validity/offline/nms_ooi__adv_cat.csv.gz
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.aggression_test import scene_index  # noqa: E402
from responsibility.metrics import weighted_cvar  # noqa: E402
from responsibility.records import load_record, scene_of  # noqa: E402
from responsibility.risk import cvar  # noqa: E402

HORIZONS = [10, 20, 30, 40, 50, 60, 80]
ALPHAS = [0.1, 0.5, 0.9]
D_SATS = [10.0, 5.0, 1e9]


def grid(run_dir: Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    weighted = cfg["responsibility"]["motion_set"] != "sampled"
    overlap = {k: v.get("first_overlap_with_logged_ego") for k, v in scene_index(Path(cfg["scenes"])).items()}
    rows = []
    for rp in sorted((run_dir / "records").glob("*.pkl"), key=lambda p: int(p.stem)):
        rec = load_record(rp)
        scene = scene_of(rec)
        a = rec["agent"]
        ov = overlap.get(rp.stem)
        for fr in rec["frames"]:
            if not fr["metric_samples"]:
                continue
            k = fr["step"]
            for tid in fr["observation"]["per_neighbour"]:
                b = scene.index(tid)
                s = torch.as_tensor(fr["samples"][:, :, :2], dtype=torch.float32)
                lw = fr["sample_log_prob"]
                if tid in fr["kept"]:  # the alternatives the run's filters kept for this neighbour
                    s, lw = s[fr["kept"][tid]], lw[fr["kept"][tid]]
                if len(s) == 0:
                    continue
                w = torch.as_tensor(np.exp(lw.astype(np.float64)), dtype=torch.float32) if weighted else None
                T = min(80, scene.n_steps - 1 - k)
                fut = slice(k + 1, k + 1 + T)
                act = torch.as_tensor(scene.position[a, fut, :2], dtype=torch.float32)
                bb = torch.as_tensor(scene.position[b, fut, :2], dtype=torch.float32)
                vb = torch.as_tensor(scene.valid[a, fut] & scene.valid[b, fut])
                ds = torch.linalg.norm(s[:, :T] - bb, dim=-1)  # [N, T]
                da = torch.linalg.norm(act - bb, dim=-1)  # [T]
                for H in HORIZONS:
                    if H > T:
                        continue
                    m = vb[:H]
                    if not m.any():
                        continue
                    ds_h = ds[:, :H].masked_fill(~m, np.inf).min(-1).values
                    da_h = da[:H].masked_fill(~m, np.inf).min()
                    for d_sat in D_SATS:
                        c = ds_h.clamp(max=d_sat) - da_h.clamp(max=d_sat)
                        for al in ALPHAS:
                            beta = float(weighted_cvar(c, w, al)) if weighted else float(cvar(c, al))
                            rows.append((int(rp.stem), k, tid, H, al, d_sat if d_sat < 1e8 else -1, beta,
                                         float(da_h), float(ds_h.mean()), ov if ov is not None else -1,
                                         float(np.linalg.norm(scene.velocity[a, k]))))
    return pd.DataFrame(rows, columns=["scene", "step", "other", "H", "alpha", "dsat", "beta", "gap_actual",
                                       "gap_alt_mean", "overlap", "speed"])


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run_dir, out = sys.argv[1], Path(sys.argv[2])
    df = grid(run_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(run_dir, len(df))


if __name__ == "__main__":
    main()
