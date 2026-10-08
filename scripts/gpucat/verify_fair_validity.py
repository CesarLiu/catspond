"""Checks whether the fair adversary's two tests rest on valid
counterfactuals (gpucat/PLAN.md, before M3's fair run).

The fair rule (responsibility/adversarial.py) keeps a candidate j only if

  beta_j  <= tau  the adversary's safety responsibility toward the ego: the
                  CVaR over the adversary's 40 DenseTNT samples of how much
                  more distance to the ego they keep than j does;
  avoid_j >= rho  the share of the ego's 40 DenseTNT samples that never
                  touch j.

Both sets are DenseTNT's raw samples. Some of them are not things a driver
in that place would do: another route, off the road, a car's impossible
acceleration. This script filters them with the rules that
responsibility/motion_filter.py applies to the responsibility metrics, and
re-runs the selection:

  adversary  lane_route, drivable (3 m), kinematics, and no drive through a
             third agent (the ego excepted): --valid-counterfactuals
  ego        drivable (3 m) and kinematics: a physically valid escape (one
             onto another route still avoids the collision)

It reports what the filters keep, how beta and avoid move, and how often the
fair choice changes, against the logged ego route (CAT's state before
training) and against five simulated histories per scene (as
verify_selection.py builds them).

    python -m scripts.gpucat.verify_fair_validity --bank logs/gpucat/candidates.npz --out logs/gpucat/fair_validity.json
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from responsibility.adversarial import adversary_responsibility, cat_candidate_probs, cat_collision_scores, select  # noqa: E402
from responsibility.metrics import ResponsibilityConfig  # noqa: E402
from scripts.gpucat.precompute_valid import HORIZON, RHO, TAU, load_scene, make_generator, valid_sets  # noqa: E402
from scripts.gpucat.verify_selection import histories  # noqa: E402


def fair(score, min_dist, beta, avoid):
    return select("fair", score, min_dist, beta, TAU, 1.0, avoid, RHO)[0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="logs/gpucat/fair_validity.json")
    args = p.parse_args(argv)
    z = np.load(REPO / args.bank, allow_pickle=True)
    stems = list(z["stems"])[: args.n] if args.n else list(z["stems"])
    gen = make_generator(args.device)
    cfg = ResponsibilityConfig(n_safety_samples=40)
    rng = np.random.default_rng(0)
    rows = []
    for i, stem in enumerate(stems):
        stem = str(stem)
        cand = z["candidates"][i].astype(np.float64)
        samples = z["samples"][i].astype(np.float64)
        ov = {"l": z["ov_size"][i][0], "w": z["ov_size"][i][1]}
        av = {"l": z["av_size"][i][0], "w": z["av_size"][i][1]}
        a_keep, avoid_raw, avoid_f, e_keep = valid_sets(gen, load_scene(REPO / args.scenes, stem), z, i, stem)
        probs = cat_candidate_probs(z["log_scores"][i])
        row = {"scene": stem, "adv_kept": int(len(a_keep)), "ego_kept": int(len(e_keep)),
               "avoid_matches_bank": bool(np.allclose(avoid_raw, z["avoid"][i]))}
        cases = {"logged": ([z["ego_route"][i]], [1.0])}
        h, _, hp = histories(z["ego_route"][i], rng)
        cases["closed"] = (h, list(hp))
        for name, (trajs, pav) in cases.items():
            score, md = cat_collision_scores(cand, probs, trajs, pav, ov, av)
            b_raw = adversary_responsibility(samples, cand, trajs, pav, cfg, HORIZON)
            b_f = adversary_responsibility(samples[a_keep], cand, trajs, pav, cfg, HORIZON)
            c_raw = fair(score, md, b_raw, avoid_raw)
            c_beta = fair(score, md, b_f, avoid_raw)
            c_avoid = fair(score, md, b_raw, avoid_f)
            c_both = fair(score, md, b_f, avoid_f)
            row[name] = {
                "cat": int(np.argmax(score)) if (score > 0).any() else int(np.argmin(md)),
                "raw": int(c_raw), "beta_filtered": int(c_beta), "avoid_filtered": int(c_avoid), "both": int(c_both),
                "raw_beta": float(b_raw[c_raw]), "raw_beta_filtered": float(b_f[c_raw]),
                "raw_avoid": float(avoid_raw[c_raw]), "raw_avoid_filtered": float(avoid_f[c_raw]),
                "raw_collides": bool(score[c_raw] > 0), "both_collides": bool(score[c_both] > 0),
                "beta_shift_median": float(np.median(b_f - b_raw)),
                "ok_raw": int(((b_raw <= TAU) & (avoid_raw >= RHO)).sum()),
                "ok_both": int(((b_f <= TAU) & (avoid_f >= RHO)).sum()),
            }
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"[{i + 1}/{len(stems)}]", flush=True)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    summarize(rows)


def summarize(rows):
    a = np.array([r["adv_kept"] for r in rows])
    e = np.array([r["ego_kept"] for r in rows])
    print(f"scenes {len(rows)}; ego samples match the bank's avoid in {sum(r['avoid_matches_bank'] for r in rows)}")
    print(f"adversary samples kept of 40: median {np.median(a):.0f}, p10 {np.percentile(a, 10):.0f}, <10 in {(a < 10).sum()}")
    print(f"ego samples kept of 40: median {np.median(e):.0f}, p10 {np.percentile(e, 10):.0f}, <10 in {(e < 10).sum()}")
    for name in ("logged", "closed"):
        c = [r[name] for r in rows]
        n = len(c)
        same = {k: sum(x[k] == x["raw"] for x in c) for k in ("beta_filtered", "avoid_filtered", "both")}
        rejected = sum((x["raw_beta_filtered"] > TAU) or (x["raw_avoid_filtered"] < RHO) for x in c)
        print(f"{name}: fair choice unchanged -- beta filtered {same['beta_filtered']}/{n}, avoid filtered "
              f"{same['avoid_filtered']}/{n}, both {same['both']}/{n}; raw choice fails a filtered test in {rejected}; "
              f"collides raw {sum(x['raw_collides'] for x in c)} / both {sum(x['both_collides'] for x in c)}; "
              f"fair = cat raw {sum(x['raw'] == x['cat'] for x in c)} / both {sum(x['both'] == x['cat'] for x in c)}; "
              f"median beta shift {np.median([x['beta_shift_median'] for x in c]):+.3f} m; "
              f"candidates passing both tests median raw {np.median([x['ok_raw'] for x in c]):.0f} / "
              f"filtered {np.median([x['ok_both'] for x in c]):.0f}")


if __name__ == "__main__":
    main()
