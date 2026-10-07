"""Checks gpucat/adversary.py against the numpy implementation that
reproduces CAT's AdvGenerator.generate (responsibility/adversarial.py), on
every scene of precompute_candidates.py's .npz:

  logged    the logged SDC route as the only ego trajectory (CAT's state
            before any rollout): scores, min_dist and beta against the
            stored numpy values, and the choices of the cat and fair rules
  closed    five ego histories per scene, as a policy leaves them: the
            logged route driven faster or slower (x0.5-1.3), shifted
            sideways (up to 1.5 m) and cut short (10-80 steps), with
            unequal probabilities; numpy's cat_collision_scores and
            adversary_responsibility recomputed on them
  plan      the 91-row plan of the chosen candidate against CAT's
            get_polyline_vel / get_polyline_yaw

Runs in this repository's environment (the numpy side imports advgen):
    python -m scripts.gpucat.verify_selection --bank logs/gpucat/candidates.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gpucat import adversary as ga  # noqa: E402
from responsibility.adversarial import adversary_responsibility, cat_candidate_probs, cat_collision_scores, select  # noqa: E402
from responsibility.metrics import ResponsibilityConfig  # noqa: E402


def histories(route: np.ndarray, rng: np.random.Generator, m: int = 5):
    """m ego trajectories derived from the logged route, with their lengths and probabilities."""
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    d = np.gradient(route, axis=0)
    normal = np.stack([-d[:, 1], d[:, 0]], -1) / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-6, None)
    out, lengths = [], []
    for _ in range(m):
        speed, shift = rng.uniform(0.5, 1.3), rng.uniform(-1.5, 1.5)
        n = int(rng.integers(10, 81))
        si = np.clip(s * speed, 0, s[-1])
        xy = np.stack([np.interp(si, s, route[:, 0]), np.interp(si, s, route[:, 1])], -1) + shift * normal
        out.append(xy[:n])
        lengths.append(n)
    return out, lengths, rng.uniform(0.2, 1.0, m)


def torch_inputs(trajs, lengths, probs, device):
    m, T = len(trajs), 80
    pad = np.zeros((m, T, 2))
    for i, (t, n) in enumerate(zip(trajs, lengths)):
        pad[i, :n] = t
    f = torch.float64
    return (torch.as_tensor(pad, dtype=f, device=device)[None], torch.as_tensor(lengths, device=device)[None],
            torch.as_tensor(probs, dtype=f, device=device)[None])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default="logs/gpucat/candidates.npz")
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--min-avoid", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args(argv)
    z = np.load(args.bank)
    S = len(z["stems"]) if args.n is None else min(args.n, len(z["stems"]))
    dev, f = args.device, torch.float64
    cfg = ResponsibilityConfig(n_safety_samples=40, d_sat=10.0)
    rng = np.random.default_rng(0)
    stats = {k: 0 for k in ("logged_cat", "logged_fair", "closed_cat", "closed_fair", "closed_cat_f32")}
    worst = {"score": 0.0, "min_dist": 0.0, "beta_logged": 0.0, "beta_closed": 0.0, "plan": 0.0}
    for s in range(S):
        cand = torch.as_tensor(z["candidates"][s], dtype=f, device=dev)[None]
        probs_ov = ga.candidate_probs(torch.as_tensor(z["log_scores"][s], dtype=f, device=dev))[None]
        ov = torch.as_tensor(z["ov_size"][s], dtype=f, device=dev)[None]
        av = torch.as_tensor(z["av_size"][s], dtype=f, device=dev)[None]
        samples = torch.as_tensor(z["samples"][s], dtype=f, device=dev)[None]
        avoid = torch.as_tensor(z["avoid"][s], dtype=f, device=dev)[None]
        route = z["ego_route"][s]
        # logged route
        tr, ln, pr = torch_inputs([route], [80], [1.0], dev)
        score, md = ga.cat_scores(cand, probs_ov, tr, ln, pr, ov, av)
        beta = ga.adversary_beta(samples, cand, tr, ln, pr)
        worst["score"] = max(worst["score"], float((score[0].cpu() - torch.as_tensor(z["score"][s])).abs().max()))
        worst["min_dist"] = max(worst["min_dist"], float((md[0].cpu() - torch.as_tensor(z["min_dist"][s])).abs().max()))
        worst["beta_logged"] = max(worst["beta_logged"], float((beta[0].cpu() - torch.as_tensor(z["beta"][s])).abs().max()))
        stats["logged_cat"] += int(ga.select("cat", score, md)[0]) == int(z["chosen_cat"][s])
        stats["logged_fair"] += int(ga.select("fair", score, md, beta, avoid, args.threshold, args.min_avoid)[0]) == int(
            z["chosen_fair"][s])
        # closed-loop-like histories
        trajs, lengths, probs = histories(route, rng)
        sizes = ({"l": float(z["ov_size"][s][0]), "w": float(z["ov_size"][s][1])},
                 {"l": float(z["av_size"][s][0]), "w": float(z["av_size"][s][1])})
        np_score, np_md = cat_collision_scores(z["candidates"][s], cat_candidate_probs(z["log_scores"][s]), trajs, probs,
                                               *sizes)
        np_beta = adversary_responsibility(z["samples"][s], z["candidates"][s], trajs, probs, cfg, 80)
        tr, ln, pr = torch_inputs(trajs, lengths, probs, dev)
        score, md = ga.cat_scores(cand, probs_ov, tr, ln, pr, ov, av)
        beta = ga.adversary_beta(samples, cand, tr, ln, pr)
        worst["score"] = max(worst["score"], float((score[0].cpu() - torch.as_tensor(np_score)).abs().max()))
        worst["min_dist"] = max(worst["min_dist"], float((md[0].cpu() - torch.as_tensor(np_md, dtype=f)).abs().max()))
        worst["beta_closed"] = max(worst["beta_closed"], float((beta[0].cpu() - torch.as_tensor(np_beta)).abs().max()))
        np_cat = select("cat", np_score, np_md, None)[0]
        np_fair = select("fair", np_score, np_md, np_beta, args.threshold, avoid=z["avoid"][s], min_avoid=args.min_avoid)[0]
        chosen = int(ga.select("cat", score, md)[0])
        stats["closed_cat"] += chosen == np_cat
        stats["closed_fair"] += int(ga.select("fair", score, md, beta, avoid, args.threshold, args.min_avoid)[0]) == np_fair
        s32, m32 = ga.cat_scores(cand.float(), probs_ov.float(), tr.float(), ln, pr.float(), ov.float(), av.float())
        stats["closed_cat_f32"] += int(ga.select("cat", s32, m32)[0]) == np_cat
        # plan
        from advgen.adv_generator import get_polyline_vel, get_polyline_yaw
        pos = np.concatenate([z["adv_past"][s], z["candidates"][s][chosen]], 0)
        ref = np.concatenate([pos, get_polyline_vel(pos), get_polyline_yaw(pos).reshape(-1, 1)], 1)
        mine = ga.plan(torch.as_tensor(z["adv_past"][s], dtype=f, device=dev)[None], cand[:, chosen])[0].cpu().numpy()
        worst["plan"] = max(worst["plan"], float(np.abs(mine - ref).max()))
    print(f"{S} scenes: same choice as numpy -- logged route: cat {stats['logged_cat']}, fair {stats['logged_fair']}; "
          f"closed-loop histories: cat {stats['closed_cat']} (float32: {stats['closed_cat_f32']}), fair {stats['closed_fair']}")
    print("largest differences: " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))


if __name__ == "__main__":
    main()
