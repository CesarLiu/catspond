"""The mechanism probes of the aggression test (probe_inertia.py,
probe_courtesy.py; responsibility/aggression_test.py) with CAT-K's SMART
(catk, clsft_E9, neighbor_future=hidden: everyone's history up to k, nobody's
future). At 3, 2, 1 and 0.5 s before CAT's adversary first overlaps the ego,
in CAT's scenes and at the same steps in the logged scenes:

  alt_close   share of the adversary's 40 own alternatives that come within
              3 m (centre distance) of the ego's logged path in the next 2 s
  act_close   whether the adversary's actual trajectory does
  gain        mean over alternatives of min(D_alt, 10) - min(D_act, 10) (m):
              the margin its alternatives would have kept in addition
  d_prog      the ego's predicted progress along its heading at 2 s with the
              adversary present minus without it (m; < 0: held back by it)
  d_end       distance between the ego's mean predicted end points (2 s)
              with and without the adversary (m)

SMART reacts where DenseTNT barely does: 0.5 s before the overlap it
predicts the ego to hold back by about 0.7 m (DenseTNT: +0.07 m), while the
adversary's own alternatives are about as committed (61% within 3 m in both).

Run in catk's environment from the catk root, on catk caches made by
run_smart.sh (export_catk.py) of CAT's scenes and of the logged scenes:

  cd ~/catk && ~/venvs/catk/bin/python <repo>/scripts/responsibility/probe_smart.py \\
      --cat-cache <VAL>/smart/cat/cache --logged-cache <repo>/logs/catk/scenes/cache \\
      --index <repo>/adv_scenes/cat.index.json --out <VAL>/smart_probe.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")  # the catk root (run from it)
from src.responsibility.motion_set import (  # noqa: E402
    NEIGHBOR_FUTURE_HIDDEN, MotionSetSynthesizer, RolloutJob, shifted_context)
from src.responsibility.pipeline import (  # noqa: E402
    compose_responsibility_config, default_sampling_scheme, iter_scenarios, load_smart_model)

NH = 11  # num_historical_steps
H = 20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cat-cache", required=True)
    p.add_argument("--logged-cache", required=True)
    p.add_argument("--index", required=True, help="CAT's index.json (first overlap per scene)")
    p.add_argument("--ckpt", default=str(Path.home() / "catk/ckpts/clsft_E9.ckpt"))
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=100)
    a = p.parse_args()
    torch.manual_seed(0)
    cfg = compose_responsibility_config(overrides=["model=smart"])
    model = load_smart_model(cfg.model.model_config, a.ckpt, device="cuda")
    syn = MotionSetSynthesizer(model, default_sampling_scheme(), rollout_batch_size=a.batch)
    horizon = syn.round_horizon(H)
    index = json.loads(Path(a.index).read_text())
    rows = []
    for src, cache in (("CAT", a.cat_cache), ("logged", a.logged_cache)):
        files = sorted(Path(cache).glob("*"))
        pos = {f.stem: i for i, f in enumerate(files)}
        for s in range(a.limit):
            e = index[str(s)].get("first_overlap_with_logged_ego")
            if e is None or e - 30 < NH - 1 or str(s) not in pos:
                continue
            for _, _, data in iter_scenarios(cache, "cuda", indices=[pos[str(s)]]):
                role = data["agent"]["role"]
                ego = int(torch.nonzero(role[:, 0])[0])
                adv = [int(i) for i in torch.nonzero(role[:, 1]).flatten() if int(i) != ego][0]
                n_total = data["agent"]["position"].shape[1]
                shift = syn.token_shift
                k_last = n_total + NH - 1 - (shift * ((NH - 1) // shift + horizon // shift) + 1)  # catk's last usable window
                for lead in (30, 20, 10, 5):
                    k = e - lead
                    if k > k_last:
                        continue
                    with shifted_context(data, k, NH):
                        valid = data["agent"]["valid_mask"]
                        if not (valid[ego, NH - 1] and valid[adv, NH - 1]):
                            continue
                        tm, ta = syn.tokenize(data)
                        jobs = [RolloutJob(free=(adv,), n_samples=a.n),
                                RolloutJob(free=(ego,), n_samples=a.n),
                                RolloutJob(free=(ego,), n_samples=a.n, removed=(adv,))]
                        res = syn.run_jobs(tm, ta, jobs, rollout_horizon=horizon, neighbor_future=NEIGHBOR_FUTURE_HIDDEN)
                        P = data["agent"]["position"][:, :, :2]
                        fut = slice(NH, NH + H)
                        ego_f, adv_f = P[ego, fut], P[adv, fut]
                        alt = res[0]["pred_traj_10hz"][:, 0, :H]  # [n, H, 2]
                        d_alt = torch.linalg.norm(alt - ego_f, dim=-1).min(-1).values
                        d_act = torch.linalg.norm(adv_f - ego_f, dim=-1).min()
                        h = data["agent"]["heading"][ego, NH - 1]
                        u = torch.stack([torch.cos(h), torch.sin(h)])
                        p0 = P[ego, NH - 1]
                        end_w = res[1]["pred_traj_10hz"][:, 0, H - 1]
                        end_wo = res[2]["pred_traj_10hz"][:, 0, H - 1]
                        rows.append({
                            "scene": s, "src": src, "lead_s": lead / 10,
                            "gap": float(torch.linalg.norm(P[adv, NH - 1] - p0)),
                            "alt_close": float((d_alt < 3).float().mean()), "act_close": bool(d_act < 3),
                            "gain": float((d_alt.clamp(max=10) - d_act.clamp(max=10)).mean()),
                            "d_prog": float(((end_w - p0) @ u).mean() - ((end_wo - p0) @ u).mean()),
                            "d_end": float(torch.linalg.norm(end_w.mean(0) - end_wo.mean(0))),
                            "replay_err": float(torch.linalg.norm(res[1]["pred_traj_10hz"][:, 0, :H].mean(0)[0] - P[ego, NH])),
                        })
        print(src, "done", len(rows), flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    print(d.groupby(["src", "lead_s"]).agg(n=("gap", "size"), gap_med=("gap", "median"),
          actual_within_3m=("act_close", "mean"), alternatives_within_3m=("alt_close", "mean"),
          margin_gain_m=("gain", "mean"), d_prog_med=("d_prog", "median"), d_end_med=("d_end", "median"),
          first_step_err=("replay_err", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
