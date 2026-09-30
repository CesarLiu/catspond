"""Checks, and fixes by a softmax temperature, how well a trained MTR's 64
intention scores are calibrated -- the responsibility metrics use them as a
probability distribution (courtesy is a KL between two of them), not just to
rank modes.

On WOMD validation scenes (ScenarioNet format; CAT's 500 scenes, listed in
responsibility/unitraj_configs/cat_scenario_ids.txt, are left out) every
agent WOMD asks to predict is predicted at step 10 through the adapter
(responsibility/unitraj.py, i.e. exactly the inputs the metrics use). Its
target is the intention point nearest to its last valid future position, as
in MTR's classification loss. Reported, before and after scaling the
log-scores by 1/T:

  nll       mean negative log-probability of the target intention (nats)
  mass      mean probability of the target intention
  top1      how often the most probable intention is the target
  ece       expected calibration error of the top-1 probability (15 bins)

T is fitted by minimising the NLL and written to <checkpoint>.calibration.json,
which the adapter picks up (responsibility.unitraj.load_calibration).

Example (server, UniTraj environment, from the cat repository root):
    python -m scripts.responsibility.calibrate_unitraj --checkpoint ckpt/mtr_womd.ckpt \\
        --scenes /data/womd_scenarionet/validation --n 3000
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CAT_IDS = Path(__file__).resolve().parents[2] / "responsibility" / "unitraj_configs" / "cat_scenario_ids.txt"
STEP = 10  # WOMD's current step with 1.1 s of history


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--scenes", required=True, help="ScenarioNet directory of WOMD validation scenes.")
    p.add_argument("--n", type=int, default=3000, help="Scenes to use.")
    p.add_argument("--unitraj-root", default=None)
    p.add_argument("--unitraj-method", default="MTR_womd")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="Default: <checkpoint>.calibration.json")
    return p.parse_args(argv)


def excluded_ids(path=CAT_IDS):
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")}


def scenario_files(directory):
    """Scenario pickles of a ScenarioNet directory (its summary and mapping files left out)."""
    return sorted(p for p in Path(directory).rglob("*.pkl") if not p.name.startswith("dataset_"))


def ece(probs: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of the top-1 prediction."""
    conf = probs.max(-1)
    correct = probs.argmax(-1) == targets
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def summary(log_scores: np.ndarray, targets: np.ndarray, temperature: float) -> dict:
    lp = torch.log_softmax(torch.as_tensor(log_scores, dtype=torch.float64) / temperature, dim=-1).numpy()
    p = np.exp(lp)
    rows = np.arange(len(targets))
    return {"nll": float(-lp[rows, targets].mean()), "mass": float(p[rows, targets].mean()),
            "top1": float((p.argmax(-1) == targets).mean()), "ece": ece(p, targets)}


def fit_temperature(log_scores: np.ndarray, targets: np.ndarray) -> float:
    """The T > 0 minimising the NLL of the targets under softmax(log_scores / T)."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda log_t: summary(log_scores, targets, float(np.exp(log_t)))["nll"],
                          bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def nearest_intention(points: np.ndarray, goal: np.ndarray) -> int:
    return int(np.argmin(np.linalg.norm(points - goal[None], axis=-1)))


def collect(inputs, predictor, files, intention_points, n, batch_size=128, exclude=frozenset(), step=STEP):
    """(log-scores [M, K], target intention [M]) over the agents to predict in
    up to ``n`` scenes (``exclude``: scenario ids left out)."""
    from responsibility.scene import Scene

    pending, log_scores, targets = [], [], []

    def flush():
        if not pending:
            return
        scores, _ = predictor(inputs.collate([inst for inst, _ in pending]))
        log_scores.append(torch.log(torch.as_tensor(scores, dtype=torch.float64).cpu().clamp_min(1e-12)).numpy())
        targets.extend(t for _, t in pending)
        pending.clear()

    used = 0
    for path in files:
        if used >= n:
            break
        with open(path, "rb") as f:
            sd = pickle.load(f)
        if str(sd["metadata"].get("scenario_id", sd.get("id"))) in exclude:
            continue
        scene = Scene.from_description(sd)
        used += 1
        for tid in (sd["metadata"].get("tracks_to_predict") or {}):
            if str(tid) not in scene.track_ids:
                continue
            inst = inputs.instance(scene, step, scene.index(str(tid)))
            if inst is None or not inst.sample["center_gt_trajs_mask"].any():
                continue
            final = int(inst.sample["center_gt_final_valid_idx"])
            goal = np.asarray(inst.sample["center_gt_trajs"][final, :2], dtype=np.float64)
            kind = scene.types[scene.index(str(tid))]
            pending.append((inst, nearest_intention(intention_points[kind], goal)))
            if len(pending) >= batch_size:
                flush()
    flush()
    if not log_scores:
        raise SystemExit("no agents to predict in the given scenes")
    return np.concatenate(log_scores), np.asarray(targets), used


def main(argv=None):
    args = parse_args(argv)
    from responsibility.unitraj import Inputs, load_config, mtr_predictor, unitraj_root

    config = load_config(method=args.unitraj_method, root=args.unitraj_root)
    inputs = Inputs(config, root=args.unitraj_root)
    predictor = mtr_predictor(args.checkpoint, config, device=args.device, root=args.unitraj_root)
    points_file = unitraj_root(args.unitraj_root) / "unitraj" / config["MOTION_DECODER"]["INTENTION_POINTS_FILE"]
    with open(points_file, "rb") as f:
        points = {k: np.asarray(v, dtype=np.float64).reshape(-1, 2) for k, v in pickle.load(f).items()}
    log_scores, targets, used = collect(inputs, predictor, scenario_files(args.scenes), points, args.n,
                                        args.batch_size, excluded_ids())
    temperature = fit_temperature(log_scores, targets)
    before, after = summary(log_scores, targets, 1.0), summary(log_scores, targets, temperature)
    result = {"temperature": temperature, "agents": int(len(targets)), "scenes": used,
              "before": before, "after": after, "checkpoint": str(Path(args.checkpoint).resolve())}
    out = Path(args.out or str(args.checkpoint) + ".calibration.json")
    out.write_text(json.dumps(result, indent=2))
    print(f"{len(targets)} agents in {used} scenes; fitted temperature {temperature:.3f}")
    for name, s in (("T = 1", before), (f"T = {temperature:.2f}", after)):
        print(f"  {name:>9}: nll {s['nll']:.3f}, target mass {s['mass']:.3f}, top-1 {s['top1']:.3f}, ece {s['ece']:.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
