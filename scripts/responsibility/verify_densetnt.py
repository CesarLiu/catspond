"""Checks the DenseTNT wrapper against CAT's own pipeline on real scenes.

  features   womd_features at step 10 with CAT's agent order equals what
             AdvGenerator._parse builds, field by field
  cat_modes  goal distribution + NMS + completion through the wrapper gives
             CAT's 32 adversary trajectories and scores (AdvGenerator.generate's
             model call, run on the adversary alone: batched with the ego,
             DenseTNT's padding leaks into the shorter instance; that
             deviation is printed for information)
  sampling   goals drawn from the distribution hit each region of the goal grid
             as often as its probability says
  repeat     the same context twice gives the same distribution (KL 0)
  removal    leaving an agent out removes exactly its polyline from the input
             (also in scenes with more than the 128 agents the input holds),
             keeps the candidate goals, and moves a neighbour's distribution
             more when the removed agent is near than when it is far
  partner    which agent fills DenseTNT's second object-of-interest slot
             barely matters (it is held fixed across with/without anyway)

Example (from the repository root):
    python -m scripts.responsibility.verify_densetnt --scenes raw_scenes_500 --n 3
"""

import argparse
import copy
import pickle
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.densetnt import DenseTNT, cat_instance  # noqa: E402  (sets up the pickle5 shim)
from responsibility.metrics import goal_kl  # noqa: E402
from responsibility.scene import Scene, cat_agent_order, scene_files, womd_features  # noqa: E402


class Report:
    def __init__(self):
        self.ok = True

    def check(self, name, passed, detail):
        self.ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)


def cat_parse(description):
    """AdvGenerator._parse on a scenario dict, without a MetaDrive env."""
    from advgen.adv_generator import AdvGenerator

    fake = types.SimpleNamespace(env=types.SimpleNamespace(
        current_seed=0,
        engine=types.SimpleNamespace(data_manager=types.SimpleNamespace(_scenario={0: copy.deepcopy(description)})),
    ))
    features, adv_agent, _, _ = AdvGenerator._parse(fake)
    return features, adv_agent


def check_features(description, scene, report):
    cat, _ = cat_parse(description)
    ours = womd_features(scene, 10, cat_agent_order(scene))
    mismatched = []
    for key, value in cat.items():
        a = value.numpy()
        b = np.asarray(ours.get(key))
        if key == "scenario/id":
            continue
        if b is None or a.shape != b.shape or not np.array_equal(a.astype(np.float64), b.astype(np.float64)):
            mismatched.append(key)
    missing = sorted(set(cat) - set(ours))
    report.check("features", not mismatched and not missing,
                 f"{len(cat)} fields; mismatched {mismatched[:5]}; missing {missing[:5]}")


def check_cat_modes(model, description, scene, report):
    from advgen.adv_utils import process_data

    cat, _ = cat_parse(description)
    pair = process_data(cat, model.args)[0]
    with torch.no_grad():
        np.random.seed(0)
        cat_trajs, cat_scores, _ = model.model([pair[1]], model.device)
        np.random.seed(0)
        pair_trajs, _, _ = model.model(process_data(cat, model.args)[0], model.device)
    mapping = cat_instance(model, scene, select=1)
    dist = model.goal_distributions([mapping])[0]
    np.random.seed(0)
    trajs, scores = model.nms_modes(dist)
    dt = float(np.abs(trajs - cat_trajs[0]).max())
    ds = float(np.abs(scores - cat_scores[0]).max())
    mass = float(dist.log_prob.exp().sum())
    report.check("cat_modes", dt < 1e-3 and ds < 1e-4,
                 f"adversary: max |traj diff| {dt:.2e} m, max |score diff| {ds:.2e}; "
                 f"{len(dist.goals)} goals, total probability {mass:.4f} "
                 f"(info: CAT's ego+adversary batch differs from alone by {float(np.abs(pair_trajs[1] - cat_trajs[0]).max()):.2e} m)")


def check_sampling(model, scene, report, n=4000):
    dist = model.distribution(scene, 10, scene.sdc)
    probs = dist.log_prob.exp().cpu().numpy()
    gen = torch.Generator().manual_seed(0)
    idx, log_prob, trajs = model.sample(dist, n, generator=gen)
    # compare mass on the 10 most likely goals and on everything else
    top = np.argsort(-probs)[:10]
    expected = probs[top].sum()
    observed = np.isin(idx.numpy(), top).mean()
    se = np.sqrt(expected * (1 - expected) / n)
    endpoint = float(np.linalg.norm(trajs[:, -1] - dist.to_global(dist.goals[idx.numpy()]), axis=-1).mean())
    report.check("sampling", abs(observed - expected) < 4 * se + 1e-3,
                 f"top-10 goal mass {expected:.3f}, sampled {observed:.3f} (n={n}); "
                 f"mean |trajectory end - goal| {endpoint:.2f} m")


def _vehicles_by_distance(scene, step, agent):
    """Other vehicles observed at ``step``, nearest to ``agent`` first."""
    others = [i for i in range(scene.n_agents)
              if i != agent and scene.valid[i, step] and scene.types[i] == "VEHICLE"]
    d = np.linalg.norm(scene.position[others, step, :2] - scene.position[agent, step, :2], axis=-1)
    return [others[j] for j in np.argsort(d)], np.sort(d)


def check_counterfactuals(model, scene, report, step=10):
    ego = scene.sdc
    neighbours, dist_to_ego = _vehicles_by_distance(scene, step, ego)
    if len(neighbours) < 3:
        report.check("removal", True, "skipped: fewer than 3 other vehicles")
        return
    b = neighbours[0]  # the ego's nearest vehicle: the natural courtesy target

    first = model.distribution(scene, step, b, avoid_partner=(ego,))
    again = model.distribution(scene, step, b, avoid_partner=(ego,))
    report.check("repeat", (goal_kl(first.log_prob, again.log_prob)) == 0.0,
                 f"KL of a repeated pass {(goal_kl(first.log_prob, again.log_prob)):.1e}")

    with_ego = model.instance(scene, step, b, avoid_partner=(ego,))
    without_ego = model.instance(scene, step, b, excluded=(ego,))
    dropped = with_ego["map_start_polyline_idx"] - without_ego["map_start_polyline_idx"]
    near = model.with_and_without(scene, step, b, ego)
    kl_near = (goal_kl(near[0].log_prob, near[1].log_prob))
    others_of_b, d_b = _vehicles_by_distance(scene, step, b)
    far = others_of_b[-1]
    far_pair = model.with_and_without(scene, step, b, far)
    kl_far = (goal_kl(far_pair[0].log_prob, far_pair[1].log_prob))
    report.check("removal", dropped == 1 and kl_near > kl_far,
                 f"removing the ego drops {dropped} agent polyline(s); KL for vehicle {b} "
                 f"({dist_to_ego[0]:.1f} m from the ego): without the ego {kl_near:.4f}, "
                 f"without the farthest vehicle ({d_b[-1]:.0f} m) {kl_far:.4f} nats")

    # partner: the ego's nearest vehicle predicted with two different partners
    partner = model._partner(scene, step, b, avoid=(ego,))
    p1 = model.distribution(scene, step, b, avoid_partner=(ego,))
    p2 = model.distribution(scene, step, b, avoid_partner=(ego, partner))
    kl_partner = (goal_kl(p1.log_prob, p2.log_prob))
    report.check("partner", kl_partner < 0.1 * max(kl_near, 1e-3),
                 f"KL between two partner choices {kl_partner:.2e} nats (vs {kl_near:.4f} for removing the ego)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--n", type=int, default=3, help="Number of scenes to check.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model = DenseTNT(device=args.device)
    report = Report()
    for path in scene_files(args.scenes)[: args.n]:
        with open(path, "rb") as f:
            description = pickle.load(f)
        scene = Scene.from_description(copy.deepcopy(description))
        print(f"scene {path.name} ({scene.scenario_id}): {scene.n_agents} agents", flush=True)
        check_features(description, scene, report)
        check_cat_modes(model, description, scene, report)
        check_sampling(model, scene, report)
        check_counterfactuals(model, scene, report)
    print("\nALL CHECKS PASSED" if report.ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
