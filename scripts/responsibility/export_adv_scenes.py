"""Writes the adversarial counterpart of CAT's scenes as scene files: each
scene with its adversary's logged future replaced by the trajectory an
adversary rule generates against the logged ego -- what cat_advgen.py's
second round plays in MetaDrive, kept as data instead.

The generator is CAT's (ResponsibleAdvGenerator with --adv_selection cat
chooses the same candidate as AdvGenerator.generate in all 500 scenes,
benchmark_advgen.py), run against a stand-in for MetaDrive's env as in
benchmark_advgen.py: the ego's trajectory is its logged route (CAT's first
round, AV_traj_num 1). Other rules (fair, constrained, penalized, and near:
a near miss at --near_gap instead of a collision) choose among the same 32
DenseTNT candidates.

Each output file is a deep copy of the original scene description in which
only the adversary's track from step 11 on changes: position (x, y; z kept),
heading, velocity and validity (valid from step 11: the plan covers every
step). Steps 0-10 are the logged history CAT plans from. So the folder can
be read like raw_scenes_500 -- by Scene.load, compute_responsibility.py,
visualize_responsibility.py (--scenes) or MetaDrive. ``metadata.adversary``
records the generation: rule, adversary id, chosen candidate and why,
predicted collision score, the adversary's beta and the ego's avoidability
(adversarial.py), the candidate's predicted closest approach (near), the
first step at which the plan's footprint overlaps the logged ego's (None: no
overlap with the logged, non-reacting ego) and the smallest gap between
the two footprints (m, circle covers; <= 0 is overlap).
OUT/<rule>.index.json, next to the folder, holds the same per scene: the
folder holds only scenes, since MetaDrive asserts that every file in a scene
folder is one (an index left inside by an older export is moved out).

Example (from the repository root):
    python -m scripts.responsibility.export_adv_scenes --n 10 --out-dir adv_scenes
    python -m scripts.responsibility.export_adv_scenes --rule fair --resp_threshold 2 --resp_avoid 0.1 \\
        --out-dir adv_scenes
    python -m scripts.responsibility.export_adv_scenes --rule near --near_gap 1.0 --near_tol 0.5 \\
        --out-dir adv_scenes
"""

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from responsibility.adversarial import ResponsibleAdvGenerator, selection_name  # noqa: E402
from responsibility.interaction import _footprint_gap  # noqa: E402
from responsibility.scene import Scene, scene_files, sidecar_index  # noqa: E402
from scripts.responsibility.benchmark_advgen import fake_env  # noqa: E402

FIRST_PLANNED = 11  # CAT's plan is the logged history up to step 10, then the generated future


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--out-dir", required=True, help="Scenes go to OUT/<rule name>/<scene>.pkl.")
    p.add_argument("--rule", default="cat", choices=["cat", "constrained", "penalized", "fair", "near"])
    p.add_argument("--resp_threshold", default="1.0", help="tau (m) for constrained / fair; inf allowed.")
    p.add_argument("--resp_avoid", default="0.3", help="rho for fair.")
    p.add_argument("--near_gap", default="1.0", help="m; near: the closest approach to aim for.")
    p.add_argument("--near_tol", default="0.5", help="m; near: the band around --near_gap that counts.")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def footprint_gaps(scene: Scene, a: int, b: int):
    """(steps, gaps): the gap between a's and b's footprints (circle
    covers, m; <= 0 overlap) at every step from FIRST_PLANNED where both are
    valid."""
    t = np.arange(FIRST_PLANNED, scene.n_steps)
    t = t[scene.valid[a, t] & scene.valid[b, t]]
    if len(t) == 0:
        return t, np.zeros(0)
    f = torch.float32
    gap = _footprint_gap(torch.as_tensor(scene.position[a, t, :2], dtype=f), torch.as_tensor(scene.heading[a, t], dtype=f),
                         torch.as_tensor(scene.shape_at(a, 10)[None], dtype=f),
                         torch.as_tensor(scene.position[b, t, :2][None], dtype=f),
                         torch.as_tensor(scene.heading[b, t][None], dtype=f),
                         torch.as_tensor(scene.shape_at(b, 10)[None], dtype=f))[0].numpy()
    return t, gap


def first_overlap(scene: Scene, a: int, b: int):
    """The first step from FIRST_PLANNED at which a's and b's footprints overlap, or None."""
    t, gap = footprint_gaps(scene, a, b)
    hit = np.flatnonzero(gap <= 0)
    return int(t[hit[0]]) if len(hit) else None


def open_index(out: Path):
    """(path, entries) of a rule folder's index, OUT/<rule>.index.json next
    to it; an index an older export kept inside the folder is moved there."""
    path = sidecar_index(out)
    if (out / "index.json").exists() and not path.exists():
        (out / "index.json").replace(path)
    return path, (json.loads(path.read_text()) if path.exists() else {})


def adversarial_description(description, adv_id: str, plan: np.ndarray):
    """A deep copy of the description with the adversary's track replaced
    from FIRST_PLANNED on by ``plan`` [91, 5] (x, y, vx, vy, yaw)."""
    out = copy.deepcopy(description)
    state = out["tracks"][adv_id]["state"]
    s = slice(FIRST_PLANNED, len(plan))
    state["position"][s, :2] = plan[s, :2]
    state["heading"][s] = plan[s, 4]
    state["velocity"][s] = plan[s, 2:4]
    state["valid"][s] = True
    return out


def main(argv=None):
    args = parse_args(argv)
    cat_parser = argparse.ArgumentParser()  # as cat_advgen.py builds it
    cat_parser.add_argument("--OV_traj_num", type=int, default=32)
    cat_parser.add_argument("--AV_traj_num", type=int, default=1)
    gen = ResponsibleAdvGenerator(cat_parser, argv=[
        "--adv_selection", args.rule, "--resp_threshold", args.resp_threshold, "--resp_avoid", args.resp_avoid,
        "--near_gap", args.near_gap, "--near_tol", args.near_tol, "--resp_device", args.device])
    name = selection_name(gen.args) if args.rule != "cat" else "cat"
    out = Path(args.out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    index_path, index = open_index(out)
    files = scene_files(args.scenes)[args.first:]
    files = files[: args.n] if args.n is not None else files
    for path in files:
        if (out / path.name).exists() and path.stem in index:
            continue
        with open(path, "rb") as f:
            description = pickle.load(f)
        scene = Scene.from_description(copy.deepcopy(description))
        seed = int(path.stem)
        gen.before_episode(fake_env(description, seed, scene))
        gen.generate()
        plan = np.array(gen.adv_traj, dtype=float)
        adv_id = str(gen.adv_agent)
        sel = gen.selections[-1]
        adv_desc = adversarial_description(description, adv_id, plan)
        adv_scene = Scene.from_description(copy.deepcopy(adv_desc))
        j = int(sel["chosen"])

        def at(key, digits):  # the chosen candidate's value (None where the rule did not compute it)
            v = sel.get(key)
            return None if v is None else round(float(np.asarray(v)[j]), digits)

        _, gaps = footprint_gaps(adv_scene, adv_scene.index(adv_id), adv_scene.sdc)
        info = {"rule": name, "adversary": adv_id, "chosen": j, "why": sel["why"], "score": at("score", 5),
                "beta": at("beta", 4), "avoid": at("avoid", 4), "gap": at("gap", 3),
                "first_overlap_with_logged_ego": first_overlap(adv_scene, adv_scene.index(adv_id), adv_scene.sdc),
                "min_gap_with_logged_ego": round(float(gaps.min()), 3) if len(gaps) else None}
        adv_desc["metadata"]["adversary"] = info
        tmp = out.parent / f".{out.name}.{path.name}.tmp"  # outside: a stray file would stop MetaDrive
        with open(tmp, "wb") as f:
            pickle.dump(adv_desc, f)
        tmp.replace(out / path.name)
        index[path.stem] = info
        index_path.write_text(json.dumps(index, indent=1, sort_keys=True))
        print(f"{path.stem}: adversary {adv_id}, candidate {info['chosen']} ({info['why']}), score {info['score']}, "
              f"beta {info['beta']} m, avoid {info['avoid']}, overlap at step {info['first_overlap_with_logged_ego']}, "
              f"min gap {info['min_gap_with_logged_ego']} m",
              flush=True)
    hits = sum(v["first_overlap_with_logged_ego"] is not None for v in index.values())
    print(f"{len(index)} scenes in {out}; the plan overlaps the logged ego in {hits}")


if __name__ == "__main__":
    main()
