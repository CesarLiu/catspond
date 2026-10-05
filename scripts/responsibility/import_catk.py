"""Turns a catk responsibility run over CAT's scenes (run_smart.sh: OUT/run,
one obs/<scene>.pkl of ResponsibilityObservation per scene) into this
repository's run format, so compare_models.py, summarize_responsibility.py
and the other readers take SMART's results as they take DenseTNT's
(SMART_PLAN.md, S4).

catk measures both objects of interest (--query interest); their windows
are split by agent into OUT/sdc and OUT/adv, the two runs
compute_responsibility.py makes with --agent sdc and --agent adv. Each gets
windows.csv with the same columns:

  step            catk's context step k (0.1 s steps, as here)
  safety          beta[0], courtesy beta[1]: the maxima over neighbours
  n_neighbours, safety_against, courtesy_toward
                  from catk's per-neighbour values, as Observation.as_row
  speed           |velocity| of the agent at the step in CAT's scene (or the
                  scene rebuilt from the rollout), as compute_responsibility
                  computes it

and config.json (model "smart", the checkpoint, catk's responsibility
settings, and the data the run used). Observations whose agent is neither
object of interest in CAT's scene -- WOMD v1.2.1 renumbered two vehicles of
one scene -- are counted and left out.

Runs in catk's environment (the observations hold torch tensors and catk's
dataclass). Example (from this repository's root):
    ~/venvs/catk/bin/python -m scripts.responsibility.import_catk --catk-root ~/catk \\
        --run logs/catk/scenes/run --out-dir logs/catk/scenes/windows --data "CAT's scenes (WOMD v1.1)"
"""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.rollouts import load_rollout, rollout_files, scene_from_rollout  # noqa: E402
from responsibility.scene import Scene  # noqa: E402

FIELDS = ["scene", "scenario_id", "agent_id", "step", "time", "speed", "safety", "courtesy",
          "n_neighbours", "safety_against", "courtesy_toward"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catk-root", default=os.environ.get("CATK_ROOT", str(Path.home() / "catk")))
    p.add_argument("--run", required=True, help="catk's output directory (obs/ and config.json).")
    p.add_argument("--scenes", default="raw_scenes_500")
    p.add_argument("--rollouts", default=None, help="The rollouts the run measured (run_smart.sh ROLLOUTS).")
    p.add_argument("--data", default="", help="What the run measured, for config.json.")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def window_row(stem, scene, agent, obs):
    """One windows.csv row of a catk observation, as Observation.as_row."""
    per = obs.per_neighbour or {}
    worst_s = max(per.items(), key=lambda kv: kv[1][0], default=(None, None))[0]
    worst_c = max(per.items(), key=lambda kv: kv[1][1], default=(None, None))[0]
    beta = [float(x) for x in obs.beta]
    return {"scene": stem, "scenario_id": scene.scenario_id, "agent_id": scene.track_ids[agent], "step": int(obs.k),
            "time": round(obs.k / 10.0, 1), "speed": round(float(np.linalg.norm(scene.velocity[agent, obs.k])), 3),
            "safety": round(beta[0], 5), "courtesy": round(beta[1], 6), "n_neighbours": len(per),
            "safety_against": worst_s, "courtesy_toward": worst_c}


def roles(scene):
    """Track id -> "sdc" / "adv" (compute_responsibility.pick_agent)."""
    out = {str(scene.track_ids[scene.sdc]): ("sdc", scene.sdc)}
    others = [i for i in scene.objects_of_interest if i != scene.sdc]
    if others:
        out[str(scene.track_ids[others[0]])] = ("adv", others[0])
    return out


def main(argv=None):
    args = parse_args(argv)
    sys.path.append(str(Path(args.catk_root).expanduser().resolve()))  # catk's dataclass, for unpickling
    run = Path(args.run)
    rollouts = {}
    if args.rollouts:
        for f in rollout_files(args.rollouts):
            r = load_rollout(f)
            rollouts[str(r["scene_file"])] = r

    rows = {"sdc": [], "adv": []}
    unmatched = Counter()
    files = sorted(run.glob("obs/*.pkl"), key=lambda f: (0, int(f.stem)) if f.stem.isdigit() else (1, f.stem))
    for f in files:
        stem = f.stem
        with open(f, "rb") as h:
            observations = pickle.load(h)
        scene = Scene.load(Path(args.scenes) / f"{stem}.pkl")
        if stem in rollouts:
            scene = scene_from_rollout(scene, rollouts[stem])
        who = roles(scene)
        for obs in observations:
            role = who.get(str(obs.agent_id))
            if role is None:
                unmatched[stem] += 1
                continue
            rows[role[0]].append(window_row(stem, scene, role[1], obs))

    catk_config = json.loads((run / "config.json").read_text())
    for role, role_rows in rows.items():
        out = Path(args.out_dir) / role
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "windows.csv", "w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(role_rows)
        config = {"agent": role, "scenes": str(Path(args.scenes).resolve()),
                  "model": {"name": "smart", "ckpt": catk_config.get("ckpt"), "data": args.data,
                            "data_root": catk_config.get("data_root")},
                  "responsibility": catk_config.get("responsibility"), "imported_from": str(run.resolve())}
        if args.rollouts:
            config["rollouts"] = str(Path(args.rollouts).resolve())
        (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{len(files)} scenes: {len(rows['sdc'])} sdc windows, {len(rows['adv'])} adv windows -> {args.out_dir}; "
          f"observations of neither object of interest: {sum(unmatched.values())} in {len(unmatched)} scene(s) "
          f"{dict(list(unmatched.items())[:5])}")


if __name__ == "__main__":
    main()
