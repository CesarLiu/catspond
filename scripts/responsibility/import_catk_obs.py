"""Converts catk's observations (SMART, run_smart.sh) into the obs/<scene>.pkl
layout of this repository's runs, split by object of interest, so that
evaluate_aggression.py reads both motion models the same way. Each catk
observation of the self-driving car goes to OUT_SDC, each of the other object
of interest to OUT_ADV, as {step, safety, courtesy, speed, per_neighbour}.
import_catk.py does the same for windows.csv; this keeps the per-neighbour
values that the aggression test needs (beta toward the target).

Run in catk's environment (unpickling needs catk's dataclass), from the catk root:

  cd ~/catk && ~/venvs/catk/bin/python <repo>/scripts/responsibility/import_catk_obs.py \\
      RUN SCENES OUT_ADV OUT_SDC [--n 100]

RUN: catk's output (RUN/obs/<scene>.pkl); SCENES: the scene folder it was exported from."""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")  # the catk root (run from it)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from responsibility.scene import Scene  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("scenes")
p.add_argument("out_adv")
p.add_argument("out_sdc")
p.add_argument("--n", type=int, default=100)
a = p.parse_args()
for out in (a.out_adv, a.out_sdc):
    (Path(out) / "obs").mkdir(parents=True, exist_ok=True)
done = 0
for s in range(a.n):
    f = Path(a.run) / "obs" / f"{s}.pkl"
    if not f.exists():
        continue
    scene = Scene.load(Path(a.scenes) / f"{s}.pkl")
    adv = [i for i in scene.objects_of_interest if i != scene.sdc][0]
    role = {str(scene.track_ids[scene.sdc]): (scene.sdc, a.out_sdc), str(scene.track_ids[adv]): (adv, a.out_adv)}
    per_role = {a.out_adv: [], a.out_sdc: []}
    for o in pickle.load(open(f, "rb")):
        r = role.get(str(o.agent_id))
        if r is None:
            continue
        agent, out = r
        beta = [float(x) for x in o.beta]
        per_role[out].append({"step": int(o.k), "safety": beta[0], "courtesy": beta[1],
                              "speed": float(np.linalg.norm(scene.velocity[agent, int(o.k)])),
                              "per_neighbour": {str(b): {"safety": float(v[0]), "courtesy": float(v[1])}
                                                for b, v in (o.per_neighbour or {}).items()}})
    for out, rows in per_role.items():
        rows.sort(key=lambda x: x["step"])
        with open(Path(out) / "obs" / f"{s}.pkl", "wb") as h:
            pickle.dump(rows, h)
    done += 1
print(f"{done} scenes -> {a.out_adv}, {a.out_sdc}")
