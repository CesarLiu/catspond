"""End reasons and crash-partner types of the MetaDrive evaluation rollouts (task A6 of the
attribution audit of 2026-10-09). CPU only; reads existing rollouts and crashes.csv.

Counts, over rollouts_eval/<policy>/<mode>/*.pkl: end reason (crash_vehicle / crash_object / crash_building /
out_of_road / arrive_dest / max_step), the type of the recorded crash_with object in the scene, the rows of
crashes.csv without a partner (no window with both sides), and the verdict mix of the rows. It shows that
only vehicle collisions end the evaluation episodes, so crashes.csv covers every crash episode.
  ~/venvs/cat39/bin/python -m scripts.gpucat.audit_attribution_ends
"""
import csv, glob, sys
from collections import Counter
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from responsibility.rollouts import load_rollout
from responsibility.scene import Scene
reasons, types, verdicts, nocrash_row = Counter(), Counter(), Counter(), 0
sc = {}
by_mode = {}
for f in glob.glob(str(REPO / "rollouts_eval/*/*/*.pkl")):
    r = load_rollout(f); mode = Path(f).parent.name
    reasons[r["end"]["reason"]] += 1
    by_mode.setdefault(mode, Counter())[r["end"]["reason"]] += 1
    cw = r["end"].get("crash_with")
    if r["end"]["reason"] == "crash_vehicle":
        s = sc.setdefault(r["scene_file"], Scene.load(REPO / "raw_scenes_500" / f"{r['scene_file']}.pkl"))
        types[s.types[s.index(cw)] if cw is not None and str(cw) in s.track_ids else "none"] += 1
for f in glob.glob(str(REPO / "logs/responsibility/eval/*/*/crashes.csv")):
    for row in csv.DictReader(open(f)):
        verdicts[row["verdict"]] += 1
print("end reasons", dict(reasons)); print("crash_with type", dict(types)); print("verdicts", dict(verdicts))
for m, c in by_mode.items(): print(m, dict(c))
