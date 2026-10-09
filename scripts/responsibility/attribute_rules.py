"""Recomputes the rule-based verdicts of a policy run's collisions -- RSS
(responsibility/rss.py) and the right of way (responsibility/right_of_way.py)
-- in its crashes*.csv, in place, leaving the counterfactual attribution
as it is. compute_responsibility.py --rollouts writes them with every
collision, but it skips finished scenes when re-run, so crash files written
before a baseline existed (or changed) stay without it; these verdicts need
only the scenes and the rollouts (named in the run's config.json), not the
motion model or MetaDrive, and take well under a second per collision.

Example:
    python -m scripts.responsibility.attribute_rules --runs logs/responsibility/policies/*/*
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from responsibility.right_of_way import rollout_right_of_way  # noqa: E402
from responsibility.rollouts import load_rollout, rollout_files, scene_from_rollout  # noqa: E402
from responsibility.rss import rollout_rss  # noqa: E402
from responsibility.scene import Scene  # noqa: E402
from scripts.responsibility.compute_responsibility import CRASH_FIELDS  # noqa: E402

RULE_COLUMNS = {"rss": "n/a", "rss_case": "", "right_of_way": "n/a", "right_of_way_case": "", "priority": "none"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="Output directories of compute_responsibility --rollouts.")
    return p.parse_args(argv)


def attribute_run(run_dir: Path) -> int:
    """Rewrites the run's crash files; returns the number of collisions."""
    config = json.loads((run_dir / "config.json").read_text())
    if "rollouts" not in config:
        return 0
    paths = sorted(run_dir.glob("crashes*.csv"))
    if not paths:
        return 0
    rollouts = {}
    for path in rollout_files(config["rollouts"]):
        rollout = load_rollout(path)
        rollouts[str(rollout["scene_file"])] = rollout
    count = 0
    for path in paths:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row.update(RULE_COLUMNS)
            rollout = rollouts.get(row["scene"])
            if rollout is None:
                continue
            scene = scene_from_rollout(Scene.load(Path(config["scenes"]) / f"{rollout['scene_file']}.pkl"), rollout)
            for baseline in (rollout_rss(scene, rollout), rollout_right_of_way(scene, rollout)):
                if baseline is not None:
                    row.update(baseline.as_row())
            count += 1
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CRASH_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({k: row.get(k, "") for k in CRASH_FIELDS} for row in rows)
        os.replace(tmp, path)
    return count


def main(argv=None):
    args = parse_args(argv)
    for run in args.runs:
        n = attribute_run(Path(run))
        print(f"{run}: {n} collisions attributed by the rules", flush=True)


if __name__ == "__main__":
    main()
