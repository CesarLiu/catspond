"""Measures responsibility/forced_yield.py on scene sets whose
assertiveness is known by construction, and calibrates its threshold on
logged driving.

Each set is a scene folder with a sidecar index (scene.sidecar_index)
naming the pair per scene: "agent" and "other" (make_go_first.py), or
"adversary" (export_adv_scenes.py: the adversary judged against the
self-driving car). For every scene it computes the braking the agent's
entry into the conflict demanded of the other, and whether the other held
the right of way (``forced``: the demand counts only then).

A rewritten scene (make_go_first.py) is judged with the right of way of
the logged scene it came from (<set>/../log/<scene>.pkl), unless
--own-priority: the rewrite keeps the paths, the map and, for the rules
it is made for, the priority, so this tests the braking demand alone; the
right of way itself is checked by validate_right_of_way.py. A delayed
holder comes from further back than its log, along a straight line the
right of way would read as its approach. Column holder_own gives the
rewritten scene's own reading, and the summary how often it agrees.

Writes OUT/pairs.csv (one row per set and scene) and OUT/summary.md: per
set, the pairs, how many have the other as holder, and of those the share
forced above 1, 2 and 4 m/s^2 (4: the yield duty is violated) and above
the --quantile of the reference set's forced demands (--reference, by
default the first set), with the ungated share above 2 m/s^2 for
comparison. Logged yielders that let the holder pass first demand 0 (or,
rarely, inf: entered behind it), so the threshold is calibrated on those
that went first (make_go_first.py's log_yielder_first): how much braking
logged drivers who take a gap leave the holder to do.

Example (make_go_first.py's sets):
    S=logs/responsibility/forced_yield/scenes
    python -m scripts.responsibility.evaluate_forced_yield --out-dir logs/responsibility/forced_yield/eval \\
        --sets log=$S/log delay_pet2=$S/delay_pet2 delay_pet1=$S/delay_pet1 retime_pet1=$S/retime_pet1 \\
               natural=$S/log_yielder_first --reference natural --workers 4
"""

import argparse
import csv
import json
import math
import multiprocessing
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.forced_yield import carry_priority, forced_yield, pair_priority  # noqa: E402
from responsibility.scene import Scene, sidecar_index  # noqa: E402

FIELDS = ["set", "scene", "agent", "other", "case", "kind", "holder", "demand", "entry", "forced", "demand_on_agent",
          "holder_own", "error"]
BANDS = (1.0, 2.0, 4.0)  # m/s^2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="+", required=True, help="name=folder, each with a sidecar index.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--reference", default=None, help="The set the threshold comes from (default: the first).")
    p.add_argument("--quantile", type=float, default=0.9)
    p.add_argument("--own-priority", action="store_true",
                   help="Judge rewritten scenes by their own right of way, not their log's.")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def judge(job):
    name, path, meta, own = job
    row = {**{k: "" for k in FIELDS}, "set": name, "scene": path.stem}
    try:
        scene = Scene.load(path)
        if "agent" in meta:
            agent, other = scene.index(meta["agent"]), scene.index(meta["other"])
        else:
            agent, other = scene.index(meta["adversary"]), scene.sdc
        row.update(agent=scene.track_ids[agent], other=scene.track_ids[other])
        mine = pair_priority(scene, agent, other)
        log = log_scene(path, meta)
        prio = mine if own or log is None else carry_priority(pair_priority(log, agent, other), scene, agent, other)
        row.update(forced_yield(scene, agent, other, prio=prio).as_row())
        if mine is not None:
            row["holder_own"] = "none" if mine.holder is None else "agent" if mine.holder == agent else "other"
    except Exception as e:  # one malformed scene must not stop the run
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def log_scene(path: Path, meta) -> Optional[Scene]:
    """The logged scene a rewritten one (make_go_first.py: "mode" in its
    index) came from, or None."""
    source = path.parent.parent / "log" / path.name
    return Scene.load(source) if "mode" in meta and source.exists() else None


def _value(x: str) -> float:
    return math.inf if x == "inf" else float(x)


def summarize(rows, reference: str, quantile: float):
    """[(set, pairs, other holds, {band: share forced above}, share above
    the reference quantile, ungated share above 2 m/s^2, entries, share of
    pairs whose own right of way agrees)], and that quantile."""
    sets = list(dict.fromkeys(r["set"] for r in rows))
    ref = [_value(r["forced"]) for r in rows if r["set"] == reference and r["forced"] != ""]
    threshold = float(np.quantile(ref, quantile, method="higher")) if ref else math.nan
    out = []
    for name in sets:
        rs = [r for r in rows if r["set"] == name and not r["error"]]
        forced = [_value(r["forced"]) for r in rs if r["forced"] != ""]
        demand = [_value(r["demand"]) for r in rs if r["demand"] != ""]
        share = {b: (sum(f > b for f in forced) / len(forced) if forced else math.nan) for b in BANDS}
        above = sum(f > threshold for f in forced) / len(forced) if forced else math.nan
        ungated = sum(d > 2.0 for d in demand) / len(demand) if demand else math.nan
        agree = sum(r["holder_own"] == r["holder"] for r in rs) / len(rs) if rs else math.nan
        out.append((name, len(rs), len(forced), share, above, ungated, len(demand), agree))
    return out, threshold


def markdown(rows, reference: str, quantile: float) -> str:
    table, threshold = summarize(rows, reference, quantile)
    lines = [f"threshold: the {quantile:g} quantile of {reference}'s forced demands = {threshold:.2f} m/s^2", "",
             "| set | pairs | other holds, agent entered | forced > 1 | > 2 | > 4 (duty violated) "
             f"| > threshold | ungated demand > 2 (of entries) | own right of way agrees |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, n, k, share, above, ungated, entries, agree in table:
        def pct(x):
            return "–" if math.isnan(x) else f"{100 * x:.0f}%"
        lines.append(f"| {name} | {n} | {k} | {pct(share[1.0])} | {pct(share[2.0])} | {pct(share[4.0])} "
                     f"| {pct(above)} | {pct(ungated)} ({entries}) | {pct(agree)} |")
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for spec in args.sets:
        name, folder = spec.split("=", 1)
        index = json.loads(sidecar_index(folder).read_text())
        jobs += [(name, Path(folder) / f"{key}.pkl", meta, args.own_priority) for key, meta in index.items()
                 if (Path(folder) / f"{key}.pkl").exists()]
    rows = list(map(judge, jobs) if args.workers <= 1
                else multiprocessing.Pool(args.workers).imap(judge, jobs, chunksize=4))
    with open(out / "pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    reference = args.reference or args.sets[0].split("=", 1)[0]
    table = markdown(rows, reference, args.quantile)
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return rows


if __name__ == "__main__":
    main()
