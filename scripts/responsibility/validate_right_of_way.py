"""Checks the right-of-way rules (responsibility/right_of_way.py) on logged
driving, at scale: WOMD's validation_interactive split converted with
ScenarioNet, or any folder of scene files.

Logged drivers mostly keep the right of way, so the holder the rules name
should usually pass the conflict point P* first. For every scene the pair
is the two objects of interest (--pair ooi; in the interactive split the
two agents WOMD marks as interacting, not necessarily the self-driving
car) or the self-driving car and the other object of interest (--pair sdc,
as in CAT's scenes). It is judged at the pair's closest approach, and the
passing order is read over the whole clip: who reached P* first, and by
how many seconds. A pair counts when its paths cross or merge and one
passed at least --min-lead seconds before the other; a yielder may still
go first lawfully, with the holder far away, so agreement below 100% is
expected even for a correct rule.

For pairs the default rules leave "uncontrolled", the CVC's order for
uncontrolled intersections (first-in, then yield to the right; off by
default, RightOfWayParams.uncontrolled_order) is evaluated too, in the
columns case_cvc / holder_cvc, so one pass measures both choices.

The scenes CAT's adversarial training uses were the development set of
the rules; --exclude responsibility/unitraj_configs/cat_scenario_ids.txt
leaves them out, so the rest is held out.

Writes OUT/pairs.csv (one row per scene; re-running skips the scenes in
it, so a run resumes) and OUT/summary.md (per rule: pairs, pairs with a
passing order, how often the holder passed first, with a 95% Wilson
interval). Scene files are found recursively, ScenarioNet's dataset_*.pkl
index files skipped.

Example (the server, 32 processes):
    python -m scripts.responsibility.validate_right_of_way --scenes /data/womd_sn/validation_interactive \\
        --exclude responsibility/unitraj_configs/cat_scenario_ids.txt --workers 32 \\
        --out-dir logs/responsibility/right_of_way/validation_interactive
"""

import argparse
import csv
import multiprocessing
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from responsibility.right_of_way import (  # noqa: E402
    RightOfWayParams,
    _enters,
    _Path,
    _signals,
    _window,
    priority,
    yield_duty,
)
from responsibility.scene import Scene  # noqa: E402

FIELDS = ["file", "scenario_id", "a", "b", "at", "gap", "kind", "case", "holder", "case_cvc", "holder_cvc",
          "control_a", "control_b", "turn_a", "turn_b", "first", "lead", "yield_kept", "yield_swapped_kept", "error"]
DEFAULT = RightOfWayParams()
CVC = RightOfWayParams(uncontrolled_order=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", required=True, help="Folder of scene files (searched recursively).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pair", choices=["ooi", "sdc"], default="ooi")
    p.add_argument("--exclude", default=None, help="File of scenario ids to leave out (one per line, # comments).")
    p.add_argument("--min-lead", type=float, default=0.5, help="s: passing orders closer than this are not counted.")
    p.add_argument("--n", type=int, default=None, help="Only the first n scene files (a pilot).")
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def scene_paths(directory):
    return sorted(p for p in Path(directory).rglob("*.pkl") if not p.name.startswith("dataset_"))


def _pair(scene: Scene, mode: str):
    if mode == "sdc":
        others = [i for i in scene.objects_of_interest if i != scene.sdc]
        return (scene.sdc, others[0]) if others else None
    return tuple(scene.objects_of_interest[:2]) if len(scene.objects_of_interest) >= 2 else None


def passing_order(scene: Scene, a: int, b: int, conflict):
    """("a" / "b" / "neither", lead in s): who reached P* first over the
    whole clip, and how long before the other (inf when only one did)."""
    first = []
    for x, s_star in ((a, conflict.s[0]), (b, conflict.s[1])):
        path = _Path(scene, x, scene.n_steps - 1, 0.0)
        reached = np.flatnonzero(path.s[:path.n_real] >= s_star)
        first.append(float(path.steps[reached[0]]) if reached.size else np.inf)
    if not np.isfinite(min(first)):
        return "neither", float("nan")
    return ("a" if first[0] < first[1] else "b"), abs(first[0] - first[1]) * 0.1


def yields_kept(scene: Scene, a: int, b: int, prio):
    """Whether the yield duty held over the whole clip for the yielder, and
    would have for the holder had it been the yielder (None: nobody entered)."""
    steps = _window(scene, a, b, scene.n_steps - 1)
    if steps.size < 2:
        return None, None
    c = prio.conflict
    sig = {x: _signals(scene, x, o, _Path(scene, x, int(steps[-1]), DEFAULT.extend), s, c.angle, steps, DEFAULT)
           for x, o, s in ((a, b, c.s[0]), (b, a, c.s[1]))}
    h = prio.holder
    y = b if h == a else a
    if not (np.any(_enters(sig[y][0]) > 0) or np.any(_enters(sig[h][0]) > 0)):
        return None, None
    return (bool(yield_duty(sig[y][0], sig[h][1], sig[h][2])[0] >= 0),
            bool(yield_duty(sig[h][0], sig[y][1], sig[y][2])[0] >= 0))


def judge(job):
    path, mode, exclude = job
    row = {k: "" for k in FIELDS}
    row["file"] = str(path)
    try:
        scene = Scene.load(path)
        row["scenario_id"] = scene.scenario_id
        if scene.scenario_id in exclude:
            row["case"] = "excluded"
            return row
        pair = _pair(scene, mode)
        both = None if pair is None else scene.valid[pair[0]] & scene.valid[pair[1]]
        if pair is None or not both.any():
            row["case"] = "no-pair"
            return row
        a, b = pair
        gap = np.where(both, np.linalg.norm(scene.position[a, :, :2] - scene.position[b, :, :2], axis=-1), np.inf)
        at = int(np.argmin(gap))
        prio = priority(scene, a, b, at, DEFAULT)
        row.update(a=scene.track_ids[a], b=scene.track_ids[b], at=at, gap=round(float(gap[at]), 2), case=prio.case,
                   holder="" if prio.holder is None else "a" if prio.holder == a else "b",
                   case_cvc=prio.case, kind="" if prio.conflict is None else prio.conflict.kind)
        row["holder_cvc"] = row["holder"]
        if prio.case == "uncontrolled":
            cvc = priority(scene, a, b, at, CVC)
            row.update(case_cvc=cvc.case, holder_cvc="" if cvc.holder is None else "a" if cvc.holder == a else "b")
        if prio.approaches is not None:
            pa, pb = prio.approaches
            row.update(control_a=pa.control, control_b=pb.control, turn_a=pa.turn, turn_b=pb.turn)
        if prio.conflict is not None and prio.conflict.kind != "closest":
            row["first"], lead = passing_order(scene, a, b, prio.conflict)
            row["lead"] = "" if not np.isfinite(lead) else round(lead, 1)
            if prio.holder is not None:
                kept, swapped = yields_kept(scene, a, b, prio)
                row.update(yield_kept="" if kept is None else int(kept),
                           yield_swapped_kept="" if swapped is None else int(swapped))
    except Exception as e:  # one malformed scene must not stop a run over tens of thousands
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval of a proportion k / n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def summarize(rows, min_lead: float):
    """[(case, pairs, ordered, holder first, (low, high))] per rule, default
    rules first, then the CVC order where the default says "uncontrolled"."""
    def ordered(r):
        return r["first"] in ("a", "b") and (r["lead"] == "" or float(r["lead"]) >= min_lead)

    judged = [r for r in rows if r["case"] not in ("excluded", "no-pair", "") and not r["error"]]
    out = []
    for column, holder, prefix, subset in (("case", "holder", "", judged),
                                           ("case_cvc", "holder_cvc", "CVC: ",
                                            [r for r in judged if r["case"] == "uncontrolled"])):
        cases = {}
        for r in subset:
            cases.setdefault(r[column], []).append(r)
        for case, rs in sorted(cases.items(), key=lambda kv: -len(kv[1])):
            counted = [r for r in rs if ordered(r) and r[holder]]
            k = sum(r[holder] == r["first"] for r in counted)
            out.append((prefix + case, len(rs), len(counted), k, wilson(k, len(counted))))
    return out


def markdown(rows, min_lead: float) -> str:
    judged = [r for r in rows if r["case"] not in ("excluded", "no-pair", "") and not r["error"]]
    decided = [r for r in judged if r["holder"] and r["first"] in ("a", "b")
               and (r["lead"] == "" or float(r["lead"]) >= min_lead)]
    k = sum(r["holder"] == r["first"] for r in decided)
    lo, hi = wilson(k, len(decided))
    kept = [r for r in judged if r["yield_kept"] != ""]
    lines = [f"scenes: {len(rows)} (excluded {sum(r['case'] == 'excluded' for r in rows)}, "
             f"no pair {sum(r['case'] == 'no-pair' for r in rows)}, errors {sum(bool(r['error']) for r in rows)}); "
             f"pairs judged: {len(judged)}",
             f"holder passed P* first: {k} of {len(decided)} pairs with a holder and a passing order "
             f"({100 * k / max(len(decided), 1):.1f}%, 95% CI {100 * lo:.1f}-{100 * hi:.1f}%)",
             f"yield duty kept by the yielder: {sum(int(r['yield_kept']) for r in kept)} of {len(kept)}; "
             f"by the holder had it been the yielder: {sum(int(r['yield_swapped_kept']) for r in kept)}",
             "", "| rule | pairs | with a holder and an order | holder first | 95% CI |", "|---|---|---|---|---|"]
    for case, n, counted, k, (lo, hi) in summarize(rows, min_lead):
        rate = "–" if counted == 0 else f"{100 * k / counted:.0f}%"
        ci = "–" if counted == 0 else f"{100 * lo:.0f}–{100 * hi:.0f}%"
        lines.append(f"| {case} | {n} | {counted} | {rate} | {ci} |")
    return "\n".join(lines)


def read_ids(path):
    if path is None:
        return frozenset()
    return frozenset(line.split("#")[0].strip() for line in Path(path).read_text().splitlines()
                     if line.split("#")[0].strip())


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs_path = out / "pairs.csv"
    done = set()
    if pairs_path.exists():
        with open(pairs_path) as f:
            done = {r["file"] for r in csv.DictReader(f)}
    paths = scene_paths(args.scenes)[:args.n]
    exclude = read_ids(args.exclude)
    todo = [(p, args.pair, exclude) for p in paths if str(p) not in done]
    print(f"{len(paths)} scene files, {len(paths) - len(todo)} done, {len(todo)} to go", flush=True)
    new = not pairs_path.exists()
    with open(pairs_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        results = (map(judge, todo) if args.workers <= 1
                   else multiprocessing.Pool(args.workers).imap_unordered(judge, todo, chunksize=16))
        for n, row in enumerate(results, 1):
            writer.writerow(row)
            if n % 500 == 0:
                f.flush()
                print(f"[{n}/{len(todo)}]", flush=True)
    with open(pairs_path) as f:
        rows = list(csv.DictReader(f))
    table = markdown(rows, args.min_lead)
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return rows


if __name__ == "__main__":
    main()
