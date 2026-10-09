"""Converts a whole WOMD split (e.g. validation_interactive, v1.2.1) into
scene files of the format of CAT's raw_scenes_500, so every reader of this
repository (Scene, DenseTNT, compute_responsibility, validate_right_of_way)
takes them as it takes CAT's 500 scenes.

The conversion is CAT's own (scripts/convert_WOMD_to_MD.py,
convert_scenario: MetaDrive's Waymo extractors), the code that produced
raw_scenes_500. Its selection differs: CAT kept the scenarios whose
self-driving car is one of the two objects of interest
(scripts/select_cases.py); this keeps every scenario whose two objects of
interest are vehicles, and records in the index whether the self-driving
car is one of them (``sdc_in_ooi``: the scenes CAT's closed loop can use,
ego = SDC, adversary = the other) and whether the scenario is one of CAT's
(``in_cat``, from responsibility/unitraj_configs/cat_scenario_ids.txt).

Each shard is converted by one process: OUT/scenes/<scenario id>.pkl, then
OUT/shards/<shard>.json, which marks it done (a re-run skips it, so a run
resumes). OUT/index.csv, rebuilt from the shard files at the end, lists
every scene. The index stays outside the scene folder, since MetaDrive
asserts that a scene folder holds scenes only. --limit converts only the
first N kept scenarios of each shard (a test); a later run without it
redoes those shards.

The records are read without TensorFlow (extract_womd_scenarios.records)
and parsed with the WOMD protos, taken from waymo_open_dataset if
installed, else from --protos (default: the catk environment's copy, same
protobuf version as this one). Runs in the cat39 environment (MetaDrive).

Example (test: two shards, 20 scenes each):
    ~/venvs/cat39/bin/python -m scripts.responsibility.convert_womd_split \\
        --tfrecords ~/womd_v1_2_1/validation_interactive --out-dir ~/womd_v1_2_1/cat_format/validation_interactive \\
        --shards 2 --limit 20 --workers 2
"""

import argparse
import csv
import json
import multiprocessing
import os
import pickle
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.responsibility.extract_womd_scenarios import records  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CAT_IDS = REPO / "responsibility" / "unitraj_configs" / "cat_scenario_ids.txt"
CATK_PROTOS = Path.home() / "venvs" / "catk" / "lib" / "python3.11" / "site-packages" / "waymo_open_dataset" / "protos"
INDEX_FIELDS = ["file", "scenario_id", "shard", "record", "sdc_id", "ooi_a", "ooi_b", "sdc_in_ooi", "in_cat",
                "n_agents", "n_lights"]
VEHICLE = 1  # scenario_pb2.Track.TYPE_VEHICLE


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tfrecords", required=True, help="Directory of the split's tfrecord shards.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shards", type=int, default=None, help="Only the first N shards (a test).")
    p.add_argument("--limit", type=int, default=None, help="Only the first N kept scenarios of each shard (a test).")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--protos", default=str(CATK_PROTOS), help="Directory of scenario_pb2.py, if waymo_open_dataset "
                                                             "is not installed.")
    return p.parse_args(argv)


def load_converter(protos: str):
    """CAT's converter module, its WOMD protos importable (a stand-in
    waymo_open_dataset package over ``protos`` when the real one is
    missing) and TensorFlow kept out (only parse_data uses it)."""
    try:
        from waymo_open_dataset.protos import scenario_pb2  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("waymo_open_dataset")
        pkg.__path__ = [str(Path(protos).parent)]
        sub = types.ModuleType("waymo_open_dataset.protos")
        sub.__path__ = [str(protos)]
        sys.modules["waymo_open_dataset"], sys.modules["waymo_open_dataset.protos"] = pkg, sub
        pkg.protos = sub
    sys.modules.setdefault("tensorflow", None)  # `import tensorflow` raises ImportError, which the module catches
    from scripts import convert_WOMD_to_MD as converter
    return converter


def read_ids(path) -> frozenset:
    return frozenset(line.split("#")[0].strip() for line in Path(path).read_text().splitlines()
                     if line.split("#")[0].strip())


def keep(scenario) -> bool:
    """Two objects of interest, both vehicles, over the 91-step clip."""
    if len(scenario.objects_of_interest) != 2 or len(scenario.timestamps_seconds) != 91:
        return False
    types_by_id = {t.id: t.object_type for t in scenario.tracks}
    return all(types_by_id.get(i) == VEHICLE for i in scenario.objects_of_interest)


def convert_shard(job):
    shard, out, limit, protos, cat_ids = job
    converter = load_converter(protos)
    scenario = converter.scenario_pb2.Scenario()
    rows, scanned, t0 = [], 0, time.time()
    for k, (_, data) in enumerate(records(shard)):
        scanned += 1
        scenario.ParseFromString(data)
        if not keep(scenario):
            continue
        _, description = converter.convert_scenario(scenario, shard.name)
        meta = description["metadata"]
        sid = str(scenario.scenario_id)
        ooi = [str(i) for i in scenario.objects_of_interest]
        path = out / "scenes" / f"{sid}.pkl"
        tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            pickle.dump(description, f)
        os.replace(tmp, path)
        rows.append({"file": path.name, "scenario_id": sid, "shard": shard.name, "record": k,
                     "sdc_id": meta["sdc_id"], "ooi_a": ooi[0], "ooi_b": ooi[1],
                     "sdc_in_ooi": int(meta["sdc_id"] in ooi), "in_cat": int(sid in cat_ids),
                     "n_agents": len(description["tracks"]), "n_lights": len(description["dynamic_map_states"])})
        if limit is not None and len(rows) >= limit:
            break
    marker = out / "shards" / f"{shard.name}.json"
    tmp = marker.with_name(f"{marker.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps({"limit": limit, "scanned": scanned, "rows": rows}))
    os.replace(tmp, marker)
    return shard.name, scanned, len(rows), time.time() - t0


def write_index(out: Path) -> list:
    rows = []
    for marker in sorted((out / "shards").glob("*.json")):
        rows += json.loads(marker.read_text())["rows"]
    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir).expanduser()
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    (out / "shards").mkdir(exist_ok=True)
    shards = sorted(p for p in Path(args.tfrecords).expanduser().iterdir() if p.is_file() and "tfrecord" in p.name)
    shards = shards[:args.shards]

    def done(shard):
        marker = out / "shards" / f"{shard.name}.json"
        return marker.exists() and json.loads(marker.read_text())["limit"] in (None, args.limit)

    todo = [s for s in shards if not done(s)]
    print(f"{len(shards)} shards, {len(shards) - len(todo)} done, {len(todo)} to go", flush=True)
    cat_ids = read_ids(CAT_IDS)
    jobs = [(s, out, args.limit, args.protos, cat_ids) for s in todo]
    load_converter(args.protos)  # fail here, not in every worker, if the protos are missing
    results = (map(convert_shard, jobs) if args.workers <= 1
               else multiprocessing.Pool(args.workers).imap_unordered(convert_shard, jobs))
    for name, scanned, kept, seconds in results:
        print(f"{name}: {kept} of {scanned} scenarios kept ({seconds:.0f} s)", flush=True)
    rows = write_index(out)
    print(f"{len(rows)} scenes in {out / 'scenes'}: {sum(int(r['sdc_in_ooi']) for r in rows)} with the SDC "
          f"among the objects of interest, {sum(int(r['in_cat']) for r in rows)} of CAT's; index {out / 'index.csv'}")
    return rows


if __name__ == "__main__":
    main()
