"""Builds catk's native cache of CAT's scenes from the original WOMD v1.2.1
data (SMART_PLAN.md): the same scenarios, but as SMART was trained on them
(v1.2.1 maps with driveways), instead of exported from CAT's v1.1 files
(export_catk.py).

CAT's scenes come from WOMD's validation_interactive split; each CAT file
records its source shard. This scans the given tfrecords for the scenarios'
ids and runs catk's own preprocessing on every hit -- the body of catk's
src/data_preprocess.py wm2argo, unchanged -- writing OUT/cache/<scene>.pkl
(scenario id inside: the CAT scene's file stem, as export_catk.py does) and
OUT/index.json. Scenes CAT duplicated get a file each. Reports which
scenarios were not found, and, when reading WOMD's own shards, which were
found in another shard than CAT's v1.1 file named. The tfrecords may also
be one file of just CAT's scenarios (extract_womd_scenarios.py).

Runs in catk's environment (TensorFlow and the WOMD protos). Example:
    ~/venvs/catk/bin/python -m scripts.responsibility.cache_womd_for_cat --catk-root ~/catk \\
        --tfrecords ~/womd_v1_2_1/validation_interactive --out-dir logs/catk/v121
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catk-root", default=os.environ.get("CATK_ROOT", str(Path.home() / "catk")))
    p.add_argument("--tfrecords", required=True, help="Directory of WOMD v1.2.1 scenario tfrecords.")
    p.add_argument("--scenes", default="raw_scenes_500", help="CAT's scenes (for their scenario ids and sources).")
    p.add_argument("--out-dir", required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.path.append(str(Path(args.catk_root).expanduser().resolve()))
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2

    from src.data_preprocess import (decode_dynamic_map_states_from_proto, decode_map_features_from_proto,
                                     decode_tracks_from_proto, get_agent_features, get_map_features,
                                     process_dynamic_map)
    from src.smart.utils.preprocess import preprocess_map

    targets, expected = defaultdict(list), {}
    for f in sorted(Path(args.scenes).glob("*.pkl")):
        with open(f, "rb") as h:
            meta = pickle.load(h)["metadata"]
        targets[str(meta["scenario_id"])].append(f.stem)
        expected[str(meta["scenario_id"])] = str(meta.get("source_file", "")).rsplit(".", 1)[-1]

    out = Path(args.out_dir)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    index, moved = {}, []
    for record in sorted(Path(args.tfrecords).expanduser().iterdir()):
        for raw in tf.data.TFRecordDataset(str(record), compression_type=""):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytes(raw.numpy()))
            sid = scenario.scenario_id
            if sid not in targets or sid in index:
                continue
            # catk's wm2argo, unchanged
            track_infos = decode_tracks_from_proto(scenario)
            map_infos = decode_map_features_from_proto(scenario.map_features)
            lights = process_dynamic_map(decode_dynamic_map_states_from_proto(scenario.dynamic_map_states))
            current = scenario.current_time_index
            data = preprocess_map(get_map_features(map_infos, lights.loc[lights["time_step"] == current]))
            data["agent"] = get_agent_features(track_infos, split="validation", num_historical_steps=current + 1,
                                               num_steps=91)
            shard = record.name.rsplit(".", 1)[-1]
            for stem in targets[sid]:
                data["scenario_id"] = stem
                tmp = out / "cache" / f"{stem}.pkl.tmp{os.getpid()}"
                with open(tmp, "wb") as h:
                    pickle.dump(data, h)
                os.replace(tmp, out / "cache" / f"{stem}.pkl")
            index[sid] = {"scenes": targets[sid], "file": record.name, "agents": int(data["agent"]["num_nodes"])}
            # only meaningful for WOMD's own shard files, not an extract (extract_womd_scenarios.py)
            if "-of-" in shard and expected[sid] and expected[sid] != shard:
                moved.append((sid, expected[sid], shard))
        print(f"{record.name}: {sum(1 for v in index.values() if v['file'] == record.name)} CAT scenarios", flush=True)

    missing = sorted(set(targets) - set(index))
    (out / "index.json").write_text(json.dumps(
        {stem: {"scenario_id": sid, **v} for sid, v in index.items() for stem in v["scenes"]}, indent=1, sort_keys=True))
    print(f"found {len(index)} of {len(targets)} scenarios ({sum(len(v['scenes']) for v in index.values())} CAT scenes); "
          f"missing {len(missing)}: {missing[:10]}; in another shard than CAT recorded: {len(moved)} {moved[:5]}")


if __name__ == "__main__":
    main()
