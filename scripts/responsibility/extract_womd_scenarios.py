"""Copies the WOMD scenarios with the given ids out of scenario tfrecords
into one tfrecord file, so only CAT's scenarios have to be transferred.

Pure Python 3, no TensorFlow or Waymo packages: it reads the TFRecord
framing (length, CRCs, data) directly and takes each record's scenario id
from the Scenario protobuf's field 5 (``string scenario_id = 5``). Matching
records are copied byte for byte, framing included, so the output is a
valid tfrecord file.

Example (on the machine with the WOMD data):
    python3 extract_womd_scenarios.py --ids cat_scenario_ids.txt \\
        --inputs /data/womd_v1_2_1/validation_interactive --out cat_scenarios_v121.tfrecord
"""

import argparse
import struct
import sys
from pathlib import Path


def _varint(buf, pos):
    result, shift = 0, 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def scenario_id(data: bytes):
    """Field 5 (length-delimited) of a serialized Scenario, or None."""
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            _, pos = _varint(data, pos)
        elif wire == 1:
            pos += 8
        elif wire == 2:
            length, pos = _varint(data, pos)
            if field == 5:
                return data[pos:pos + length].decode()
            pos += length
        elif wire == 5:
            pos += 4
        else:
            return None
    return None


def records(path):
    """(raw framed record, data) of every record in a tfrecord file."""
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            (length,) = struct.unpack("<Q", head[:8])
            data = f.read(length)
            tail = f.read(4)
            yield head + data + tail, data


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", required=True, help="Text file, one scenario id per line ('#' lines are comments).")
    p.add_argument("--inputs", nargs="+", required=True, help="tfrecord files or directories of them.")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    wanted = {line.strip() for line in open(args.ids) if line.strip() and not line.startswith("#")}
    files = []
    for item in args.inputs:
        item = Path(item).expanduser()
        files += sorted(x for x in item.iterdir() if x.is_file()) if item.is_dir() else [item]
    found = set()
    with open(args.out, "wb") as out:
        for i, path in enumerate(files, 1):
            hits = 0
            for framed, data in records(path):
                sid = scenario_id(data)
                if sid in wanted and sid not in found:
                    out.write(framed)
                    found.add(sid)
                    hits += 1
            print(f"[{i}/{len(files)}] {path.name}: {hits} (total {len(found)}/{len(wanted)})", flush=True)
            if found == wanted:
                break
    missing = sorted(wanted - found)
    print(f"wrote {len(found)} scenarios to {args.out}; missing {len(missing)}" + (f": {missing[:10]}" if missing else ""))
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
