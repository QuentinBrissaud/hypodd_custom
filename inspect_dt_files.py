#!/usr/bin/env python
"""
Inspect HypoDD dt.ct and dt.cc files.

The parser accepts both catalog and cross-correlation differential-time files:

    # event_id_1 event_id_2 ...
    station value_1 value_2_or_weight weight_or_phase phase

For dt.ct rows, the usual columns are:
    STA TT1 TT2 WEIGHT PHASE

For dt.cc rows, the usual columns are:
    STA DT_CC WEIGHT PHASE

Some dt.cc files can contain event-pair headers but no accepted
cross-correlation observations. This script reports that clearly.
"""

import argparse
from collections import Counter
from pathlib import Path


def read_dt_file(path):
    path = Path(path)
    rows = []
    headers = []
    bad_rows = []
    event_id_1 = None
    event_id_2 = None

    with path.open("r", encoding="utf-8", errors="replace") as open_file:
        for line_number, line in enumerate(open_file, 1):
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "#":
                if len(parts) >= 3:
                    event_id_1 = int(parts[1])
                    event_id_2 = int(parts[2])
                    headers.append(
                        {
                            "line_number": line_number,
                            "event_id_1": event_id_1,
                            "event_id_2": event_id_2,
                        }
                    )
                continue

            if event_id_1 is None or event_id_2 is None:
                bad_rows.append((line_number, line.rstrip("\n"), "no header"))
                continue

            if len(parts) == 4:
                rows.append(
                    {
                        "line_number": line_number,
                        "file_type": "cc",
                        "event_id_1": event_id_1,
                        "event_id_2": event_id_2,
                        "station_id": parts[0],
                        "dt_s": float(parts[1]),
                        "weight": float(parts[2]),
                        "phase": parts[3].upper(),
                    }
                )
            elif len(parts) >= 5:
                rows.append(
                    {
                        "line_number": line_number,
                        "file_type": "ct",
                        "event_id_1": event_id_1,
                        "event_id_2": event_id_2,
                        "station_id": parts[0],
                        "time_1_s": float(parts[1]),
                        "time_2_s": float(parts[2]),
                        "weight": float(parts[3]),
                        "phase": parts[4].upper(),
                    }
                )
            else:
                bad_rows.append((line_number, line.rstrip("\n"), "too few columns"))

    return headers, rows, bad_rows


def summarize_dt_file(path):
    headers, rows, bad_rows = read_dt_file(path)
    phase_counts = Counter(row["phase"] for row in rows)
    type_counts = Counter(row["file_type"] for row in rows)
    weight_values = [row["weight"] for row in rows]

    summary = {
        "path": str(path),
        "event_pair_headers": len(headers),
        "data_rows": len(rows),
        "bad_rows": len(bad_rows),
        "type_counts": dict(type_counts),
        "phase_counts": dict(phase_counts),
        "min_weight": min(weight_values) if weight_values else None,
        "max_weight": max(weight_values) if weight_values else None,
        "mean_weight": (
            sum(weight_values) / len(weight_values) if weight_values else None
        ),
    }
    return summary, rows, bad_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    for path in args.paths:
        summary, rows, bad_rows = summarize_dt_file(path)
        print("\n%s" % summary["path"])
        print("  event-pair headers: %s" % summary["event_pair_headers"])
        print("  data rows:          %s" % summary["data_rows"])
        print("  bad rows:           %s" % summary["bad_rows"])
        print("  row types:          %s" % summary["type_counts"])
        print("  phase counts:       %s" % summary["phase_counts"])
        print("  weight min/mean/max: %s / %s / %s" % (
            summary["min_weight"],
            summary["mean_weight"],
            summary["max_weight"],
        ))
        if summary["data_rows"] == 0:
            print("  WARNING: no observation rows found in this file.")
        if rows:
            print("  first data row:     %s" % rows[0])
        if bad_rows:
            print("  first bad row:      %s" % (bad_rows[0],))


if __name__ == "__main__":
    main()
