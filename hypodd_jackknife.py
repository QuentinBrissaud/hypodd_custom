#!/usr/bin/env python
"""
Leave-one-out jackknife diagnostics for completed HypoDDPy runs.

Two jackknife modes are implemented:

* leave-one-station-out: remove all dt.ct/dt.cc observations at one station
* leave-one-event-out: remove one event from event.sel and all event pairs
  involving that event

The trials are compared against the completed full-data relocation in
``hypoDD.reloc``. This is a sensitivity diagnostic, not a residual-noise
uncertainty model.
"""

from collections import defaultdict
from pathlib import Path
import csv
import json
import math
import shutil
import subprocess

from hypodd_bootstrap import (
    _copy_if_exists,
    _hypodd_executable,
    _local_offsets_km,
    _resolve_input_dir,
    _resolve_output_dir,
    _write_trial_hypodd_input,
    read_dt_file,
    read_hypodd_locations,
    write_dt_file,
)


def _stdev(values):
    values = list(values)
    if len(values) < 2:
        return math.nan
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quantile(values, q):
    values = sorted(values)
    if not values:
        return math.nan
    return values[int(q * (len(values) - 1))]


def _median(values):
    values = sorted(values)
    if not values:
        return math.nan
    n = len(values)
    if n % 2:
        return values[n // 2]
    return 0.5 * (values[n // 2 - 1] + values[n // 2])


def _write_csv(path, rows):
    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _all_stations(blocks_by_type):
    stations = set()
    for blocks in blocks_by_type.values():
        for block in blocks:
            for row in block["rows"]:
                stations.add(row["station_id"])
    return sorted(stations)


def _all_events(blocks_by_type):
    event_ids = set()
    for blocks in blocks_by_type.values():
        for block in blocks:
            event_ids.add(block["event_id_1"])
            event_ids.add(block["event_id_2"])
    return sorted(event_ids)


def _filter_blocks_for_station(blocks, station_id):
    filtered = []
    removed_rows = 0
    kept_rows = 0
    removed_blocks = 0
    for block in blocks:
        rows = [row.copy() for row in block["rows"] if row["station_id"] != station_id]
        removed_rows += len(block["rows"]) - len(rows)
        if rows:
            new_block = block.copy()
            new_block["rows"] = rows
            filtered.append(new_block)
            kept_rows += len(rows)
        else:
            removed_blocks += 1
    return filtered, {
        "removed_rows": removed_rows,
        "kept_rows": kept_rows,
        "removed_blocks": removed_blocks,
    }


def _filter_blocks_for_event(blocks, event_id):
    filtered = []
    removed_rows = 0
    removed_blocks = 0
    kept_rows = 0
    for block in blocks:
        if block["event_id_1"] == event_id or block["event_id_2"] == event_id:
            removed_blocks += 1
            removed_rows += len(block["rows"])
            continue
        new_block = block.copy()
        new_block["rows"] = [row.copy() for row in block["rows"]]
        filtered.append(new_block)
        kept_rows += len(new_block["rows"])
    return filtered, {
        "removed_rows": removed_rows,
        "kept_rows": kept_rows,
        "removed_blocks": removed_blocks,
    }


def _copy_filtered_station_file(source, destination, station_id=None):
    lines = []
    removed = 0
    for line in Path(source).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if station_id is not None and parts and parts[0] == station_id:
            removed += 1
            continue
        lines.append(line)
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return removed


def _copy_filtered_event_file(source, destination, event_id=None):
    lines = []
    removed = 0
    for line in Path(source).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if event_id is not None and parts:
            try:
                current_id = int(float(parts[0]))
            except ValueError:
                current_id = None
            if current_id == event_id:
                removed += 1
                continue
        lines.append(line)
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return removed


def _run_jackknife_trial(
    base_working_dir,
    input_dir,
    trial_dir,
    ct_blocks,
    cc_blocks=None,
    use_cross_correlation=None,
    removed_station=None,
    removed_event=None,
):
    trial_dir = Path(trial_dir).resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)

    write_dt_file(ct_blocks, trial_dir / "dt.ct")
    if use_cross_correlation:
        if cc_blocks is not None:
            write_dt_file(cc_blocks, trial_dir / "dt.cc")
        else:
            _copy_if_exists(input_dir / "dt.cc", trial_dir / "dt.cc")

    _copy_filtered_station_file(
        input_dir / "station.sel",
        trial_dir / "station.sel",
        station_id=removed_station,
    )
    _copy_filtered_event_file(
        input_dir / "event.sel",
        trial_dir / "event.sel",
        event_id=removed_event,
    )
    _write_trial_hypodd_input(
        input_dir / "hypoDD.inp",
        trial_dir / "hypoDD.inp",
        use_cross_correlation,
    )

    hypodd_path = _hypodd_executable(base_working_dir)
    completed = subprocess.run(
        [str(hypodd_path), "hypoDD.inp"],
        cwd=str(trial_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    (trial_dir / "hypoDD.stdout.txt").write_text(
        completed.stdout or "", encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "HypoDD failed in %s with return code %s"
            % (trial_dir, completed.returncode)
        )
    if not (trial_dir / "hypoDD.reloc").exists():
        raise RuntimeError("HypoDD did not create hypoDD.reloc in %s" % trial_dir)
    return read_hypodd_locations(trial_dir / "hypoDD.reloc")


def _trial_impact_row(
    reference_locations,
    trial_locations,
    trial_type,
    target_id,
    removed_rows,
    removed_blocks,
    kept_rows,
):
    common_ids = sorted(set(reference_locations).intersection(trial_locations))
    lost_ids = sorted(set(reference_locations) - set(trial_locations))
    horizontal = []
    dlon = []
    dlat = []
    dz = []
    dt = []
    for event_id in common_ids:
        reference = reference_locations[event_id]
        location = trial_locations[event_id]
        dx_km, dy_km = _local_offsets_km(reference, location)
        horizontal.append(math.hypot(dx_km, dy_km))
        dlon.append(location["longitude"] - reference["longitude"])
        dlat.append(location["latitude"] - reference["latitude"])
        dz.append(location["depth_km"] - reference["depth_km"])
        dt.append((location["time"] - reference["time"]).total_seconds())
    return {
        "trial_type": trial_type,
        "target_id": target_id,
        "status": "success",
        "common_event_count": len(common_ids),
        "lost_reference_event_count": len(lost_ids),
        "trial_event_count": len(trial_locations),
        "removed_observation_rows": removed_rows,
        "removed_event_pair_blocks": removed_blocks,
        "kept_observation_rows": kept_rows,
        "median_horizontal_shift_km": _median(horizontal),
        "p95_horizontal_shift_km": _quantile(horizontal, 0.95),
        "max_horizontal_shift_km": max(horizontal) if horizontal else math.nan,
        "mean_abs_lon_shift_deg": (
            sum(abs(value) for value in dlon) / len(dlon) if dlon else math.nan
        ),
        "mean_abs_lat_shift_deg": (
            sum(abs(value) for value in dlat) / len(dlat) if dlat else math.nan
        ),
        "lon_shift_std_deg": _stdev(dlon),
        "lat_shift_std_deg": _stdev(dlat),
        "median_abs_depth_shift_km": _median([abs(value) for value in dz]),
        "median_abs_origin_time_shift_s": _median([abs(value) for value in dt]),
    }


def _event_variability_rows(reference_locations, trial_records, trial_type):
    values = defaultdict(lambda: defaultdict(list))
    appearances = defaultdict(int)
    for record in trial_records:
        if record["status"] != "success":
            continue
        locations = record["locations"]
        for event_id, reference in reference_locations.items():
            location = locations.get(event_id)
            if location is None:
                continue
            appearances[event_id] += 1
            dx_km, dy_km = _local_offsets_km(reference, location)
            values[event_id]["longitude"].append(location["longitude"])
            values[event_id]["latitude"].append(location["latitude"])
            values[event_id]["depth_km"].append(location["depth_km"])
            values[event_id]["origin_time_shift_s"].append(
                (location["time"] - reference["time"]).total_seconds()
            )
            values[event_id]["dx_km"].append(dx_km)
            values[event_id]["dy_km"].append(dy_km)
            values[event_id]["horizontal_shift_km"].append(math.hypot(dx_km, dy_km))

    rows = []
    successful_trials = sum(1 for record in trial_records if record["status"] == "success")
    for event_id, columns in sorted(values.items()):
        reference = reference_locations[event_id]
        horizontal = columns["horizontal_shift_km"]
        row = {
            "trial_type": trial_type,
            "event_id": event_id,
            "reference_longitude": reference["longitude"],
            "reference_latitude": reference["latitude"],
            "reference_depth_km": reference["depth_km"],
            "n_trials_with_event": appearances[event_id],
            "n_successful_trials": successful_trials,
            "n_trials_missing_event": successful_trials - appearances[event_id],
            "lon_std_deg": _stdev(columns["longitude"]),
            "lat_std_deg": _stdev(columns["latitude"]),
            "depth_std_km": _stdev(columns["depth_km"]),
            "dx_std_km": _stdev(columns["dx_km"]),
            "dy_std_km": _stdev(columns["dy_km"]),
            "origin_time_shift_std_s": _stdev(columns["origin_time_shift_s"]),
            "median_horizontal_shift_km": _median(horizontal),
            "p95_horizontal_shift_km": _quantile(horizontal, 0.95),
        }
        rows.append(row)
    return rows


def run_jackknife_impact(
    working_dir,
    output_dir=None,
    run_station_jackknife=True,
    run_event_jackknife=True,
    stations=None,
    events=None,
    include_cc=True,
    use_cross_correlation=None,
    keep_trial_files="first",
    overwrite=False,
):
    """
    Run station and/or event jackknife sensitivity tests.

    Parameters
    ----------
    working_dir
        Completed HypoDDPy working directory containing ``input_files`` and
        ``output_files``.
    output_dir
        Destination for trial folders and CSV summaries. Defaults to
        ``working_dir/jackknife``.
    run_station_jackknife, run_event_jackknife
        Enable leave-one-station-out and leave-one-event-out trials.
    stations, events
        Optional subsets to test. If omitted, all stations/events found in the
        dt files are tested.
    include_cc
        Whether dt.cc should be filtered too when it exists.
    use_cross_correlation
        Whether HypoDD should use dt.cc in the trials. Defaults to whether
        ``include_cc`` is true and dt.cc exists.
    keep_trial_files
        ``"all"``, ``"first"``, ``"failed"``, or ``"none"``. The default keeps
        the first successful trial per jackknife type plus failed trials.
    overwrite
        Remove an existing output directory first.
    """
    if keep_trial_files is True:
        keep_trial_files = "all"
    elif keep_trial_files is False:
        keep_trial_files = "first"
    keep_trial_files = str(keep_trial_files).lower()
    if keep_trial_files not in ("all", "first", "failed", "none"):
        raise ValueError(
            "keep_trial_files must be one of 'all', 'first', 'failed', or 'none'."
        )

    working_dir = Path(working_dir).resolve()
    input_dir = _resolve_input_dir(working_dir)
    output_files = _resolve_output_dir(working_dir)
    if output_dir is None:
        output_dir = working_dir / "jackknife"
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_locations = read_hypodd_locations(output_files / "hypoDD.reloc")
    ct_blocks = read_dt_file(input_dir / "dt.ct", file_type="ct")
    cc_blocks = None
    if include_cc and (input_dir / "dt.cc").exists():
        cc_blocks = read_dt_file(input_dir / "dt.cc", file_type="cc")
    blocks_by_type = {"ct": ct_blocks}
    if cc_blocks is not None:
        blocks_by_type["cc"] = cc_blocks
    if use_cross_correlation is None:
        use_cross_correlation = bool(cc_blocks)

    if stations is None:
        stations = _all_stations(blocks_by_type)
    else:
        stations = [str(station) for station in stations]
    if events is None:
        events = _all_events(blocks_by_type)
    else:
        events = [int(event) for event in events]

    status_rows = []
    station_impact = []
    event_impact = []
    station_records = []
    event_records = []

    for trial_type, targets in [
        ("station", stations if run_station_jackknife else []),
        ("event", events if run_event_jackknife else []),
    ]:
        kept_first_success = False
        for index, target_id in enumerate(targets, 1):
            safe_target = str(target_id).replace("/", "_").replace("\\", "_")
            trial_dir = output_dir / ("%s_%04d_%s" % (trial_type, index, safe_target))
            if trial_dir.exists() and overwrite:
                shutil.rmtree(trial_dir)
            trial_dir.mkdir(parents=True, exist_ok=True)

            try:
                if trial_type == "station":
                    trial_ct, ct_counts = _filter_blocks_for_station(ct_blocks, target_id)
                    trial_cc = None
                    cc_counts = {"removed_rows": 0, "kept_rows": 0, "removed_blocks": 0}
                    if cc_blocks is not None:
                        trial_cc, cc_counts = _filter_blocks_for_station(
                            cc_blocks, target_id
                        )
                    locations = _run_jackknife_trial(
                        working_dir,
                        input_dir,
                        trial_dir,
                        trial_ct,
                        cc_blocks=trial_cc,
                        use_cross_correlation=use_cross_correlation,
                        removed_station=target_id,
                    )
                else:
                    trial_ct, ct_counts = _filter_blocks_for_event(ct_blocks, target_id)
                    trial_cc = None
                    cc_counts = {"removed_rows": 0, "kept_rows": 0, "removed_blocks": 0}
                    if cc_blocks is not None:
                        trial_cc, cc_counts = _filter_blocks_for_event(
                            cc_blocks, target_id
                        )
                    locations = _run_jackknife_trial(
                        working_dir,
                        input_dir,
                        trial_dir,
                        trial_ct,
                        cc_blocks=trial_cc,
                        use_cross_correlation=use_cross_correlation,
                        removed_event=target_id,
                    )

                removed_rows = ct_counts["removed_rows"] + cc_counts["removed_rows"]
                kept_rows = ct_counts["kept_rows"] + cc_counts["kept_rows"]
                removed_blocks = (
                    ct_counts["removed_blocks"] + cc_counts["removed_blocks"]
                )
                impact = _trial_impact_row(
                    reference_locations,
                    locations,
                    trial_type,
                    target_id,
                    removed_rows=removed_rows,
                    removed_blocks=removed_blocks,
                    kept_rows=kept_rows,
                )
                impact["trial_dir"] = str(trial_dir)
                if trial_type == "station":
                    station_impact.append(impact)
                    station_records.append(
                        {"status": "success", "target_id": target_id, "locations": locations}
                    )
                else:
                    event_impact.append(impact)
                    event_records.append(
                        {"status": "success", "target_id": target_id, "locations": locations}
                    )
                status_rows.append(impact.copy())

                keep_this_success = (
                    keep_trial_files == "all"
                    or (keep_trial_files == "first" and not kept_first_success)
                )
                if keep_this_success and keep_trial_files == "first":
                    kept_first_success = True
                if not keep_this_success and trial_dir.exists():
                    shutil.rmtree(trial_dir)
            except Exception as exc:
                failed = {
                    "trial_type": trial_type,
                    "target_id": target_id,
                    "status": "failed",
                    "trial_dir": str(trial_dir),
                    "error": "%s: %s" % (exc.__class__.__name__, exc),
                }
                status_rows.append(failed)
                if trial_type == "station":
                    station_impact.append(failed)
                    station_records.append(
                        {"status": "failed", "target_id": target_id, "locations": {}}
                    )
                else:
                    event_impact.append(failed)
                    event_records.append(
                        {"status": "failed", "target_id": target_id, "locations": {}}
                    )
                if keep_trial_files == "none" and trial_dir.exists():
                    shutil.rmtree(trial_dir)

    station_variability = _event_variability_rows(
        reference_locations, station_records, "station"
    )
    event_variability = _event_variability_rows(
        reference_locations, event_records, "event"
    )

    _write_csv(output_dir / "station_jackknife_impact.csv", station_impact)
    _write_csv(output_dir / "event_jackknife_impact.csv", event_impact)
    _write_csv(
        output_dir / "station_jackknife_event_variability.csv", station_variability
    )
    _write_csv(output_dir / "event_jackknife_event_variability.csv", event_variability)
    _write_csv(output_dir / "jackknife_trial_status.csv", status_rows)

    metadata = {
        "working_dir": str(working_dir),
        "output_dir": str(output_dir),
        "reference_location_count": len(reference_locations),
        "run_station_jackknife": bool(run_station_jackknife),
        "run_event_jackknife": bool(run_event_jackknife),
        "station_trial_count_requested": len(stations) if run_station_jackknife else 0,
        "event_trial_count_requested": len(events) if run_event_jackknife else 0,
        "station_trial_count_successful": sum(
            1 for row in station_impact if row.get("status") == "success"
        ),
        "event_trial_count_successful": sum(
            1 for row in event_impact if row.get("status") == "success"
        ),
        "include_cc": bool(include_cc),
        "use_cross_correlation": bool(use_cross_correlation),
        "keep_trial_files": keep_trial_files,
        "files": {
            "station_impact": str(output_dir / "station_jackknife_impact.csv"),
            "event_impact": str(output_dir / "event_jackknife_impact.csv"),
            "station_event_variability": str(
                output_dir / "station_jackknife_event_variability.csv"
            ),
            "event_event_variability": str(
                output_dir / "event_jackknife_event_variability.csv"
            ),
            "trial_status": str(output_dir / "jackknife_trial_status.csv"),
        },
    }
    (output_dir / "jackknife_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return {
        "metadata": metadata,
        "station_impact": station_impact,
        "event_impact": event_impact,
        "station_event_variability": station_variability,
        "event_event_variability": event_variability,
        "trial_status": status_rows,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("working_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--stations", nargs="*")
    parser.add_argument("--events", nargs="*", type=int)
    parser.add_argument("--no-stations", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-cc", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-trial-files",
        choices=["all", "first", "failed", "none"],
        default="first",
    )
    args = parser.parse_args()

    result = run_jackknife_impact(
        working_dir=args.working_dir,
        output_dir=args.output_dir,
        run_station_jackknife=not args.no_stations,
        run_event_jackknife=not args.no_events,
        stations=args.stations,
        events=args.events,
        include_cc=not args.no_cc,
        overwrite=args.overwrite,
        keep_trial_files=args.keep_trial_files,
    )
    print(json.dumps(result["metadata"], indent=2))
