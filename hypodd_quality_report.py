#!/usr/bin/env python
"""
Quality diagnostics for a hypoddpy/HypoDD relocation.

This module reads files produced by hypoddpy/HypoDD and writes CSV summaries
and optional plots. It intentionally avoids depth-distribution diagnostics.
"""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path


def _float(value):
    try:
        return float(value)
    except ValueError:
        return math.nan


def _hypodd_log_float(value):
    """
    Parse numeric fields from HypoDD logs.

    HypoDD sometimes emits fixed-width overflow fields with asterisks, e.g.
    0*****. Keep a clean leading numeric prefix if present; otherwise NaN.
    """
    try:
        return float(value)
    except ValueError:
        match = re.match(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)", value)
        if match:
            return float(match.group(0))
        return math.nan


def _hypodd_log_int(value):
    parsed = _hypodd_log_float(value)
    if not math.isfinite(parsed):
        return ""
    return int(parsed)


def _percentile(values, percentile):
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return math.nan
    index = (len(values) - 1) * percentile / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def _summary(values):
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return {
            "count": 0,
            "mean": math.nan,
            "median": math.nan,
            "p05": math.nan,
            "p95": math.nan,
            "min": math.nan,
            "max": math.nan,
            "rms": math.nan,
        }
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 50),
        "p05": _percentile(values, 5),
        "p95": _percentile(values, 95),
        "min": min(values),
        "max": max(values),
        "rms": math.sqrt(sum(v * v for v in values) / len(values)),
    }


def _residual_summary(prefix, residuals):
    values = [row["residual_s"] for row in residuals]
    abs_values = [abs(value) for value in values]
    if not values:
        return {
            "%s_count" % prefix: 0,
            "%s_mean_abs_s" % prefix: math.nan,
            "%s_median_abs_s" % prefix: math.nan,
            "%s_rms_s" % prefix: math.nan,
            "%s_mean_s" % prefix: math.nan,
            "%s_p05_abs_s" % prefix: math.nan,
            "%s_p95_abs_s" % prefix: math.nan,
        }
    sorted_abs = sorted(abs_values)
    return {
        "%s_count" % prefix: len(values),
        "%s_mean_abs_s" % prefix: sum(abs_values) / len(abs_values),
        "%s_median_abs_s" % prefix: _percentile(sorted_abs, 50),
        "%s_rms_s" % prefix: math.sqrt(
            sum(value * value for value in values) / len(values)
        ),
        "%s_mean_s" % prefix: sum(values) / len(values),
        "%s_p05_abs_s" % prefix: _percentile(sorted_abs, 5),
        "%s_p95_abs_s" % prefix: _percentile(sorted_abs, 95),
    }


def event_pick_residual_summary_rows(pick_rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in pick_rows:
        grouped[row["event_id"]][row["dataset"]].append(row)

    rows = []
    for event_id in sorted(grouped):
        row = {"event_id": event_id}
        for dataset in ["original", "relocated"]:
            dataset_rows = grouped[event_id].get(dataset, [])
            row.update(_residual_summary("%s_pick" % dataset, dataset_rows))
            row["%s_pick_p_count" % dataset] = sum(
                1 for item in dataset_rows if item.get("phase") == "P"
            )
            row["%s_pick_s_count" % dataset] = sum(
                1 for item in dataset_rows if item.get("phase") == "S"
            )
        rows.append(row)
    return rows


def event_double_difference_summary_rows(dd_rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in dd_rows:
        for key in ["event_id_1", "event_id_2"]:
            grouped[row[key]][row["dataset"]].append(row)

    rows = []
    for event_id in sorted(grouped):
        row = {"event_id": event_id}
        for dataset in ["original", "relocated"]:
            dataset_rows = grouped[event_id].get(dataset, [])
            row.update(_residual_summary("%s_dd" % dataset, dataset_rows))
            linked_events = set()
            for item in dataset_rows:
                linked_events.add(item["event_id_1"])
                linked_events.add(item["event_id_2"])
            linked_events.discard(event_id)
            row["%s_dd_linked_event_count" % dataset] = len(linked_events)
            row["%s_dd_station_count" % dataset] = len(
                {item["station_id"] for item in dataset_rows}
            )
            row["%s_dd_p_count" % dataset] = sum(
                1 for item in dataset_rows if item.get("phase") == "P"
            )
            row["%s_dd_s_count" % dataset] = sum(
                1 for item in dataset_rows if item.get("phase") == "S"
            )
        rows.append(row)
    return rows


def paired_double_difference_rows(original_rows, relocated_rows):
    """
    Return original/relocated DD rows restricted to observations in both sets.

    HypoDD can skip observations during later iterations, so initial and final
    diagnostic files do not necessarily contain identical rows. Matching by
    event pair, station, and type makes before/after comparisons meaningful.
    """
    def key(row):
        return (
            row["event_id_1"],
            row["event_id_2"],
            row["station_id"],
            row["type_index"],
        )

    original_by_key = defaultdict(deque)
    for row in original_rows:
        original_by_key[key(row)].append(row)

    paired_original = []
    paired_relocated = []
    for row in relocated_rows:
        row_key = key(row)
        if not original_by_key[row_key]:
            continue
        paired_original.append(original_by_key[row_key].popleft())
        paired_relocated.append(row)

    return paired_original, paired_relocated


def combined_event_residual_summary_rows(pick_event_rows, dd_event_rows):
    by_event = {}
    for row in pick_event_rows:
        by_event.setdefault(row["event_id"], {"event_id": row["event_id"]}).update(row)
    for row in dd_event_rows:
        by_event.setdefault(row["event_id"], {"event_id": row["event_id"]}).update(row)

    for row in by_event.values():
        original_pick = row.get("original_pick_mean_abs_s", math.nan)
        relocated_pick = row.get("relocated_pick_mean_abs_s", math.nan)
        original_dd = row.get("original_dd_mean_abs_s", math.nan)
        relocated_dd = row.get("relocated_dd_mean_abs_s", math.nan)
        row["pick_mean_abs_change_s"] = relocated_pick - original_pick
        row["dd_mean_abs_change_s"] = relocated_dd - original_dd

    return [by_event[event_id] for event_id in sorted(by_event)]


def _write_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["no_rows"]
    with open(path, "w", newline="", encoding="utf-8") as open_file:
        writer = csv.DictWriter(open_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hypodd_time(parts, first_index):
    year = int(parts[first_index])
    month = int(parts[first_index + 1])
    day = int(parts[first_index + 2])
    hour = int(parts[first_index + 3])
    minute = int(parts[first_index + 4])
    second = float(parts[first_index + 5])
    whole_second = int(second)
    microsecond = int(round((second - whole_second) * 1_000_000))
    if whole_second >= 60:
        minute += 1
        whole_second -= 60
    return datetime(year, month, day, hour, minute, whole_second, microsecond)


def read_hypodd_locations(path):
    """
    Read hypoDD.loc or hypoDD.reloc.

    Returns a dict keyed by event id. Units are km for depth and seconds for
    residual fields if present in the source file.
    """
    events = {}
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if len(parts) < 17:
                continue
            event_id = int(parts[0])
            lat = float(parts[1])
            lon = float(parts[2])
            depth_km = float(parts[3])
            time = _hypodd_time(parts, 10)
            row = {
                "event_id": event_id,
                "latitude": lat,
                "longitude": lon,
                "depth_km": depth_km,
                "time": time,
            }
            if len(parts) >= 24:
                row.update(
                    {
                        "nccp": int(float(parts[17])),
                        "nccs": int(float(parts[18])),
                        "nctp": int(float(parts[19])),
                        "ncts": int(float(parts[20])),
                        "rms_cc": float(parts[21]),
                        "rms_ct": float(parts[22]),
                        "cluster_id": int(float(parts[23])),
                    }
                )
            events[event_id] = row
    return events


def read_station_dat(path):
    stations = {}
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if len(parts) < 3:
                continue
            stations[parts[0]] = {
                "station_id": parts[0],
                "latitude": float(parts[1]),
                "longitude": float(parts[2]),
                "elevation_km": float(parts[3]) / 1000.0 if len(parts) == 4 else 0.0,
            }
    return stations


def read_phase_dat(path):
    """
    Read hypoddpy phase.dat.

    Pick rows contain absolute observed travel times relative to the event
    origin time in the preceding event header.
    """
    events = {}
    current = None
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "#":
                event_id = int(parts[-1])
                second = float(parts[6])
                whole_second = int(second)
                microsecond = int(round((second - whole_second) * 1_000_000))
                current = {
                    "event_id": event_id,
                    "time": datetime(
                        int(parts[1]),
                        int(parts[2]),
                        int(parts[3]),
                        int(parts[4]),
                        int(parts[5]),
                        whole_second,
                        microsecond,
                    ),
                    "latitude": float(parts[7]),
                    "longitude": float(parts[8]),
                    "depth_km": float(parts[9]),
                    "picks": [],
                }
                events[event_id] = current
            elif current is not None and len(parts) >= 4:
                current["picks"].append(
                    {
                        "station_id": parts[0],
                        "travel_time": float(parts[1]),
                        "weight": float(parts[2]),
                        "phase": parts[3].upper(),
                    }
                )
    return events


def read_velocity_model(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8-sig") as open_file:
        reader = csv.DictReader(open_file)
        for row in reader:
            depth_key = next(k for k in row if k.lower().startswith("depth"))
            vp_key = next(k for k in row if k.lower().startswith("vp"))
            vs_key = next(k for k in row if k.lower().startswith("vs"))
            rows.append(
                {
                    "depth_km": float(row[depth_key]),
                    "vp": float(row[vp_key]),
                    "vs": float(row[vs_key]),
                }
            )
    return sorted(rows, key=lambda item: item["depth_km"])


def velocity_at_depth(velocity_model, depth_km, phase):
    phase = phase.upper()
    key = "vp" if phase == "P" else "vs"
    chosen = velocity_model[0]
    for layer in velocity_model:
        if depth_km >= layer["depth_km"]:
            chosen = layer
        else:
            break
    return chosen[key]


def horizontal_distance_km(lat1, lon1, lat2, lon2):
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111.0 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 111.0
    return math.sqrt(dx * dx + dy * dy)


def distance_3d_km(event, station):
    horizontal = horizontal_distance_km(
        event["latitude"],
        event["longitude"],
        station["latitude"],
        station["longitude"],
    )
    vertical = event["depth_km"] + station["elevation_km"]
    return math.sqrt(horizontal * horizontal + vertical * vertical)


def inter_event_distances(events):
    values = []
    event_list = list(events.values())
    for i, event_1 in enumerate(event_list):
        for event_2 in event_list[i + 1 :]:
            values.append(
                horizontal_distance_km(
                    event_1["latitude"],
                    event_1["longitude"],
                    event_2["latitude"],
                    event_2["longitude"],
                )
            )
    return values


def nearest_neighbor_distances(events):
    event_list = list(events.values())
    values = []
    for i, event in enumerate(event_list):
        distances = [
            horizontal_distance_km(
                event["latitude"],
                event["longitude"],
                other["latitude"],
                other["longitude"],
            )
            for j, other in enumerate(event_list)
            if i != j
        ]
        if distances:
            values.append(min(distances))
    return values


def location_shift_rows(original, relocated):
    rows = []
    for event_id, rel_event in relocated.items():
        if event_id not in original:
            continue
        orig_event = original[event_id]
        horizontal_shift_km = horizontal_distance_km(
            orig_event["latitude"],
            orig_event["longitude"],
            rel_event["latitude"],
            rel_event["longitude"],
        )
        rows.append(
            {
                "event_id": event_id,
                "horizontal_shift_km": horizontal_shift_km,
                "time_shift_s": (rel_event["time"] - orig_event["time"]).total_seconds(),
                "cluster_id": rel_event.get("cluster_id", ""),
            }
        )
    return rows


def pick_residual_rows(phase_events, locations, stations, velocity_model, label):
    rows = []
    for event_id, event in phase_events.items():
        if event_id not in locations:
            continue
        location = locations[event_id]
        origin_time_shift = (
            location["time"] - event["time"]
        ).total_seconds()
        for pick in event["picks"]:
            station = stations.get(pick["station_id"])
            if station is None:
                continue
            velocity = velocity_at_depth(
                velocity_model, location["depth_km"], pick["phase"]
            )
            theoretical = distance_3d_km(location, station) / velocity
            observed_relative = pick["travel_time"] - origin_time_shift
            residual = observed_relative - theoretical
            rows.append(
                {
                    "dataset": label,
                    "event_id": event_id,
                    "station_id": pick["station_id"],
                    "phase": pick["phase"],
                    "observed_s": observed_relative,
                    "theoretical_s": theoretical,
                    "residual_s": residual,
                    "absolute_residual_s": abs(residual),
                }
            )
    return rows


def read_hypodd_travel_times(path):
    """
    Read patched HypoDD source-station P/S travel-time diagnostics.
    """
    travel_times = {}
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) < 4:
                continue
            event_id = int(parts[0])
            station_id = parts[1]
            travel_times[(event_id, station_id, "P")] = float(parts[2])
            travel_times[(event_id, station_id, "S")] = float(parts[3])
    return travel_times


def pick_residual_rows_from_hypodd_travel_times(
    phase_events,
    locations,
    travel_times,
    label,
):
    """
    Compute absolute pick residuals using HypoDD ray-traced travel times.
    """
    rows = []
    for event_id, event in phase_events.items():
        if event_id not in locations:
            continue
        location = locations[event_id]
        origin_time_shift = (location["time"] - event["time"]).total_seconds()
        for pick in event["picks"]:
            phase = pick["phase"].upper()
            theoretical = travel_times.get((event_id, pick["station_id"], phase))
            if theoretical is None:
                continue
            observed_relative = pick["travel_time"] - origin_time_shift
            residual = observed_relative - theoretical
            rows.append(
                {
                    "dataset": label,
                    "event_id": event_id,
                    "station_id": pick["station_id"],
                    "phase": phase,
                    "observed_s": observed_relative,
                    "theoretical_s": theoretical,
                    "residual_s": residual,
                    "absolute_residual_s": abs(residual),
                    "source": "hypodd_ray_tracing",
                }
            )
    return rows


def read_catalog_differential_times(path):
    """
    Read ph2dt/HypoDD catalog differential-time observations from dt.ct.
    """
    rows = []
    event_id_1 = None
    event_id_2 = None
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "#":
                if len(parts) >= 3:
                    event_id_1 = int(parts[1])
                    event_id_2 = int(parts[2])
                continue
            if event_id_1 is None or len(parts) < 5:
                continue
            rows.append(
                {
                    "event_id_1": event_id_1,
                    "event_id_2": event_id_2,
                    "station_id": parts[0],
                    "observed_1_s": float(parts[1]),
                    "observed_2_s": float(parts[2]),
                    "weight": float(parts[3]),
                    "phase": parts[4].upper(),
                }
            )
    return rows


def model_double_difference_residual_rows(
    differential_rows,
    phase_events,
    locations,
    stations,
    velocity_model,
    label,
):
    """
    Recompute catalog double-difference residuals from a simple travel-time model.
    """
    rows = []
    for row in differential_rows:
        event_1 = locations.get(row["event_id_1"])
        event_2 = locations.get(row["event_id_2"])
        phase_event_1 = phase_events.get(row["event_id_1"])
        phase_event_2 = phase_events.get(row["event_id_2"])
        station = stations.get(row["station_id"])
        if (
            event_1 is None
            or event_2 is None
            or phase_event_1 is None
            or phase_event_2 is None
            or station is None
        ):
            continue

        shift_1 = (event_1["time"] - phase_event_1["time"]).total_seconds()
        shift_2 = (event_2["time"] - phase_event_2["time"]).total_seconds()
        observed_difference = (
            row["observed_1_s"] - shift_1
        ) - (
            row["observed_2_s"] - shift_2
        )

        velocity_1 = velocity_at_depth(velocity_model, event_1["depth_km"], row["phase"])
        velocity_2 = velocity_at_depth(velocity_model, event_2["depth_km"], row["phase"])
        theoretical_1 = distance_3d_km(event_1, station) / velocity_1
        theoretical_2 = distance_3d_km(event_2, station) / velocity_2
        theoretical_difference = theoretical_1 - theoretical_2
        residual = observed_difference - theoretical_difference
        rows.append(
            {
                "dataset": label,
                "event_id_1": row["event_id_1"],
                "event_id_2": row["event_id_2"],
                "station_id": row["station_id"],
                "phase": row["phase"],
                "weight": row["weight"],
                "observed_difference_s": observed_difference,
                "theoretical_difference_s": theoretical_difference,
                "residual_s": residual,
                "absolute_residual_s": abs(residual),
            }
        )
    return rows


def read_hypodd_residuals(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if len(parts) < 9:
                continue
            idx = int(parts[4])
            rows.append(
                {
                    "station_id": parts[0],
                    "observed_difference_s": float(parts[1]),
                    "residual_s": float(parts[6]) / 1000.0,
                    "absolute_residual_s": abs(float(parts[6]) / 1000.0),
                    "event_id_1": int(parts[2]),
                    "event_id_2": int(parts[3]),
                    "type_index": idx,
                    "quality": float(parts[5]),
                    "weight": float(parts[7]),
                    "offset_km": float(parts[8]),
                    "phase": "P" if idx in [1, 3] else "S",
                    "data_type": "cc" if idx in [1, 2] else "ct",
                }
            )
    return rows


def read_hypodd_diagnostic_residuals(path, label):
    """
    Read patched HypoDD diagnostic residual files.

    Columns are seconds for observed, calculated, and residual differential
    times. These are written directly by HypoDD after its ray tracing.
    """
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) < 10:
                continue
            idx = int(parts[6])
            rows.append(
                {
                    "dataset": label,
                    "station_id": parts[0],
                    "observed_difference_s": float(parts[1]),
                    "theoretical_difference_s": float(parts[2]),
                    "residual_s": float(parts[3]),
                    "absolute_residual_s": abs(float(parts[3])),
                    "event_id_1": int(parts[4]),
                    "event_id_2": int(parts[5]),
                    "type_index": idx,
                    "phase": "P" if idx in [1, 3] else "S",
                    "data_type": "cc" if idx in [1, 2] else "ct",
                    "weight": float(parts[8]),
                    "offset_km": float(parts[9]),
                    "source": "hypodd_ray_tracing",
                }
            )
    return rows


def read_station_residuals(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if len(parts) < 11:
                continue
            rows.append(
                {
                    "station_id": parts[0],
                    "latitude": float(parts[1]),
                    "longitude": float(parts[2]),
                    "mean_cc_p_s": float(parts[3]),
                    "mean_cc_s_s": float(parts[4]),
                    "n_cc_p": int(float(parts[5])),
                    "n_cc_s": int(float(parts[6])),
                    "n_ct_p": int(float(parts[7])),
                    "n_ct_s": int(float(parts[8])),
                    "rms_cc_s": float(parts[9]),
                    "rms_ct_s": float(parts[10]),
                    "cluster_id": int(float(parts[11])) if len(parts) > 11 else "",
                }
            )
    return rows


def parse_hypodd_log(path):
    rows = []
    current = None
    summary_mode = None
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            match = re.search(r"===ITERATION\s+(\d+)\s+\(\s*(\d+)\)", line)
            if match:
                current = {
                    "iteration": int(match.group(1)),
                    "global_iteration": int(match.group(2)),
                }
                summary_mode = None
                continue
            if current is None:
                continue

            if "Residual summary of initial data:" in line:
                summary_mode = "initial"
                continue
            if line.strip() == "Residual summary:":
                summary_mode = "post"
                continue

            if summary_mode is not None:
                match = re.search(r"absolute mean \[s\] =\s*([0-9.Ee+*\-]+)", line)
                if match:
                    current["%s_absolute_mean_s" % summary_mode] = (
                        _hypodd_log_float(match.group(1))
                    )
                    continue
                match = re.search(r"weighted mean \[s\] =\s*([0-9.Ee+*\-]+)", line)
                if match:
                    current["%s_weighted_mean_s" % summary_mode] = (
                        _hypodd_log_float(match.group(1))
                    )
                    continue
                match = re.search(r"absolute variance \[s\] =\s*([0-9.Ee+*\-]+)", line)
                if match:
                    current["%s_absolute_variance_s" % summary_mode] = (
                        _hypodd_log_float(match.group(1))
                    )
                    continue
                match = re.search(r"weighted variance \[s\] =\s*([0-9.Ee+*\-]+)", line)
                if match:
                    current["%s_weighted_variance_s" % summary_mode] = (
                        _hypodd_log_float(match.group(1))
                    )
                    continue
                match = re.search(r"absolute cc rms \[s\] =\s*([0-9.Ee+*\-]+)", line)
                if match:
                    current["%s_absolute_cc_rms_s" % summary_mode] = (
                        _hypodd_log_float(match.group(1))
                    )
                    continue
                match = re.search(r"weighted cc rms \[s\](?: \(RMSCC\))? =\s*([0-9.Ee+*\-]+)", line)
                if match:
                    current["%s_weighted_cc_rms_s" % summary_mode] = (
                        _hypodd_log_float(match.group(1))
                    )
                    current["rms_cc_s"] = _hypodd_log_float(match.group(1))
                    continue
                match = re.search(r"absolute ct rms \[s\] =\s*([0-9.Ee+*\-]+)", line)
                if match:
                    current["%s_absolute_ct_rms_s" % summary_mode] = (
                        _hypodd_log_float(match.group(1))
                    )
                    continue
                match = re.search(r"weighted ct rms \[s\](?: \(RMSCT\))? =\s*([0-9.Ee+*\-]+)", line)
                if match:
                    current["%s_weighted_ct_rms_s" % summary_mode] = (
                        _hypodd_log_float(match.group(1))
                    )
                    current["rms_ct_s"] = _hypodd_log_float(match.group(1))
                    continue

            match = re.search(r"^\s*(\d+)\s+\d+\s+\d+\s+", line)
            if match:
                parts = line.split()
                if len(parts) >= 14:
                    current["table_iteration"] = int(parts[0])
                    current["events_percent"] = _hypodd_log_float(parts[2])
                    current["ct_percent"] = _hypodd_log_float(parts[3])
                    current["cc_percent"] = _hypodd_log_float(parts[4])
                    current["table_rms_ct_ms"] = _hypodd_log_float(parts[5])
                    current["table_rms_ct_percent"] = _hypodd_log_float(parts[6])
                    current["table_rms_cc_ms"] = _hypodd_log_float(parts[7])
                    current["table_rms_cc_percent"] = _hypodd_log_float(parts[8])
                    current["rms_station_ms"] = _hypodd_log_float(parts[9])
                    current["mean_abs_dx_m"] = _hypodd_log_float(parts[10])
                    current["mean_abs_dy_m"] = _hypodd_log_float(parts[11])
                    current["mean_abs_dz_m"] = _hypodd_log_float(parts[12])
                    current["mean_abs_dt_ms"] = _hypodd_log_float(parts[13])
                    if len(parts) > 14:
                        current["origin_shift_m"] = _hypodd_log_float(parts[14])
                    if len(parts) > 15:
                        current["airquake_count"] = _hypodd_log_int(parts[15])
                    if "rms_ct_s" not in current:
                        current["rms_ct_s"] = current["table_rms_ct_ms"] / 1000.0
                    if "rms_cc_s" not in current:
                        current["rms_cc_s"] = current["table_rms_cc_ms"] / 1000.0
                    rows.append(current)
                    current = None
                    summary_mode = None
                elif len(parts) >= 12:
                    current["table_iteration"] = int(parts[0])
                    current["events_percent"] = _hypodd_log_float(parts[2])
                    current["ct_percent"] = _hypodd_log_float(parts[3])
                    current["table_rms_ct_ms"] = _hypodd_log_float(parts[4])
                    current["table_rms_ct_percent"] = _hypodd_log_float(parts[5])
                    current["rms_station_ms"] = _hypodd_log_float(parts[6])
                    current["mean_abs_dx_m"] = _hypodd_log_float(parts[7])
                    current["mean_abs_dy_m"] = _hypodd_log_float(parts[8])
                    current["mean_abs_dz_m"] = _hypodd_log_float(parts[9])
                    current["mean_abs_dt_ms"] = _hypodd_log_float(parts[10])
                    current["origin_shift_m"] = _hypodd_log_float(parts[11])
                    if len(parts) > 12:
                        current["airquake_count"] = _hypodd_log_int(parts[12])
                    if "rms_ct_s" not in current:
                        current["rms_ct_s"] = current["table_rms_ct_ms"] / 1000.0
                    rows.append(current)
                    current = None
                    summary_mode = None
                continue
    block = 1
    previous_iteration = None
    for row in rows:
        current_iteration = row.get("global_iteration")
        if (
            previous_iteration is not None
            and current_iteration is not None
            and current_iteration <= previous_iteration
        ):
            block += 1
        row["convergence_block"] = block
        previous_iteration = current_iteration
    return rows


def _month_label(time):
    if time is None:
        return ""
    return time.strftime("%Y-%m")


def _midpoint_time(time_1, time_2):
    return time_1 + (time_2 - time_1) / 2


def add_pick_plot_metadata(rows, locations):
    """
    Add month and cluster metadata to pick residual rows.
    """
    enriched = []
    for row in rows:
        event = locations.get(row["event_id"])
        if event is None:
            continue
        item = dict(row)
        item["month"] = _month_label(event["time"])
        item["cluster_id"] = event.get("cluster_id", "")
        enriched.append(item)
    return enriched


def add_dd_plot_metadata(rows, locations):
    """
    Add midpoint month and cluster metadata to DD residual rows.

    A DD residual is assigned to a cluster only when both events belong to the
    same cluster. Cross-cluster residuals are skipped for cluster boxplots.
    """
    enriched = []
    for row in rows:
        event_1 = locations.get(row["event_id_1"])
        event_2 = locations.get(row["event_id_2"])
        if event_1 is None or event_2 is None:
            continue
        item = dict(row)
        item["month"] = _month_label(_midpoint_time(event_1["time"], event_2["time"]))
        cluster_1 = event_1.get("cluster_id", "")
        cluster_2 = event_2.get("cluster_id", "")
        item["cluster_id"] = cluster_1 if cluster_1 == cluster_2 else ""
        enriched.append(item)
    return enriched


def _selected_cluster_ids(cluster_rows, max_clusters_to_plot):
    cluster_rows = [
        row for row in cluster_rows if row.get("cluster_id") not in ("", None)
    ]
    cluster_rows = sorted(
        cluster_rows,
        key=lambda row: (-int(row.get("event_count", 0)), row.get("cluster_id")),
    )
    if max_clusters_to_plot is not None:
        cluster_rows = cluster_rows[:max_clusters_to_plot]
    return [row["cluster_id"] for row in cluster_rows]


def _boxplot_values(rows, group_key, value_key, groups, phase=None):
    values = []
    for group in groups:
        group_values = []
        for row in rows:
            if row.get(group_key) != group:
                continue
            if phase is not None and row.get("phase") != phase:
                continue
            value = row.get(value_key, math.nan)
            if math.isfinite(value):
                group_values.append(value)
        values.append(group_values)
    return values


def _draw_boxplot(
    ax,
    rows,
    group_key,
    value_key,
    groups,
    title,
    ylabel,
    phase=None,
    signed=False,
    ylim=None,
):
    values = _boxplot_values(rows, group_key, value_key, groups, phase=phase)
    non_empty = [(group, vals) for group, vals in zip(groups, values) if vals]
    if not non_empty:
        ax.set_axis_off()
        return False
    labels, data = zip(*non_empty)
    ax.boxplot(data, labels=[str(label) for label in labels], showfliers=False)
    for index, vals in enumerate(data, 1):
        if not vals:
            continue
        # Deterministic jitter: enough to reveal dense groups without making
        # repeated report runs visually different.
        offsets = [
            ((item % 17) - 8) / 8.0 * 0.08
            for item in range(len(vals))
        ]
        ax.scatter(
            [index + offset for offset in offsets],
            vals,
            s=5,
            color="0.20",
            alpha=0.22,
            linewidths=0,
            zorder=2,
        )
    if signed:
        ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=45)
    return True


def _finite_values(rows, key, group_key=None, groups=None, phase=None):
    values = []
    for row in rows:
        if group_key is not None and row.get(group_key) not in groups:
            continue
        if phase is not None and row.get("phase") != phase:
            continue
        value = row.get(key, math.nan)
        if math.isfinite(value):
            values.append(value)
    return values


def _shared_ylim(values, signed=False):
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return None
    if signed:
        limit = max(abs(value) for value in values)
        if limit == 0:
            limit = 1.0
        return (-limit * 1.08, limit * 1.08)
    upper = max(values)
    if upper == 0:
        upper = 1.0
    return (0.0, upper * 1.08)


def _residual_boxplot_limits(original_dd, original_pick, relocated_dd, relocated_pick):
    return {
        "dd_month": _shared_ylim(
            _finite_values(original_dd + relocated_dd, "residual_s"),
            signed=True,
        ),
        "pick_month": _shared_ylim(
            _finite_values(original_pick + relocated_pick, "absolute_residual_s")
        ),
        "dd_p_month": _shared_ylim(
            _finite_values(original_dd + relocated_dd, "residual_s", phase="P"),
            signed=True,
        ),
        "dd_s_month": _shared_ylim(
            _finite_values(original_dd + relocated_dd, "residual_s", phase="S"),
            signed=True,
        ),
        "pick_p_month": _shared_ylim(
            _finite_values(
                original_pick + relocated_pick, "absolute_residual_s", phase="P"
            )
        ),
        "pick_s_month": _shared_ylim(
            _finite_values(
                original_pick + relocated_pick, "absolute_residual_s", phase="S"
            )
        ),
    }


def _add_cluster_limits(
    limits,
    original_dd,
    original_pick,
    relocated_dd,
    relocated_pick,
    cluster_ids,
):
    limits.update(
        {
            "dd_cluster": _shared_ylim(
                _finite_values(
                    original_dd + relocated_dd,
                    "residual_s",
                    group_key="cluster_id",
                    groups=set(cluster_ids),
                ),
                signed=True,
            ),
            "pick_cluster": _shared_ylim(
                _finite_values(
                    original_pick + relocated_pick,
                    "absolute_residual_s",
                    group_key="cluster_id",
                    groups=set(cluster_ids),
                )
            ),
            "dd_p_cluster": _shared_ylim(
                _finite_values(
                    original_dd + relocated_dd,
                    "residual_s",
                    group_key="cluster_id",
                    groups=set(cluster_ids),
                    phase="P",
                ),
                signed=True,
            ),
            "dd_s_cluster": _shared_ylim(
                _finite_values(
                    original_dd + relocated_dd,
                    "residual_s",
                    group_key="cluster_id",
                    groups=set(cluster_ids),
                    phase="S",
                ),
                signed=True,
            ),
            "pick_p_cluster": _shared_ylim(
                _finite_values(
                    original_pick + relocated_pick,
                    "absolute_residual_s",
                    group_key="cluster_id",
                    groups=set(cluster_ids),
                    phase="P",
                )
            ),
            "pick_s_cluster": _shared_ylim(
                _finite_values(
                    original_pick + relocated_pick,
                    "absolute_residual_s",
                    group_key="cluster_id",
                    groups=set(cluster_ids),
                    phase="S",
                )
            ),
        }
    )
    return limits


def make_residual_boxplots_for_dataset(
    output_dir,
    dd_rows,
    pick_rows,
    cluster_ids,
    dataset_label,
    filename_prefix,
    limits,
):
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    plot_paths = []

    def save(fig, filename):
        path = output_dir / filename
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path))

    months = sorted(
        {
            row.get("month")
            for row in dd_rows + pick_rows
            if row.get("month") not in ("", None)
        }
    )

    if months:
        fig, axes = plt.subplots(2, 1, figsize=(max(8, len(months) * 0.45), 7))
        drawn = [
            _draw_boxplot(
                axes[0],
                dd_rows,
                "month",
                "residual_s",
                months,
                "%s Signed Double-Difference Residuals By Month" % dataset_label,
                "DD residual (s)",
                signed=True,
                ylim=limits.get("dd_month"),
            ),
            _draw_boxplot(
                axes[1],
                pick_rows,
                "month",
                "absolute_residual_s",
                months,
                "%s Absolute Pick Residuals By Month" % dataset_label,
                "|pick residual| (s)",
                ylim=limits.get("pick_month"),
            ),
        ]
        if any(drawn):
            save(fig, "%s_residual_boxplots_by_month.png" % filename_prefix)
        else:
            plt.close(fig)

        fig, axes = plt.subplots(
            2, 2, figsize=(max(10, len(months) * 0.55), 8), sharex=True
        )
        drawn = []
        for row_index, phase in enumerate(["P", "S"]):
            drawn.append(
                _draw_boxplot(
                    axes[row_index][0],
                    dd_rows,
                    "month",
                    "residual_s",
                    months,
                    "%s %s Signed DD Residuals By Month" % (dataset_label, phase),
                    "DD residual (s)",
                    phase=phase,
                    signed=True,
                    ylim=limits.get("dd_%s_month" % phase.lower()),
                )
            )
            drawn.append(
                _draw_boxplot(
                    axes[row_index][1],
                    pick_rows,
                    "month",
                    "absolute_residual_s",
                    months,
                    "%s %s Absolute Pick Residuals By Month"
                    % (dataset_label, phase),
                    "|pick residual| (s)",
                    phase=phase,
                    ylim=limits.get("pick_%s_month" % phase.lower()),
                )
            )
        if any(drawn):
            save(fig, "%s_residual_boxplots_by_month_by_phase.png" % filename_prefix)
        else:
            plt.close(fig)

    if cluster_ids:
        fig, axes = plt.subplots(
            2, 1, figsize=(max(8, len(cluster_ids) * 0.55), 7)
        )
        drawn = [
            _draw_boxplot(
                axes[0],
                dd_rows,
                "cluster_id",
                "residual_s",
                cluster_ids,
                "%s Signed Double-Difference Residuals By Cluster" % dataset_label,
                "DD residual (s)",
                signed=True,
                ylim=limits.get("dd_cluster"),
            ),
            _draw_boxplot(
                axes[1],
                pick_rows,
                "cluster_id",
                "absolute_residual_s",
                cluster_ids,
                "%s Absolute Pick Residuals By Cluster" % dataset_label,
                "|pick residual| (s)",
                ylim=limits.get("pick_cluster"),
            ),
        ]
        if any(drawn):
            save(fig, "%s_residual_boxplots_by_cluster.png" % filename_prefix)
        else:
            plt.close(fig)

        fig, axes = plt.subplots(
            2, 2, figsize=(max(10, len(cluster_ids) * 0.65), 8), sharex=True
        )
        drawn = []
        for row_index, phase in enumerate(["P", "S"]):
            drawn.append(
                _draw_boxplot(
                    axes[row_index][0],
                    dd_rows,
                    "cluster_id",
                    "residual_s",
                    cluster_ids,
                    "%s %s Signed DD Residuals By Cluster" % (dataset_label, phase),
                    "DD residual (s)",
                    phase=phase,
                    signed=True,
                    ylim=limits.get("dd_%s_cluster" % phase.lower()),
                )
            )
            drawn.append(
                _draw_boxplot(
                    axes[row_index][1],
                    pick_rows,
                    "cluster_id",
                    "absolute_residual_s",
                    cluster_ids,
                    "%s %s Absolute Pick Residuals By Cluster"
                    % (dataset_label, phase),
                    "|pick residual| (s)",
                    phase=phase,
                    ylim=limits.get("pick_%s_cluster" % phase.lower()),
                )
            )
        if any(drawn):
            save(fig, "%s_residual_boxplots_by_cluster_by_phase.png" % filename_prefix)
        else:
            plt.close(fig)

    return plot_paths


def make_residual_boxplots(
    output_dir,
    original_dd_rows,
    original_pick_rows,
    relocated_dd_rows,
    relocated_pick_rows,
    cluster_rows,
    max_clusters_to_plot=12,
):
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        return []

    cluster_ids = _selected_cluster_ids(cluster_rows, max_clusters_to_plot)
    limits = _residual_boxplot_limits(
        original_dd_rows,
        original_pick_rows,
        relocated_dd_rows,
        relocated_pick_rows,
    )
    _add_cluster_limits(
        limits,
        original_dd_rows,
        original_pick_rows,
        relocated_dd_rows,
        relocated_pick_rows,
        cluster_ids,
    )
    plot_paths = []
    plot_paths.extend(
        make_residual_boxplots_for_dataset(
            output_dir,
            original_dd_rows,
            original_pick_rows,
            cluster_ids,
            "Original",
            "original",
            limits,
        )
    )
    plot_paths.extend(
        make_residual_boxplots_for_dataset(
            output_dir,
            relocated_dd_rows,
            relocated_pick_rows,
            cluster_ids,
            "Relocated",
            "relocated",
            limits,
        )
    )
    return plot_paths


def make_plots(output_dir, summaries, residual_rows, convergence_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    def save_current(name):
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))

    raw = summaries.get("_raw", {})

    for original_key, relocated_key, title, filename in [
        (
            "interevent_original_km",
            "interevent_relocated_km",
            "Inter-Event Distances",
            "interevent_original_vs_relocated_km.png",
        ),
        (
            "nearest_original_km",
            "nearest_relocated_km",
            "Nearest-Neighbor Distances",
            "nearest_original_vs_relocated_km.png",
        ),
        (
            "pick_abs_residual_original_s",
            "pick_abs_residual_relocated_s",
            "Absolute Pick Residuals From Travel-Time Model",
            "pick_abs_residual_original_vs_relocated_s.png",
        ),
        (
            "model_dd_abs_residual_original_s",
            "model_dd_abs_residual_relocated_s",
            "Absolute Catalog Double-Difference Residuals From Travel-Time Model",
            "model_dd_abs_residual_original_vs_relocated_s.png",
        ),
        (
            "event_pick_mean_abs_original_s",
            "event_pick_mean_abs_relocated_s",
            "Event-Wise Mean Absolute Pick Residuals",
            "event_pick_mean_abs_original_vs_relocated_s.png",
        ),
        (
            "event_dd_mean_abs_original_s",
            "event_dd_mean_abs_relocated_s",
            "Event-Wise Mean Absolute Double-Difference Residuals",
            "event_dd_mean_abs_original_vs_relocated_s.png",
        ),
    ]:
        original_values = raw.get(original_key, [])
        relocated_values = raw.get(relocated_key, [])
        if not original_values and not relocated_values:
            continue
        plt.figure()
        plt.hist(
            original_values,
            bins=50,
            alpha=0.5,
            label="Original",
            density=False,
        )
        plt.hist(
            relocated_values,
            bins=50,
            alpha=0.5,
            label="Relocated",
            density=False,
        )
        xlabel = "km" if original_key.endswith("_km") else "seconds"
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.title(title)
        plt.legend()
        save_current(filename)

    for key, title, xlabel in [
        ("horizontal_shift_km", "Horizontal Relocation Shifts", "km"),
        ("time_shift_s", "Origin-Time Relocation Shifts", "seconds"),
        ("abs_time_shift_s", "Absolute Origin-Time Relocation Shifts", "seconds"),
    ]:
        values = raw.get(key, [])
        if not values:
            continue
        plt.figure()
        plt.hist(values, bins=50)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.title(title)
        save_current("%s.png" % key)

    mean_pairs = [
        (
            "pick_abs_residual_original_s",
            "pick_abs_residual_relocated_s",
            "Absolute pick residual",
        ),
        (
            "model_dd_abs_residual_original_s",
            "model_dd_abs_residual_relocated_s",
            "Catalog double-difference residual",
        ),
        (
            "event_pick_mean_abs_original_s",
            "event_pick_mean_abs_relocated_s",
            "Event-wise pick residual",
        ),
        (
            "event_dd_mean_abs_original_s",
            "event_dd_mean_abs_relocated_s",
            "Event-wise DD residual",
        ),
    ]
    labels = []
    original_means = []
    relocated_means = []
    for original_key, relocated_key, label in mean_pairs:
        original_values = raw.get(original_key, [])
        relocated_values = raw.get(relocated_key, [])
        if not original_values and not relocated_values:
            continue
        labels.append(label)
        original_means.append(
            sum(original_values) / len(original_values) if original_values else math.nan
        )
        relocated_means.append(
            sum(relocated_values) / len(relocated_values)
            if relocated_values
            else math.nan
        )
    if labels:
        x = list(range(len(labels)))
        width = 0.36
        plt.figure(figsize=(8, 4.5))
        plt.bar([item - width / 2 for item in x], original_means, width, label="Original")
        plt.bar([item + width / 2 for item in x], relocated_means, width, label="Relocated")
        plt.xticks(x, labels, rotation=15, ha="right")
        plt.ylabel("mean absolute residual (s)")
        plt.title("Model-Based Mean Absolute Error")
        plt.legend()
        save_current("model_based_mean_absolute_errors.png")

    if residual_rows:
        plt.figure()
        plt.hist([row["residual_s"] for row in residual_rows], bins=80)
        plt.xlabel("residual (s)")
        plt.ylabel("count")
        plt.title("Final Double-Difference Residuals")
        save_current("double_difference_residuals.png")

    if convergence_rows:
        plt.figure()
        def plot_segmented_series(value_key, label, color=None):
            blocks = defaultdict(list)
            for row in convergence_rows:
                x_value = row.get("global_iteration")
                y_value = row.get(value_key, math.nan)
                if x_value is None or not math.isfinite(y_value):
                    continue
                blocks[row.get("convergence_block", 1)].append((x_value, y_value))
            if not blocks:
                return False
            label_used = False
            for _, points in sorted(blocks.items()):
                if not points:
                    continue
                x_values = [point[0] for point in points]
                y_values = [point[1] for point in points]
                plt.plot(
                    x_values,
                    y_values,
                    marker="o",
                    markersize=3,
                    linewidth=1.0 if len(points) > 1 else 0.0,
                    alpha=0.85,
                    color=color,
                    label=label if not label_used else None,
                )
                label_used = True
            return True

        plot_segmented_series(
            "initial_weighted_ct_rms_s",
            "Initial RMSCT",
            color="tab:orange",
        )
        has_post_ct = plot_segmented_series(
            "post_weighted_ct_rms_s",
            "Post-solve RMSCT",
            color="tab:red",
        )
        if not has_post_ct:
            plot_segmented_series("rms_ct_s", "RMSCT", color="tab:red")

        plot_segmented_series(
            "initial_weighted_cc_rms_s",
            "Initial RMSCC",
            color="tab:blue",
        )
        has_post_cc = plot_segmented_series(
            "post_weighted_cc_rms_s",
            "Post-solve RMSCC",
            color="tab:green",
        )
        if not has_post_cc:
            plot_segmented_series("rms_cc_s", "RMSCC", color="tab:green")
        plt.xlabel("iteration")
        plt.ylabel("weighted RMS (s)")
        plt.title("HypoDD Residual Convergence")
        plt.legend()
        save_current("convergence_rms.png")

    return plot_paths


def make_ukraine_cartopy_plot(output_dir, original, relocated, stations):
    """
    Plot original and relocated epicenters on a Ukraine map with Cartopy.

    The function is optional: if Cartopy or Natural Earth data are unavailable,
    it returns an empty list instead of failing the report.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ukraine_original_vs_relocated_map.png"
    relocation_path = output_dir / "ukraine_relocation_vectors_map.png"

    cities = [
        ("Kyiv", 50.4501, 30.5234),
        ("Korosten", 50.95937, 28.63855),
        ("Zhytomyr", 50.26235, 28.67913),
        ("Malyn", 50.77233, 29.23833),
        ("Chornobyl", 51.26667, 30.21667),
        ("Fastiv", 50.07670, 29.91770),
        ("Kharkiv", 49.9935, 36.2304),
        ("Odesa", 46.4825, 30.7233),
        ("Dnipro", 48.4647, 35.0462),
        ("Donetsk", 48.0159, 37.8028),
        ("Lviv", 49.8397, 24.0297),
        ("Zaporizhzhia", 47.8388, 35.1396),
        ("Kryvyi Rih", 47.9105, 33.3918),
        ("Mykolaiv", 46.9750, 31.9946),
        ("Mariupol", 47.0971, 37.5434),
        ("Luhansk", 48.5740, 39.3078),
        ("Vinnytsia", 49.2331, 28.4682),
        ("Chernihiv", 51.4982, 31.2893),
        ("Poltava", 49.5883, 34.5514),
        ("Sumy", 50.9077, 34.7981),
        ("Kherson", 46.6354, 32.6169),
    ]

    datasets = [
        ("Original Events", original),
        ("Relocated Events", relocated),
    ]

    cluster_ids = sorted(
        {
            event.get("cluster_id")
            for event in list(original.values()) + list(relocated.values())
            if event.get("cluster_id") not in ("", None)
        }
    )
    if not cluster_ids:
        cluster_ids = sorted(
            {
                relocated[event_id].get("cluster_id")
                for event_id in set(original).intersection(relocated)
                if relocated[event_id].get("cluster_id") not in ("", None)
            }
        )
    cmap = plt.get_cmap("tab20")
    cluster_colors = {
        cluster_id: cmap(index % cmap.N)
        for index, cluster_id in enumerate(cluster_ids)
    }
    unknown_cluster_color = "0.55"

    def event_cluster(event_id, event):
        cluster_id = event.get("cluster_id")
        if cluster_id in ("", None) and event_id in relocated:
            cluster_id = relocated[event_id].get("cluster_id")
        return cluster_id

    def event_color(event_id, event):
        cluster_id = event_cluster(event_id, event)
        return cluster_colors.get(cluster_id, unknown_cluster_color)

    all_lats = (
        [event["latitude"] for event in original.values()]
        + [event["latitude"] for event in relocated.values()]
        + [station["latitude"] for station in stations.values()]
    )
    all_lons = (
        [event["longitude"] for event in original.values()]
        + [event["longitude"] for event in relocated.values()]
        + [station["longitude"] for station in stations.values()]
    )
    if not all_lats or not all_lons:
        return []
    lat_pad = max(0.25, (max(all_lats) - min(all_lats)) * 0.12)
    lon_pad = max(0.25, (max(all_lons) - min(all_lons)) * 0.12)
    extent = [
        min(all_lons) - lon_pad,
        max(all_lons) + lon_pad,
        min(all_lats) - lat_pad,
        max(all_lats) + lat_pad,
    ]

    def add_base_map(ax, projection):
        ax.set_extent(extent, crs=projection)
        ax.add_feature(cfeature.LAND, facecolor="0.96")
        ax.add_feature(cfeature.OCEAN, facecolor="0.90")
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor="0.45")
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "cultural",
                "admin_0_countries",
                "10m",
                facecolor="none",
                edgecolor="black",
            ),
            linewidth=0.8,
        )

    def add_stations(ax, projection):
        ax.scatter(
            [station["longitude"] for station in stations.values()],
            [station["latitude"] for station in stations.values()],
            s=34,
            marker="^",
            c="black",
            edgecolors="white",
            linewidths=0.4,
            transform=projection,
            label="Stations",
            zorder=4,
        )

    def add_cities(ax, projection):
        for city, lat, lon in cities:
            if not (
                extent[0] <= lon <= extent[1]
                and extent[2] <= lat <= extent[3]
            ):
                continue
            ax.plot(lon, lat, marker="o", markersize=2.5, color="black")
            ax.text(
                lon + 0.12,
                lat + 0.08,
                city,
                fontsize=7,
                transform=projection,
            )

    def add_grid(ax):
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            color="0.5",
            alpha=0.5,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False

    def add_cluster_legend(ax, include_markers=True):
        handles = []
        if include_markers:
            handles.extend(
                [
                    Line2D(
                        [0],
                        [0],
                        marker="x",
                        color="0.35",
                        linestyle="none",
                        markersize=5,
                        label="Original event",
                    ),
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="none",
                        markerfacecolor="0.35",
                        markeredgecolor="0.35",
                        markersize=5,
                        label="Relocated event",
                    ),
                ]
            )
        for cluster_id in cluster_ids[:12]:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=cluster_colors[cluster_id],
                    markeredgecolor=cluster_colors[cluster_id],
                    markersize=5,
                    label="Cluster %s" % cluster_id,
                )
            )
        if len(cluster_ids) > 12:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="none",
                    label="+%i more clusters" % (len(cluster_ids) - 12),
                )
            )
        if handles:
            ax.legend(handles=handles, loc="lower left", fontsize=7)

    def scatter_events(ax, events, projection, marker="o", label=None, zorder=3):
        event_items = sorted(events.items())
        if not event_items:
            return
        ax.scatter(
            [event["longitude"] for _, event in event_items],
            [event["latitude"] for _, event in event_items],
            s=10,
            c=[event_color(event_id, event) for event_id, event in event_items],
            alpha=0.65,
            marker=marker,
            linewidths=0.4 if marker == "x" else 0,
            transform=projection,
            label=label,
            zorder=zorder,
        )

    plot_paths = []

    try:
        projection = ccrs.PlateCarree()
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 7),
            subplot_kw={"projection": projection},
        )
        for ax, (title, events) in zip(axes, datasets):
            add_base_map(ax, projection)
            marker = "x" if title.startswith("Original") else "o"
            scatter_events(ax, events, projection, marker=marker, label=title)
            add_stations(ax, projection)
            add_cities(ax, projection)
            add_grid(ax)
            ax.set_title(
                "%s (%i events, %i clusters)"
                % (title, len(events), len(cluster_ids))
            )
            add_cluster_legend(ax, include_markers=False)

        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(path))

        common_event_ids = sorted(set(original).intersection(relocated))
        fig = plt.figure(figsize=(9, 8))
        ax = plt.axes(projection=projection)
        add_base_map(ax, projection)
        for event_id in common_event_ids:
            original_event = original[event_id]
            relocated_event = relocated[event_id]
            color = event_color(event_id, relocated_event)
            ax.plot(
                [original_event["longitude"], relocated_event["longitude"]],
                [original_event["latitude"], relocated_event["latitude"]],
                color=color,
                linewidth=0.35,
                alpha=0.45,
                transform=projection,
                zorder=2,
            )
        scatter_events(
            ax,
            {event_id: original[event_id] for event_id in common_event_ids},
            projection,
            marker="x",
            label="Original Events",
            zorder=3,
        )
        scatter_events(
            ax,
            {event_id: relocated[event_id] for event_id in common_event_ids},
            projection,
            marker="o",
            label="Relocated Events",
            zorder=4,
        )
        add_stations(ax, projection)
        add_cities(ax, projection)
        add_grid(ax)
        ax.set_title(
            "Event Relocation Vectors (%i matched events, %i clusters)"
            % (len(common_event_ids), len(cluster_ids))
        )
        add_cluster_legend(ax, include_markers=True)
        plt.tight_layout()
        plt.savefig(relocation_path, dpi=180)
        plt.close(fig)
        plot_paths.append(str(relocation_path))
    except Exception:
        plt.close("all")
        return []

    return plot_paths


def create_quality_report(
    working_dir,
    velocity_model_csv,
    output_dir=None,
    create_plots=True,
    max_clusters_to_plot=12,
):
    working_dir = Path(working_dir)
    output_dir = Path(output_dir) if output_dir else working_dir / "quality_report"
    input_dir = working_dir / "input_files"
    output_files = working_dir / "output_files"
    if not (output_files / "hypoDD.loc").exists():
        output_files = working_dir
    if not (input_dir / "phase.dat").exists():
        input_dir = working_dir

    original = read_hypodd_locations(output_files / "hypoDD.loc")
    relocated = read_hypodd_locations(output_files / "hypoDD.reloc")
    phase_path = input_dir / "phase.dat"
    station_path = input_dir / "station.sel"
    if not station_path.exists():
        station_path = output_files / "hypoDD.sta"
    dt_ct_path = input_dir / "dt.ct"
    if not dt_ct_path.exists():
        dt_ct_path = output_files / "dt.ct"
    phase_events = read_phase_dat(phase_path) if phase_path.exists() else {}
    stations = read_station_dat(station_path)
    velocity_model = read_velocity_model(velocity_model_csv)
    differential_rows = (
        read_catalog_differential_times(dt_ct_path) if dt_ct_path.exists() else []
    )

    residual_rows = read_hypodd_residuals(output_files / "hypoDD.res")
    station_rows = read_station_residuals(output_files / "hypoDD.sta")
    initial_diag_path = output_files / "hypoDD.initial.res"
    final_diag_path = output_files / "hypoDD.final.res"
    initial_tt_path = output_files / "hypoDD.initial.tt"
    final_tt_path = output_files / "hypoDD.final.tt"
    log_path = working_dir / "hypoDD_log.txt"
    if not log_path.exists():
        log_path = output_files / "hypoDD.log"
    convergence_rows = parse_hypodd_log(log_path) if log_path.exists() else []
    shifts = location_shift_rows(original, relocated)

    pick_original = []
    pick_relocated = []
    if phase_events:
        if initial_tt_path.exists() and final_tt_path.exists():
            pick_original = pick_residual_rows_from_hypodd_travel_times(
                phase_events,
                original,
                read_hypodd_travel_times(initial_tt_path),
                "original",
            )
            pick_relocated = pick_residual_rows_from_hypodd_travel_times(
                phase_events,
                relocated,
                read_hypodd_travel_times(final_tt_path),
                "relocated",
            )
        else:
            pick_original = pick_residual_rows(
                phase_events, original, stations, velocity_model, "original"
            )
            pick_relocated = pick_residual_rows(
                phase_events, relocated, stations, velocity_model, "relocated"
            )
    pick_rows = pick_original + pick_relocated
    model_dd_original = []
    model_dd_relocated = []
    if initial_diag_path.exists() and final_diag_path.exists():
        model_dd_original = read_hypodd_diagnostic_residuals(
            initial_diag_path, "original"
        )
        model_dd_relocated = read_hypodd_diagnostic_residuals(
            final_diag_path, "relocated"
        )
    elif phase_events and differential_rows:
        model_dd_original = model_double_difference_residual_rows(
            differential_rows,
            phase_events,
            original,
            stations,
            velocity_model,
            "original",
        )
        model_dd_relocated = model_double_difference_residual_rows(
            differential_rows,
            phase_events,
            relocated,
            stations,
            velocity_model,
            "relocated",
        )
    model_dd_rows = model_dd_original + model_dd_relocated
    comparison_dd_original, comparison_dd_relocated = (
        paired_double_difference_rows(model_dd_original, model_dd_relocated)
    )
    comparison_dd_rows = comparison_dd_original + comparison_dd_relocated
    event_pick_rows = event_pick_residual_summary_rows(pick_rows)
    event_dd_rows = event_double_difference_summary_rows(comparison_dd_rows)
    event_residual_rows = combined_event_residual_summary_rows(
        event_pick_rows,
        event_dd_rows,
    )

    raw = {
        "interevent_original_km": inter_event_distances(original),
        "interevent_relocated_km": inter_event_distances(relocated),
        "nearest_original_km": nearest_neighbor_distances(original),
        "nearest_relocated_km": nearest_neighbor_distances(relocated),
        "horizontal_shift_km": [row["horizontal_shift_km"] for row in shifts],
        "time_shift_s": [row["time_shift_s"] for row in shifts],
        "abs_time_shift_s": [abs(row["time_shift_s"]) for row in shifts],
        "pick_abs_residual_original_s": [
            row["absolute_residual_s"] for row in pick_original
        ],
        "pick_abs_residual_relocated_s": [
            row["absolute_residual_s"] for row in pick_relocated
        ],
        "model_dd_abs_residual_original_s": [
            row["absolute_residual_s"] for row in comparison_dd_original
        ],
        "model_dd_abs_residual_relocated_s": [
            row["absolute_residual_s"] for row in comparison_dd_relocated
        ],
        "event_pick_mean_abs_original_s": [
            row["original_pick_mean_abs_s"]
            for row in event_pick_rows
            if math.isfinite(row["original_pick_mean_abs_s"])
        ],
        "event_pick_mean_abs_relocated_s": [
            row["relocated_pick_mean_abs_s"]
            for row in event_pick_rows
            if math.isfinite(row["relocated_pick_mean_abs_s"])
        ],
        "event_dd_mean_abs_original_s": [
            row["original_dd_mean_abs_s"]
            for row in event_dd_rows
            if math.isfinite(row["original_dd_mean_abs_s"])
        ],
        "event_dd_mean_abs_relocated_s": [
            row["relocated_dd_mean_abs_s"]
            for row in event_dd_rows
            if math.isfinite(row["relocated_dd_mean_abs_s"])
        ],
    }

    summary_rows = []
    summary_items = {
        "interevent_original_km": raw["interevent_original_km"],
        "interevent_relocated_km": raw["interevent_relocated_km"],
        "nearest_original_km": raw["nearest_original_km"],
        "nearest_relocated_km": raw["nearest_relocated_km"],
        "horizontal_shift_km": raw["horizontal_shift_km"],
        "time_shift_s": raw["time_shift_s"],
        "abs_time_shift_s": raw["abs_time_shift_s"],
        "dd_residual_s": [row["residual_s"] for row in residual_rows],
        "dd_abs_residual_s": [abs(row["residual_s"]) for row in residual_rows],
        "pick_abs_residual_original_s": raw["pick_abs_residual_original_s"],
        "pick_abs_residual_relocated_s": raw["pick_abs_residual_relocated_s"],
        "model_dd_abs_residual_original_s": raw[
            "model_dd_abs_residual_original_s"
        ],
        "model_dd_abs_residual_relocated_s": raw[
            "model_dd_abs_residual_relocated_s"
        ],
        "model_dd_original_all_count": [float(len(model_dd_original))],
        "model_dd_relocated_all_count": [float(len(model_dd_relocated))],
        "model_dd_common_count": [float(len(comparison_dd_original))],
    }
    for metric, values in summary_items.items():
        row = {"metric": metric}
        row.update(_summary(values))
        summary_rows.append(row)

    cluster_counts = Counter(
        row.get("cluster_id") for row in relocated.values() if row.get("cluster_id") != ""
    )
    cluster_rows = [
        {"cluster_id": cluster_id, "event_count": count}
        for cluster_id, count in sorted(cluster_counts.items())
    ]
    original_for_cluster_plots = {
        event_id: dict(event) for event_id, event in original.items()
    }
    for event_id, event in original_for_cluster_plots.items():
        if event_id in relocated:
            event["cluster_id"] = relocated[event_id].get("cluster_id", "")

    pick_plot_original = add_pick_plot_metadata(
        pick_original, original_for_cluster_plots
    )
    dd_plot_original = add_dd_plot_metadata(
        model_dd_original, original_for_cluster_plots
    )
    pick_plot_relocated = add_pick_plot_metadata(pick_relocated, relocated)
    dd_plot_relocated = add_dd_plot_metadata(model_dd_relocated, relocated)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "summary_metrics.csv", summary_rows)
    _write_rows(output_dir / "location_shifts.csv", shifts)
    _write_rows(output_dir / "pick_residuals.csv", pick_rows)
    _write_rows(output_dir / "double_difference_residuals.csv", residual_rows)
    _write_rows(
        output_dir / "model_double_difference_residuals.csv", model_dd_rows
    )
    _write_rows(
        output_dir / "model_double_difference_residuals_common.csv",
        comparison_dd_rows,
    )
    _write_rows(output_dir / "event_pick_residual_summary.csv", event_pick_rows)
    _write_rows(
        output_dir / "event_double_difference_residual_summary.csv",
        event_dd_rows,
    )
    _write_rows(output_dir / "event_residual_summary.csv", event_residual_rows)
    _write_rows(output_dir / "station_residuals.csv", station_rows)
    _write_rows(output_dir / "cluster_sizes.csv", cluster_rows)
    _write_rows(output_dir / "iteration_convergence.csv", convergence_rows)
    _write_rows(output_dir / "plot_pick_residuals_original.csv", pick_plot_original)
    _write_rows(output_dir / "plot_dd_residuals_original.csv", dd_plot_original)
    _write_rows(output_dir / "plot_pick_residuals_relocated.csv", pick_plot_relocated)
    _write_rows(output_dir / "plot_dd_residuals_relocated.csv", dd_plot_relocated)

    plot_paths = []
    if create_plots:
        plot_paths = make_plots(
            output_dir,
            {"_raw": raw},
            residual_rows,
            convergence_rows,
        )
        plot_paths.extend(
            make_residual_boxplots(
                output_dir,
                dd_plot_original,
                pick_plot_original,
                dd_plot_relocated,
                pick_plot_relocated,
                cluster_rows,
                max_clusters_to_plot=max_clusters_to_plot,
            )
        )
        plot_paths.extend(
            make_ukraine_cartopy_plot(output_dir, original, relocated, stations)
        )

    return {
        "summary": summary_rows,
        "clusters": cluster_rows,
        "convergence": convergence_rows,
        "output_dir": str(output_dir),
        "plots": plot_paths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("working_dir")
    parser.add_argument("velocity_model_csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--max-clusters-to-plot",
        type=int,
        default=12,
        help="Maximum number of largest clusters to show in cluster boxplots.",
    )
    args = parser.parse_args()

    result = create_quality_report(
        working_dir=args.working_dir,
        velocity_model_csv=args.velocity_model_csv,
        output_dir=args.output_dir,
        create_plots=not args.no_plots,
        max_clusters_to_plot=args.max_clusters_to_plot,
    )
    print("Wrote quality report to %s" % result["output_dir"])


if __name__ == "__main__":
    main()
