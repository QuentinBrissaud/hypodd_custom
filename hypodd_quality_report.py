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
from collections import Counter, defaultdict
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


def _write_rows(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
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


def read_hypodd_residuals(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append(
                {
                    "station_id": parts[0],
                    "residual_s": float(parts[1]),
                    "event_id_1": int(parts[2]),
                    "event_id_2": int(parts[3]),
                    "type_index": int(parts[4]),
                    "weight": float(parts[5]),
                    "offset_km": float(parts[7]),
                    "phase": "P" if int(parts[4]) in [1, 3] else "S",
                    "data_type": "cc" if int(parts[4]) in [1, 2] else "ct",
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
    rmscc = math.nan
    rmsct = math.nan
    with open(path, "r", encoding="utf-8", errors="replace") as open_file:
        for line in open_file:
            match = re.search(r"===ITERATION\s+(\d+)\s+\(\s*(\d+)\)", line)
            if match:
                current = {
                    "iteration": int(match.group(1)),
                    "global_iteration": int(match.group(2)),
                }
                rmscc = math.nan
                rmsct = math.nan
                continue
            if current is None:
                continue
            match = re.search(r"weighted cc rms \[s\] \(RMSCC\) =\s*([0-9.Ee+-]+)", line)
            if match:
                rmscc = float(match.group(1))
                continue
            match = re.search(r"weighted ct rms \[s\] \(RMSCT\) =\s*([0-9.Ee+-]+)", line)
            if match:
                rmsct = float(match.group(1))
                continue
            parts = line.split()
            if len(parts) >= 10 and parts[0].isdigit():
                row = dict(current)
                row.update({"rms_cc_s": rmscc, "rms_ct_s": rmsct})
                if len(parts) >= 15:
                    row.update(
                        {
                            "events": int(parts[1]),
                            "ct_percent": _hypodd_log_float(parts[2]),
                            "cc_percent": _hypodd_log_float(parts[3]),
                            "rms_ct_percent": _hypodd_log_float(parts[4]),
                            "rms_ct_ms": _hypodd_log_float(parts[5]),
                            "rms_cc_percent": _hypodd_log_float(parts[6]),
                            "rms_cc_ms": _hypodd_log_float(parts[7]),
                            "rms_station_percent": _hypodd_log_float(parts[8]),
                            "rms_station_ms": _hypodd_log_float(parts[9]),
                            "mean_abs_dx_m": _hypodd_log_float(parts[10]),
                            "mean_abs_dy_m": _hypodd_log_float(parts[11]),
                            "mean_abs_dz_m": _hypodd_log_float(parts[12]),
                            "mean_abs_dt_ms": _hypodd_log_float(parts[13]),
                            "origin_shift_m": _hypodd_log_float(parts[14]),
                            "airquake_count": _hypodd_log_int(parts[15]),
                        }
                    )
                    if len(parts) > 16:
                        row["condition_number"] = _hypodd_log_float(parts[16])
                elif len(parts) >= 13:
                    row.update(
                        {
                            "events": int(parts[1]),
                            "ct_percent": _hypodd_log_float(parts[2]),
                            "rms_ct_percent": _hypodd_log_float(parts[3]),
                            "rms_ct_ms": _hypodd_log_float(parts[4]),
                            "rms_station_percent": _hypodd_log_float(parts[5]),
                            "rms_station_ms": _hypodd_log_float(parts[6]),
                            "mean_abs_dx_m": _hypodd_log_float(parts[7]),
                            "mean_abs_dy_m": _hypodd_log_float(parts[8]),
                            "mean_abs_dz_m": _hypodd_log_float(parts[9]),
                            "mean_abs_dt_ms": _hypodd_log_float(parts[10]),
                            "origin_shift_m": _hypodd_log_float(parts[11]),
                            "airquake_count": _hypodd_log_int(parts[12]),
                        }
                    )
                    if len(parts) > 13:
                        row["condition_number"] = _hypodd_log_float(parts[13])
                rows.append(row)
                current = None
    return rows


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

    for key, title in [
        ("interevent_original_km", "Inter-Event Distances: Original"),
        ("interevent_relocated_km", "Inter-Event Distances: Relocated"),
        ("nearest_original_km", "Nearest-Neighbor Distances: Original"),
        ("nearest_relocated_km", "Nearest-Neighbor Distances: Relocated"),
        ("horizontal_shift_km", "Horizontal Relocation Shifts"),
    ]:
        values = summaries.get("_raw", {}).get(key, [])
        if not values:
            continue
        plt.figure()
        plt.hist(values, bins=50)
        plt.xlabel("km")
        plt.ylabel("count")
        plt.title(title)
        save_current("%s.png" % key)

    if residual_rows:
        plt.figure()
        plt.hist([row["residual_s"] for row in residual_rows], bins=80)
        plt.xlabel("residual (s)")
        plt.ylabel("count")
        plt.title("Final Double-Difference Residuals")
        save_current("double_difference_residuals.png")

    if convergence_rows:
        plt.figure()
        x = [row["global_iteration"] for row in convergence_rows]
        if any(math.isfinite(row.get("rms_ct_s", math.nan)) for row in convergence_rows):
            plt.plot(x, [row.get("rms_ct_s", math.nan) for row in convergence_rows], label="RMSCT")
        if any(math.isfinite(row.get("rms_cc_s", math.nan)) for row in convergence_rows):
            plt.plot(x, [row.get("rms_cc_s", math.nan) for row in convergence_rows], label="RMSCC")
        plt.xlabel("iteration")
        plt.ylabel("weighted RMS (s)")
        plt.title("HypoDD Residual Convergence")
        plt.legend()
        save_current("convergence_rms.png")

    return plot_paths


def create_quality_report(
    working_dir,
    velocity_model_csv,
    output_dir=None,
    create_plots=True,
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
    phase_events = read_phase_dat(phase_path) if phase_path.exists() else {}
    stations = read_station_dat(station_path)
    velocity_model = read_velocity_model(velocity_model_csv)

    residual_rows = read_hypodd_residuals(output_files / "hypoDD.res")
    station_rows = read_station_residuals(output_files / "hypoDD.sta")
    log_path = working_dir / "hypoDD_log.txt"
    if not log_path.exists():
        log_path = output_files / "hypoDD.log"
    convergence_rows = parse_hypodd_log(log_path) if log_path.exists() else []
    shifts = location_shift_rows(original, relocated)

    pick_original = []
    pick_relocated = []
    if phase_events:
        pick_original = pick_residual_rows(
            phase_events, original, stations, velocity_model, "original"
        )
        pick_relocated = pick_residual_rows(
            phase_events, relocated, stations, velocity_model, "relocated"
        )
    pick_rows = pick_original + pick_relocated

    raw = {
        "interevent_original_km": inter_event_distances(original),
        "interevent_relocated_km": inter_event_distances(relocated),
        "nearest_original_km": nearest_neighbor_distances(original),
        "nearest_relocated_km": nearest_neighbor_distances(relocated),
        "horizontal_shift_km": [row["horizontal_shift_km"] for row in shifts],
    }

    summary_rows = []
    summary_items = {
        "interevent_original_km": raw["interevent_original_km"],
        "interevent_relocated_km": raw["interevent_relocated_km"],
        "nearest_original_km": raw["nearest_original_km"],
        "nearest_relocated_km": raw["nearest_relocated_km"],
        "horizontal_shift_km": raw["horizontal_shift_km"],
        "dd_residual_s": [row["residual_s"] for row in residual_rows],
        "dd_abs_residual_s": [abs(row["residual_s"]) for row in residual_rows],
        "pick_abs_residual_original_s": [
            row["absolute_residual_s"] for row in pick_original
        ],
        "pick_abs_residual_relocated_s": [
            row["absolute_residual_s"] for row in pick_relocated
        ],
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

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "summary_metrics.csv", summary_rows)
    _write_rows(output_dir / "location_shifts.csv", shifts)
    _write_rows(output_dir / "pick_residuals.csv", pick_rows)
    _write_rows(output_dir / "double_difference_residuals.csv", residual_rows)
    _write_rows(output_dir / "station_residuals.csv", station_rows)
    _write_rows(output_dir / "cluster_sizes.csv", cluster_rows)
    _write_rows(output_dir / "iteration_convergence.csv", convergence_rows)

    plot_paths = []
    if create_plots:
        plot_paths = make_plots(
            output_dir,
            {"_raw": raw},
            residual_rows,
            convergence_rows,
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
    args = parser.parse_args()

    result = create_quality_report(
        working_dir=args.working_dir,
        velocity_model_csv=args.velocity_model_csv,
        output_dir=args.output_dir,
        create_plots=not args.no_plots,
    )
    print("Wrote quality report to %s" % result["output_dir"])


if __name__ == "__main__":
    main()
