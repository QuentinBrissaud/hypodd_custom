#!/usr/bin/env python
"""
Residual-bootstrap uncertainty estimates for completed HypoDDPy runs.

This implements the Waldhauser & Ellsworth style residual bootstrap:

    synthetic_observed_DD_i = predicted_DD_i + sampled_final_residual

where final residuals are sampled with replacement from the observed residual
distribution. The event-pair/station/phase geometry is kept fixed.

Typical use:

    from hypodd_bootstrap import run_residual_bootstrap

    result = run_residual_bootstrap(
        working_dir="relocator_working_dir",
        output_dir="relocator_working_dir/bootstrap",
        n_trials=200,
        random_seed=1234,
    )
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import json
import math
import os
import random
import shutil
import subprocess


OUTPUT_FILES = [
    "hypoDD.loc",
    "hypoDD.reloc",
    "hypoDD.sta",
    "hypoDD.res",
    "hypoDD.src",
    "hypoDD.initial.res",
    "hypoDD.final.res",
    "hypoDD.initial.tt",
    "hypoDD.final.tt",
]


def _float(value):
    try:
        return float(value)
    except ValueError:
        return math.nan


def _format_float(value, precision=6):
    text = ("%%.%df" % precision) % float(value)
    text = text.rstrip("0").rstrip(".")
    if text in ("-0", ""):
        text = "0"
    return text


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


def _local_offsets_km(reference, location):
    mean_lat = math.radians((reference["latitude"] + location["latitude"]) / 2.0)
    dx = (
        (location["longitude"] - reference["longitude"])
        * 111.32
        * math.cos(mean_lat)
    )
    dy = (location["latitude"] - reference["latitude"]) * 111.32
    return dx, dy


def _observed_dd(row):
    if row["file_type"] == "ct":
        return row["time_1_s"] - row["time_2_s"]
    return row["dt_s"]


def read_dt_file(path, file_type=None):
    """
    Read a HypoDD dt.ct or dt.cc file while preserving event-pair blocks.
    """
    path = Path(path)
    if file_type is None:
        file_type = "cc" if path.name.endswith(".cc") else "ct"
    blocks = []
    current = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "#":
                if len(parts) < 3:
                    continue
                current = {
                    "header": line.rstrip("\n"),
                    "event_id_1": int(parts[1]),
                    "event_id_2": int(parts[2]),
                    "rows": [],
                }
                blocks.append(current)
                continue
            if current is None:
                continue
            if file_type == "ct":
                if len(parts) < 5:
                    continue
                row = {
                    "line_number": line_number,
                    "file_type": "ct",
                    "event_id_1": current["event_id_1"],
                    "event_id_2": current["event_id_2"],
                    "station_id": parts[0],
                    "time_1_s": float(parts[1]),
                    "time_2_s": float(parts[2]),
                    "weight": float(parts[3]),
                    "phase": parts[4].upper(),
                    "parts": parts,
                }
            else:
                if len(parts) < 4:
                    continue
                row = {
                    "line_number": line_number,
                    "file_type": "cc",
                    "event_id_1": current["event_id_1"],
                    "event_id_2": current["event_id_2"],
                    "station_id": parts[0],
                    "dt_s": float(parts[1]),
                    "weight": float(parts[2]),
                    "phase": parts[3].upper(),
                    "parts": parts,
                }
            row["observed_dd_s"] = _observed_dd(row)
            current["rows"].append(row)
    return blocks


def write_dt_file(blocks, path):
    """
    Write dt.ct/dt.cc blocks. Rows with ``bootstrap_observed_dd_s`` are replaced.
    """
    path = Path(path)
    lines = []
    for block in blocks:
        rows = block.get("rows", [])
        if not rows:
            continue
        lines.append(block["header"])
        for row in rows:
            synthetic_dd = row.get("bootstrap_observed_dd_s")
            if row["file_type"] == "ct":
                time_2 = row["time_2_s"]
                time_1 = (
                    time_2 + synthetic_dd
                    if synthetic_dd is not None
                    else row["time_1_s"]
                )
                lines.append(
                    "%s %s %s %s %s"
                    % (
                        row["station_id"],
                        _format_float(time_1),
                        _format_float(time_2),
                        _format_float(row["weight"], precision=4),
                        row["phase"],
                    )
                )
            else:
                dt = synthetic_dd if synthetic_dd is not None else row["dt_s"]
                lines.append(
                    "%s %s %s %s"
                    % (
                        row["station_id"],
                        _format_float(dt),
                        _format_float(row["weight"], precision=4),
                        row["phase"],
                    )
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_hypodd_final_residuals(path):
    """
    Read patched ``hypoDD.final.res`` rows.

    Expected columns:

        STA OBS_S CALC_S RES_S C1 C2 IDX QUAL WT OFFS
    """
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) < 10:
                continue
            rows.append(
                {
                    "line_number": line_number,
                    "station_id": parts[0],
                    "observed_dd_s": float(parts[1]),
                    "calculated_dd_s": float(parts[2]),
                    "residual_s": float(parts[3]),
                    "event_id_1": int(float(parts[4])),
                    "event_id_2": int(float(parts[5])),
                    "idx": int(float(parts[6])),
                    "quality": _float(parts[7]),
                    "weight": _float(parts[8]),
                    "offset_m": _float(parts[9]),
                }
            )
    return rows


def _residual_index(residual_rows, ndigits=5):
    index = defaultdict(list)
    for row in residual_rows:
        key = (
            row["event_id_1"],
            row["event_id_2"],
            row["station_id"],
            round(row["observed_dd_s"], ndigits),
        )
        index[key].append(row)
    return index


def attach_final_residuals(blocks_by_type, residual_rows):
    """
    Match final residual rows to dt rows by event pair, station, and observed DD.

    The residual diagnostic file does not reliably store the original dt-file
    phase label, so the phase is taken from the matched dt row.
    """
    index = _residual_index(residual_rows)
    matched = 0
    unmatched = 0
    for file_type, blocks in blocks_by_type.items():
        for block in blocks:
            for row in block["rows"]:
                key = (
                    row["event_id_1"],
                    row["event_id_2"],
                    row["station_id"],
                    round(row["observed_dd_s"], 5),
                )
                candidates = index.get(key, [])
                if not candidates:
                    unmatched += 1
                    continue
                residual = candidates.pop(0)
                row["calculated_dd_s"] = residual["calculated_dd_s"]
                row["final_residual_s"] = residual["residual_s"]
                row["residual_pool"] = (file_type, row["phase"])
                matched += 1
    return {"matched": matched, "unmatched": unmatched}


def residual_pools(blocks_by_type):
    pools = defaultdict(list)
    for file_type, blocks in blocks_by_type.items():
        for block in blocks:
            for row in block["rows"]:
                if "final_residual_s" not in row:
                    continue
                pools[(file_type, row["phase"])].append(row["final_residual_s"])
    return dict(pools)


def create_bootstrap_blocks(blocks, pools, rng, keep_unmatched=True):
    """
    Deep-ish copy dt blocks and replace matched rows with bootstrap DD values.
    """
    new_blocks = []
    for block in blocks:
        new_block = {
            "header": block["header"],
            "event_id_1": block["event_id_1"],
            "event_id_2": block["event_id_2"],
            "rows": [],
        }
        for row in block["rows"]:
            new_row = dict(row)
            pool_key = row.get("residual_pool")
            pool = pools.get(pool_key, [])
            if "calculated_dd_s" in row and pool:
                sampled_residual = rng.choice(pool)
                new_row["bootstrap_sampled_residual_s"] = sampled_residual
                new_row["bootstrap_observed_dd_s"] = (
                    row["calculated_dd_s"] + sampled_residual
                )
            elif not keep_unmatched:
                continue
            new_block["rows"].append(new_row)
        if new_block["rows"]:
            new_blocks.append(new_block)
    return new_blocks


def read_hypodd_locations(path):
    """
    Read hypoDD.loc/hypoDD.reloc rows into a dict keyed by internal event id.
    """
    rows = {}
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 17:
                continue
            event_id = int(float(parts[0]))
            row = {
                "event_id": event_id,
                "latitude": float(parts[1]),
                "longitude": float(parts[2]),
                "depth_km": float(parts[3]),
                "time": _hypodd_time(parts, 10),
                "cluster_id": int(float(parts[23])) if len(parts) > 23 else None,
            }
            if len(parts) > 7:
                row["raw_col_4"] = _float(parts[4])
                row["raw_col_5"] = _float(parts[5])
                row["raw_col_6"] = _float(parts[6])
                row["raw_col_7"] = _float(parts[7])
            rows[event_id] = row
    return rows


def _hypodd_executable(working_dir):
    candidates = [
        Path(working_dir) / "bin" / "hypoDD",
        Path(working_dir) / "bin" / "hypoDD.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find compiled hypoDD binary in %s/bin" % working_dir)


def _copy_if_exists(source, destination):
    source = Path(source)
    if source.exists():
        shutil.copyfile(source, destination)
        return True
    return False


def _resolve_input_dir(working_dir):
    working_dir = Path(working_dir)
    candidates = [working_dir / "input_files", working_dir]
    for candidate in candidates:
        if (candidate / "dt.ct").exists() and (candidate / "hypoDD.inp").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find dt.ct and hypoDD.inp in %s/input_files or %s"
        % (working_dir, working_dir)
    )


def _resolve_output_dir(working_dir):
    working_dir = Path(working_dir)
    candidates = [working_dir / "output_files", working_dir]
    for candidate in candidates:
        if (candidate / "hypoDD.reloc").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find hypoDD.reloc in %s/output_files or %s"
        % (working_dir, working_dir)
    )


def _write_trial_hypodd_input(source, destination, use_cross_correlation):
    """
    Copy hypoDD.inp and optionally force IDAT to 2 or 3.
    """
    lines = Path(source).read_text(encoding="utf-8", errors="replace").splitlines()
    if use_cross_correlation is not None and len(lines) > 10:
        parts = lines[10].split()
        if len(parts) >= 3:
            parts[0] = "3" if use_cross_correlation else "2"
            lines[10] = " ".join(parts)
    Path(destination).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_hypodd_trial(base_working_dir, trial_dir, use_cross_correlation=None):
    """
    Run HypoDD once in ``trial_dir`` using files already copied there.
    """
    base_working_dir = Path(base_working_dir)
    trial_dir = Path(trial_dir)
    hypodd_path = _hypodd_executable(base_working_dir)
    input_dir = _resolve_input_dir(base_working_dir)
    if use_cross_correlation is None:
        use_cross_correlation = (input_dir / "dt.cc").exists()

    for filename in ["event.sel", "station.sel"]:
        shutil.copyfile(input_dir / filename, trial_dir / filename)
    _write_trial_hypodd_input(
        input_dir / "hypoDD.inp",
        trial_dir / "hypoDD.inp",
        use_cross_correlation,
    )

    if use_cross_correlation and not (trial_dir / "dt.cc").exists():
        _copy_if_exists(input_dir / "dt.cc", trial_dir / "dt.cc")

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


def summarize_bootstrap_locations(reference_locations, trial_location_paths):
    """
    Summarize bootstrap scatter in HypoDD local x/y/z/time coordinates.
    """
    trial_locations = [read_hypodd_locations(path) for path in trial_location_paths]
    return summarize_bootstrap_location_sets(reference_locations, trial_locations)


def summarize_bootstrap_location_sets(reference_locations, trial_locations):
    """
    Summarize bootstrap scatter from already-read location dictionaries.
    """
    values = defaultdict(lambda: defaultdict(list))
    for locations in trial_locations:
        for event_id, location in locations.items():
            reference = reference_locations.get(event_id)
            if reference is None:
                continue
            dx_km, dy_km = _local_offsets_km(reference, location)
            values[event_id]["dx_km"].append(dx_km)
            values[event_id]["dy_km"].append(dy_km)
            values[event_id]["dz_km"].append(
                location["depth_km"] - reference["depth_km"]
            )
            values[event_id]["dt_s"].append(
                (location["time"] - reference["time"]).total_seconds()
            )

    summary = []
    for event_id, columns in sorted(values.items()):
        row = {"event_id": event_id, "n_trials": len(columns["dx_km"])}
        for key, vals in columns.items():
            if len(vals) < 2:
                row["%s_std" % key] = math.nan
                row["%s_p025" % key] = math.nan
                row["%s_p975" % key] = math.nan
                continue
            sorted_vals = sorted(vals)
            mean = sum(vals) / len(vals)
            variance = sum((value - mean) ** 2 for value in vals) / (len(vals) - 1)
            row["%s_mean" % key] = mean
            row["%s_std" % key] = math.sqrt(variance)
            row["%s_p025" % key] = sorted_vals[int(0.025 * (len(vals) - 1))]
            row["%s_p975" % key] = sorted_vals[int(0.975 * (len(vals) - 1))]
        dx = columns["dx_km"]
        dy = columns["dy_km"]
        if len(dx) >= 2:
            horizontal = [math.hypot(x, y) for x, y in zip(dx, dy)]
            row["horizontal_shift_km_median"] = sorted(horizontal)[len(horizontal) // 2]
            row["horizontal_shift_km_p95"] = sorted(horizontal)[
                int(0.95 * (len(horizontal) - 1))
            ]
        summary.append(row)
    return summary


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


def run_residual_bootstrap(
    working_dir,
    output_dir=None,
    n_trials=200,
    random_seed=None,
    include_ct=True,
    include_cc=True,
    keep_unmatched=True,
    overwrite=False,
    use_cross_correlation=None,
    keep_trial_files="first",
    **kwargs,
):
    """
    Run residual-bootstrap HypoDD trials from an existing completed run.

    Parameters
    ----------
    working_dir
        Existing HypoDDPy working directory with ``input_files`` and
        ``output_files``.
    output_dir
        Directory where bootstrap trials and summaries are written.
    n_trials
        Number of bootstrap realizations.
    random_seed
        Optional random seed for reproducibility.
    include_ct, include_cc
        Whether to bootstrap catalog and/or cross-correlation differential data.
    keep_unmatched
        Keep original observations that cannot be matched to final residuals.
    overwrite
        Remove an existing output directory before running.
    use_cross_correlation
        Whether HypoDD should require/use dt.cc. Defaults to whether dt.cc exists.
    keep_trial_files
        Controls how much per-trial data is kept on disk:

        ``"all"``
            Keep every trial directory.
        ``"first"``
            Keep the first successful trial directory and failed trial
            directories, remove other successful trial directories after reading
            ``hypoDD.reloc``. This is the default.
        ``"failed"``
            Keep only failed trial directories.
        ``"none"``
            Remove all successful and failed trial directories after recording
            status. Useful for large bootstrap runs.
        ``True``/``False`` are accepted as aliases for ``"all"``/``"first"``.

    Extra keyword aliases are accepted for notebook-style calls:

        ``save_all_trials`` -> ``keep_trial_files="all"`` if true, otherwise
        ``"first"``.
        ``n_bootstrap`` or ``n_bootstrap_trials`` -> ``n_trials``.
    """
    if "n_bootstrap" in kwargs:
        n_trials = kwargs.pop("n_bootstrap")
    if "n_bootstrap_trials" in kwargs:
        n_trials = kwargs.pop("n_bootstrap_trials")
    if "save_all_trials" in kwargs:
        keep_trial_files = "all" if kwargs.pop("save_all_trials") else "first"
    if "save_trial_files" in kwargs:
        keep_trial_files = "all" if kwargs.pop("save_trial_files") else "first"
    if kwargs:
        raise TypeError(
            "Unexpected bootstrap option(s): %s" % ", ".join(sorted(kwargs))
        )

    if keep_trial_files is True:
        keep_trial_files = "all"
    elif keep_trial_files is False:
        keep_trial_files = "first"
    keep_trial_files = str(keep_trial_files).lower()
    if keep_trial_files not in ("all", "first", "failed", "none"):
        raise ValueError(
            "keep_trial_files must be one of 'all', 'first', 'failed', or 'none'."
        )

    working_dir = Path(working_dir)
    input_dir = _resolve_input_dir(working_dir)
    output_files = _resolve_output_dir(working_dir)
    if output_dir is None:
        output_dir = working_dir / "bootstrap"
    output_dir = Path(output_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blocks_by_type = {}
    if include_ct:
        blocks_by_type["ct"] = read_dt_file(input_dir / "dt.ct", file_type="ct")
    if include_cc and (input_dir / "dt.cc").exists():
        blocks_by_type["cc"] = read_dt_file(input_dir / "dt.cc", file_type="cc")
    if use_cross_correlation is None:
        use_cross_correlation = bool(include_cc and "cc" in blocks_by_type)

    residual_path = output_files / "hypoDD.final.res"
    residual_rows = read_hypodd_final_residuals(residual_path)
    match_summary = attach_final_residuals(blocks_by_type, residual_rows)
    pools = residual_pools(blocks_by_type)
    if not any(pools.values()):
        raise RuntimeError(
            "No residual pools could be built. Check %s and dt.ct/dt.cc matching."
            % residual_path
        )

    rng = random.Random(random_seed)
    trial_location_paths = []
    trial_locations = []
    trial_status = []
    kept_first_success = False
    for trial_index in range(1, int(n_trials) + 1):
        trial_dir = output_dir / ("trial_%04d" % trial_index)
        if trial_dir.exists() and overwrite:
            shutil.rmtree(trial_dir)
        trial_dir.mkdir(parents=True, exist_ok=True)

        try:
            if "ct" in blocks_by_type:
                trial_ct = create_bootstrap_blocks(
                    blocks_by_type["ct"], pools, rng, keep_unmatched=keep_unmatched
                )
                write_dt_file(trial_ct, trial_dir / "dt.ct")
            else:
                _copy_if_exists(input_dir / "dt.ct", trial_dir / "dt.ct")

            if "cc" in blocks_by_type and use_cross_correlation:
                trial_cc = create_bootstrap_blocks(
                    blocks_by_type["cc"], pools, rng, keep_unmatched=keep_unmatched
                )
                write_dt_file(trial_cc, trial_dir / "dt.cc")
            elif use_cross_correlation and (input_dir / "dt.cc").exists():
                _copy_if_exists(input_dir / "dt.cc", trial_dir / "dt.cc")

            run_hypodd_trial(
                working_dir,
                trial_dir,
                use_cross_correlation=use_cross_correlation,
            )
            trial_reloc_path = trial_dir / "hypoDD.reloc"
            locations = read_hypodd_locations(trial_reloc_path)
            trial_locations.append(locations)
            trial_location_paths.append(trial_reloc_path)
            trial_status.append(
                {
                    "trial": trial_index,
                    "status": "success",
                    "trial_dir": str(trial_dir),
                    "location_count": len(locations),
                    "error": "",
                }
            )
            keep_this_success = (
                keep_trial_files == "all"
                or (
                    keep_trial_files == "first"
                    and not kept_first_success
                )
            )
            if keep_this_success and keep_trial_files == "first":
                kept_first_success = True
            if not keep_this_success and trial_dir.exists():
                shutil.rmtree(trial_dir)
        except Exception as exc:
            trial_status.append(
                {
                    "trial": trial_index,
                    "status": "failed",
                    "trial_dir": str(trial_dir),
                    "location_count": 0,
                    "error": "%s: %s" % (exc.__class__.__name__, exc),
                }
            )
            if keep_trial_files == "none" and trial_dir.exists():
                shutil.rmtree(trial_dir)

    reference_locations = read_hypodd_locations(output_files / "hypoDD.reloc")
    location_summary = summarize_bootstrap_location_sets(
        reference_locations, trial_locations
    )
    reference_event_ids = set(reference_locations)
    trial_event_ids = set()
    overlap_event_ids = set()
    for locations in trial_locations:
        current_event_ids = set(locations)
        trial_event_ids.update(current_event_ids)
        overlap_event_ids.update(reference_event_ids.intersection(current_event_ids))
    successful_location_counts = [
        row.get("location_count", 0)
        for row in trial_status
        if row.get("status") == "success"
    ]

    metadata = {
        "working_dir": str(working_dir),
        "output_dir": str(output_dir),
        "n_trials_requested": int(n_trials),
        "n_trials_successful": len(trial_location_paths),
        "random_seed": random_seed,
        "include_ct": include_ct,
        "include_cc": include_cc,
        "use_cross_correlation": use_cross_correlation,
        "keep_unmatched": keep_unmatched,
        "keep_trial_files": keep_trial_files,
        "reference_location_count": len(reference_locations),
        "unique_trial_location_count": len(trial_event_ids),
        "reference_trial_overlap_event_count": len(overlap_event_ids),
        "successful_trial_location_count_min": (
            min(successful_location_counts) if successful_location_counts else 0
        ),
        "successful_trial_location_count_max": (
            max(successful_location_counts) if successful_location_counts else 0
        ),
        "location_summary_event_count": len(location_summary),
        "match_summary": match_summary,
        "residual_pool_sizes": {
            "%s_%s" % (key[0], key[1]): len(value)
            for key, value in sorted(pools.items())
        },
    }
    (output_dir / "bootstrap_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "trial_status.json").write_text(
        json.dumps(trial_status, indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "bootstrap_location_uncertainty.csv", location_summary)
    if not location_summary:
        diagnostic_rows = [
            {
                "reason": (
                    "No event had at least one successful bootstrap location "
                    "matching the reference relocation."
                ),
                "reference_location_count": len(reference_locations),
                "unique_trial_location_count": len(trial_event_ids),
                "reference_trial_overlap_event_count": len(overlap_event_ids),
                "n_trials_successful": len(trial_location_paths),
                "successful_trial_location_count_min": (
                    min(successful_location_counts)
                    if successful_location_counts
                    else 0
                ),
                "successful_trial_location_count_max": (
                    max(successful_location_counts)
                    if successful_location_counts
                    else 0
                ),
            }
        ]
        _write_csv(output_dir / "bootstrap_empty_summary_diagnostic.csv", diagnostic_rows)
    return {
        "metadata": metadata,
        "trial_status": trial_status,
        "location_summary": location_summary,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("working_dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--no-ct", action="store_true")
    parser.add_argument("--no-cc", action="store_true")
    parser.add_argument("--drop-unmatched", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-trial-files",
        choices=["all", "first", "failed", "none"],
        default="first",
        help=(
            "How many per-trial folders to keep. Default keeps only the first "
            "successful trial and failed trials."
        ),
    )
    args = parser.parse_args()

    result = run_residual_bootstrap(
        working_dir=args.working_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        random_seed=args.random_seed,
        include_ct=not args.no_ct,
        include_cc=not args.no_cc,
        keep_unmatched=not args.drop_unmatched,
        overwrite=args.overwrite,
        keep_trial_files=args.keep_trial_files,
    )
    print(json.dumps(result["metadata"], indent=2))
