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
    working_dir = Path(working_dir).resolve()
    candidates = [
        Path(working_dir) / "bin" / "hypoDD",
        Path(working_dir) / "bin" / "hypoDD.exe",
        Path(working_dir) / "hypoDD",
        Path(working_dir) / "hypoDD.exe",
        Path("hypoDD"),
        Path("hypoDD.exe"),
    ]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    resolved = shutil.which("hypoDD") or shutil.which("hypoDD.exe")
    if resolved:
        return Path(resolved)
    raise FileNotFoundError(
        "Could not find compiled hypoDD binary. Checked: %s. "
        "Also checked PATH."
        % ", ".join(str(candidate.resolve()) for candidate in candidates)
    )


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


def run_hypodd_trial(
    base_working_dir,
    trial_dir,
    use_cross_correlation=None,
):
    """
    Run HypoDD once in ``trial_dir`` using files already copied there.
    """
    base_working_dir = Path(base_working_dir).resolve()
    trial_dir = Path(trial_dir).resolve()
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
            row.update(_horizontal_uncertainty_ellipse(dx, dy))
        summary.append(row)
    return summary


def _horizontal_uncertainty_ellipse(dx_km, dy_km):
    """
    Return event-wise horizontal covariance sigmas and major-axis azimuth.

    ``dx_km`` is positive east and ``dy_km`` is positive north. The azimuth is
    the major-axis direction in degrees clockwise from north, modulo 180.
    """
    n_values = len(dx_km)
    if n_values < 2:
        return {
            "sigma_x_km": math.nan,
            "sigma_y_km": math.nan,
            "cov_xy_km2": math.nan,
            "sigma_major_km": math.nan,
            "sigma_minor_km": math.nan,
            "ellipse_azimuth_deg": math.nan,
            "ellipse_68_major_km": math.nan,
            "ellipse_68_minor_km": math.nan,
            "ellipse_95_major_km": math.nan,
            "ellipse_95_minor_km": math.nan,
        }

    mean_x = sum(dx_km) / n_values
    mean_y = sum(dy_km) / n_values
    var_x = sum((value - mean_x) ** 2 for value in dx_km) / (n_values - 1)
    var_y = sum((value - mean_y) ** 2 for value in dy_km) / (n_values - 1)
    cov_xy = sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(dx_km, dy_km)
    ) / (n_values - 1)

    trace = var_x + var_y
    difference = var_x - var_y
    discriminant = math.sqrt(max(0.0, difference * difference + 4.0 * cov_xy * cov_xy))
    eigen_major = max(0.0, 0.5 * (trace + discriminant))
    eigen_minor = max(0.0, 0.5 * (trace - discriminant))

    if abs(cov_xy) < 1e-15 and var_x >= var_y:
        major_x, major_y = 1.0, 0.0
    elif abs(cov_xy) < 1e-15:
        major_x, major_y = 0.0, 1.0
    else:
        major_x = cov_xy
        major_y = eigen_major - var_x
        norm = math.hypot(major_x, major_y)
        major_x /= norm
        major_y /= norm

    azimuth = math.degrees(math.atan2(major_x, major_y)) % 180.0
    sigma_major = math.sqrt(eigen_major)
    sigma_minor = math.sqrt(eigen_minor)

    # Scale factors for a 2-D Gaussian confidence ellipse:
    # sqrt(chi2.ppf(0.68, 2)) ~= 1.5096, sqrt(chi2.ppf(0.95, 2)) ~= 2.4477.
    scale_68 = 1.5095921854516636
    scale_95 = 2.447746830680816
    return {
        "sigma_x_km": math.sqrt(max(0.0, var_x)),
        "sigma_y_km": math.sqrt(max(0.0, var_y)),
        "cov_xy_km2": cov_xy,
        "sigma_major_km": sigma_major,
        "sigma_minor_km": sigma_minor,
        "ellipse_azimuth_deg": azimuth,
        "ellipse_68_major_km": scale_68 * sigma_major,
        "ellipse_68_minor_km": scale_68 * sigma_minor,
        "ellipse_95_major_km": scale_95 * sigma_major,
        "ellipse_95_minor_km": scale_95 * sigma_minor,
    }


def _read_bootstrap_uncertainty_csv(path):
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float_or_nan(value):
    if value in ("", None):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _ellipse_width_height_degrees(lat, major_km, minor_km):
    km_per_degree_lat = 111.32
    km_per_degree_lon = 111.32 * math.cos(math.radians(lat))
    if abs(km_per_degree_lon) < 1e-12:
        km_per_degree_lon = 1e-12
    return 2.0 * major_km / km_per_degree_lon, 2.0 * minor_km / km_per_degree_lat


def plot_bootstrap_event_uncertainties(
    working_dir,
    uncertainty_csv=None,
    confidence="95",
    scale=1.0,
    min_n_trials=2,
    max_ellipses=None,
    ellipse_stride=1,
    color_by="ellipse_95_major_km",
    cmap="magma_r",
    use_cartopy=True,
    show_events=True,
    show_ellipses=True,
    ellipse_alpha=0.28,
    marker_size=12,
    ax=None,
    output_path=None,
):
    """
    Plot relocated events with event-wise bootstrap horizontal uncertainty.

    Parameters
    ----------
    working_dir
        Completed HypoDDPy run folder.
    uncertainty_csv
        Optional path to ``bootstrap_location_uncertainty.csv``. Defaults to
        ``working_dir/bootstrap/bootstrap_location_uncertainty.csv``.
    confidence
        Which ellipse columns to plot: ``"sigma"``, ``"68"``, or ``"95"``.
    scale
        Extra visual scale factor applied to ellipse axes.
    min_n_trials
        Only plot uncertainty for events present in at least this many
        successful bootstrap trials.
    max_ellipses, ellipse_stride
        Optional thinning controls for crowded maps. Event points are still
        plotted for all rows that pass ``min_n_trials``.

    Returns ``(fig, ax, plotted_rows)``.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError as exc:
        raise ImportError("matplotlib is required to plot bootstrap uncertainty") from exc

    working_dir = Path(working_dir)
    output_dir = _resolve_output_dir(working_dir)
    if uncertainty_csv is None:
        uncertainty_csv = working_dir / "bootstrap" / "bootstrap_location_uncertainty.csv"
    uncertainty_rows = _read_bootstrap_uncertainty_csv(uncertainty_csv)
    locations = read_hypodd_locations(output_dir / "hypoDD.reloc")

    confidence = str(confidence).lower()
    if confidence in ("1", "1sigma", "sigma"):
        major_column = "sigma_major_km"
        minor_column = "sigma_minor_km"
        label = "1-sigma"
    elif confidence in ("68", "0.68", "68%"):
        major_column = "ellipse_68_major_km"
        minor_column = "ellipse_68_minor_km"
        label = "68%"
    elif confidence in ("95", "0.95", "95%"):
        major_column = "ellipse_95_major_km"
        minor_column = "ellipse_95_minor_km"
        label = "95%"
    else:
        raise ValueError("confidence must be 'sigma', '68', or '95'.")

    required = {major_column, minor_column, "ellipse_azimuth_deg", "event_id"}
    missing = sorted(required.difference(uncertainty_rows[0] if uncertainty_rows else {}))
    if missing:
        raise ValueError(
            "Missing bootstrap uncertainty column(s): %s. Rerun bootstrap after "
            "the event-wise ellipse update." % ", ".join(missing)
        )

    plotted_rows = []
    for row in uncertainty_rows:
        event_id = int(float(row["event_id"]))
        location = locations.get(event_id)
        if location is None:
            continue
        n_trials = int(float(row.get("n_trials") or 0))
        if n_trials < min_n_trials:
            continue
        major_km = _float_or_nan(row.get(major_column))
        minor_km = _float_or_nan(row.get(minor_column))
        azimuth = _float_or_nan(row.get("ellipse_azimuth_deg"))
        if not all(math.isfinite(value) for value in [major_km, minor_km, azimuth]):
            continue
        current = dict(row)
        current.update(location)
        current["uncertainty_major_km"] = major_km
        current["uncertainty_minor_km"] = minor_km
        current["uncertainty_azimuth_deg"] = azimuth
        plotted_rows.append(current)

    if not plotted_rows:
        raise ValueError("No events had usable bootstrap uncertainty rows.")

    projection = None
    transform = None
    if use_cartopy:
        try:
            import cartopy.crs as ccrs

            projection = ccrs.PlateCarree()
            transform = ccrs.PlateCarree()
        except ImportError:
            projection = None
            transform = None

    if ax is None:
        if projection is None:
            fig, ax = plt.subplots(figsize=(9, 8))
        else:
            fig = plt.figure(figsize=(9, 8))
            ax = plt.axes(projection=projection)
    else:
        fig = ax.figure

    lons = [row["longitude"] for row in plotted_rows]
    lats = [row["latitude"] for row in plotted_rows]
    values = [_float_or_nan(row.get(color_by)) for row in plotted_rows]
    if not any(math.isfinite(value) for value in values):
        color_by = "uncertainty_major_km"
        values = [row["uncertainty_major_km"] for row in plotted_rows]

    lon_pad = max(0.25, (max(lons) - min(lons)) * 0.12)
    lat_pad = max(0.25, (max(lats) - min(lats)) * 0.12)
    extent = [min(lons) - lon_pad, max(lons) + lon_pad, min(lats) - lat_pad, max(lats) + lat_pad]

    if projection is not None:
        ax.set_extent(extent, crs=transform)
        try:
            import cartopy.feature as cfeature

            ax.add_feature(cfeature.LAND, facecolor="0.96")
            ax.add_feature(cfeature.OCEAN, facecolor="0.90")
            ax.add_feature(cfeature.BORDERS, linewidth=0.8)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
            ax.add_feature(cfeature.RIVERS, linewidth=0.45, edgecolor="0.55")
        except Exception:
            pass
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    scatter = None
    transform_kwargs = {"transform": transform} if transform is not None else {}
    if show_events:
        scatter = ax.scatter(
            lons,
            lats,
            c=values,
            cmap=cmap,
            s=marker_size,
            edgecolors="none",
            zorder=5,
            **transform_kwargs,
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label(color_by)

    ellipse_rows = plotted_rows[:: max(1, int(ellipse_stride))]
    if max_ellipses is not None:
        ellipse_rows = ellipse_rows[: int(max_ellipses)]

    if show_ellipses:
        for row in ellipse_rows:
            width, height = _ellipse_width_height_degrees(
                row["latitude"],
                row["uncertainty_major_km"] * scale,
                row["uncertainty_minor_km"] * scale,
            )
            # Matplotlib ellipse angle is counterclockwise from east; our
            # azimuth is clockwise from north.
            angle = 90.0 - row["uncertainty_azimuth_deg"]
            ellipse = Ellipse(
                (row["longitude"], row["latitude"]),
                width=width,
                height=height,
                angle=angle,
                facecolor="none",
                edgecolor="black",
                linewidth=0.6,
                alpha=ellipse_alpha,
                zorder=6,
                transform=transform,
            )
            ax.add_patch(ellipse)

    ax.set_title(
        "Bootstrap event uncertainty (%s ellipses, %i events)" % (label, len(plotted_rows))
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=200)
    return fig, ax, plotted_rows


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

    working_dir = Path(working_dir).resolve()
    input_dir = _resolve_input_dir(working_dir)
    output_files = _resolve_output_dir(working_dir)
    if output_dir is None:
        output_dir = working_dir / "bootstrap"
    output_dir = Path(output_dir).resolve()
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
