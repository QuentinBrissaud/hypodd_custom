#!/usr/bin/env python
"""
Compute event distances to geographic context layers for a HypoDD run.

The script reads original and relocated locations from a run directory
containing either ``output_files/hypoDD.loc`` and ``output_files/hypoDD.reloc``
or those files directly. It writes one CSV row per event with distances from
both the original and relocated locations to the closest frontline, town, road,
and river.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path

from hypodd_quality_report import read_hypodd_locations


UKRAINE_CITIES = [
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


@dataclass(frozen=True)
class LineSegment:
    lon1: float
    lat1: float
    lon2: float
    lat2: float
    label: str = ""


def resolve_output_files(path):
    """
    Return the directory containing ``hypoDD.loc`` and ``hypoDD.reloc``.
    """
    path = Path(path).expanduser()
    output_files = path / "output_files"
    if (output_files / "hypoDD.loc").exists():
        return output_files
    if (path / "hypoDD.loc").exists():
        return path
    raise FileNotFoundError(
        "Could not find hypoDD.loc in %s or %s." % (path, output_files)
    )


def horizontal_distance_km(lat1, lon1, lat2, lon2):
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111.0 * math.cos(mean_lat)
    dy = (lat2 - lat1) * 111.0
    return math.sqrt(dx * dx + dy * dy)


def point_to_segment_distance_km(lat, lon, segment):
    """
    Approximate point-to-segment distance using a local km projection.
    """
    cos_lat = math.cos(math.radians(lat))
    x1 = (segment.lon1 - lon) * 111.0 * cos_lat
    y1 = (segment.lat1 - lat) * 111.0
    x2 = (segment.lon2 - lon) * 111.0 * cos_lat
    y2 = (segment.lat2 - lat) * 111.0
    dx = x2 - x1
    dy = y2 - y1
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return math.sqrt(x1 * x1 + y1 * y1)
    t = max(0.0, min(1.0, -(x1 * dx + y1 * dy) / length2))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.sqrt(closest_x * closest_x + closest_y * closest_y)


def nearest_segment_distance_km(lat, lon, segments):
    best_distance = math.nan
    best_label = ""
    for segment in segments:
        distance = point_to_segment_distance_km(lat, lon, segment)
        if math.isnan(best_distance) or distance < best_distance:
            best_distance = distance
            best_label = segment.label
    return best_distance, best_label


def nearest_town_distance_km(lat, lon, towns=UKRAINE_CITIES):
    best_distance = math.nan
    best_name = ""
    for name, town_lat, town_lon in towns:
        distance = horizontal_distance_km(lat, lon, town_lat, town_lon)
        if math.isnan(best_distance) or distance < best_distance:
            best_distance = distance
            best_name = name
    return best_distance, best_name


def geojson_geometry_lines(geometry):
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates") or []
    if geometry_type == "LineString":
        return [coordinates]
    if geometry_type == "MultiLineString":
        return list(coordinates)
    if geometry_type == "Polygon":
        return list(coordinates)
    if geometry_type == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    if geometry_type == "GeometryCollection":
        lines = []
        for item in geometry.get("geometries") or []:
            lines.extend(geojson_geometry_lines(item))
        return lines
    return []


def line_to_segments(line, label=""):
    segments = []
    for first, second in zip(line, line[1:]):
        if len(first) < 2 or len(second) < 2:
            continue
        lon1, lat1 = float(first[0]), float(first[1])
        lon2, lat2 = float(second[0]), float(second[1])
        segments.append(LineSegment(lon1, lat1, lon2, lat2, label=label))
    return segments


def load_frontline_segments(path, date_property="date"):
    """
    Load dated frontline segments from a GeoJSON FeatureCollection.

    Returns a dictionary keyed by date string.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError("Frontline GeoJSON does not exist: %s" % path)
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    by_date = {}
    for feature in data.get("features", []):
        properties = feature.get("properties") or {}
        date = properties.get(date_property)
        if date in ("", None):
            continue
        date = str(date)[:10]
        geometry = feature.get("geometry") or {}
        for line in geojson_geometry_lines(geometry):
            by_date.setdefault(date, []).extend(line_to_segments(line, label=date))
    return {date: segments for date, segments in by_date.items() if segments}


def _parse_date(value):
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value)[:10]).date()


def select_frontline_segments(by_date, event_time, mode, fixed_date=None, start=None, end=None):
    if not by_date:
        return []
    if mode == "all":
        return [segment for segments in by_date.values() for segment in segments]
    if mode == "fixed-date":
        if fixed_date is None:
            raise ValueError("--frontline-date is required with fixed-date mode")
        return by_date.get(str(fixed_date)[:10], [])
    if mode == "date-range":
        if start is None or end is None:
            raise ValueError(
                "--frontline-start-date and --frontline-end-date are required "
                "with date-range mode"
            )
        start_date = _parse_date(start)
        end_date = _parse_date(end)
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        segments = []
        for date, date_segments in by_date.items():
            parsed = _parse_date(date)
            if start_date <= parsed <= end_date:
                segments.extend(date_segments)
        return segments

    event_date = event_time.date().isoformat()
    if event_date in by_date:
        return by_date[event_date]

    parsed_event_date = _parse_date(event_date)
    nearest_date = min(
        by_date,
        key=lambda date: abs((_parse_date(date) - parsed_event_date).days),
    )
    return by_date[nearest_date]


def shapely_geometry_lines(geometry):
    """
    Extract lon/lat coordinate sequences from shapely geometries.
    """
    geom_type = getattr(geometry, "geom_type", None)
    if geom_type == "LineString":
        return [list(geometry.coords)]
    if geom_type == "MultiLineString":
        return [list(item.coords) for item in geometry.geoms]
    if geom_type == "Polygon":
        return [list(geometry.exterior.coords)]
    if geom_type == "MultiPolygon":
        return [list(item.exterior.coords) for item in geometry.geoms]
    if geom_type == "GeometryCollection":
        lines = []
        for item in geometry.geoms:
            lines.extend(shapely_geometry_lines(item))
        return lines
    return []


def _segment_intersects_bbox(segment, bbox):
    min_lon, max_lon, min_lat, max_lat = bbox
    seg_min_lon = min(segment.lon1, segment.lon2)
    seg_max_lon = max(segment.lon1, segment.lon2)
    seg_min_lat = min(segment.lat1, segment.lat2)
    seg_max_lat = max(segment.lat1, segment.lat2)
    return not (
        seg_max_lon < min_lon
        or seg_min_lon > max_lon
        or seg_max_lat < min_lat
        or seg_min_lat > max_lat
    )


def filter_segments_to_bbox(segments, bbox):
    filtered = [segment for segment in segments if _segment_intersects_bbox(segment, bbox)]
    return filtered if filtered else list(segments)


def load_natural_earth_segments(category, names, label, bbox=None, resolution="10m"):
    """
    Load Natural Earth linework through Cartopy.

    Returns ``(segments, warning)``. If Cartopy or the requested dataset is not
    available, ``segments`` is empty and ``warning`` explains why.
    """
    try:
        import cartopy.io.shapereader as shpreader
    except ImportError as exc:
        return [], "Cartopy is not available, so %s distances were skipped: %s" % (
            label,
            exc,
        )

    errors = []
    for name in names:
        try:
            shp_path = shpreader.natural_earth(
                resolution=resolution,
                category=category,
                name=name,
            )
            reader = shpreader.Reader(shp_path)
            segments = []
            for geometry in reader.geometries():
                for line in shapely_geometry_lines(geometry):
                    segments.extend(line_to_segments(line, label=label))
            if bbox is not None:
                segments = filter_segments_to_bbox(segments, bbox)
            return segments, ""
        except Exception as exc:  # pragma: no cover - depends on local Cartopy data
            errors.append("%s: %s" % (name, exc))
    return [], "Could not load Natural Earth %s data (%s)." % (label, "; ".join(errors))


def event_bbox(original_events, relocated_events, padding_degrees=2.0):
    lons = []
    lats = []
    for event in list(original_events.values()) + list(relocated_events.values()):
        lons.append(event["longitude"])
        lats.append(event["latitude"])
    return (
        min(lons) - padding_degrees,
        max(lons) + padding_degrees,
        min(lats) - padding_degrees,
        max(lats) + padding_degrees,
    )


def _rounded(value):
    if value is None or math.isnan(value):
        return ""
    return "%.4f" % value


def _distance_change(original, relocated):
    if math.isnan(original) or math.isnan(relocated):
        return math.nan
    return relocated - original


def compute_event_context_distances(
    run_folder,
    frontline_geojson="data/daily_red_boundaries.geojson",
    output_csv=None,
    frontline_mode="event-date",
    frontline_date=None,
    frontline_start_date=None,
    frontline_end_date=None,
    include_natural_earth=True,
):
    """
    Compute distances and write a CSV file.

    Distances are horizontal distances in kilometers. Positive ``*_change_km``
    values mean the relocated event is farther from the feature than the
    original event.
    """
    output_files = resolve_output_files(run_folder)
    original_events = read_hypodd_locations(output_files / "hypoDD.loc")
    relocated_events = read_hypodd_locations(output_files / "hypoDD.reloc")
    common_event_ids = sorted(set(original_events) & set(relocated_events))
    if not common_event_ids:
        raise ValueError("No common events found between hypoDD.loc and hypoDD.reloc.")

    frontline_by_date = load_frontline_segments(frontline_geojson)
    bbox = event_bbox(
        {event_id: original_events[event_id] for event_id in common_event_ids},
        {event_id: relocated_events[event_id] for event_id in common_event_ids},
    )

    road_segments = []
    river_segments = []
    warnings = []
    if include_natural_earth:
        road_segments, warning = load_natural_earth_segments(
            "cultural",
            ["roads"],
            "road",
            bbox=bbox,
        )
        if warning:
            warnings.append(warning)
        river_segments, warning = load_natural_earth_segments(
            "physical",
            ["rivers_lake_centerlines", "rivers_lake_centerlines_scale_rank"],
            "river",
            bbox=bbox,
        )
        if warning:
            warnings.append(warning)

    rows = []
    for event_id in common_event_ids:
        original = original_events[event_id]
        relocated = relocated_events[event_id]
        frontline_segments = select_frontline_segments(
            frontline_by_date,
            original["time"],
            frontline_mode,
            fixed_date=frontline_date,
            start=frontline_start_date,
            end=frontline_end_date,
        )

        original_frontline, original_frontline_date = nearest_segment_distance_km(
            original["latitude"], original["longitude"], frontline_segments
        )
        relocated_frontline, relocated_frontline_date = nearest_segment_distance_km(
            relocated["latitude"], relocated["longitude"], frontline_segments
        )
        original_town, original_town_name = nearest_town_distance_km(
            original["latitude"], original["longitude"]
        )
        relocated_town, relocated_town_name = nearest_town_distance_km(
            relocated["latitude"], relocated["longitude"]
        )
        original_road, _ = nearest_segment_distance_km(
            original["latitude"], original["longitude"], road_segments
        )
        relocated_road, _ = nearest_segment_distance_km(
            relocated["latitude"], relocated["longitude"], road_segments
        )
        original_river, _ = nearest_segment_distance_km(
            original["latitude"], original["longitude"], river_segments
        )
        relocated_river, _ = nearest_segment_distance_km(
            relocated["latitude"], relocated["longitude"], river_segments
        )

        cluster_id = relocated.get("cluster_id", original.get("cluster_id", ""))
        rows.append(
            {
                "event_id": event_id,
                "event_time": original["time"].isoformat(),
                "event_date": original["time"].date().isoformat(),
                "cluster_id": cluster_id,
                "original_latitude": "%.6f" % original["latitude"],
                "original_longitude": "%.6f" % original["longitude"],
                "original_depth_km": "%.3f" % original["depth_km"],
                "relocated_latitude": "%.6f" % relocated["latitude"],
                "relocated_longitude": "%.6f" % relocated["longitude"],
                "relocated_depth_km": "%.3f" % relocated["depth_km"],
                "original_frontline_distance_km": _rounded(original_frontline),
                "relocated_frontline_distance_km": _rounded(relocated_frontline),
                "frontline_distance_change_km": _rounded(
                    _distance_change(original_frontline, relocated_frontline)
                ),
                "original_frontline_date": original_frontline_date,
                "relocated_frontline_date": relocated_frontline_date,
                "original_town_distance_km": _rounded(original_town),
                "original_town_name": original_town_name,
                "relocated_town_distance_km": _rounded(relocated_town),
                "relocated_town_name": relocated_town_name,
                "town_distance_change_km": _rounded(
                    _distance_change(original_town, relocated_town)
                ),
                "original_road_distance_km": _rounded(original_road),
                "relocated_road_distance_km": _rounded(relocated_road),
                "road_distance_change_km": _rounded(
                    _distance_change(original_road, relocated_road)
                ),
                "original_river_distance_km": _rounded(original_river),
                "relocated_river_distance_km": _rounded(relocated_river),
                "river_distance_change_km": _rounded(
                    _distance_change(original_river, relocated_river)
                ),
            }
        )

    if output_csv is None:
        report_dir = Path(run_folder).expanduser() / "quality_report"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_csv = report_dir / "event_context_distances.csv"
    output_csv = Path(output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "output_csv": output_csv,
        "n_events": len(rows),
        "warnings": warnings,
    }


def _read_distance_rows(table):
    if isinstance(table, (str, Path)):
        with Path(table).expanduser().open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if hasattr(table, "to_dict"):
        return table.to_dict(orient="records")
    return list(table)


def _numeric_column(rows, column):
    values = []
    for row in rows:
        value = row.get(column, "")
        if value in ("", None):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value):
            values.append(value)
    return values


def _auto_bins(values, max_bins=60):
    if not values:
        return 10
    unique_values = sorted(set(values))
    if len(unique_values) <= 1:
        center = unique_values[0]
        width = max(abs(center) * 0.05, 0.5)
        return [center - width, center + width]
    return min(max_bins, max(12, int(math.sqrt(len(values)) * 2)))


def plot_distance_distributions(
    distance_table,
    features=("frontline", "town", "road", "river"),
    bins=None,
    density=False,
    output_path=None,
    title="Event distances to geographic context",
):
    """
    Plot 1-D distributions of original and relocated distances.

    ``distance_table`` can be the CSV path written by
    :func:`compute_event_context_distances`, a pandas DataFrame, or an iterable
    of row dictionaries. The function returns ``(fig, axes)``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for plotting distributions") from exc

    rows = _read_distance_rows(distance_table)
    if not rows:
        raise ValueError("No distance rows were provided.")

    features = list(features)
    n_features = len(features)
    ncols = 2 if n_features > 1 else 1
    nrows = int(math.ceil(n_features / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 3.8 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    for ax, feature in zip(axes.ravel(), features):
        original_column = "original_%s_distance_km" % feature
        relocated_column = "relocated_%s_distance_km" % feature
        original_values = _numeric_column(rows, original_column)
        relocated_values = _numeric_column(rows, relocated_column)
        all_values = original_values + relocated_values

        if not all_values:
            ax.text(
                0.5,
                0.5,
                "No %s distances" % feature,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        feature_bins = bins if bins is not None else _auto_bins(all_values)
        ax.hist(
            original_values,
            bins=feature_bins,
            density=density,
            histtype="stepfilled",
            alpha=0.35,
            color="#4C78A8",
            label="Original",
        )
        ax.hist(
            relocated_values,
            bins=feature_bins,
            density=density,
            histtype="step",
            linewidth=1.8,
            color="#F58518",
            label="Relocated",
        )
        ax.axvline(
            sum(original_values) / len(original_values),
            color="#4C78A8",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )
        ax.axvline(
            sum(relocated_values) / len(relocated_values),
            color="#F58518",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
        )
        ax.set_title("%s distance" % feature.capitalize())
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Density" if density else "Event count")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False)

    for ax in axes.ravel()[n_features:]:
        ax.set_axis_off()

    if title:
        fig.suptitle(title)
    if output_path is not None:
        fig.savefig(output_path, dpi=200)
    return fig, axes


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute original and relocated event distances to frontlines, "
            "towns, roads, and rivers."
        )
    )
    parser.add_argument("run_folder", help="HypoDD run folder or output_files folder.")
    parser.add_argument(
        "--frontlines",
        default="data/daily_red_boundaries.geojson",
        help="Dated frontline GeoJSON file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output CSV path. Defaults to "
            "<run_folder>/quality_report/event_context_distances.csv."
        ),
    )
    parser.add_argument(
        "--frontline-mode",
        choices=["event-date", "all", "fixed-date", "date-range"],
        default="event-date",
        help=(
            "Which frontline geometry to compare with. event-date uses the "
            "event date, falling back to the nearest available frontline date."
        ),
    )
    parser.add_argument("--frontline-date", default=None)
    parser.add_argument("--frontline-start-date", default=None)
    parser.add_argument("--frontline-end-date", default=None)
    parser.add_argument(
        "--skip-natural-earth",
        action="store_true",
        help="Skip roads and rivers from Cartopy/Natural Earth.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    result = compute_event_context_distances(
        args.run_folder,
        frontline_geojson=args.frontlines,
        output_csv=args.output,
        frontline_mode=args.frontline_mode,
        frontline_date=args.frontline_date,
        frontline_start_date=args.frontline_start_date,
        frontline_end_date=args.frontline_end_date,
        include_natural_earth=not args.skip_natural_earth,
    )
    print("Wrote %i events to %s" % (result["n_events"], result["output_csv"]))
    for warning in result["warnings"]:
        print("WARNING: %s" % warning)


if __name__ == "__main__":
    main()
