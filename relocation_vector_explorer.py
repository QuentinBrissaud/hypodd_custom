#!/usr/bin/env python
"""
Interactive Matplotlib explorer for HypoDD relocation vectors.

The main entry points are:

* load_relocation_vector_dataframe()
* plot_relocation_vectors()
* create_relocation_vector_interface()

The module reuses parsers from hypodd_quality_report.py so it reads the same
files that are used to build ukraine_relocation_vectors_map.png.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np

from hypodd_quality_report import read_hypodd_locations, read_station_residuals

try:
    import pandas as pd
except ImportError:
    pd = None


def _require_pandas():
    if pd is None:
        raise ImportError(
            "pandas is required for relocation_vector_explorer. Install it in "
            "the notebook environment with conda install pandas or pip install pandas."
        )
    return pd


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


def resolve_output_files(path):
    """
    Return the directory containing HypoDD output files.

    Pass either a relocation working directory or the output_files directory
    itself.
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


def _locations_to_dataframe(events, prefix):
    pd = _require_pandas()
    rows = []
    for event_id, event in sorted(events.items()):
        row = {
            "event_id": int(event_id),
            "%s_latitude" % prefix: event["latitude"],
            "%s_longitude" % prefix: event["longitude"],
            "%s_depth_km" % prefix: event["depth_km"],
            "%s_time" % prefix: event["time"],
        }
        if "cluster_id" in event:
            row["%s_cluster_id" % prefix] = event["cluster_id"]
        if "rms_ct" in event:
            row["%s_rms_ct_s" % prefix] = event["rms_ct"]
        if "rms_cc" in event:
            row["%s_rms_cc_s" % prefix] = event["rms_cc"]
        rows.append(row)
    return pd.DataFrame(rows)


def _read_optional_csv(path):
    pd = _require_pandas()
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_station_dataframe(path):
    """
    Load station coordinates from hypoDD.sta when available.
    """
    pd = _require_pandas()
    output_files = resolve_output_files(path)
    station_path = output_files / "hypoDD.sta"
    if not station_path.exists():
        return pd.DataFrame(columns=["station_id", "latitude", "longitude"])
    return pd.DataFrame(read_station_residuals(station_path))


def load_relocation_vector_dataframe(path):
    """
    Load original/relocated event locations and optional residual summaries.

    Returned columns include:
    event_id, original_latitude, original_longitude, relocated_latitude,
    relocated_longitude, cluster_id, rms_ct_s, rms_cc_s, horizontal_shift_km,
    depth_shift_km, and any columns from event_residual_summary.csv.
    """
    pd = _require_pandas()
    output_files = resolve_output_files(path)
    original = read_hypodd_locations(output_files / "hypoDD.loc")
    relocated = read_hypodd_locations(output_files / "hypoDD.reloc")

    original_df = _locations_to_dataframe(original, "original")
    relocated_df = _locations_to_dataframe(relocated, "relocated")
    df = original_df.merge(relocated_df, on="event_id", how="inner")

    if "relocated_cluster_id" in df:
        df["cluster_id"] = df["relocated_cluster_id"]
    elif "original_cluster_id" in df:
        df["cluster_id"] = df["original_cluster_id"]
    else:
        df["cluster_id"] = np.nan

    if "relocated_rms_ct_s" in df:
        df["rms_ct_s"] = df["relocated_rms_ct_s"]
    if "relocated_rms_cc_s" in df:
        df["rms_cc_s"] = df["relocated_rms_cc_s"]

    df["horizontal_shift_km"] = _approx_horizontal_distance_km(
        df["original_latitude"],
        df["original_longitude"],
        df["relocated_latitude"],
        df["relocated_longitude"],
    )
    df["depth_shift_km"] = df["relocated_depth_km"] - df["original_depth_km"]
    df["original_time"] = pd.to_datetime(df["original_time"])
    df["relocated_time"] = pd.to_datetime(df["relocated_time"])
    df["origin_time_shift_s"] = (
        df["relocated_time"] - df["original_time"]
    ).dt.total_seconds()

    residual_summary = _read_optional_csv(output_files / "event_residual_summary.csv")
    if residual_summary is not None and "event_id" in residual_summary:
        residual_summary["event_id"] = residual_summary["event_id"].astype(int)
        df = df.merge(residual_summary, on="event_id", how="left")

    return df


def available_color_columns(df):
    """
    Return columns that make sense for coloring relocated events.
    """
    pd = _require_pandas()
    preferred = [
        "cluster_id",
        "rms_ct_s",
        "rms_cc_s",
        "relocated_dd_mean_abs_s",
        "relocated_dd_rms_s",
        "dd_mean_abs_change_s",
        "relocated_pick_mean_abs_s",
        "pick_mean_abs_change_s",
        "horizontal_shift_km",
        "depth_shift_km",
        "origin_time_shift_s",
    ]
    columns = []
    for column in preferred:
        if column in df.columns and df[column].notna().any():
            columns.append(column)
    for column in df.columns:
        if column in columns or column == "event_id":
            continue
        if df[column].notna().any() and pd.api.types.is_numeric_dtype(df[column]):
            columns.append(column)
    return columns


def plot_relocation_vectors(
    df,
    stations=None,
    color_by="cluster_id",
    ax=None,
    use_cartopy=True,
    basemap="natural_earth",
    osm_zoom=9,
    show_original=True,
    show_stations=True,
    show_cities=True,
    show_roads=True,
    show_scale_bar=True,
    vector_alpha=0.45,
    marker_size=16,
    cmap=None,
):
    """
    Plot original-to-relocated vectors and color relocated events by a column.

    The returned Matplotlib figure can be zoomed/panned with the notebook
    toolbar when using an interactive backend such as `%matplotlib widget`.
    """
    projection, transform, basemap = _cartopy_projection(use_cartopy, basemap)
    if ax is None:
        if projection is None:
            fig, ax = plt.subplots(figsize=(9, 8))
        else:
            fig = plt.figure(figsize=(9, 8))
            ax = plt.axes(projection=projection)
    else:
        fig = ax.figure

    if color_by not in df.columns:
        raise ValueError("Unknown color column: %s" % color_by)

    if cmap is None:
        cmap = _default_colormap_for_column(df[color_by], color_by)

    _add_base_map(ax, df, stations, projection, transform, basemap, osm_zoom)
    if show_roads:
        _add_roads(ax, projection, transform)
    _plot_vectors(ax, df, color_by, cmap, transform, vector_alpha)

    if show_original:
        ax.scatter(
            df["original_longitude"],
            df["original_latitude"],
            s=marker_size,
            marker="x",
            c="0.25",
            linewidths=0.7,
            label="Original events",
            zorder=4,
            **_transform_kwargs(transform),
        )

    scatter = _plot_relocated_events(
        ax, df, color_by, cmap, transform, marker_size
    )

    if show_stations and stations is not None and len(stations):
        ax.scatter(
            stations["longitude"],
            stations["latitude"],
            s=34,
            marker="^",
            c="black",
            edgecolors="white",
            linewidths=0.4,
            label="Stations",
            zorder=5,
            **_transform_kwargs(transform),
        )

    if show_cities:
        _add_cities(ax, df, stations, transform)

    _format_axes(ax, projection)
    if show_scale_bar:
        _add_dynamic_scale_bar(ax, transform)
    _add_color_legend_or_bar(fig, ax, scatter, df, color_by, cmap)
    ax.set_title(
        "Event Relocation Vectors (%i matched events), colored by %s"
        % (len(df), color_by)
    )
    fig.tight_layout()
    return fig, ax, scatter


def create_relocation_vector_interface(
    path,
    default_color_by="cluster_id",
    use_cartopy=True,
    default_basemap="natural_earth",
    osm_zoom=9,
):
    """
    Create an ipywidgets interface for changing event colors in a notebook.

    Use `%matplotlib widget` before calling this function if you want the map
    toolbar to support zooming and panning inside the notebook.
    """
    df = load_relocation_vector_dataframe(path)
    stations = load_station_dataframe(path)
    color_columns = available_color_columns(df)
    if default_color_by not in color_columns:
        default_color_by = color_columns[0]
    default_basemap = _normalize_basemap(default_basemap)

    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError:
        fig, ax, scatter = plot_relocation_vectors(
            df,
            stations=stations,
            color_by=default_color_by,
            use_cartopy=use_cartopy,
            basemap=default_basemap,
            osm_zoom=osm_zoom,
        )
        plt.show()
        print(
            "ipywidgets is not installed, so the dropdown interface is disabled. "
            "Use plot_relocation_vectors(df, stations=stations, color_by=...) "
            "to change colors manually."
        )
        return {
            "data": df,
            "stations": stations,
            "figure": fig,
            "axes": ax,
            "scatter": scatter,
            "color_columns": color_columns,
        }

    color_dropdown = widgets.Dropdown(
        options=color_columns,
        value=default_color_by,
        description="Color by",
    )
    basemap_dropdown = widgets.Dropdown(
        options=[
            ("Natural Earth", "natural_earth"),
            ("OpenStreetMap", "osm"),
            ("None", "none"),
        ],
        value=default_basemap,
        description="Basemap",
    )
    original_checkbox = widgets.Checkbox(value=True, description="Original")
    stations_checkbox = widgets.Checkbox(value=True, description="Stations")
    cities_checkbox = widgets.Checkbox(value=True, description="Cities")
    roads_checkbox = widgets.Checkbox(value=True, description="Roads")
    scale_bar_checkbox = widgets.Checkbox(value=True, description="Scale bar")
    output = widgets.Output()

    def redraw(*_):
        with output:
            clear_output(wait=True)
            plot_relocation_vectors(
                df,
                stations=stations,
                color_by=color_dropdown.value,
                use_cartopy=use_cartopy,
                basemap=basemap_dropdown.value,
                osm_zoom=osm_zoom,
                show_original=original_checkbox.value,
                show_stations=stations_checkbox.value,
                show_cities=cities_checkbox.value,
                show_roads=roads_checkbox.value,
                show_scale_bar=scale_bar_checkbox.value,
            )
            plt.show()

    for widget in [
        color_dropdown,
        original_checkbox,
        stations_checkbox,
        cities_checkbox,
        roads_checkbox,
        scale_bar_checkbox,
    ]:
        widget.observe(redraw, names="value")
    basemap_dropdown.observe(redraw, names="value")

    controls = widgets.HBox(
        [
            color_dropdown,
            basemap_dropdown,
            original_checkbox,
            stations_checkbox,
            cities_checkbox,
            roads_checkbox,
            scale_bar_checkbox,
        ]
    )
    display(widgets.VBox([controls, output]))
    redraw()
    return {
        "data": df,
        "stations": stations,
        "color_dropdown": color_dropdown,
        "basemap_dropdown": basemap_dropdown,
        "output": output,
    }


def _approx_horizontal_distance_km(lat1, lon1, lat2, lon2):
    mean_lat = np.deg2rad((lat1 + lat2) / 2.0)
    dx = (lon2 - lon1) * 111.32 * np.cos(mean_lat)
    dy = (lat2 - lat1) * 110.57
    return np.sqrt(dx * dx + dy * dy)


def _cartopy_projection(use_cartopy, basemap):
    if not use_cartopy:
        return None, None, "none"
    try:
        import cartopy.crs as ccrs
    except ImportError:
        return None, None, "none"
    basemap = _normalize_basemap(basemap)
    transform = ccrs.PlateCarree()
    if basemap == "osm":
        try:
            import cartopy.io.img_tiles as cimgt

            tiler = cimgt.OSM()
            return tiler.crs, transform, "osm"
        except Exception:
            return transform, transform, "natural_earth"
    return transform, transform, basemap


def _normalize_basemap(basemap):
    if basemap is None:
        return "natural_earth"
    basemap = str(basemap).strip().lower().replace("-", "_")
    aliases = {
        "naturalearth": "natural_earth",
        "natural_earth": "natural_earth",
        "ne": "natural_earth",
        "osm": "osm",
        "openstreetmap": "osm",
        "open_street_map": "osm",
        "none": "none",
        "off": "none",
        "false": "none",
    }
    if basemap not in aliases:
        raise ValueError(
            "Unsupported basemap %s. Use 'natural_earth', 'osm', or 'none'."
            % basemap
        )
    return aliases[basemap]


def _plot_vectors(ax, df, color_by, cmap, transform, vector_alpha):
    colors, _, _ = _colors_for_column(df[color_by], cmap, color_by)
    for (_, row), color in zip(df.iterrows(), colors):
        ax.plot(
            [row["original_longitude"], row["relocated_longitude"]],
            [row["original_latitude"], row["relocated_latitude"]],
            color=color,
            linewidth=0.45,
            alpha=vector_alpha,
            zorder=2,
            **_transform_kwargs(transform),
        )


def _plot_relocated_events(ax, df, color_by, cmap, transform, marker_size):
    colors, values, norm = _colors_for_column(df[color_by], cmap, color_by)
    if _is_categorical(df[color_by], color_by):
        return ax.scatter(
            df["relocated_longitude"],
            df["relocated_latitude"],
            s=marker_size,
            c=colors,
            edgecolors="none",
            label="Relocated events",
            zorder=6,
            **_transform_kwargs(transform),
        )
    return ax.scatter(
        df["relocated_longitude"],
        df["relocated_latitude"],
        s=marker_size,
        c=values,
        cmap=cmap,
        norm=norm,
        edgecolors="none",
        label="Relocated events",
        zorder=6,
        **_transform_kwargs(transform),
    )


def _is_categorical(series, color_by=None):
    if color_by == "cluster_id":
        return True
    pd = _require_pandas()
    if not pd.api.types.is_numeric_dtype(series):
        return True
    values = series.dropna().unique()
    if len(values) <= 20 and np.allclose(values, np.round(values)):
        return True
    return False


def _default_colormap_for_column(series, color_by=None):
    if _is_categorical(series, color_by):
        return "tab10"
    return "viridis"


def _colors_for_column(series, cmap, color_by=None):
    pd = _require_pandas()
    if _is_categorical(series, color_by):
        categories = sorted(series.dropna().unique(), key=str)
        by_category = _category_color_lookup(categories, cmap)
        colors = [by_category.get(value, "0.55") for value in series]
        return colors, None, None

    values = pd.to_numeric(series, errors="coerce")
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        norm = plt.Normalize(0.0, 1.0)
    else:
        norm = plt.Normalize(finite_values.min(), finite_values.max())
    color_map = plt.get_cmap(cmap)
    colors = [
        color_map(norm(value)) if np.isfinite(value) else "0.55"
        for value in values
    ]
    return colors, values, norm


def _add_color_legend_or_bar(fig, ax, scatter, df, color_by, cmap):
    if _is_categorical(df[color_by], color_by):
        import matplotlib.lines as mlines

        categories = sorted(df[color_by].dropna().unique(), key=str)
        by_category = _category_color_lookup(categories, cmap)
        handles = [
            mlines.Line2D(
                [],
                [],
                color=by_category[category],
                marker="o",
                linestyle="None",
                markersize=5,
                label="%s" % category,
            )
            for category in categories[:16]
        ]
        if len(categories) > 16:
            handles.append(
                mlines.Line2D(
                    [], [], color="none", label="+%i more" % (len(categories) - 16)
                )
            )
        if handles:
            ax.legend(handles=handles, title=color_by, loc="lower left", fontsize=8)
        return
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label(color_by)


def _category_color_lookup(categories, cmap):
    """
    Return one discrete color assignment per category.
    """
    color_map = plt.get_cmap(cmap)
    if len(categories) <= getattr(color_map, "N", 256):
        return {
            category: color_map(index % color_map.N)
            for index, category in enumerate(categories)
        }

    sampled_map = plt.get_cmap("nipy_spectral")
    return {
        category: sampled_map((index + 0.5) / len(categories))
        for index, category in enumerate(categories)
    }


def _add_base_map(ax, df, stations, projection, transform, basemap, osm_zoom):
    extent = _data_extent(df, stations)
    if projection is None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        return
    ax.set_extent(extent, crs=transform)
    if basemap == "none":
        return
    if basemap == "osm":
        try:
            import cartopy.io.img_tiles as cimgt

            ax.add_image(cimgt.OSM(), osm_zoom)
        except Exception:
            pass
        return
    try:
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.LAND, facecolor="0.96")
        ax.add_feature(cfeature.OCEAN, facecolor="0.90")
        ax.add_feature(cfeature.BORDERS, linewidth=0.8)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor="0.45", color='blue')
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
    except Exception:
        pass


def _add_roads(ax, projection, transform, resolution="10m"):
    """
    Add Natural Earth roads when Cartopy is active and the dataset is available.
    """
    if projection is None:
        return
    try:
        import cartopy.io.shapereader as shpreader

        roads_shp = shpreader.natural_earth(
            resolution=resolution,
            category="cultural",
            name="roads",
        )
        roads = shpreader.Reader(roads_shp)
        ax.add_geometries(
            roads.geometries(),
            crs=transform,
            facecolor="none",
            edgecolor="gray",
            alpha=0.5,
            linewidth=0.5,
            zorder=1.5,
        )
    except Exception:
        return


def _data_extent(df, stations):
    lats = list(df["original_latitude"]) + list(df["relocated_latitude"])
    lons = list(df["original_longitude"]) + list(df["relocated_longitude"])
    if stations is not None and len(stations):
        lats += list(stations["latitude"])
        lons += list(stations["longitude"])
    lat_pad = max(0.25, (max(lats) - min(lats)) * 0.12)
    lon_pad = max(0.25, (max(lons) - min(lons)) * 0.12)
    return [
        min(lons) - lon_pad,
        max(lons) + lon_pad,
        min(lats) - lat_pad,
        max(lats) + lat_pad,
    ]


def _add_cities(ax, df, stations, transform):
    extent = _data_extent(df, stations)
    for city, lat, lon in UKRAINE_CITIES:
        if not (extent[0] <= lon <= extent[1] and extent[2] <= lat <= extent[3]):
            continue
        ax.plot(
            lon,
            lat,
            marker="o",
            markersize=2.5,
            color="black",
            **_transform_kwargs(transform),
        )
        ax.text(
            lon + 0.12,
            lat + 0.08,
            city,
            fontsize=7,
            **_transform_kwargs(transform),
        )


def _transform_kwargs(transform):
    if transform is None:
        return {}
    return {"transform": transform}


def _nice_scale_length_km(width_km):
    target = width_km / 5.0
    if target <= 0 or not math.isfinite(target):
        return 1.0
    exponent = math.floor(math.log10(target))
    fraction = target / (10 ** exponent)
    if fraction < 1.5:
        nice = 1.0
    elif fraction < 3.5:
        nice = 2.0
    elif fraction < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    return nice * (10 ** exponent)


def _current_extent_lonlat(ax, transform):
    if transform is None:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]
    try:
        return ax.get_extent(crs=transform)
    except Exception:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return [min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1)]


def _remove_scale_bar_artists(ax):
    for artist in list(getattr(ax, "_hypodd_scale_bar_artists", [])):
        try:
            artist.remove()
        except ValueError:
            pass
    ax._hypodd_scale_bar_artists = []


def _draw_scale_bar(ax, transform):
    _remove_scale_bar_artists(ax)
    lon_min, lon_max, lat_min, lat_max = _current_extent_lonlat(ax, transform)
    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span <= 0 or lon_span <= 0:
        return

    bar_lat = lat_min + 0.08 * lat_span
    bar_lon = lon_min + 0.08 * lon_span
    km_per_degree_lon = 111.32 * math.cos(math.radians(bar_lat))
    if abs(km_per_degree_lon) < 1e-6:
        return
    width_km = lon_span * km_per_degree_lon
    bar_km = _nice_scale_length_km(width_km)
    bar_degrees = bar_km / km_per_degree_lon
    if bar_degrees > 0.6 * lon_span:
        bar_km = _nice_scale_length_km(width_km / 2.0)
        bar_degrees = bar_km / km_per_degree_lon

    y_offset = 0.012 * lat_span
    text_y = bar_lat + 0.018 * lat_span
    kwargs = _transform_kwargs(transform)
    line = ax.plot(
        [bar_lon, bar_lon + bar_degrees],
        [bar_lat, bar_lat],
        color="black",
        linewidth=3.0,
        solid_capstyle="butt",
        zorder=20,
        **kwargs,
    )[0]
    tick_1 = ax.plot(
        [bar_lon, bar_lon],
        [bar_lat - y_offset, bar_lat + y_offset],
        color="black",
        linewidth=2.0,
        zorder=20,
        **kwargs,
    )[0]
    tick_2 = ax.plot(
        [bar_lon + bar_degrees, bar_lon + bar_degrees],
        [bar_lat - y_offset, bar_lat + y_offset],
        color="black",
        linewidth=2.0,
        zorder=20,
        **kwargs,
    )[0]
    label = ax.text(
        bar_lon + 0.5 * bar_degrees,
        text_y,
        "%g km" % bar_km,
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        zorder=21,
        **kwargs,
    )
    ax._hypodd_scale_bar_artists = [line, tick_1, tick_2, label]


def _add_dynamic_scale_bar(ax, transform):
    _draw_scale_bar(ax, transform)

    def refresh_scale_bar(_):
        _draw_scale_bar(ax, transform)
        if ax.figure.canvas is not None:
            ax.figure.canvas.draw_idle()

    for signal in ["xlim_changed", "ylim_changed"]:
        ax.callbacks.connect(signal, refresh_scale_bar)


def _format_axes(ax, projection):
    if projection is not None:
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        return
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
