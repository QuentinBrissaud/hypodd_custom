#!/usr/bin/env python
"""
Spatial clustering helpers for previewing event clusters before HypoDD.

The routines in this module are intentionally independent from HypoDD. They
cluster original epicenters only, so they are useful for quickly testing whether
spatial pre-clustering might avoid bridge-event cluster merges.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088


@dataclass
class ClusteringConfig:
    method: str = "dbscan"
    eps_km: float = 3.0
    min_samples: int = 8
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 8
    agglomerative_distance_threshold_km: float = 3.0
    agglomerative_linkage: str = "single"
    kmeans_n_clusters: int = 4
    kmeans_random_state: int = 42


def catalog_to_dataframe(path):
    """
    Load an ObsPy-readable event catalog into a dataframe.

    The returned dataframe has at least:
    event_id, event_index, time, latitude, longitude, depth_km, magnitude.
    """
    try:
        from obspy import read_events
    except ImportError as exc:
        raise ImportError(
            "ObsPy is required to read QuakeML. Install obspy or use "
            "csv_to_dataframe()."
        ) from exc

    catalog = read_events(str(path))
    rows = []
    for index, event in enumerate(catalog):
        origin = event.preferred_origin() or (
            event.origins[0] if event.origins else None
        )
        if origin is None or origin.latitude is None or origin.longitude is None:
            continue
        magnitude = event.preferred_magnitude()
        rows.append(
            {
                "event_id": str(event.resource_id),
                "event_index": index,
                "time": origin.time.datetime if origin.time else pd.NaT,
                "latitude": float(origin.latitude),
                "longitude": float(origin.longitude),
                "depth_km": float(origin.depth or 0.0) / 1000.0,
                "magnitude": (
                    float(magnitude.mag)
                    if magnitude and magnitude.mag is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def csv_to_dataframe(path):
    """Load a simple event CSV containing latitude and longitude columns."""
    df = pd.read_csv(path)
    lower_to_original = {column.lower(): column for column in df.columns}
    lat_col = lower_to_original.get("latitude") or lower_to_original.get("lat")
    lon_col = lower_to_original.get("longitude") or lower_to_original.get("lon")
    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not find latitude/longitude columns in %s. Columns: %s"
            % (path, list(df.columns))
        )
    event_id_col = (
        lower_to_original.get("id")
        or lower_to_original.get("eventid")
        or lower_to_original.get("event_id")
    )
    depth_col = lower_to_original.get("depth") or lower_to_original.get("depth_km")
    time_col = lower_to_original.get("isotime") or lower_to_original.get("time")
    out = pd.DataFrame(
        {
            "event_id": (
                df[event_id_col].astype(str) if event_id_col else df.index.astype(str)
            ),
            "event_index": np.arange(len(df)),
            "time": pd.to_datetime(df[time_col], errors="coerce")
            if time_col
            else pd.NaT,
            "latitude": pd.to_numeric(df[lat_col], errors="coerce"),
            "longitude": pd.to_numeric(df[lon_col], errors="coerce"),
            "depth_km": pd.to_numeric(df[depth_col], errors="coerce")
            if depth_col
            else np.nan,
        }
    )
    return out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)


def load_events(event_xml=None, event_csv=None):
    """
    Load events from QuakeML if available, otherwise from CSV.

    Returns ``(events_dataframe, source_path)``.
    """
    if event_xml is not None and Path(event_xml).exists():
        return catalog_to_dataframe(event_xml), Path(event_xml)
    if event_csv is None:
        raise ValueError("event_csv is required when event_xml is absent.")
    return csv_to_dataframe(event_csv), Path(event_csv)


def inventory_to_dataframe(path):
    """Load station coordinates from an ObsPy-readable StationXML file."""
    path = Path(path) if path is not None else None
    if path is None or not path.exists():
        return pd.DataFrame(columns=["station_id", "latitude", "longitude"])
    try:
        from obspy import read_inventory
    except ImportError:
        return pd.DataFrame(columns=["station_id", "latitude", "longitude"])

    inventory = read_inventory(str(path))
    rows = []
    for network in inventory:
        for station in network:
            rows.append(
                {
                    "station_id": "%s.%s" % (network.code, station.code),
                    "latitude": float(station.latitude),
                    "longitude": float(station.longitude),
                }
            )
    return pd.DataFrame(rows)


def local_xy_km(events_df):
    """Approximate lon/lat as local x/y in km around the event centroid."""
    lat = events_df["latitude"].to_numpy(dtype=float)
    lon = events_df["longitude"].to_numpy(dtype=float)
    lat0 = np.radians(np.nanmean(lat))
    x = np.radians(lon - np.nanmean(lon)) * EARTH_RADIUS_KM * np.cos(lat0)
    y = np.radians(lat - np.nanmean(lat)) * EARTH_RADIUS_KM
    return np.column_stack([x, y])


def run_clustering(events_df, config=None, **overrides):
    """
    Cluster event epicenters.

    Supported methods are ``dbscan``, ``hdbscan``, ``agglomerative``, and
    ``kmeans``. Returns ``(clustered_dataframe, description)``. The dataframe
    contains a ``cluster`` column. For methods with noise, cluster ``-1`` means
    unclustered/noise.
    """
    config = config or ClusteringConfig()
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError("Unknown clustering config key: %s" % key)
        setattr(config, key, value)

    method = config.method.lower()
    clustered = events_df.copy()

    if method == "dbscan":
        from sklearn.cluster import DBSCAN

        coords_rad = np.radians(events_df[["latitude", "longitude"]].to_numpy())
        model = DBSCAN(
            eps=config.eps_km / EARTH_RADIUS_KM,
            min_samples=config.min_samples,
            metric="haversine",
        )
        labels = model.fit_predict(coords_rad)
        description = "DBSCAN eps=%g km, min_samples=%i" % (
            config.eps_km,
            config.min_samples,
        )

    elif method == "hdbscan":
        try:
            import hdbscan
        except ImportError as exc:
            raise ImportError("HDBSCAN requires `pip install hdbscan`.") from exc
        coords_rad = np.radians(events_df[["latitude", "longitude"]].to_numpy())
        model = hdbscan.HDBSCAN(
            min_cluster_size=config.hdbscan_min_cluster_size,
            min_samples=config.hdbscan_min_samples,
            metric="haversine",
        )
        labels = model.fit_predict(coords_rad)
        description = "HDBSCAN min_cluster_size=%i, min_samples=%i" % (
            config.hdbscan_min_cluster_size,
            config.hdbscan_min_samples,
        )

    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        xy = local_xy_km(events_df)
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=config.agglomerative_distance_threshold_km,
            linkage=config.agglomerative_linkage,
        )
        labels = model.fit_predict(xy)
        description = "Agglomerative threshold=%g km, linkage=%s" % (
            config.agglomerative_distance_threshold_km,
            config.agglomerative_linkage,
        )

    elif method == "kmeans":
        from sklearn.cluster import KMeans

        xy = local_xy_km(events_df)
        model = KMeans(
            n_clusters=config.kmeans_n_clusters,
            random_state=config.kmeans_random_state,
            n_init=10,
        )
        labels = model.fit_predict(xy)
        description = "K-means n_clusters=%i" % config.kmeans_n_clusters

    else:
        raise ValueError("Unknown clustering method: %s" % config.method)

    clustered["cluster"] = labels
    clustered["clustering_method"] = method
    return clustered, description


def cluster_summary(clustered):
    """Return event counts per cluster."""
    counts = clustered["cluster"].value_counts().sort_index()
    rows = []
    for cluster_id, count in counts.items():
        rows.append(
            {
                "cluster": cluster_id,
                "event_count": int(count),
                "is_noise": cluster_id == -1,
            }
        )
    return pd.DataFrame(rows)


def dbscan_eps_sweep(events_df, eps_values_km, min_samples=8):
    """Run DBSCAN for several eps values and return clustered results/summary."""
    summaries = []
    clustered_by_eps = {}
    for eps_km in eps_values_km:
        config = ClusteringConfig(
            method="dbscan", eps_km=eps_km, min_samples=min_samples
        )
        clustered, _ = run_clustering(events_df, config)
        clustered_by_eps[eps_km] = clustered
        counts = clustered["cluster"].value_counts()
        summaries.append(
            {
                "eps_km": eps_km,
                "min_samples": min_samples,
                "clusters": int((counts.index != -1).sum()),
                "noise_events": int(counts.get(-1, 0)),
                "largest_cluster": (
                    int(counts[counts.index != -1].max())
                    if (counts.index != -1).any()
                    else 0
                ),
            }
        )
    return clustered_by_eps, pd.DataFrame(summaries)


def plot_cluster_map(clustered, stations=None, title=None, ax=None):
    """Plot event clusters and optional station locations."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    clusters = sorted(clustered["cluster"].unique())
    non_noise = [cluster for cluster in clusters if cluster != -1]
    cmap = plt.get_cmap("tab20")

    noise = clustered[clustered["cluster"] == -1]
    if len(noise):
        ax.scatter(
            noise["longitude"],
            noise["latitude"],
            marker="x",
            s=16,
            c="0.65",
            linewidths=0.8,
            label="Noise (%i)" % len(noise),
        )

    for index, cluster_id in enumerate(non_noise):
        subset = clustered[clustered["cluster"] == cluster_id]
        ax.scatter(
            subset["longitude"],
            subset["latitude"],
            marker="o",
            s=14,
            alpha=0.75,
            color=cmap(index % cmap.N),
            label="Cluster %s (%i)" % (cluster_id, len(subset)),
        )

    if stations is not None and len(stations):
        ax.scatter(
            stations["longitude"],
            stations["latitude"],
            marker="^",
            s=45,
            c="black",
            edgecolors="white",
            linewidths=0.5,
            label="Stations",
            zorder=5,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or "Spatial event clusters")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7, markerscale=1.2)
    return ax


def safe_output_name(description, prefix="cluster_preview"):
    """Build a filesystem-friendly CSV name from a clustering description."""
    safe_description = (
        description.lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("=", "-")
        .replace("/", "-")
    )
    return "%s_%s.csv" % (prefix, safe_description)
