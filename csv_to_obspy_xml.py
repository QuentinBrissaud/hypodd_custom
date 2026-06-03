#!/usr/bin/env python
"""
Convert Malyn event/station/pick CSV files to ObsPy QuakeML and StationXML.

The script creates two ObsPy objects:
    * Catalog, written as QuakeML, containing events, origins, magnitudes, picks
    * Inventory, written as StationXML, containing networks and stations
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from obspy import UTCDateTime
from obspy.core.event import (
    Catalog,
    CreationInfo,
    Event,
    Magnitude,
    Origin,
    Pick,
    QuantityError,
    ResourceIdentifier,
    WaveformStreamID,
)
from obspy.core.inventory import Inventory, Network, Site, Station


DEFAULT_EVENT_CSV = (
    r"O:\Staff\quentin\Documents\Projects\2026_hypodd\data"
    r"\starting-event-Malyn.csv"
)
DEFAULT_STATION_CSV = (
    r"O:\Staff\quentin\Documents\Projects\2026_hypodd\data"
    r"\starting-station-Malyn.csv"
)
DEFAULT_PHASE_CSV = (
    r"O:\Staff\quentin\Documents\Projects\2026_hypodd\data"
    r"\starting-phase-Malyn.csv"
)


def _read_csv(filename):
    with open(filename, "r", newline="", encoding="utf-8-sig") as open_file:
        return list(csv.DictReader(open_file))


def _dataframe_rows(dataframe):
    return dataframe.to_dict(orient="records")


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _clean_str(value):
    if _is_missing(value):
        return ""
    return str(value).strip()


def _to_utcdatetime(value):
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    return UTCDateTime(value)


def _float_or_none(value):
    if _is_missing(value) or value == "":
        return None
    return float(value)


def _station_key(network_code, station_code):
    return "%s.%s" % (network_code or "", station_code or "")


def build_inventory_from_rows(station_rows):
    """Build an ObsPy Inventory from station rows."""
    stations_by_network = defaultdict(list)
    for row in station_rows:
        network_code = _clean_str(row["networkCode"])
        station_code = _clean_str(row["stationCode"])
        station = Station(
            code=station_code,
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            elevation=float(row.get("elevation") or 0.0),
            site=Site(name=station_code),
        )
        stations_by_network[network_code].append(station)

    networks = [
        Network(code=network_code, stations=stations)
        for network_code, stations in sorted(stations_by_network.items())
    ]
    return Inventory(networks=networks, source="csv_to_obspy_xml.py")


def build_inventory(station_csv):
    """Build an ObsPy Inventory from the station CSV."""
    return build_inventory_from_rows(_read_csv(station_csv))


def build_inventory_from_dataframe(station_df):
    """Build an ObsPy Inventory from a pandas station DataFrame."""
    return build_inventory_from_rows(_dataframe_rows(station_df))


def _pick_uncertainty(row):
    lower = _float_or_none(row.get("lowerUncertainty"))
    upper = _float_or_none(row.get("upperUncertainty"))
    values = [value for value in (lower, upper) if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _build_pick(row, event_id, pick_number):
    network_code = _clean_str(row.get("networkCode")) or None
    station_code = _clean_str(row.get("stationCode")) or None
    location_code = _clean_str(row.get("locationCode")) or None
    channel_code = _clean_str(row.get("channelCode")) or None
    phase_hint = _clean_str(row["type"]).upper()

    pick = Pick(
        resource_id=ResourceIdentifier(
            "smi:local/event/%s/pick/%06d" % (event_id, pick_number)
        ),
        time=_to_utcdatetime(row["isotime"]),
        waveform_id=WaveformStreamID(
            network_code=network_code,
            station_code=station_code,
            location_code=location_code,
            channel_code=channel_code,
        ),
        phase_hint=phase_hint,
        evaluation_mode=_clean_str(row.get("evalMode")) or None,
    )
    uncertainty = _pick_uncertainty(row)
    if uncertainty is not None:
        pick.time_errors = QuantityError(uncertainty=uncertainty)
    return pick


def build_catalog_from_rows(event_rows, phase_rows, fixed_depth_km=None):
    """Build an ObsPy Catalog from event and phase rows."""
    phases_by_event = defaultdict(list)
    for row in phase_rows:
        phases_by_event[_clean_str(row["eventId"])].append(row)

    catalog = Catalog()
    catalog.creation_info = CreationInfo(author="csv_to_obspy_xml.py")

    for event_row in event_rows:
        event_id = _clean_str(event_row["id"])
        event = Event(
            resource_id=ResourceIdentifier("smi:local/event/%s" % event_id)
        )

        depth_km = (
            float(fixed_depth_km)
            if fixed_depth_km is not None
            else float(event_row.get("depth") or 0.0)
        )
        origin = Origin(
            resource_id=ResourceIdentifier(
                "smi:local/event/%s/origin/initial" % event_id
            ),
            time=_to_utcdatetime(event_row["isotime"]),
            latitude=float(event_row["latitude"]),
            longitude=float(event_row["longitude"]),
            depth=depth_km * 1000.0,
        )
        event.origins.append(origin)
        event.preferred_origin_id = origin.resource_id

        magnitude_value = _float_or_none(event_row.get("magnitude"))
        if magnitude_value is not None:
            magnitude = Magnitude(
                resource_id=ResourceIdentifier(
                    "smi:local/event/%s/magnitude/initial" % event_id
                ),
                mag=magnitude_value,
                origin_id=origin.resource_id,
            )
            event.magnitudes.append(magnitude)
            event.preferred_magnitude_id = magnitude.resource_id

        for pick_number, phase_row in enumerate(phases_by_event[event_id], 1):
            event.picks.append(_build_pick(phase_row, event_id, pick_number))

        catalog.events.append(event)

    return catalog


def build_catalog(event_csv, phase_csv, fixed_depth_km=None):
    """Build an ObsPy Catalog from event and phase CSV files."""
    return build_catalog_from_rows(
        _read_csv(event_csv),
        _read_csv(phase_csv),
        fixed_depth_km=fixed_depth_km,
    )


def build_catalog_from_dataframes(event_df, phase_df, fixed_depth_km=None):
    """Build an ObsPy Catalog from pandas event and phase DataFrames."""
    return build_catalog_from_rows(
        _dataframe_rows(event_df),
        _dataframe_rows(phase_df),
        fixed_depth_km=fixed_depth_km,
    )


def convert_csv_to_xml(
    event_csv,
    station_csv,
    phase_csv,
    event_xml,
    station_xml,
    fixed_depth_km=None,
):
    """Create Catalog/Inventory objects and write QuakeML/StationXML files."""
    catalog = build_catalog(event_csv, phase_csv, fixed_depth_km=fixed_depth_km)
    inventory = build_inventory(station_csv)

    Path(event_xml).parent.mkdir(parents=True, exist_ok=True)
    Path(station_xml).parent.mkdir(parents=True, exist_ok=True)
    catalog.write(str(event_xml), format="QUAKEML")
    inventory.write(str(station_xml), format="STATIONXML")
    return catalog, inventory


def convert_dataframes_to_xml(
    event_df,
    station_df,
    phase_df,
    event_xml,
    station_xml,
    fixed_depth_km=None,
):
    """Create ObsPy objects from pandas DataFrames and write XML files."""
    catalog = build_catalog_from_dataframes(
        event_df, phase_df, fixed_depth_km=fixed_depth_km
    )
    inventory = build_inventory_from_dataframe(station_df)

    Path(event_xml).parent.mkdir(parents=True, exist_ok=True)
    Path(station_xml).parent.mkdir(parents=True, exist_ok=True)
    catalog.write(str(event_xml), format="QUAKEML")
    inventory.write(str(station_xml), format="STATIONXML")
    return catalog, inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default=DEFAULT_EVENT_CSV)
    parser.add_argument("--stations", default=DEFAULT_STATION_CSV)
    parser.add_argument("--phases", default=DEFAULT_PHASE_CSV)
    parser.add_argument("--event-xml", default="events.xml")
    parser.add_argument("--station-xml", default="stations.xml")
    parser.add_argument(
        "--fixed-depth-km",
        type=float,
        default=None,
        help="Optional fixed depth in km to assign to every event.",
    )
    args = parser.parse_args()

    catalog, inventory = convert_csv_to_xml(
        event_csv=args.events,
        station_csv=args.stations,
        phase_csv=args.phases,
        event_xml=args.event_xml,
        station_xml=args.station_xml,
        fixed_depth_km=args.fixed_depth_km,
    )
    print("Wrote %i events to %s" % (len(catalog), args.event_xml))
    print("Wrote %i networks to %s" % (len(inventory), args.station_xml))


if __name__ == "__main__":
    main()
