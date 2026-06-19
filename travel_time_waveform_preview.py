#!/usr/bin/env python
"""
Preview observed and predicted arrivals on waveform snippets.

This module reads HypoDD travel-time diagnostics such as:

    hypoDD.initial.tt
    hypoDD.final.tt

and overlays arrival markers on event waveform files named like:

    eventid_P.mseed
    eventid_S.mseed

The HypoDD event ids in .tt and phase.dat are internal numeric ids. If your
waveform filenames use QuakeML event ids, pass the same events.xml used by
hypoddpy; the mapping is reconstructed by sorting events by origin time.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import re

import numpy as np


def default_filename_event_id(event_id):
    """Convert an event id to a filesystem-friendly filename token."""
    event_id = str(event_id)
    event_id = event_id.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", event_id)


def read_event_xml_internal_id_map(event_xml):
    """Reconstruct hypoddpy's internal numeric id -> QuakeML event id mapping."""
    from obspy import read_events

    catalog = read_events(str(event_xml))
    rows = []
    for event in catalog:
        origin = event.preferred_origin() or (
            event.origins[0] if event.origins else None
        )
        if origin is None:
            continue
        rows.append((origin.time, str(event.resource_id)))
    rows.sort(key=lambda item: item[0])
    return {str(index + 1): event_id for index, (_, event_id) in enumerate(rows)}


def read_event_xml_pick_uncertainties(event_xml, id_map=None):
    """
    Read pick time uncertainties from QuakeML.

    Returns rows keyed by HypoDD internal event id when ``id_map`` is supplied,
    otherwise keyed by the QuakeML event resource id.
    """
    from obspy import read_events

    catalog = read_events(str(event_xml))
    reverse_id_map = {}
    if id_map is not None:
        reverse_id_map = {str(value): str(key) for key, value in id_map.items()}

    uncertainties = defaultdict(list)
    for event in catalog:
        catalog_event_id = str(event.resource_id)
        event_id = reverse_id_map.get(catalog_event_id, catalog_event_id)
        for pick in event.picks:
            waveform_id = pick.waveform_id
            if waveform_id is None:
                continue
            station_code = waveform_id.station_code
            network_code = waveform_id.network_code
            if station_code is None:
                continue
            if network_code:
                station_id = "%s.%s" % (network_code, station_code)
            else:
                station_id = station_code
            uncertainty = None
            if pick.time_errors is not None:
                uncertainty = pick.time_errors.uncertainty
            if uncertainty is None:
                continue
            phase = (pick.phase_hint or "").upper()
            uncertainties[str(event_id)].append(
                {
                    "station_id": station_id,
                    "phase": phase,
                    "pick_time": pick.time,
                    "uncertainty_s": float(uncertainty),
                }
            )
    return uncertainties


def read_phase_dat(path):
    """
    Read hypoddpy/HypoDD phase.dat.

    Pick travel times are seconds after the original event origin time.
    """
    events = {}
    current = None
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "#":
                event_id = str(int(parts[-1]))
                second = float(parts[6])
                whole_second = int(second)
                microsecond = int(round((second - whole_second) * 1_000_000))
                current = {
                    "event_id": event_id,
                    "origin_time": datetime(
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
                        "travel_time_s": float(parts[1]),
                        "weight": float(parts[2]),
                        "phase": parts[3].upper(),
                    }
                )
    return events


def _hypodd_time(parts, first_index):
    second = float(parts[first_index + 5])
    whole_second = int(second)
    microsecond = int(round((second - whole_second) * 1_000_000))
    return datetime(
        int(parts[first_index]),
        int(parts[first_index + 1]),
        int(parts[first_index + 2]),
        int(parts[first_index + 3]),
        int(parts[first_index + 4]),
        whole_second,
        microsecond,
    )


def read_hypodd_locations(path):
    """Read hypoDD.loc or hypoDD.reloc and return rows keyed by internal id."""
    events = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 17:
                continue
            event_id = str(int(parts[0]))
            events[event_id] = {
                "event_id": event_id,
                "latitude": float(parts[1]),
                "longitude": float(parts[2]),
                "depth_km": float(parts[3]),
                "origin_time": _hypodd_time(parts, 10),
                "cluster_id": int(float(parts[23])) if len(parts) >= 24 else "",
            }
    return events


def read_hypodd_travel_times(path):
    """
    Read patched HypoDD travel-time diagnostics.

    Returns ``travel_times[(event_id, station_id, phase)] = travel_time_s``.
    """
    travel_times = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) < 4:
                continue
            event_id = str(int(parts[0]))
            station_id = parts[1]
            travel_times[(event_id, station_id, "P")] = float(parts[2])
            travel_times[(event_id, station_id, "S")] = float(parts[3])
    return travel_times


def _resolve_event_id(internal_event_id, id_map=None):
    if id_map is None:
        return str(internal_event_id)
    return str(id_map.get(str(internal_event_id), internal_event_id))


def _waveform_path(waveform_dir, event_id, phase, filename_event_id=None):
    if filename_event_id is None:
        filename_event_id = default_filename_event_id
    file_event_id = filename_event_id(event_id)
    path = Path(waveform_dir) / ("%s_%s.mseed" % (file_event_id, phase))
    if not path.exists():
        raise FileNotFoundError("Waveform file not found: %s" % path)
    return path


def read_event_phase_stream(
    waveform_dir,
    event_id,
    phase,
    filename_event_id=None,
):
    """Read ``eventid_phase.mseed`` with ObsPy and return a Stream."""
    from obspy import read

    return read(str(_waveform_path(waveform_dir, event_id, phase, filename_event_id)))


def _trace_station_id(trace):
    if trace.stats.network:
        return "%s.%s" % (trace.stats.network, trace.stats.station)
    return trace.stats.station


def _station_aliases(station_id):
    station_id = str(station_id)
    aliases = {station_id}
    if "." in station_id:
        aliases.add(station_id.split(".")[-1])
    return aliases


def _arrival_lookup_keys(station_id):
    aliases = _station_aliases(station_id)
    keys = set(aliases)
    for alias in aliases:
        if "." not in alias:
            keys.add(alias)
    return keys


def _preprocess_data(trace, demean=True, normalize=True):
    data = np.asarray(trace.data, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return data
    if demean:
        data = data - np.mean(data)
    if normalize:
        scale = np.max(np.abs(data))
        if scale > 0:
            data = data / scale
    return data


def _seconds_from_trace_start(trace, absolute_time):
    from obspy import UTCDateTime

    return float(UTCDateTime(absolute_time) - trace.stats.starttime)


def _seconds_from_reference(reference_time, absolute_time):
    from obspy import UTCDateTime

    return float(UTCDateTime(absolute_time) - UTCDateTime(reference_time))


def _add_seconds(origin_time, seconds):
    from datetime import timedelta

    return origin_time + timedelta(seconds=float(seconds))


def _time_difference_seconds(time_1, time_2):
    from obspy import UTCDateTime

    return float(UTCDateTime(time_1) - UTCDateTime(time_2))


def _attach_pick_uncertainties(arrivals, uncertainty_rows, max_time_difference_s=0.05):
    for arrival in arrivals:
        best_row = None
        best_difference = None
        for row in uncertainty_rows:
            if row["station_id"] != arrival["station_id"]:
                continue
            if row["phase"] and row["phase"] != arrival["phase"]:
                continue
            difference = abs(
                _time_difference_seconds(row["pick_time"], arrival["observed_pick_time"])
            )
            if best_difference is None or difference < best_difference:
                best_row = row
                best_difference = difference
        if best_row is not None and best_difference <= max_time_difference_s:
            arrival["pick_time_uncertainty_s"] = best_row["uncertainty_s"]
        else:
            arrival["pick_time_uncertainty_s"] = None
    return arrivals


def event_arrival_rows(
    internal_event_id,
    phase,
    phase_events,
    locations,
    travel_times,
):
    """
    Build observed/predicted arrival rows for one event and phase.
    """
    internal_event_id = str(internal_event_id)
    phase = phase.upper()
    phase_event = phase_events.get(internal_event_id)
    location = locations.get(internal_event_id)
    if phase_event is None:
        raise KeyError("Event %s not found in phase.dat" % internal_event_id)
    if location is None:
        raise KeyError("Event %s not found in location file" % internal_event_id)

    rows = []
    for pick in phase_event["picks"]:
        if pick["phase"] != phase:
            continue
        theoretical = travel_times.get(
            (internal_event_id, pick["station_id"], phase)
        )
        if theoretical is None:
            continue
        observed_time = _add_seconds(
            phase_event["origin_time"], pick["travel_time_s"]
        )
        predicted_time = _add_seconds(location["origin_time"], theoretical)
        rows.append(
            {
                "event_id": internal_event_id,
                "station_id": pick["station_id"],
                "phase": phase,
                "observed_pick_time": observed_time,
                "predicted_arrival_time": predicted_time,
                "observed_travel_time_s": pick["travel_time_s"],
                "predicted_travel_time_s": theoretical,
                "residual_s": (observed_time - predicted_time).total_seconds(),
            }
        )
    return rows


def _arrival_by_station(rows):
    grouped = defaultdict(list)
    for row in rows:
        for key in _station_aliases(row["station_id"]):
            grouped[key].append(row)
    return grouped


def plot_event_waveforms_with_arrivals(
    internal_event_id,
    phase,
    tt_path,
    phase_dat_path,
    location_path,
    waveform_dir,
    event_xml=None,
    id_map=None,
    filename_event_id=None,
    max_traces=30,
    title=None,
    save_path=None,
    show_pick_uncertainty=True,
):
    """
    Plot one event/phase waveform file with observed and predicted arrivals.

    ``location_path`` should correspond to the travel-time file:
    use ``hypoDD.loc`` with ``hypoDD.initial.tt`` and ``hypoDD.reloc`` with
    ``hypoDD.final.tt``.
    """
    import matplotlib.pyplot as plt

    if id_map is None and event_xml is not None:
        id_map = read_event_xml_internal_id_map(event_xml)

    internal_event_id = str(internal_event_id)
    catalog_event_id = _resolve_event_id(internal_event_id, id_map)
    phase = phase.upper()

    phase_events = read_phase_dat(phase_dat_path)
    locations = read_hypodd_locations(location_path)
    travel_times = read_hypodd_travel_times(tt_path)
    arrivals = event_arrival_rows(
        internal_event_id,
        phase,
        phase_events,
        locations,
        travel_times,
    )
    if show_pick_uncertainty and event_xml is not None:
        uncertainty_rows = read_event_xml_pick_uncertainties(event_xml, id_map)
        arrivals = _attach_pick_uncertainties(
            arrivals, uncertainty_rows.get(internal_event_id, [])
        )
    arrivals_by_station = _arrival_by_station(arrivals)

    stream = read_event_phase_stream(
        waveform_dir,
        catalog_event_id,
        phase,
        filename_event_id=filename_event_id,
    )

    traces = list(stream)
    if max_traces is not None:
        traces = traces[:max_traces]

    reference_time = min(trace.stats.starttime for trace in traces)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(traces) + 2)))
    offset_step = 1.4
    observed_label_used = False
    uncertainty_label_used = False
    predicted_label_used = False
    diagnostics = {
        "event_id": internal_event_id,
        "catalog_event_id": catalog_event_id,
        "phase": phase,
        "arrival_rows": len(arrivals),
        "trace_count": len(traces),
        "traces_with_matching_arrivals": 0,
        "observed_markers_plotted": 0,
        "predicted_markers_plotted": 0,
        "observed_markers_outside_window": 0,
        "predicted_markers_outside_window": 0,
        "trace_station_ids": [],
        "arrival_station_ids": sorted({row["station_id"] for row in arrivals}),
    }
    for index, trace in enumerate(traces):
        data = _preprocess_data(trace)
        if data.size == 0:
            continue
        offset = index * offset_step
        trace_start_offset = _seconds_from_reference(
            reference_time, trace.stats.starttime
        )
        time_axis = (
            trace_start_offset
            + np.arange(len(data), dtype=float) / trace.stats.sampling_rate
        )
        ax.plot(time_axis, data + offset, color="black", linewidth=0.8)

        station_id = _trace_station_id(trace)
        diagnostics["trace_station_ids"].append(station_id)
        matching_arrivals = []
        for key in _arrival_lookup_keys(station_id):
            matching_arrivals.extend(arrivals_by_station.get(key, []))
        if len(matching_arrivals) > 0:
            diagnostics["traces_with_matching_arrivals"] += 1
        seen_arrivals = set()
        for arrival in matching_arrivals:
            arrival_key = (
                arrival["station_id"],
                arrival["phase"],
                arrival["observed_pick_time"],
                arrival["predicted_arrival_time"],
            )
            if arrival_key in seen_arrivals:
                continue
            seen_arrivals.add(arrival_key)
            observed_x = _seconds_from_trace_start(
                trace, arrival["observed_pick_time"]
            ) + trace_start_offset
            predicted_x = _seconds_from_trace_start(
                trace, arrival["predicted_arrival_time"]
            ) + trace_start_offset
            if 0 <= observed_x <= time_axis[-1]:
                diagnostics["observed_markers_plotted"] += 1
                uncertainty = arrival.get("pick_time_uncertainty_s")
                if (
                    show_pick_uncertainty
                    and uncertainty is not None
                    and uncertainty > 0
                ):
                    sigma = float(uncertainty)
                    gaussian_x = np.linspace(
                        observed_x - 3.0 * sigma,
                        observed_x + 3.0 * sigma,
                        80,
                    )
                    gaussian = np.exp(
                        -0.5 * ((gaussian_x - observed_x) / sigma) ** 2
                    )
                    gaussian_y = offset + 0.55 * gaussian
                    ax.plot(
                        gaussian_x,
                        gaussian_y,
                        color="tab:blue",
                        linewidth=1.1,
                        label=(
                            "observed pick uncertainty"
                            if not uncertainty_label_used
                            else None
                        ),
                    )
                    ax.fill_between(
                        gaussian_x,
                        offset,
                        gaussian_y,
                        color="tab:blue",
                        alpha=0.18,
                    )
                    uncertainty_label_used = True
                else:
                    ax.plot(
                        [observed_x, observed_x],
                        [offset - 0.55, offset + 0.55],
                        color="tab:blue",
                        linewidth=1.1,
                        label=(
                            "observed pick" if not observed_label_used else None
                        ),
                    )
                    observed_label_used = True
            else:
                diagnostics["observed_markers_outside_window"] += 1
            if 0 <= predicted_x <= time_axis[-1]:
                diagnostics["predicted_markers_plotted"] += 1
                ax.plot(
                    [predicted_x, predicted_x],
                    [offset - 0.55, offset + 0.55],
                    color="tab:red",
                    linewidth=1.1,
                    linestyle="--",
                    label=(
                        "HypoDD predicted arrival"
                        if not predicted_label_used
                        else None
                    ),
                )
                predicted_label_used = True
            else:
                diagnostics["predicted_markers_outside_window"] += 1

        ax.text(
            1.01,
            offset,
            trace.id,
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            va="center",
        )

    if title is None:
        title = "Event %s (%s), phase %s, %s" % (
            internal_event_id,
            catalog_event_id,
            phase,
            Path(tt_path).name,
        )
    ax.set_title(title)
    ax.set_xlabel("time since earliest trace start (s)")
    ax.set_yticks([])
    if observed_label_used or predicted_label_used:
        ax.legend(loc="upper right")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return {
        "figure": fig,
        "axes": ax,
        "arrivals": arrivals,
        "catalog_event_id": catalog_event_id,
        "diagnostics": diagnostics,
    }


def preview_event_arrivals(
    internal_event_ids,
    phases,
    tt_path,
    phase_dat_path,
    location_path,
    waveform_dir,
    event_xml=None,
    id_map=None,
    filename_event_id=None,
    max_traces=30,
    output_dir=None,
    show_pick_uncertainty=True,
):
    """
    Plot several events/phases with observed and predicted arrival markers.
    """
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(phases, str):
        phases = [phases]
    if id_map is None and event_xml is not None:
        id_map = read_event_xml_internal_id_map(event_xml)

    outputs = []
    for event_id in internal_event_ids:
        for phase in phases:
            save_path = None
            if output_dir is not None:
                save_path = output_dir / (
                    "event_%s_%s_%s.png"
                    % (event_id, phase.upper(), Path(tt_path).stem)
                )
            outputs.append(
                plot_event_waveforms_with_arrivals(
                    event_id,
                    phase,
                    tt_path,
                    phase_dat_path,
                    location_path,
                    waveform_dir,
                    event_xml=event_xml,
                    id_map=id_map,
                    filename_event_id=filename_event_id,
                    max_traces=max_traces,
                    save_path=save_path,
                    show_pick_uncertainty=show_pick_uncertainty,
                )
            )
    return outputs


def plot_travel_time_waveform_preview(
    working_dir,
    waveform_dir,
    internal_event_ids,
    phases=("P", "S"),
    dataset="final",
    event_xml=None,
    filename_event_id=None,
    max_traces=30,
    output_dir=None,
    show_pick_uncertainty=True,
):
    """
    Convenience wrapper for plotting observed picks against HypoDD predictions.

    Parameters
    ----------
    working_dir
        HypoDDPy working directory containing ``input_files`` and
        ``output_files``.
    waveform_dir
        Directory containing waveform files named like ``eventid_P.mseed`` and
        ``eventid_S.mseed``.
    internal_event_ids
        HypoDD internal numeric event ids, as strings or integers.
    phases
        Phase or phases to plot.
    dataset
        ``"initial"`` compares against ``hypoDD.initial.tt`` and
        ``hypoDD.loc``. ``"final"`` compares against ``hypoDD.final.tt`` and
        ``hypoDD.reloc``.
    event_xml
        Optional QuakeML catalog used to map HypoDD internal ids back to the
        event ids used in waveform filenames.
    """
    working_dir = Path(working_dir)
    dataset = str(dataset).lower()
    if dataset == "initial":
        tt_path = working_dir / "output_files" / "hypoDD.initial.tt"
        location_path = working_dir / "output_files" / "hypoDD.loc"
    elif dataset in ("final", "relocated"):
        tt_path = working_dir / "output_files" / "hypoDD.final.tt"
        location_path = working_dir / "output_files" / "hypoDD.reloc"
    else:
        raise ValueError("dataset must be 'initial' or 'final'")

    phase_dat_path = working_dir / "input_files" / "phase.dat"
    return preview_event_arrivals(
        internal_event_ids=internal_event_ids,
        phases=phases,
        tt_path=tt_path,
        phase_dat_path=phase_dat_path,
        location_path=location_path,
        waveform_dir=waveform_dir,
        event_xml=event_xml,
        filename_event_id=filename_event_id,
        max_traces=max_traces,
        output_dir=output_dir,
        show_pick_uncertainty=show_pick_uncertainty,
    )
