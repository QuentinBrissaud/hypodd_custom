#!/usr/bin/env python
"""
Preview waveform cross-correlations for selected HypoDD event pairs.

This module is independent from HypoDD. It reads event-pair ids from a dt.ct
file, finds waveform files named like ``eventid_phase.mseed``, and plots:

    event 1 waveforms | cross-correlation curves | event 2 waveforms

Waveforms are plotted in one panel with vertical offsets. Cross-correlation is
computed trace-by-trace for matching station/channel ids.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np


def read_dt_event_pairs(dt_ct_path):
    """
    Read event-pair headers from a HypoDD dt.ct file.

    Returns a list of dictionaries with a zero-based ``pair_index``. Only header
    lines are needed here:

        # event_id_1 event_id_2 ...
    """
    pairs = []
    with Path(dt_ct_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 3 or parts[0] != "#":
                continue
            pairs.append(
                {
                    "pair_index": len(pairs),
                    "event_id_1": parts[1],
                    "event_id_2": parts[2],
                }
            )
    return pairs


def _waveform_path(waveform_dir, event_id, phase):
    path = Path(waveform_dir) / ("%s_%s.mseed" % (event_id, phase))
    if not path.exists():
        raise FileNotFoundError("Waveform file not found: %s" % path)
    return path


def read_event_phase_stream(waveform_dir, event_id, phase):
    """
    Read ``eventid_phase.mseed`` with ObsPy and return a Stream.
    """
    from obspy import read

    return read(str(_waveform_path(waveform_dir, event_id, phase)))


def _trace_key(trace):
    stats = trace.stats
    return (
        stats.network,
        stats.station,
        stats.location,
        stats.channel,
    )


def matching_trace_pairs(stream_1, stream_2):
    """
    Match traces by network, station, location, and channel.
    """
    traces_2 = defaultdict(list)
    for trace in stream_2:
        traces_2[_trace_key(trace)].append(trace)

    pairs = []
    for trace_1 in stream_1:
        key = _trace_key(trace_1)
        for trace_2 in traces_2.get(key, []):
            pairs.append((key, trace_1, trace_2))
    return pairs


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


def normalized_cross_correlation(trace_1, trace_2, max_lag_s=None):
    """
    Compute normalized cross-correlation versus lag.

    Positive lag means trace_2 is shifted later relative to trace_1.
    """
    if trace_1.stats.sampling_rate != trace_2.stats.sampling_rate:
        raise ValueError(
            "Sampling-rate mismatch for %s and %s" % (trace_1.id, trace_2.id)
        )

    data_1 = _preprocess_data(trace_1)
    data_2 = _preprocess_data(trace_2)
    n = min(len(data_1), len(data_2))
    if n < 3:
        raise ValueError("Need at least 3 samples to compute correlation.")
    data_1 = data_1[:n]
    data_2 = data_2[:n]

    corr = np.correlate(data_2, data_1, mode="full")
    norm = np.sqrt(np.sum(data_1 ** 2) * np.sum(data_2 ** 2))
    if norm > 0:
        corr = corr / norm
    lags_samples = np.arange(-n + 1, n)
    lags_s = lags_samples / float(trace_1.stats.sampling_rate)

    if max_lag_s is not None:
        keep = np.abs(lags_s) <= max_lag_s
        lags_s = lags_s[keep]
        corr = corr[keep]

    best_index = int(np.argmax(np.abs(corr)))
    return {
        "lags_s": lags_s,
        "correlation": corr,
        "best_lag_s": float(lags_s[best_index]),
        "best_correlation": float(corr[best_index]),
        "best_abs_correlation": float(abs(corr[best_index])),
    }


def compute_pair_correlations(
    dt_ct_path,
    waveform_dir,
    pair_index,
    phase="P",
    max_lag_s=None,
):
    """
    Compute cross-correlation curves for one dt.ct event-pair index.
    """
    pairs = read_dt_event_pairs(dt_ct_path)
    if pair_index < 0 or pair_index >= len(pairs):
        raise IndexError(
            "pair_index %s outside valid range 0..%s"
            % (pair_index, len(pairs) - 1)
        )

    pair = pairs[pair_index]
    stream_1 = read_event_phase_stream(waveform_dir, pair["event_id_1"], phase)
    stream_2 = read_event_phase_stream(waveform_dir, pair["event_id_2"], phase)
    trace_pairs = matching_trace_pairs(stream_1, stream_2)

    results = []
    for key, trace_1, trace_2 in trace_pairs:
        try:
            corr = normalized_cross_correlation(
                trace_1,
                trace_2,
                max_lag_s=max_lag_s,
            )
        except Exception as exc:
            results.append(
                {
                    "key": key,
                    "trace_1": trace_1,
                    "trace_2": trace_2,
                    "error": str(exc),
                }
            )
            continue
        results.append(
            {
                "key": key,
                "trace_1": trace_1,
                "trace_2": trace_2,
                **corr,
            }
        )

    return {
        "pair": pair,
        "phase": phase,
        "stream_1": stream_1,
        "stream_2": stream_2,
        "results": results,
    }


def _time_axis(trace):
    return np.arange(trace.stats.npts, dtype=float) / trace.stats.sampling_rate


def _plot_stream_with_offsets(ax, traces, title, max_traces=None):
    traces = list(traces)
    if max_traces is not None:
        traces = traces[:max_traces]
    offset_step = 1.4
    for index, trace in enumerate(traces):
        data = _preprocess_data(trace)
        if data.size == 0:
            continue
        offset = index * offset_step
        ax.plot(_time_axis(trace)[: len(data)], data + offset, linewidth=0.8)
        ax.text(
            1.01,
            offset,
            trace.id,
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            va="center",
        )
    ax.set_title(title)
    ax.set_xlabel("time since trace start (s)")
    ax.set_yticks([])


def plot_pair_correlations(
    pair_correlation,
    max_traces=None,
    sort_by_abs_correlation=True,
    title=None,
    save_path=None,
):
    """
    Plot event 1 waveforms, correlation curves, and event 2 waveforms.
    """
    import matplotlib.pyplot as plt

    pair = pair_correlation["pair"]
    phase = pair_correlation["phase"]
    results = [
        result for result in pair_correlation["results"] if "correlation" in result
    ]
    if sort_by_abs_correlation:
        results = sorted(
            results,
            key=lambda item: item["best_abs_correlation"],
            reverse=True,
        )
    if max_traces is not None:
        results = results[:max_traces]

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, 0.35 * len(results) + 2)))
    _plot_stream_with_offsets(
        axes[0],
        [result["trace_1"] for result in results],
        "Event %s %s waveforms" % (pair["event_id_1"], phase),
        max_traces=max_traces,
    )

    offset_step = 1.4
    for index, result in enumerate(results):
        offset = index * offset_step
        axes[1].plot(
            result["lags_s"],
            result["correlation"] + offset,
            linewidth=0.8,
        )
        axes[1].axvline(
            result["best_lag_s"],
            color="0.4",
            linewidth=0.4,
            alpha=0.5,
        )
        label = "%s.%s.%s.%s  lag=%.3fs  cc=%.3f" % (
            result["key"][0],
            result["key"][1],
            result["key"][2],
            result["key"][3],
            result["best_lag_s"],
            result["best_correlation"],
        )
        axes[1].text(
            1.01,
            offset,
            label,
            transform=axes[1].get_yaxis_transform(),
            fontsize=7,
            va="center",
        )
    axes[1].axvline(0.0, color="black", linewidth=0.6)
    axes[1].set_title("Normalized cross-correlation")
    axes[1].set_xlabel("lag (s)")
    axes[1].set_yticks([])

    _plot_stream_with_offsets(
        axes[2],
        [result["trace_2"] for result in results],
        "Event %s %s waveforms" % (pair["event_id_2"], phase),
        max_traces=max_traces,
    )

    if title is None:
        title = "dt.ct pair %s: event %s vs %s, phase %s" % (
            pair["pair_index"],
            pair["event_id_1"],
            pair["event_id_2"],
            phase,
        )
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    return fig, axes


def preview_pair_indexes(
    dt_ct_path,
    waveform_dir,
    pair_indexes,
    phase="P",
    max_lag_s=None,
    max_traces=20,
    output_dir=None,
):
    """
    Compute and plot several pair indexes.

    Returns a list of dictionaries with pair metadata, correlation summaries,
    and matplotlib figures.
    """
    outputs = []
    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for pair_index in pair_indexes:
        result = compute_pair_correlations(
            dt_ct_path,
            waveform_dir,
            pair_index,
            phase=phase,
            max_lag_s=max_lag_s,
        )
        save_path = None
        if output_dir is not None:
            pair = result["pair"]
            save_path = output_dir / (
                "pair_%06d_%s_%s_%s.png"
                % (pair_index, pair["event_id_1"], pair["event_id_2"], phase)
            )
        fig, axes = plot_pair_correlations(
            result,
            max_traces=max_traces,
            save_path=save_path,
        )
        summaries = []
        for item in result["results"]:
            if "correlation" not in item:
                summaries.append(
                    {
                        "key": item["key"],
                        "error": item.get("error"),
                    }
                )
                continue
            summaries.append(
                {
                    "key": item["key"],
                    "best_lag_s": item["best_lag_s"],
                    "best_correlation": item["best_correlation"],
                    "best_abs_correlation": item["best_abs_correlation"],
                }
            )
        outputs.append(
            {
                "pair": result["pair"],
                "phase": phase,
                "summaries": summaries,
                "figure": fig,
                "axes": axes,
                "save_path": str(save_path) if save_path is not None else None,
            }
        )
    return outputs
