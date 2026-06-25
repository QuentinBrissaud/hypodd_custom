import copy
import configparser
import fnmatch
import glob
import json
import logging
import math
import numpy as np
from obspy.core import read, Stream, UTCDateTime
from obspy.core.inventory import read_inventory
from obspy.core.event import (
    Catalog,
    Comment,
    Origin,
    read_events,
    ResourceIdentifier,
    Magnitude,
)
from obspy.signal.cross_correlation import xcorr_pick_correction
from obspy.io.xseed import Parser
import os
import progressbar
import shutil
import subprocess
import sys
import warnings

from .hypodd_compiler import HypoDDCompiler


# Global variable for StationXML or XSEED inventory files (WCC)
stations_XSEED = False


class HypoDDException(Exception):
    pass


def _normalize_cc_backend(cc_backend):
    cc_backend = str(cc_backend).strip().lower()
    if cc_backend in ["numpy", "fast", "fast-numpy", "fast_numpy"]:
        cc_backend = "numpy_cache"
    if cc_backend not in ["obspy", "numpy_cache"]:
        msg = "cc_backend must be 'obspy' or 'numpy_cache'."
        raise HypoDDException(msg)
    return cc_backend


def _numpy_cache_xcorr_pick_correction(
    pick_1_time,
    trace_1,
    pick_2_time,
    trace_2,
    t_before,
    t_after,
    cc_maxlag,
    filter=None,
    filter_options=None,
    trace_cache=None,
):
    """
    Vectorized integer-sample normalized cross correlation around two picks.

    Returns the time correction to add to pick_2 and the corresponding
    normalized correlation coefficient. This backend intentionally does not do
    sub-sample interpolation.
    """
    if trace_1.stats.sampling_rate != trace_2.stats.sampling_rate:
        raise ValueError("Sampling rates must match.")
    sampling_rate = float(trace_1.stats.sampling_rate)

    n_before = int(round(float(t_before) * sampling_rate))
    n_after = int(round(float(t_after) * sampling_rate))
    n_samples = n_before + n_after + 1
    max_lag_samples = int(round(float(cc_maxlag) * sampling_rate))
    if n_samples < 3:
        raise ValueError("Less than 3 samples selected for cross correlation.")
    if max_lag_samples < 0:
        raise ValueError("cc_maxlag must be non-negative.")

    data_1 = _preprocessed_trace_data(trace_1, filter, filter_options, trace_cache)
    data_2 = _preprocessed_trace_data(trace_2, filter, filter_options, trace_cache)
    if data_1.size == 0 or data_2.size == 0:
        raise ValueError("empty array")

    pick_1_index = int(round((pick_1_time - trace_1.stats.starttime) * sampling_rate))
    pick_2_index = int(round((pick_2_time - trace_2.stats.starttime) * sampling_rate))
    start_1 = pick_1_index - n_before
    end_1 = start_1 + n_samples
    if start_1 < 0 or end_1 > data_1.size:
        raise ValueError("empty array")
    window_1 = data_1[start_1:end_1].astype(np.float64)
    window_1 -= window_1.mean()
    norm_1 = np.linalg.norm(window_1)
    if norm_1 == 0.0 or not np.isfinite(norm_1):
        raise ValueError("empty array")

    lag_min = max(-max_lag_samples, n_before - pick_2_index)
    lag_max = min(max_lag_samples, data_2.size - pick_2_index - n_after - 1)
    if lag_min > lag_max:
        raise ValueError("empty array")

    segment_start = pick_2_index + lag_min - n_before
    segment_end = pick_2_index + lag_max + n_after + 1
    segment = data_2[segment_start:segment_end].astype(np.float64, copy=False)
    windows_2 = np.lib.stride_tricks.sliding_window_view(segment, n_samples)
    windows_2 = windows_2.astype(np.float64, copy=True)
    windows_2 -= windows_2.mean(axis=1, keepdims=True)
    norms_2 = np.linalg.norm(windows_2, axis=1)
    valid = np.isfinite(norms_2) & (norms_2 > 0.0)
    if not valid.any():
        raise ValueError("empty array")

    coefficients = np.full(windows_2.shape[0], -np.inf, dtype=np.float64)
    coefficients[valid] = windows_2[valid].dot(window_1) / (norms_2[valid] * norm_1)
    best_index = int(np.argmax(coefficients))
    best_coeff = float(coefficients[best_index])
    if not np.isfinite(best_coeff):
        raise ValueError("empty array")
    best_lag_samples = best_index + lag_min
    # Positive lag_samples means the matching waveform in trace_2 is later than
    # pick_2, so the correction added to pick_2 is positive.
    return best_lag_samples / sampling_rate, best_coeff


def _preprocessed_trace_data(trace, filter_name, filter_options, trace_cache):
    """
    Return a cached float64 trace array after the configured CC preprocessing.
    """
    if filter_name in [None, "none"]:
        filter_key = ("none",)
    elif filter_name == "bandpass":
        filter_options = filter_options or {}
        filter_key = (
            "bandpass",
            float(filter_options["freqmin"]),
            float(filter_options["freqmax"]),
        )
    else:
        raise ValueError(
            "Unsupported numpy_cache cross-correlation filter: %s" % filter_name
        )

    key = (
        getattr(trace.stats, "hypoddpy_source_file", None) or id(trace.data),
        trace.id,
        str(trace.stats.starttime),
        int(trace.stats.npts),
        float(trace.stats.sampling_rate),
        filter_key,
    )
    if trace_cache is not None and key in trace_cache:
        return trace_cache[key]

    if filter_name == "bandpass":
        filtered = trace.copy()
        filtered.detrend("demean")
        filtered.filter("bandpass", **filter_options)
        data = np.asarray(filtered.data, dtype=np.float64)
    else:
        data = np.asarray(trace.data, dtype=np.float64)

    if trace_cache is not None:
        trace_cache[key] = data
    return data


class HypoDDRelocator(object):
    def __init__(
        self,
        working_dir,
        cc_time_before=0.05,
        cc_time_after=0.2,
        cc_maxlag=0.1,
        cc_backend="obspy",
        cc_write_pair_files=True,
        cc_filter="bandpass",
        cc_filter_min_freq=1.0,
        cc_filter_max_freq=20.0,
        cc_p_phase_weighting=None,
        cc_s_phase_weighting=None,
        cc_min_allowed_cross_corr_coeff=0.4,
        supress_warning_traces=False,
        shift_stations=False,
        use_cross_correlation=True,
        event_fix=None,
        fixed_depth_km=None,
        enforce_mean_shift_constraint=False,
        mean_shift_constraint_weight=10.0,
        lsqr_constraint_weight=100.0,
        lsqr_xyz_constraint_weight=10.0,
        lsqr_time_constraint_weight=1000.0,
    ):
        """
        :param working_dir: The working directory where all temporary and final
            files will be placed.
        :param cc_time_before: Time to start cross correlation before pick time
            in seconds. Defaults to 0.05.
        :param cc_time_after: Time to start cross correlation after pick time
            in seconds. Defaults to 0.2.
        :param cc_maxlag: Maximum lag time tested during cross correlation.
            Defaults to 0.1.
        :param cc_backend: Cross-correlation implementation. Use "obspy" for
            ObsPy's xcorr_pick_correction or "numpy_cache" for cached waveform
            reads plus vectorized integer-sample correlation. Defaults to
            "obspy".
        :param cc_write_pair_files: If True, write one intermediate text file
            per event pair. If False, write only the final dt.cc file, which is
            faster on filesystems where many tiny files are expensive.
        :param cc_filter: Filter to apply during cross correlation. Use
            "bandpass" or "none". Defaults to "bandpass".
        :param cc_filter_min_freq: Lower corner frequency for the Butterworth
            bandpass filter to be applied during cross correlation.
            Defaults to 1.0.
        :param cc_filter_max_freq: Upper corner frequency for the Butterworth
            bandpass filter to be applied during cross correlation.
            Defaults to 20.0.
        :param cc_p_phase_weighting: The cross correlation travel time
            differences can be calculated on several channels. This dict
            specified which channels to calculate it for and how to weight the
            channels to determine the final cross correlated traveltime between
            two events. This assumes the waveform data adheres to the SEED
            naming convention.
            This dict applies to all P phase picks. Defaults to {"Z": 1.0}.
            Examples:
                {"Z": 1.0} - Only use the vertical channel.
                {"E": 1.0, "N": 1.0} - Use east and north channel and weight
                                       them equally.
        :param cc_s_phase_weighting: See cc_p_phase_weighting. Just for S
            phases. Defaults to {"Z": 1.0, "E": 1.0, "N": 1.0}.
        :param cc_min_allowed_cross_corr_coeff: The minimum allowed
            cross-correlation coefficient for a differential travel time to be
            accepted. Defaults to 0.4.
        :param supress_warning_traces: Supress warning about traces not being
            found (useful if you mix stations with different types of
            component codes (ZNE versus 123, for example))
        :param shift_stations: Shift station (and model) depths so that
            the deepest station is at elev=0 (useful for networks with negative
            elevations (HypoDD can't handle them)
        :param use_cross_correlation: If False, skip waveform parsing and
            cross correlation and run HypoDD with catalog differential times
            only (IDATA=2).
        :param event_fix: Optional HypoDD ev_fix value to append to event.sel.
            For HypoDD 2.x, 1 fixes depth, 2 fixes depth and time, 3 fixes
            x/y/depth, 4 fixes x/y/depth/time, 5 fixes time, and 6 fixes x/y.
        :param fixed_depth_km: Optional depth in km to write for all events in
            phase.dat and event.sel. Use 0.01 with event_fix=1 for a near
            surface fixed-depth relocation.
        :param enforce_mean_shift_constraint: If True, patch HypoDD's LSQR
            solver to add four rows enforcing zero mean x/y/z/origin-time
            update per cluster, matching the SVD mean-shift constraint.
        :param mean_shift_constraint_weight: Weight for the LSQR mean-shift
            constraint rows.
        :param lsqr_constraint_weight: Default weight for LSQR fixed-parameter
            constraints from ev_fix.
        :param lsqr_xyz_constraint_weight: Lower weight used by HypoDD for the
            ev_fix=3 x/y/z fixed-parameter constraints.
        :param lsqr_time_constraint_weight: Weight used by HypoDD for the
            ev_fix=5 origin-time fixed-parameter constraint.
        """
        self.working_dir = working_dir
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        cc_backend = _normalize_cc_backend(cc_backend)
        cc_filter = str(cc_filter).lower()
        if cc_filter in ["", "none", "false", "off", "no"]:
            cc_filter = None
        elif cc_filter != "bandpass":
            msg = "cc_filter must be 'bandpass' or 'none'."
            raise HypoDDException(msg)
        # Some sanity checks.
        if cc_filter == "bandpass" and cc_filter_min_freq >= cc_filter_max_freq:
            msg = "cc_filter_min_freq has to smaller then cc_filter_max_freq."
            raise HypoDDException(msg)
        # Fill the phase weighting dict if necessary.
        if cc_p_phase_weighting is None:
            cc_p_phase_weighting = {"Z": 1.0}
        if cc_s_phase_weighting is None:
            cc_s_phase_weighting = {"Z": 1.0, "E": 1.0, "N": 1.0}
        cc_p_phase_weighting = copy.copy(cc_p_phase_weighting)
        cc_s_phase_weighting = copy.copy(cc_s_phase_weighting)
        for phase in ["Z", "E", "N"]:
            cc_p_phase_weighting[phase] = float(
                cc_p_phase_weighting.get(phase, 0.0)
            )
            cc_s_phase_weighting[phase] = float(
                cc_s_phase_weighting.get(phase, 0.0)
            )
        # set an equal phase weighting scheme by uncertainty
        self.phase_weighting = lambda sta_id, ph_type, time, uncertainty: 1.0
        # Set the cross correlation parameters.
        self.cc_param = {
            "cc_time_before": cc_time_before,
            "cc_time_after": cc_time_after,
            "cc_maxlag": cc_maxlag,
            "cc_backend": cc_backend,
            "cc_write_pair_files": bool(cc_write_pair_files),
            "cc_filter": cc_filter,
            "cc_filter_min_freq": cc_filter_min_freq,
            "cc_filter_max_freq": cc_filter_max_freq,
            "cc_p_phase_weighting": cc_p_phase_weighting,
            "cc_s_phase_weighting": cc_s_phase_weighting,
            "cc_min_allowed_cross_corr_coeff": cc_min_allowed_cross_corr_coeff,
        }
        self.cc_results = {}
        self.cc_diagnostics = {}
        self.supress_warnings = {"no_matching_trace": supress_warning_traces}
        self.shift_stations = shift_stations
        self.min_elev = 0  # Minimum station elevation (used for shifting)
        self.use_cross_correlation = bool(use_cross_correlation)
        if event_fix is not None:
            event_fix = int(event_fix)
            if event_fix not in [-1, 0, 1, 2, 3, 4, 5, 6]:
                msg = "event_fix must be one of -1, 0, 1, 2, 3, 4, 5, or 6."
                raise HypoDDException(msg)
        self.event_fix = event_fix
        if fixed_depth_km is not None:
            fixed_depth_km = float(fixed_depth_km)
            if fixed_depth_km < 0:
                msg = "fixed_depth_km must be non-negative."
                raise HypoDDException(msg)
        self.fixed_depth_km = fixed_depth_km
        self.enforce_mean_shift_constraint = bool(
            enforce_mean_shift_constraint
        )
        self.mean_shift_constraint_weight = float(mean_shift_constraint_weight)
        self.lsqr_constraint_weight = float(lsqr_constraint_weight)
        self.lsqr_xyz_constraint_weight = float(lsqr_xyz_constraint_weight)
        self.lsqr_time_constraint_weight = float(lsqr_time_constraint_weight)

        # Setup logging.
        logging.basicConfig(
            level=logging.DEBUG,
            filename=os.path.join(self.working_dir, "log.txt"),
            format="[%(asctime)s] %(message)s",
        )

        self.event_files = []
        self.station_files = []
        self.waveform_files = []

        # Dictionary to store forced configuration values.
        self.forced_configuration_values = {}
        self.compiler_configuration_values = {}

        # Configure the paths.
        self._configure_paths()

    def start_relocation(
        self,
        output_event_file,
        output_cross_correlation_file=None,
        create_plots=True,
    ):
        """
        Start the relocation with HypoDD and write the output to
        output_event_file.

        :type output_event_file: str
        :param output_event_file: The filename of the final QuakeML file.
        :param output_cross_correlation_file: Filename of cross correlation
            results output.
        :type output_cross_correlation_file: str
        :param create_plots: If true, some plots will be created in
            working_dir/output_files. Defaults to True.
        """
        self.output_event_file = output_event_file
        if os.path.exists(self.output_event_file):
            msg = "The output_event_file already exists. Nothing to do."
            self.log(msg)
            return

        self.log("Starting relocator...")
        self._parse_station_files()
        self._write_station_input_file()
        self._read_event_information()
        self._write_ph2dt_inp_file()
        self._create_event_id_map()
        self._write_catalog_input_file()
        self._compile_hypodd()
        self._run_ph2dt()
        self._apply_event_fix_to_event_selection()
        if self.use_cross_correlation:
            self._parse_waveform_files()
            self._cross_correlate_picks(outfile=output_cross_correlation_file)
        self._write_hypoDD_inp_file()
        self._run_hypodd()
        self._create_output_event_file()
        if create_plots:
            self._create_plots()

    def add_event_files(self, event_files):
        """
        Adds all files in event_files to self.event_files. All files will be
        verified to exist but no further checks are done.
        """
        if isinstance(event_files, str):
            event_files = [event_files]
        for event_file in event_files:
            if not isinstance(event_file, str):
                msg = "%s is not a filename." % event_file
                warnings.warn(msg)
                continue
            if not os.path.exists(event_file):
                msg = "Warning: File %s does not exists." % event_file
                warnings.warn(msg)
                continue
            self.event_files.append(event_file)

    def add_station_files(self, station_files):
        """
        Adds all files in station_files to self.station_files. All files will
        be verified to exist but no further checks are done.
        """
        if isinstance(station_files, str):
            station_files = [station_files]
        for station_file in station_files:
            if not isinstance(station_file, str):
                msg = "%s is not a filename." % station_file
                warnings.warn(msg)
                continue
            if not os.path.exists(station_file):
                msg = "Warning: File %s does not exists."
                warnings.warn(msg)
                continue
            self.station_files.append(station_file)

    def add_waveform_files(self, waveform_files):
        """
        Adds all files in waveform_files to self.waveform_files. All files will
        be verified to exist but no further checks are done.
        """
        if isinstance(waveform_files, str):
            waveform_files = [waveform_files]
        for waveform_file in waveform_files:
            if not isinstance(waveform_file, str):
                msg = "%s is not a filename." % waveform_file
                warnings.warn(msg)
                continue
            if not os.path.exists(waveform_file):
                msg = "Warning: File %s does not exists."
                warnings.warn(msg)
                continue
            self.waveform_files.append(waveform_file)

    def set_forced_configuration_value(self, key, value):
        """
        Force a configuration key to a certain value. This will overwrite any
        automatically determined values. Use with caution.

        Possible keys (refer to the HypoDD manual for more information) - if
        no value for a certain key, the reasoning in the brackets will be
        applied:
            For ph2dt.inp
            * MINWGHT (Will be set to 0.0)
            * MAXDIST (Will be set so that all event_pair-station pairs are
                       included.)
            * MAXSEP  (This is rather difficult to determine automatically.
                       Will be set to the lower quartile of all inter-event
                       distances.)
            * MAXNGH  (Will be set to 10.)
            * MINLNK  (Will be set to 8. Should not be set lower.)
            * MINOBS  (Will be set to 8.)
            * MAXOBS  (Will be set to 50.)
            * MAXTIMESEP_DAYS (Wrapper-only dt.ct pair filter. -999 disables.)

            For hypoDD.inp
            * MAXDIST - DIST in the hypoDD manual for hypoDD.inp. Same as
                MAXDIST in ph2dt.inp. (Will be set so that all
                event_pair-station pairs are included.)
        """
        allowed_keys = [
            "MINWGHT",
            "MAXDIST",
            "MAXSEP",
            "MAXNGH",
            "MINLNK",
            "MINOBS",
            "MAXOBS",
            "MAXTIMESEP_DAYS",
            "IPHA",
            "OBSCC",
            "OBSCT",
            "MINDS",
            "MAXDS",
            "MAXGAP",
            "ISTART",
            "ISOLV",
            "IAQ",
            "CID",
            "ID",
            "DATA_WEIGHTING_AND_REWEIGHTING",
            "ITERATIONS",
        ]
        if not isinstance(key, str):
            msg = "The configuration key needs to be a string"
            warnings.warn(msg)
            return
        key = key.upper()
        if key not in allowed_keys:
            msg = "Key {key} is not an allowed key an will ignored. "
            msg += "Allowed keys:\n{all_keys}"
            warnings.warn(msg.format(key=key, all_keys=allowed_keys))
            return
        self.forced_configuration_values[key] = value

    def load_configuration_file(self, filename):
        """
        Load ph2dt/hypoDD wrapper settings from an INI-style config file.

        Supported sections:
            [relocator]
                use_cross_correlation, event_fix, fixed_depth_km,
                enforce_mean_shift_constraint, mean_shift_constraint_weight,
                lsqr_constraint_weight, lsqr_xyz_constraint_weight,
                lsqr_time_constraint_weight

            [cross_correlation]
                cc_time_before, cc_time_after, cc_maxlag, cc_backend,
                cc_write_pair_files, cc_filter,
                cc_filter_min_freq, cc_filter_max_freq,
                cc_min_allowed_cross_corr_coeff, cc_p_phase_weighting,
                cc_s_phase_weighting

            [ph2dt]
                MINWGHT, MAXDIST, MAXSEP, MAXNGH, MINLNK, MINOBS, MAXOBS

            [hypodd]
                IPHA, OBSCC, OBSCT, MINDS, MAXDS, MAXGAP, ISTART, ISOLV,
                IAQ, CID, ID, ITERATIONS

        ITERATIONS may be a multiline value; each non-empty line is written as
        one HypoDD data-weighting row.
        """
        parser = configparser.ConfigParser()
        parser.optionxform = str
        read_files = parser.read(filename)
        if not read_files:
            msg = "Configuration file could not be read: %s" % filename
            raise HypoDDException(msg)

        if parser.has_section("relocator"):
            section = parser["relocator"]
            if "use_cross_correlation" in section:
                self.use_cross_correlation = section.getboolean(
                    "use_cross_correlation"
                )
            if "event_fix" in section:
                value = section.get("event_fix")
                self.event_fix = None if value == "" else int(value)
            if "fixed_depth_km" in section:
                value = section.get("fixed_depth_km")
                self.fixed_depth_km = None if value == "" else float(value)
            if "enforce_mean_shift_constraint" in section:
                self.enforce_mean_shift_constraint = section.getboolean(
                    "enforce_mean_shift_constraint"
                )
            for key in [
                "mean_shift_constraint_weight",
                "lsqr_constraint_weight",
                "lsqr_xyz_constraint_weight",
                "lsqr_time_constraint_weight",
            ]:
                if key in section:
                    setattr(self, key, section.getfloat(key))

        if parser.has_section("cross_correlation"):
            section = parser["cross_correlation"]
            if "cc_backend" in section:
                self.cc_param["cc_backend"] = _normalize_cc_backend(
                    section.get("cc_backend")
                )
            if "cc_write_pair_files" in section:
                self.cc_param["cc_write_pair_files"] = section.getboolean(
                    "cc_write_pair_files"
                )
            if "cc_filter" in section:
                cc_filter = section.get("cc_filter").strip().lower()
                if cc_filter in ["", "none", "false", "off", "no"]:
                    self.cc_param["cc_filter"] = None
                elif cc_filter == "bandpass":
                    self.cc_param["cc_filter"] = "bandpass"
                else:
                    msg = "cc_filter must be 'bandpass' or 'none'."
                    raise HypoDDException(msg)
            for key in [
                "cc_time_before",
                "cc_time_after",
                "cc_maxlag",
                "cc_filter_min_freq",
                "cc_filter_max_freq",
                "cc_min_allowed_cross_corr_coeff",
            ]:
                if key in section:
                    self.cc_param[key] = section.getfloat(key)
            for key in ["cc_p_phase_weighting", "cc_s_phase_weighting"]:
                if key in section:
                    self.cc_param[key] = self._parse_cc_phase_weighting(
                        section.get(key)
                    )

        if parser.has_section("compiler"):
            for key, value in parser.items("compiler"):
                self.compiler_configuration_values[key.upper()] = int(
                    value.strip()
                )

        for section_name in ["ph2dt", "hypodd"]:
            if not parser.has_section(section_name):
                continue
            for key, value in parser.items(section_name):
                self.set_forced_configuration_value(
                    key, self._parse_configuration_value(key, value)
                )
        self.log("Loaded configuration file: %s" % filename)

    @staticmethod
    def _parse_cc_phase_weighting(value):
        """
        Parse component weights from config strings like ``Z:1,E:0,N:0``.
        """
        weights = {}
        value = value.strip()
        if not value:
            return weights
        for item in value.split(","):
            if not item.strip():
                continue
            if ":" not in item:
                msg = (
                    "Cross-correlation phase weighting entries must be "
                    "component:weight pairs, e.g. Z:1.0,E:0.0,N:0.0."
                )
                raise HypoDDException(msg)
            component, weight = item.split(":", 1)
            weights[component.strip()] = float(weight.strip())
        for component in ["Z", "E", "N"]:
            weights[component] = float(weights.get(component, 0.0))
        return weights

    def _parse_configuration_value(self, key, value):
        """
        Parse a config value into an int, float, string, or list of strings.
        """
        key = key.upper()
        value = value.strip()
        if key in ["ITERATIONS", "DATA_WEIGHTING_AND_REWEIGHTING"]:
            return [
                line.strip()
                for line in value.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        if key == "ID":
            return value
        int_keys = [
            "MAXNGH",
            "MINLNK",
            "MINOBS",
            "MAXOBS",
            "IPHA",
            "OBSCC",
            "OBSCT",
            "ISTART",
            "ISOLV",
            "IAQ",
            "CID",
        ]
        if key in int_keys:
            return int(value)
        return float(value)

    def _configure_paths(self):
        """
        Central place to setup up all the paths needed for running HypoDD.
        """
        self.paths = {}

        # Setup some necessary directories.
        for path in ["bin", "input_files", "working_files", "output_files"]:
            self.paths[path] = os.path.join(self.working_dir, path)
            if not os.path.exists(self.paths[path]):
                os.makedirs(self.paths[path])

    def log(self, string, level="info"):
        """
        Prints a colorful and fancy string and logs the same string.

        level is the log level. Default is info which will result in a green
        output. So far everything else will be output with a red color.
        """
        logging.info(string)
        # Info is green
        if level == "info":
            print("\033[0;32m" + ">>> " + string + "\033[1;m")
        # Everything else is currently red.
        else:
            level = level.lower().capitalize()
            print("\033[1;31m" + ">>> " + level + ": " + string + "\033[1;m")
        sys.stdout.flush()

    def _parse_station_files(self):
        """
        Parse all station files and serialize the necessary information as a
        JSON object to working_dir/working_files/stations.json.
        """
        serialized_station_file = os.path.join(
            self.paths["working_files"], "stations.json"
        )
        # If already parsed before, just read the serialized station file.
        if os.path.exists(serialized_station_file):
            self.log(
                "Stations already parsed. Will load the serialized "
                + "information."
            )
            with open(serialized_station_file, "r") as open_file:
                self.stations = json.load(open_file)
                return
        self.log("Parsing stations...")
        self.stations = {}
        for station_file in self.station_files:
            if stations_XSEED:
                p = Parser(station_file)
                # In theory it would be enough to parse Blockette 50, put faulty
                # SEED files do not store enough information in them, so
                # blockettes 52 need to be parsed...
                for station in p.stations:
                    for blockette in station:
                        if blockette.id != 52:
                            continue
                        station_id = "%s.%s" % (
                            station[0].network_code,
                            station[0].station_call_letters,
                        )
                        self.stations[station_id] = {
                            "latitude": blockette.latitude,
                            "longitude": blockette.longitude,
                            "elevation": int(round(blockette.elevation)),
                        }
            else:
                inv = read_inventory(station_file, "STATIONXML")
                for net in inv:
                    for sta in net:
                        station_id = f"{net.code}.{sta.code}"
                        if len(station_id) > 7:
                            station_id = f"{sta.code}"
                        self.stations[station_id] = {
                            "latitude": sta.latitude,
                            "longitude": sta.longitude,
                            "elevation": int(round(sta.elevation)),
                        }

        with open(serialized_station_file, "w") as open_file:
            json.dump(self.stations, open_file)
        self.log("Done parsing stations.")

    def _write_station_input_file(self):
        """
        Write the station.data input file for ph2dt and hypodd.

        The format is one station per line and:
            station_label latitude longitude elevation_in_meters
        """
        station_dat_file = os.path.join(
            self.paths["input_files"], "station.dat"
        )
        if os.path.exists(station_dat_file):
            self.log("station.dat input file already exists.")
            return
        station_strings = []
        if self.shift_stations:
            self.min_elev = min(
                [s["elevation"] for s in self.stations.values()]
            )

        for key, value in self.stations.items():
            station_strings.append(
                "%-7s %9.5f %10.5f %5i"
                % (
                    key,
                    value["latitude"],
                    value["longitude"],
                    value["elevation"] - self.min_elev,
                )
            )
        station_string = "\n".join(station_strings)
        with open(station_dat_file, "w") as open_file:
            open_file.write(station_string)
        self.log("Created station.dat input file.")

    def _write_catalog_input_file(self, phase_weighting=None):
        """
        Write the phase.dat input file for ph2dt.

        The format is described in the HypoDD manual. All event ids will be
        mapped.

        :type phase_weighting: func
        :param phase_weighting: Function that returns the weighting (from 0.0
            to 1.0) for each phase depending on the pick. The following
            parameters are fed to the function as arguments: station id, phase
            type, pick time, pick uncertainty. Note that pick uncertainty input
            can be None.
        """
        if phase_weighting is None:
            phase_weighting = self.phase_weighting
        phase_dat_file = os.path.join(self.paths["input_files"], "phase.dat")
        if os.path.exists(phase_dat_file):
            self.log("phase.dat input file already exists.")
            return
        event_strings = []
        for event in self.events:
            string = (
                "# {year} {month} {day} {hour} {minute} "
                + "{second:.6f} {latitude:.6f} {longitude:.6f} "
                + "{depth:.4f} {magnitude:.6f} {horizontal_error:.6f} "
                + "{depth_error:.6f} {travel_time_residual:.6f} {event_id}"
            )
            event_string = string.format(
                year=event["origin_time"].year,
                month=event["origin_time"].month,
                day=event["origin_time"].day,
                hour=event["origin_time"].hour,
                minute=event["origin_time"].minute,
                # Seconds + microseconds
                second=float(event["origin_time"].second)
                + (event["origin_time"].microsecond / 1e6),
                latitude=event["origin_latitude"],
                longitude=event["origin_longitude"],
                # QuakeML depth is in meters. Convert to km.
                depth=(
                    self.fixed_depth_km
                    if self.fixed_depth_km is not None
                    else event["origin_depth"] / 1000.0
                ),
                magnitude=event["magnitude"],
                horizontal_error=max(
                    [
                        event["origin_latitude_error"],
                        event["origin_longitude_error"],
                    ]
                ),
                depth_error=event["origin_depth_error"] / 1000.0,
                travel_time_residual=event["origin_time_error"],
                event_id=self.event_map[event["event_id"]],
            )
            event_strings.append(event_string)
            # Now loop over every pick and add station traveltimes.
            for pick in event["picks"]:
                # Only P and S phases currently supported by HypoDD.
                if (
                    pick["phase"].upper() != "P"
                    and pick["phase"].upper() != "S"
                ):
                    continue
                string = (
                    "{station_id:7s} {travel_time:7.3f} {weight:5.2f} {phase}"
                )
                travel_time = pick["pick_time"] - event["origin_time"]
                # Simple check to assure no negative travel times are used.
                if travel_time < 0:
                    msg = (
                        "Negative absolute travel time. "
                        + "{phase} phase pick for event {event_id} at "
                        + "station {station_id} will not be used."
                    )
                    msg = msg.format(
                        phase=pick["phase"],
                        event_id=event["event_id"],
                        station_id=pick["station_id"],
                    )
                    self.log(msg, level="warning")
                    continue
                weight = phase_weighting(
                    pick["station_id"],
                    pick["phase"],
                    pick["pick_time"],
                    pick["pick_time_error"],
                )
                pick_string = string.format(
                    station_id=pick["station_id"],
                    travel_time=travel_time,
                    weight=weight,
                    phase=pick["phase"].upper(),
                )
                event_strings.append(pick_string)
        event_string = "\n".join(event_strings)
        # Write the phase.dat file.
        with open(phase_dat_file, "w") as open_file:
            open_file.write(event_string)
        self.log("Created phase.dat input file.")

    def _apply_event_fix_to_event_selection(self):
        """
        Optionally modify ph2dt's event.sel for HypoDD 2.x fixed-parameter
        runs.

        ph2dt writes the 10-column event.sel file consumed by HypoDD. HypoDD
        also accepts an optional 11th ev_fix column. This method appends or
        replaces that column and can force the selected event depth.
        """
        if self.event_fix is None and self.fixed_depth_km is None:
            return
        event_sel_path = os.path.join(self.paths["input_files"], "event.sel")
        if not os.path.exists(event_sel_path):
            msg = "event.sel does not exist; ph2dt must run before ev_fix."
            raise HypoDDException(msg)

        event_lines = []
        with open(event_sel_path, "r") as open_file:
            for line in open_file:
                stripped = line.strip()
                if not stripped:
                    event_lines.append(line.rstrip("\n"))
                    continue
                parts = stripped.split()
                if len(parts) < 10:
                    msg = "Unexpected event.sel line format: %s" % stripped
                    raise HypoDDException(msg)
                if self.fixed_depth_km is not None:
                    parts[4] = "%.4f" % self.fixed_depth_km
                if self.event_fix is not None:
                    if len(parts) >= 11:
                        parts[10] = str(self.event_fix)
                    else:
                        parts.append(str(self.event_fix))
                event_lines.append(" ".join(parts))

        with open(event_sel_path, "w") as open_file:
            open_file.write("\n".join(event_lines))
        self.log("Updated event.sel with fixed event parameters.")

    def _read_event_information(self):
        """
        Read all event files and extract the needed information and serialize
        it as a JSON object. This is not necessarily needed but eases
        development as the JSON file is just much faster to read then the full
        event files.
        """
        serialized_event_file = os.path.join(
            self.paths["working_files"], "events.json"
        )
        if os.path.exists(serialized_event_file):
            self.log(
                "Events already parsed. Will load the serialized "
                + "information."
            )
            with open(serialized_event_file, "r") as open_file:
                self.events = json.load(open_file)
            # Loop and convert all time values to UTCDateTime.
            for event in self.events:
                event["origin_time"] = UTCDateTime(event["origin_time"])
                for pick in event["picks"]:
                    pick["pick_time"] = UTCDateTime(pick["pick_time"])
            self.log("Reading serialized event file successful.")
            return
        self.log("Reading all events...")
        catalog = Catalog()
        for event in self.event_files:
            catalog += read_events(event)
        self.events = []
        # Keep track of the number of discarded picks.
        discarded_picks = 0
        # Loop over all events.
        for event in catalog:
            current_event = {}
            self.events.append(current_event)
            current_event["event_id"] = str(event.resource_id)
            # Take the value from the first event.
            if event.preferred_magnitude() is not None:
                current_event["magnitude"] = event.preferred_magnitude().mag
            else:
                current_event["magnitude"] = 0.0
            # Always take the preferred origin.
            origin = event.preferred_origin()
            current_event["origin_time"] = origin.time
            # Origin time error.
            if origin.time_errors.uncertainty is not None:
                current_event[
                    "origin_time_error"
                ] = origin.time_errors.uncertainty
            else:
                current_event["origin_time_error"] = 0.0
            current_event["origin_latitude"] = origin.latitude
            # Origin latitude error.
            if origin.latitude_errors.uncertainty is not None:
                current_event[
                    "origin_latitude_error"
                ] = origin.latitude_errors.uncertainty
            else:
                current_event["origin_latitude_error"] = 0.0
            current_event["origin_longitude"] = origin.longitude
            # Origin longitude error.
            if origin.longitude_errors.uncertainty is not None:
                current_event[
                    "origin_longitude_error"
                ] = origin.longitude_errors.uncertainty
            else:
                current_event["origin_longitude_error"] = 0.0
            current_event["origin_depth"] = origin.depth
            # Origin depth error.
            if origin.depth_errors.uncertainty is not None:
                current_event[
                    "origin_depth_error"
                ] = origin.depth_errors.uncertainty
            else:
                current_event["origin_depth_error"] = 0.0
            # Also append all picks.
            current_event["picks"] = []
            for pick in event.picks:
                current_pick = {}
                current_pick["id"] = str(pick.resource_id)
                current_pick["pick_time"] = pick.time
                if hasattr(pick.time_errors, "uncertainty"):
                    current_pick[
                        "pick_time_error"
                    ] = pick.time_errors.uncertainty
                else:
                    current_pick["pick_time_error"] = None
                current_pick["station_id"] = "{}.{}".format(
                    pick.waveform_id.network_code,
                    pick.waveform_id.station_code,
                )
                if len(current_pick["station_id"]) > 7:
                    current_pick["station_id"] = pick.waveform_id.station_code
                current_pick["phase"] = pick.phase_hint
                # Assert that information for the station of the pick is
                # available.
                if not current_pick["station_id"] in list(
                    self.stations.keys()
                ):
                    discarded_picks += 0
                    continue
                current_event["picks"].append(current_pick)
        # Sort events by origin time
        self.events.sort(key=lambda event: event["origin_time"])
        # Serialize the event dict. Copy it so the times can be converted to
        # strings.
        events = copy.deepcopy(self.events)
        for event in events:
            event["origin_time"] = str(event["origin_time"])
            for pick in event["picks"]:
                pick["pick_time"] = str(pick["pick_time"])
        with open(serialized_event_file, "w") as open_file:
            json.dump(events, open_file)
        self.log("Reading all events successful.")
        self.log(
            ("%i picks discarded because of " % discarded_picks)
            + "unavailable station information."
        )

    def _create_event_id_map(self):
        """
        HypoDD can only deal with numeric event ids. Map all events to a number
        from 1 to number_of_events.

        This method will create the self.map dictionary with a two way mapping.

        self.event_map["event_id_string"] = number
        self.event_map[number] = "event_id_string"
        """
        self.event_map = {}
        # Just create this every time as it is very fast.
        for _i, event in enumerate(self.events):
            event_id = event["event_id"]
            self.event_map[event_id] = _i + 1
            self.event_map[_i + 1] = event_id

    def _write_ph2dt_inp_file(self):
        """
        Create the ph2dt.inp file.
        """
        # MAXDIST is reused in the hypoDD.inp file. It always needs to be
        # calculated. Fake a forced configuration
        # value.
        values = {}
        if "MAXDIST" not in self.forced_configuration_values:
            # Calculate MAXDIST so that all event-station pairs are definitely
            # inluded. This is a very simple way of doing it.
            lats = []
            longs = []
            depths = []
            for event in self.events:
                lats.append(event["origin_latitude"])
                longs.append(event["origin_longitude"])
                # Convert to km.
                depths.append(event["origin_depth"] / 1000.0)
            for _, station in self.stations.items():
                lats.append(station["latitude"])
                longs.append(station["longitude"])
                # station elevation is in meter.
                depths.append(station["elevation"] / 1000.0)
            lat_range = (max(lats) - min(lats)) * 111.0
            long_range = (max(longs) - min(longs)) * 111.0
            depth_range = max(depths) - min(depths)
            maxdist = math.sqrt(
                lat_range ** 2 + long_range ** 2 + depth_range ** 2
            )
            values["MAXDIST"] = int(math.ceil(maxdist))
            self.log(
                "MAXDIST for ph2dt.inp calculated to %i." % values["MAXDIST"]
            )
            self.forced_configuration_values["MAXDIST"] = values["MAXDIST"]

        ph2dt_inp_file = os.path.join(self.paths["input_files"], "ph2dt.inp")
        if os.path.exists(ph2dt_inp_file):
            self.log("ph2dt.inp input file already exists.")
            return
        # Determine the necessary variables. See the documentation of the
        # set_forced_configuration_value method for the reasoning.
        values = {}
        values["MINWGHT"] = 0.0
        values["MAXNGH"] = 10
        values["MINLNK"] = 8
        values["MINOBS"] = 8
        values["MAXOBS"] = 50
        if "MAXSEP" not in self.forced_configuration_values:
            # Set MAXSEP to the 10-percentile of all inter-event distances.
            distances = []
            for event_1 in self.events:
                # Will produce one 0 distance pair but that should not matter.
                for event_2 in self.events:
                    lat_range = (
                        abs(
                            event_1["origin_latitude"]
                            - event_2["origin_latitude"]
                        )
                        * 111.0
                    )
                    long_range = (
                        abs(
                            event_1["origin_longitude"]
                            - event_2["origin_longitude"]
                        )
                        * 111.0
                    )
                    depth_range = (
                        abs(event_1["origin_depth"] - event_2["origin_depth"])
                        / 1000.0
                    )
                    distances.append(
                        math.sqrt(
                            lat_range ** 2 + long_range ** 2 + depth_range ** 2
                        )
                    )
            # Get the percentile value.
            distances.sort()
            maxsep = distances[int(math.floor(len(distances) * 0.10))]
            values["MAXSEP"] = maxsep
            self.log(
                "MAXSEP for ph2dt.inp calculated to %f." % values["MAXSEP"]
            )
        # Use any potential forced values to overwrite the automatically set
        # ones.
        keys = [
            "MINWGHT",
            "MAXDIST",
            "MAXSEP",
            "MAXNGH",
            "MINLNK",
            "MINOBS",
            "MAXOBS",
        ]
        for key in keys:
            if key in self.forced_configuration_values:
                values[key] = self.forced_configuration_values[key]
        # Use this construction to get rid of leading whitespaces.
        ph2dt_string = [
            "station.dat",
            "phase.dat",
            "{MINWGHT} {MAXDIST} {MAXSEP} {MAXNGH} {MINLNK} {MINOBS} {MAXOBS}",
        ]
        ph2dt_string = "\n".join(ph2dt_string)
        ph2dt_string = ph2dt_string.format(**values)
        with open(ph2dt_inp_file, "w") as open_file:
            open_file.write(ph2dt_string)
        self.log("Writing ph2dt.inp successful")

    def _compile_hypodd(self):
        """
        Compiles HypoDD and ph2dt
        """
        logfile = os.path.join(self.working_dir, "compilation.log")
        self.log("Initating HypoDD compilation (logfile: %s)..." % logfile)
        with open(logfile, "w") as fh:

            def logfunc(line):
                fh.write(line)
                fh.write(os.linesep)

            compiler = HypoDDCompiler(
                working_dir=self.working_dir,
                log_function=logfunc,
                enforce_mean_shift_constraint=(
                    self.enforce_mean_shift_constraint
                ),
                mean_shift_constraint_weight=(
                    self.mean_shift_constraint_weight
                ),
                lsqr_constraint_weight=self.lsqr_constraint_weight,
                lsqr_xyz_constraint_weight=self.lsqr_xyz_constraint_weight,
                lsqr_time_constraint_weight=self.lsqr_time_constraint_weight,
            )
            compiler_config = {
                "MAXEVE": len(self.events) + 30,
                # MAXEVE0=len(self.events) + 30,
                "MAXEVE0": 200,
                "MAXDATA": 3000000,
                "MAXDATA0": 60000,
                "MAXCL": 20,
                "MAXSTA": len(self.stations) + 10,
            }
            compiler_config.update(self.compiler_configuration_values)
            compiler.configure(**compiler_config)
            compiler.make()

    def _run_hypodd(self):
        """
        Runs HypoDD with the necessary input files.
        """
        # Check if all the hypodd output files are already existant. If they
        # do, do not run it again.
        output_files = [
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
        files_exists = True
        for o_file in output_files:
            if os.path.exists(
                os.path.join(self.paths["output_files"], o_file)
            ):
                continue
            files_exists = False
            break
        if files_exists is True:
            self.log("HypoDD output files already existant.")
            return
        # Otherwise just run it.
        self.log("Running HypoDD...")
        hypodd_path = os.path.abspath(
            os.path.join(self.paths["bin"], "hypoDD")
        )
        if not os.path.exists(hypodd_path):
            msg = "hypodd could not be found. Did the compilation succeed?"
            raise HypoDDException(msg)
        # Create directory to run ph2dt in.
        hypodd_dir = os.path.join(self.working_dir, "hypodd_temp_dir")
        if os.path.exists(hypodd_dir):
            shutil.rmtree(hypodd_dir)
        os.makedirs(hypodd_dir)
        # Check if all necessary files are there.
        necessary_files = [
            "dt.ct",
            "event.sel",
            "station.sel",
            "hypoDD.inp",
        ]
        if self.use_cross_correlation:
            necessary_files.insert(0, "dt.cc")
        for filename in necessary_files:
            if not os.path.exists(
                os.path.join(self.paths["input_files"], filename)
            ):
                msg = "{file} does not exists for HypoDD"
                raise HypoDDException(msg.format(file=filename))
        # Copy the files.
        for filename in necessary_files:
            shutil.copyfile(
                os.path.join(self.paths["input_files"], filename),
                os.path.join(hypodd_dir, filename),
            )
        # Run ph2dt
        retcode = subprocess.Popen(
            [hypodd_path, "hypoDD.inp"], cwd=hypodd_dir
        ).wait()
        if retcode != 0:
            msg = "Problem running HypoDD."
            raise HypoDDException(msg)
        # Check if all are there.
        for o_file in output_files:
            if not os.path.exists(os.path.join(hypodd_dir, o_file)):
                msg = "HypoDD output file {filename} was not created."
                msg = msg.format(filename=o_file)
                raise HypoDDException(msg)
        # Copy the output files.
        for o_file in output_files:
            shutil.copyfile(
                os.path.join(hypodd_dir, o_file),
                os.path.join(self.paths["output_files"], o_file),
            )
        # Also copy the log file.
        log_file = os.path.join(hypodd_dir, "hypoDD.log")
        if os.path.exists(log_file):
            shutil.move(
                log_file, os.path.join(self.working_dir, "hypoDD_log.txt")
            )
        # Remove the temporary ph2dt running directory.
        shutil.rmtree(hypodd_dir)
        self.log("HypoDD run was successful!")

    def _run_ph2dt(self):
        """
        Runs ph2dt with the necessary input files.
        """
        # Check if all the ph2dt output files are already existant. If they do,
        # do not run it again.
        output_files = ["station.sel", "event.sel", "event.dat", "dt.ct"]
        files_exists = True
        for o_file in output_files:
            if os.path.exists(os.path.join(self.paths["input_files"], o_file)):
                continue
            files_exists = False
            break
        if files_exists is True:
            self.log("ph2dt output files already existant.")
            self._filter_dt_ct_by_max_time_separation()
            return
        # Otherwise just run it.
        self.log("Running ph2dt...")
        ph2dt_path = os.path.abspath(os.path.join(self.paths["bin"], "ph2dt"))
        if not os.path.exists(ph2dt_path):
            msg = "ph2dt could not be found. Did the compilation succeed?"
            raise HypoDDException(msg)
        # Create directory to run ph2dt in.
        ph2dt_dir = os.path.join(self.working_dir, "ph2dt_temp_dir")
        if os.path.exists(ph2dt_dir):
            shutil.rmtree(ph2dt_dir)
        os.makedirs(ph2dt_dir)
        # Check if all necessary files are there.
        station_file = os.path.join(self.paths["input_files"], "station.dat")
        phase_file = os.path.join(self.paths["input_files"], "phase.dat")
        input_file = os.path.join(self.paths["input_files"], "ph2dt.inp")
        if not os.path.exists(station_file):
            msg = "station.dat input file is not existent for ph2dt."
            HypoDDException(msg)
        if not os.path.exists(phase_file):
            msg = "phase.dat input file is not existent for ph2dt."
            HypoDDException(msg)
        if not os.path.exists(input_file):
            msg = "ph2d.inp input file is not existent for ph2dt."
            HypoDDException(msg)
        # Copy the three files.
        shutil.copyfile(station_file, os.path.join(ph2dt_dir, "station.dat"))
        shutil.copyfile(phase_file, os.path.join(ph2dt_dir, "phase.dat"))
        shutil.copyfile(input_file, os.path.join(ph2dt_dir, "ph2dt.inp"))
        # Run ph2dt
        retcode = subprocess.Popen(
            [ph2dt_path, "ph2dt.inp"], cwd=ph2dt_dir
        ).wait()
        if retcode != 0:
            msg = "Problem running ph2dt."
            raise HypoDDException(msg)
        # Check if all are there.
        for o_file in output_files:
            if not os.path.exists(os.path.join(ph2dt_dir, o_file)):
                msg = "ph2dt output file {filename} does not exists."
                msg = msg.format(filename=o_file)
                raise HypoDDException(msg)
        # Copy the output files.
        for o_file in output_files:
            shutil.copyfile(
                os.path.join(ph2dt_dir, o_file),
                os.path.join(self.paths["input_files"], o_file),
            )
        self._filter_dt_ct_by_max_time_separation()
        # Also copy the log file.
        log_file = os.path.join(ph2dt_dir, "ph2dt.log")
        if os.path.exists(log_file):
            shutil.move(
                log_file, os.path.join(self.working_dir, "ph2dt_log.txt")
            )
        # Remove the temporary ph2dt running directory.
        shutil.rmtree(ph2dt_dir)
        self.log("ph2dt run successful.")

    def _filter_dt_ct_by_max_time_separation(self):
        """
        Optionally remove dt.ct event-pair blocks separated by too much time.

        Controlled by the wrapper-only configuration key MAXTIMESEP_DAYS in the
        [ph2dt] section. A value of -999 disables the filter.
        """
        max_time_sep_days = self.forced_configuration_values.get(
            "MAXTIMESEP_DAYS", -999
        )
        if max_time_sep_days in ("", None):
            return
        max_time_sep_days = float(max_time_sep_days)
        if max_time_sep_days == -999:
            return
        if max_time_sep_days < 0:
            msg = "MAXTIMESEP_DAYS must be positive or -999."
            raise HypoDDException(msg)

        event_times = {}
        for event in self.events:
            event_number = self.event_map[event["event_id"]]
            event_times[int(event_number)] = event["origin_time"]

        dt_ct_path = os.path.join(self.paths["input_files"], "dt.ct")
        if not os.path.exists(dt_ct_path):
            return

        kept_lines = []
        current_header = None
        current_rows = []
        current_keep = True
        total_pairs = 0
        removed_pairs = 0
        removed_rows = 0

        def flush_current():
            nonlocal removed_pairs, removed_rows
            if current_header is None:
                return
            if current_keep:
                kept_lines.append(current_header)
                kept_lines.extend(current_rows)
            else:
                removed_pairs += 1
                removed_rows += len(current_rows)

        with open(dt_ct_path, "r") as open_file:
            for line in open_file:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "#":
                    flush_current()
                    current_header = line.rstrip("\n")
                    current_rows = []
                    current_keep = True
                    total_pairs += 1
                    if len(parts) >= 3:
                        event_id_1 = int(parts[1])
                        event_id_2 = int(parts[2])
                        time_1 = event_times.get(event_id_1)
                        time_2 = event_times.get(event_id_2)
                        if time_1 is not None and time_2 is not None:
                            delta_days = abs(time_1 - time_2) / 86400.0
                            current_keep = delta_days <= max_time_sep_days
                    continue
                current_rows.append(line.rstrip("\n"))
        flush_current()

        with open(dt_ct_path, "w") as open_file:
            open_file.write("\n".join(kept_lines))
            if kept_lines:
                open_file.write("\n")
        self.log(
            "Filtered dt.ct by MAXTIMESEP_DAYS=%s: removed %i/%i event "
            "pairs and %i observations."
            % (max_time_sep_days, removed_pairs, total_pairs, removed_rows)
        )

    def _parse_waveform_files(self):
        """
        Read all specified waveform files and store information about them in
        working_dir/working_files/waveform_information.json
        """
        serialized_waveform_information_file = os.path.join(
            self.paths["working_files"], "waveform_information.json"
        )
        # If already parsed before, just read the serialized waveform file.
        if os.path.exists(serialized_waveform_information_file):
            self.log(
                "Waveforms already parsed. Will load the serialized "
                + "information."
            )
            with open(serialized_waveform_information_file, "r") as open_file:
                self.waveform_information = json.load(open_file)
                # Convert all times to UTCDateTimes.
                for value in list(self.waveform_information.values()):
                    for item in value:
                        item["starttime"] = UTCDateTime(item["starttime"])
                        item["endtime"] = UTCDateTime(item["endtime"])
                return
        file_count = len(self.waveform_files)
        self.log("Parsing %i waveform files..." % file_count)
        self.waveform_information = {}
        pbar = progressbar.ProgressBar(
            widgets=[
                progressbar.Percentage(),
                progressbar.Bar(),
                progressbar.ETA(),
            ],
            maxval=file_count,
        )
        pbar.start()
        # Use a progress bar for displaying.
        for _i, waveform_file in enumerate(self.waveform_files):
            try:
                st = read(waveform_file)
            except:
                msg = "Waveform file %s could not be read." % waveform_file
                self.log(msg, level="warning")
                continue
            for trace in st:
                # Append empty list if the id is not yet stored.
                if trace.id not in self.waveform_information:
                    self.waveform_information[trace.id] = []
                self.waveform_information[trace.id].append(
                    {
                        "starttime": trace.stats.starttime,
                        "endtime": trace.stats.endtime,
                        "filename": os.path.abspath(waveform_file),
                    }
                )
            pbar.update(_i + 1)
        pbar.finish()
        # Serialze it as a json object.
        waveform_information = copy.deepcopy(self.waveform_information)
        for value in list(waveform_information.values()):
            for item in value:
                item["starttime"] = str(item["starttime"])
                item["endtime"] = str(item["endtime"])
        with open(serialized_waveform_information_file, "w") as open_file:
            json.dump(waveform_information, open_file)
        self.log("Successfully parsed all waveform files.")

    def save_cross_correlation_results(self, filename):
        with open(filename, "w") as open_file:
            json.dump(self.cc_results, open_file)
        self.log(
            "Successfully saved cross correlation results to file: %s."
            % filename
        )

    def save_cross_correlation_diagnostics(self, filename):
        with open(filename, "w") as open_file:
            json.dump(self.cc_diagnostics, open_file, indent=2, sort_keys=True)
        self.log(
            "Successfully saved cross correlation diagnostics to file: %s."
            % filename
        )

    def load_cross_correlation_results(self, filename, purge=False):
        """
        Load previously computed and saved cross correlation results.

        :param purge: If True any already present cross correlation
            information will be discarded, if False loaded information will be
            used to update any currently present information.
        """
        with open(filename, "r") as open_file:
            cc_ = json.load(open_file)
        if purge:
            self.cc_results = cc_
        else:
            for id1, items in cc_.items():
                self.cc_results.setdefault(id1, {}).update(items)
        self.log(
            "Successfully loaded cross correlation results from file: "
            "%s." % filename
        )

    def prepare_cross_correlation_inputs(
        self,
        output_cross_correlation_file=None,
        output_cross_correlation_diagnostics_file=None,
        force_recompute=False,
    ):
        """
        Run all steps needed to create dt.cc, then stop before HypoDD.

        This is a preflight diagnostic helper for checking whether waveform
        cross-correlation produced usable observations before running the full
        relocation.

        :param output_cross_correlation_file: Optional JSON file for raw
            per-pick-pair cross-correlation results.
        :param output_cross_correlation_diagnostics_file: Optional JSON file
            with rejection/acceptance counters.
        :param force_recompute: If True, remove an existing dt.cc and cached
            per-pair cc files in the working directory before recomputing.
        :returns: A summary dictionary for the generated dt.cc file.
        """
        if not self.use_cross_correlation:
            msg = "use_cross_correlation is False; no dt.cc will be created."
            raise HypoDDException(msg)

        self.log("Preparing cross-correlation inputs...")
        self._parse_station_files()
        self._write_station_input_file()
        self._read_event_information()
        self._write_ph2dt_inp_file()
        self._create_event_id_map()
        self._write_catalog_input_file()
        self._compile_hypodd()
        self._run_ph2dt()
        self._apply_event_fix_to_event_selection()
        self._parse_waveform_files()

        dt_cc_path = os.path.join(self.paths["input_files"], "dt.cc")
        cc_dir = os.path.join(self.paths["working_files"], "cc_files")
        if force_recompute:
            if os.path.exists(dt_cc_path):
                os.remove(dt_cc_path)
            if os.path.exists(cc_dir):
                shutil.rmtree(cc_dir)

        self._cross_correlate_picks(outfile=output_cross_correlation_file)
        if output_cross_correlation_diagnostics_file:
            self.save_cross_correlation_diagnostics(
                output_cross_correlation_diagnostics_file
            )
        summary = self.summarize_dt_cc_file(dt_cc_path)
        summary["diagnostics"] = copy.deepcopy(self.cc_diagnostics)
        self.log(
            "dt.cc summary: {data_rows} observations, P={p_count}, "
            "S={s_count}, mean coefficient={mean_weight}".format(**summary)
        )
        if summary["data_rows"] == 0:
            self.log(
                "Warning: dt.cc contains no cross-correlation observations.",
                level="warning",
            )
        return summary

    @staticmethod
    def summarize_dt_cc_file(filename):
        """
        Summarize a HypoDD dt.cc file.

        Data rows are expected to contain:
            station_id differential_time_s cross_correlation_coefficient phase
        """
        summary = {
            "filename": filename,
            "event_pair_headers": 0,
            "data_rows": 0,
            "p_count": 0,
            "s_count": 0,
            "min_weight": None,
            "max_weight": None,
            "mean_weight": None,
        }
        weights = []
        if not os.path.exists(filename):
            return summary
        with open(filename, "r") as open_file:
            for line in open_file:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "#":
                    summary["event_pair_headers"] += 1
                    continue
                if len(parts) < 4:
                    continue
                summary["data_rows"] += 1
                phase = parts[3].upper()
                if phase == "P":
                    summary["p_count"] += 1
                elif phase == "S":
                    summary["s_count"] += 1
                weights.append(float(parts[2]))
        if weights:
            summary["min_weight"] = min(weights)
            summary["max_weight"] = max(weights)
            summary["mean_weight"] = sum(weights) / len(weights)
        return summary

    def _cross_correlate_picks(self, outfile=None):
        """
        Reads the event pairs matched in dt.ct which are selected by ph2dt and
        calculate cross correlated differential travel_times for every pair.

        :param outfile: Filename of cross correlation results output.
        """
        ct_file_path = os.path.join(self.paths["input_files"], "dt.cc")
        if os.path.exists(ct_file_path):
            self.log("ct.cc input file already exists")
            return
        self.cc_diagnostics = {
            "backend": self.cc_param["cc_backend"],
            "write_pair_files": self.cc_param["cc_write_pair_files"],
            "event_pairs_total": 0,
            "pick_pairs_considered": 0,
            "accepted": 0,
            "rejected": {
                "no_matching_pick": 0,
                "no_waveform_files": 0,
                "preloaded_error": 0,
                "no_matching_trace_1": 0,
                "no_matching_trace_2": 0,
                "multiple_matching_trace_1": 0,
                "multiple_matching_trace_2": 0,
                "non_matching_trace_ids": 0,
                "sampling_rate_mismatch": 0,
                "xcorr_error": 0,
                "xcorr_too_few_samples": 0,
                "xcorr_empty_window": 0,
                "no_components_correlated": 0,
                "below_min_cc_coeff": 0,
            },
            "accepted_by_phase": {"P": 0, "S": 0},
            "below_threshold_by_phase": {"P": 0, "S": 0},
            "coefficients": [],
        }
        # This is by far the lengthiest operation and will be broken up in
        # smaller steps
        cc_dir = os.path.join(self.paths["working_files"], "cc_files")
        write_pair_files = self.cc_param["cc_write_pair_files"]
        if write_pair_files and not os.path.exists(cc_dir):
            os.makedirs(cc_dir)
        # Read the dt.ct file and get all event pairs.
        dt_ct_path = os.path.join(self.paths["input_files"], "dt.ct")
        if not os.path.exists(dt_ct_path):
            msg = "dt.ct does not exists. Did ph2dt run successfully?"
            raise HypoDDException(msg)
        event_id_pairs = []
        with open(dt_ct_path, "r") as open_file:
            for line in open_file:
                line = line.strip()
                if not line.startswith("#"):
                    continue
                # Remove leading hashtag.
                line = line[1:]
                event_id_1, event_id_2 = list(map(int, line.split()))
                event_id_pairs.append((event_id_1, event_id_2))
        # Now for every event pair, calculate cross correlated differential
        # travel times for every pick.
        # Setup a progress bar.
        self.log(
            "Cross correlating arrival times for %i event_pairs..."
            % len(event_id_pairs)
        )
        pbar = progressbar.ProgressBar(
            widgets=[
                progressbar.Percentage(),
                progressbar.Bar(),
                progressbar.ETA(),
            ],
            maxval=len(event_id_pairs),
        )
        pbar_progress = 1
        pbar.start()
        final_strings = []
        waveform_stream_cache = {}
        trace_data_cache = {}

        def read_waveform_file_cached(filename):
            if filename not in waveform_stream_cache:
                waveform_stream_cache[filename] = read(filename)
                for trace in waveform_stream_cache[filename]:
                    trace.stats.hypoddpy_source_file = filename
            return waveform_stream_cache[filename].copy()

        for event_1, event_2 in event_id_pairs:
            self.cc_diagnostics["event_pairs_total"] += 1
            # Update the progress bar.
            pbar.update(pbar_progress)
            pbar_progress += 1
            # filename for event_pair
            event_pair_file = os.path.join(
                cc_dir, "%i_%i.txt" % (event_1, event_2)
            )
            if write_pair_files and os.path.exists(event_pair_file):
                continue
            current_pair_strings = []
            # Find the corresponding events.
            event_id_1 = self.event_map[event_1]
            event_id_2 = self.event_map[event_2]
            event_1_dict = event_2_dict = None
            for event in self.events:
                if event["event_id"] == event_id_1:
                    event_1_dict = event
                if event["event_id"] == event_id_2:
                    event_2_dict = event
                if event_1_dict is not None and event_2_dict is not None:
                    break
            # Some safety measures to ensure the script keeps running even if
            # something unexpected happens.
            if event_1_dict is None:
                msg = (
                    "Event %s not be found. This is likely a bug." % event_id_1
                )
                self.log(msg, level="warning")
                continue
            if event_2_dict is None:
                msg = (
                    "Event %s not be found. This is likely a bug." % event_id_2
                )
                self.log(msg, level="warning")
                continue
            # Write the leading string in the dt.cc file.
            current_pair_strings.append(
                "# {event_id_1}  {event_id_2} 0.0".format(
                    event_id_1=event_1, event_id_2=event_2
                )
            )
            # Now try to cross-correlate as many picks as possible.
            for pick_1 in event_1_dict["picks"]:
                pick_1_station_id = pick_1["station_id"]
                pick_1_phase = pick_1["phase"]
                # Try to find the corresponding pick for the second event.
                pick_2 = None
                for pick in event_2_dict["picks"]:
                    if (
                        pick["station_id"] == pick_1_station_id
                        and pick["phase"] == pick_1_phase
                    ):
                        pick_2 = pick
                        break
                # No corresponding pick could be found.
                if pick_2 is None:
                    self.cc_diagnostics["rejected"]["no_matching_pick"] += 1
                    continue
                self.cc_diagnostics["pick_pairs_considered"] += 1
                # we got some previously computed information..
                if pick_2["id"] in self.cc_results.get(pick_1["id"], {}):
                    cc_result = self.cc_results.get(pick_1["id"], {})[
                        pick_2["id"]
                    ]
                    # .. and it's actual data
                    if (
                        isinstance(cc_result, (list, tuple))
                        and len(cc_result) == 2
                    ):
                        pick2_corr, cross_corr_coeff = cc_result
                    # .. but it's only an error message or None for a silent skip
                    else:
                        self.cc_diagnostics["rejected"]["preloaded_error"] += 1
                        self.log(
                            "Skipping pick pair due to error message in preloaded cross correlation result: %s"
                            % str(cc_result)
                        )
                        continue
                # we got some previously computed information (but picks were order other way round)..
                elif pick_1["id"] in self.cc_results.get(pick_2["id"], {}):
                    cc_result = self.cc_results.get(pick_2["id"], {})[
                        pick_1["id"]
                    ]
                    # .. and it's actual data
                    if (
                        isinstance(cc_result, (list, tuple))
                        and len(cc_result) == 2
                    ):
                        # revert time correction for other pick order!
                        pick2_corr, cross_corr_coeff = (
                            -cc_result[0],
                            cc_result[1],
                        )
                    # .. but it's only an error message or None for a silent skip
                    else:
                        self.cc_diagnostics["rejected"]["preloaded_error"] += 1
                        self.log(
                            "Skipping pick pair due to error message in preloaded cross correlation result: %s"
                            % str(cc_result)
                        )
                        continue
                else:
                    station_id = pick_1["station_id"]
                    # Try to find data for both picks.
                    data_files_1 = self._find_data(
                        station_id,
                        pick_1["pick_time"] - self.cc_param["cc_time_before"],
                        self.cc_param["cc_time_before"]
                        + self.cc_param["cc_time_after"],
                        event_id=event_1_dict["event_id"],
                        phase=pick_1_phase,
                    )
                    data_files_2 = self._find_data(
                        station_id,
                        pick_2["pick_time"] - self.cc_param["cc_time_before"],
                        self.cc_param["cc_time_before"]
                        + self.cc_param["cc_time_after"],
                        event_id=event_2_dict["event_id"],
                        phase=pick_1_phase,
                    )
                    # If any pick has no data, skip this pick pair.
                    if data_files_1 is False or data_files_2 is False:
                        self.cc_diagnostics["rejected"]["no_waveform_files"] += 1
                        continue
                    # Read all files.
                    stream_1 = Stream()
                    stream_2 = Stream()
                    for waveform_file in data_files_1:
                        stream_1 += read_waveform_file_cached(waveform_file)
                    for waveform_file in data_files_2:
                        stream_2 += read_waveform_file_cached(waveform_file)
                    # Get the corresponing pick weighting dictionary.
                    if pick_1_phase == "P":
                        pick_weight_dict = self.cc_param[
                            "cc_p_phase_weighting"
                        ]
                    elif pick_1_phase == "S":
                        pick_weight_dict = self.cc_param[
                            "cc_s_phase_weighting"
                        ]
                    all_cross_correlations = []
                    # Loop over all picks and weight them.
                    for channel, channel_weight in pick_weight_dict.items():
                        if channel_weight == 0.0:
                            continue
                        # Filter the files to obtain the correct trace.
                        if "." in station_id:
                            network, station = station_id.split(".")
                        else:
                            network = "*"
                            station = station_id
                        st_1 = stream_1.select(
                            network=network,
                            station=station,
                            channel="*%s" % channel,
                        )
                        st_2 = stream_2.select(
                            network=network,
                            station=station,
                            channel="*%s" % channel,
                        )
                        max_starttime_st_1 = (
                            pick_1["pick_time"]
                            - self.cc_param["cc_time_before"]
                        )
                        min_endtime_st_1 = (
                            pick_1["pick_time"]
                            + self.cc_param["cc_time_after"]
                        )
                        max_starttime_st_2 = (
                            pick_2["pick_time"]
                            - self.cc_param["cc_time_before"]
                        )
                        min_endtime_st_2 = (
                            pick_2["pick_time"]
                            + self.cc_param["cc_time_after"]
                        )
                        # Attempt to find the correct trace.
                        for trace in st_1:
                            if (
                                trace.stats.starttime > max_starttime_st_1
                                or trace.stats.endtime < min_endtime_st_1
                            ):
                                st_1.remove(trace)
                        for trace in st_2:
                            if (
                                trace.stats.starttime > max_starttime_st_2
                                or trace.stats.endtime < min_endtime_st_2
                            ):
                                st_2.remove(trace)

                        # cleanup merges, in case the event is included in
                        # multiple traces (happens for events with very close
                        # origin times)
                        st_1.merge(-1)
                        st_2.merge(-1)

                        if len(st_1) > 1:
                            self.cc_diagnostics["rejected"][
                                "multiple_matching_trace_1"
                            ] += 1
                            msg = (
                                "More than one %s matching trace found for %s"
                                % (channel, str(pick_1))
                            )
                            self.log(msg, level="warning")
                            self.cc_results.setdefault(pick_1["id"], {})[
                                pick_2["id"]
                            ] = msg
                            continue
                        elif len(st_1) == 0:
                            self.cc_diagnostics["rejected"][
                                "no_matching_trace_1"
                            ] += 1
                            msg = f"No matching {channel} trace found for {str(pick_1)}"
                            if not self.supress_warnings["no_matching_trace"]:
                                self.log(msg, level="warning")
                            self.cc_results.setdefault(pick_1["id"], {})[
                                pick_2["id"]
                            ] = msg
                            continue
                        trace_1 = st_1[0]

                        if len(st_2) > 1:
                            self.cc_diagnostics["rejected"][
                                "multiple_matching_trace_2"
                            ] += 1
                            msg = (
                                "More than one matching %s trace found for %s"
                                % (channel, str(pick_2))
                            )
                            self.log(msg, level="warning")
                            self.cc_results.setdefault(pick_1["id"], {})[
                                pick_2["id"]
                            ] = msg
                            continue
                        elif len(st_2) == 0:
                            self.cc_diagnostics["rejected"][
                                "no_matching_trace_2"
                            ] += 1
                            msg = f"No matching {channel} trace found for {channel}  {str(pick_2)}"
                            if not self.supress_warnings["no_matching_trace"]:
                                self.log(msg, level="warning")
                            self.cc_results.setdefault(pick_1["id"], {})[
                                pick_2["id"]
                            ] = msg
                            continue
                        trace_2 = st_2[0]

                        if trace_1.id != trace_2.id:
                            self.cc_diagnostics["rejected"][
                                "non_matching_trace_ids"
                            ] += 1
                            msg = "Non matching ids during cross correlation. "
                            msg += "(%s and %s)" % (trace_1.id, trace_2.id)
                            self.log(msg, level="warning")
                            self.cc_results.setdefault(pick_1["id"], {})[
                                pick_2["id"]
                            ] = msg
                            continue
                        if (
                            trace_1.stats.sampling_rate
                            != trace_2.stats.sampling_rate
                        ):
                            self.cc_diagnostics["rejected"][
                                "sampling_rate_mismatch"
                            ] += 1
                            msg = (
                                "Non matching sampling rates during cross "
                                "correlation. "
                            )
                            msg += "(%s and %s)" % (trace_1.id, trace_2.id)
                            self.log(msg, level="warning")
                            self.cc_results.setdefault(pick_1["id"], {})[
                                pick_2["id"]
                            ] = msg
                            continue

                        # Call the cross correlation function.
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            cc_success = False
                            try:
                                filter_name = self.cc_param["cc_filter"]
                                filter_options = None
                                if filter_name == "bandpass":
                                    filter_options = {
                                        "freqmin": self.cc_param[
                                            "cc_filter_min_freq"
                                        ],
                                        "freqmax": self.cc_param[
                                            "cc_filter_max_freq"
                                        ],
                                    }
                                if self.cc_param["cc_backend"] == "numpy_cache":
                                    (
                                        pick2_corr,
                                        cross_corr_coeff,
                                    ) = _numpy_cache_xcorr_pick_correction(
                                        pick_1["pick_time"],
                                        trace_1,
                                        pick_2["pick_time"],
                                        trace_2,
                                        t_before=self.cc_param["cc_time_before"],
                                        t_after=self.cc_param["cc_time_after"],
                                        cc_maxlag=self.cc_param["cc_maxlag"],
                                        filter=filter_name,
                                        filter_options=filter_options,
                                        trace_cache=trace_data_cache,
                                    )
                                else:
                                    (
                                        pick2_corr,
                                        cross_corr_coeff,
                                    ) = xcorr_pick_correction(
                                        pick_1["pick_time"],
                                        trace_1,
                                        pick_2["pick_time"],
                                        trace_2,
                                        t_before=self.cc_param["cc_time_before"],
                                        t_after=self.cc_param["cc_time_after"],
                                        cc_maxlag=self.cc_param["cc_maxlag"],
                                        filter=filter_name,
                                        filter_options=filter_options,
                                        plot=False,
                                    )
                                cc_success = True
                            except Exception as err:
                                # XXX: Maybe maxlag is too short?
                                # if not err.message.startswith("Less than 3"):
                                err_msg = str(err)
                                if err_msg.startswith("Less than 3"):
                                    self.cc_diagnostics["rejected"][
                                        "xcorr_too_few_samples"
                                    ] += 1
                                    msg = "Error during cross correlating: "
                                    msg += err_msg
                                    self.cc_results.setdefault(
                                        pick_1["id"], {}
                                    )[pick_2["id"]] = msg
                                    continue
                                elif "empty array" in err_msg.lower():
                                    self.cc_diagnostics["rejected"][
                                        "xcorr_empty_window"
                                    ] += 1
                                    msg = "Error during cross correlating: "
                                    msg += err_msg
                                    self.cc_results.setdefault(
                                        pick_1["id"], {}
                                    )[pick_2["id"]] = msg
                                    continue
                                else:
                                    self.cc_diagnostics["rejected"][
                                        "xcorr_error"
                                    ] += 1
                                    msg = "Error during cross correlating: "
                                    msg += err_msg
                                    # msg += err.message
                                    self.log(msg, level="error")
                                    self.cc_results.setdefault(
                                        pick_1["id"], {}
                                    )[pick_2["id"]] = msg
                                    continue
                        if not cc_success:
                            self.cc_diagnostics["rejected"][
                                "xcorr_error"
                            ] += 1
                            self.cc_results.setdefault(pick_1["id"], {})[
                                pick_2["id"]
                            ] = "Cross correlation did not return a result"
                            continue
                        all_cross_correlations.append(
                            (pick2_corr, cross_corr_coeff, channel_weight)
                        )
                    if len(all_cross_correlations) == 0:
                        self.cc_diagnostics["rejected"][
                            "no_components_correlated"
                        ] += 1
                        self.cc_results.setdefault(pick_1["id"], {})[
                            pick_2["id"]
                        ] = "No cross correlations performed"
                        continue
                    # Now combine all of them based upon their weight.
                    pick2_corr = sum(
                        [_i[0] * _i[2] for _i in all_cross_correlations]
                    )
                    cross_corr_coeff = sum(
                        [_i[1] * _i[2] for _i in all_cross_correlations]
                    )
                    weight = sum([_i[2] for _i in all_cross_correlations])
                    pick2_corr /= weight
                    cross_corr_coeff /= weight
                    self.cc_results.setdefault(pick_1["id"], {})[
                        pick_2["id"]
                    ] = (
                        pick2_corr,
                        cross_corr_coeff,
                    )
                # If the cross_corr_coeff is under the allowed limit, discard
                # it.
                if (
                    cross_corr_coeff
                    < self.cc_param["cc_min_allowed_cross_corr_coeff"]
                ):
                    self.cc_diagnostics["rejected"]["below_min_cc_coeff"] += 1
                    self.cc_diagnostics["below_threshold_by_phase"][
                        pick_1["phase"]
                    ] = (
                        self.cc_diagnostics["below_threshold_by_phase"].get(
                            pick_1["phase"], 0
                        )
                        + 1
                    )
                    self.cc_diagnostics["coefficients"].append(cross_corr_coeff)
                    continue
                # Otherwise calculate the corrected differential travel time.
                diff_travel_time = (
                    (pick_1["pick_time"]
                    - event_1_dict["origin_time"]) -
                    (pick_2["pick_time"]
                    + pick2_corr
                    - event_2_dict["origin_time"]))
                string = "{station_id} {travel_time:.6f} {weight:.4f} {phase}"
                string = string.format(
                    station_id=pick_1["station_id"],
                    travel_time=diff_travel_time,
                    weight=cross_corr_coeff,
                    phase=pick_1["phase"],
                )
                current_pair_strings.append(string)
                self.cc_diagnostics["accepted"] += 1
                self.cc_diagnostics["accepted_by_phase"][pick_1["phase"]] = (
                    self.cc_diagnostics["accepted_by_phase"].get(
                        pick_1["phase"], 0
                    )
                    + 1
                )
                self.cc_diagnostics["coefficients"].append(cross_corr_coeff)
            # Write the file.
            current_pair_string = "\n".join(current_pair_strings)
            if write_pair_files:
                with open(event_pair_file, "w") as open_file:
                    open_file.write(current_pair_string)
            else:
                final_strings.append(current_pair_string)
        pbar.finish()
        self.log("Finished calculating cross correlations.")
        if outfile:
            self.save_cross_correlation_results(outfile)
        # Assemble final file.
        if write_pair_files:
            final_strings = []
            for cc_file in glob.iglob(os.path.join(cc_dir, "*.txt")):
                with open(cc_file, "r") as open_file:
                    final_strings.append(open_file.read().strip())
        final_string = "\n".join(final_strings)
        with open(ct_file_path, "w") as open_file:
            open_file.write(final_string)

    def _find_data(self, station_id, starttime, duration, event_id=None, phase=None):
        """"
        Parses the self.waveform_information dictionary and returns a list of
        filenames containing traces of the seeked information.

        Returns False if it could not find any corresponding waveforms.

        :param station_id: Station id in the form network.station
        :param starttime: The minimum starttime of the data.
        :param duration: The minimum duration of the data.
        :param event_id: Optional event resource id. If waveform files are named
            like eventid_phase.mseed, matching files are preferred.
        :param phase: Optional phase label used with event_id filename matching.
        """
        endtime = starttime + duration
        # Find all possible keys for the station_id.
        if "." in station_id:
            id_pattern = f"{station_id}.*.*"
        else:
            id_pattern = f"*.{station_id}.*.*"
        station_keys = [
            _i
            for _i in list(self.waveform_information.keys())
            if fnmatch.fnmatch(_i, id_pattern)
        ]
        filenames = []
        for key in station_keys:
            for waveform in self.waveform_information[key]:
                if waveform["starttime"] > starttime:
                    continue
                if waveform["endtime"] < endtime:
                    continue
                filenames.append(waveform["filename"])
        if len(filenames) == 0:
            return False

        if event_id is not None and phase is not None:
            phase = str(phase).upper()
            event_id = str(event_id)
            event_id_candidates = [
                event_id,
                event_id.rstrip("/").split("/")[-1],
                event_id.rstrip(":").split(":")[-1],
            ]
            preferred_filenames = []
            for filename in filenames:
                basename = os.path.basename(filename)
                stem, _ = os.path.splitext(basename)
                for candidate in event_id_candidates:
                    if not candidate:
                        continue
                    if stem == "%s_%s" % (candidate, phase):
                        preferred_filenames.append(filename)
                        break
            if len(preferred_filenames) > 0:
                filenames = preferred_filenames

        return list(set(filenames))

    def _write_hypoDD_inp_file(self):
        """
        Writes the hypoDD.inp file.
        """
        hypodd_inp_path = os.path.join(self.paths["input_files"], "hypoDD.inp")
        if os.path.exists(hypodd_inp_path):
            self.log("hypoDD.inp input file already exists.")
            return
        # Use this way of defining the string to avoid leading whitespaces.
        hypodd_inp = "\n".join(
            [
                "hypoDD_2",
                "dt.cc",
                "dt.ct",
                "event.sel",
                "station.sel",
                "",
                "",
                "hypoDD.sta",
                "hypoDD.res",
                "hypoDD.src",
                "{IDAT} {IPHA} {DIST}",
                "{OBSCC} {OBSCT} {MINDS} {MAXDS} {MAXGAP}",
                "{ISTART} {ISOLV} {IAQ} {NSET}",
                "{DATA_WEIGHTING_AND_REWEIGHTING}",
                "{FORWARD_MODEL}",
                "{CID}",
                "{ID}",
            ]
        )
        # Determine all the values.
        values = {}
        values["IDAT"] = 3 if self.use_cross_correlation else 2
        values["IPHA"] = 3
        # Max distance between centroid of event cluster and stations.
        values["DIST"] = self.forced_configuration_values["MAXDIST"]
        if self.use_cross_correlation:
            # If IDAT=3, the sum of OBSCC and OBSCT is taken for both.
            values["OBSCC"] = 8
            values["OBSCT"] = 0
        else:
            values["OBSCC"] = 0
            values["OBSCT"] = 8
        # Set min/max distances/azimuthal gap to -999 (not used)
        values["MINDS"] = -999
        values["MAXDS"] = -999
        values["MAXGAP"] = -999
        # Start from catalog locations
        values["ISTART"] = 2
        # Least squares solution via conjugate gradients
        values["ISOLV"] = 2
        values["IAQ"] = 2
        # Create the data_weighting and reweightig scheme. Currently static.
        if self.use_cross_correlation:
            # Iterative 10 times for only cross correlated travel time data and
            # then 10 times also including catalog data.
            iterations = [
                "100 1 0.5 -999 -999 0.1 0.05 -999 -999 30",
                "100 1 0.5 6 -999 0.1 0.05 6 -999 30",
            ]
        else:
            iterations = [
                "100 0 0 -999 -999 0.1 0.05 6 -999 30",
            ]
        for key in [
            "IPHA",
            "OBSCC",
            "OBSCT",
            "MINDS",
            "MAXDS",
            "MAXGAP",
            "ISTART",
            "ISOLV",
            "IAQ",
        ]:
            if key in self.forced_configuration_values:
                values[key] = self.forced_configuration_values[key]
        if "ITERATIONS" in self.forced_configuration_values:
            iterations = self.forced_configuration_values["ITERATIONS"]
        elif "DATA_WEIGHTING_AND_REWEIGHTING" in self.forced_configuration_values:
            iterations = self.forced_configuration_values[
                "DATA_WEIGHTING_AND_REWEIGHTING"
            ]
        values["NSET"] = len(iterations)
        values["DATA_WEIGHTING_AND_REWEIGHTING"] = "\n".join(iterations)
        values["FORWARD_MODEL"] = self._get_forward_model_string()
        # Allow relocating of all clusters.
        values["CID"] = 0
        # Also of all events.
        values["ID"] = ""
        if "CID" in self.forced_configuration_values:
            values["CID"] = self.forced_configuration_values["CID"]
        if "ID" in self.forced_configuration_values:
            values["ID"] = self.forced_configuration_values["ID"]
        hypodd_inp = hypodd_inp.format(**values)
        with open(hypodd_inp_path, "w") as open_file:
            open_file.write(hypodd_inp)
        self.log("Created hypoDD.inp input file.")

    def setup_velocity_model(self, model_type, **kwargs):
        """
        Defines the used velocity model for the forward simulation. The chosen
        model_type determines the available kwargs.

        Possible model_types:

        * layered_p_velocity_with_constant_vp_vs_ratio

        :param vp_vs_ratio: The vp/vs ratio for all layers
        :param layer_tops: List of (depth_of_top_of_layer, layer_velocity_km_s)
           e.g. to define five layers:
            [(0.0, 3.77), (1.0, 4.64), (3.0, 5.34), (6.0, 5.75), (14.0, 6.0)]
            
        * layered_variable_vp_vs_ratio (IMOD 1 in hypodd2.1)
        
        :param layer_tops: List of (depth_of_top_of_layer, layer_velocity_km_s,
        layer_ratios)
            e.g. to define five layers:
            [(-3.0, 3.42, 2.38), 
             (0.5, 4.6, 1.75), 
             (1.0, 5.42, 1.74), 
             (1.5, 5.52, 1.75),
             (2.13, 5.67, 1.77)]
        """
        if model_type == "layered_p_velocity_with_constant_vp_vs_ratio":
            # Check the kwargs.
            if not "layer_tops" in kwargs:
                msg = "layer_tops need to be defined"
                raise HypoDDException(msg)
            if not "vp_vs_ratio" in kwargs:
                msg = "vp_vs_ratio need to be defined"
                raise HypoDDException(msg)
            ratio = float(kwargs.get("vp_vs_ratio"))
            layers = kwargs.get("layer_tops")
            if len(layers) > 30:
                msg = "Model must have <= 30 layers"
                raise HypoDDException(msg)
            depths = [str(_i[0]) for _i in layers]
            velocities = [str(_i[1]) for _i in layers]
            # Use imod 5 which allows for negative station elevations by using
            # straight rays.
            forward_model = [
                "0",  # IMOD
                "%i %.2f" % (len(layers), ratio),
                " ".join(depths),
                " ".join(velocities),
            ]
            # forward_model = [ \
            # "0",  # IMOD
            ## If IMOD=0, number of layers and v_p/v_s ration.
            # "{layer_count} {ratio}".format(layer_count=len(layers),
            # ratio=ratio),
            ## Depth of the layer tops.
            # " ".join(depths),
            ## P wave velocity of layers.
            # " ".join(velocities)]
            self.forward_model_string = "\n".join(forward_model)
            
        elif model_type == "layered_variable_vp_vs_ratio":
            """
            *--- 1D model, variable  vp/vs ratio:
            * TOP:          depths of top of layer (km)
            * VEL:          layer velocities (km/s) end w/ -9
            * RATIO:        layer ratios  end w/ -9
            * IMOD: 1
            """
            if not "layer_tops" in kwargs:
                msg = "layer_tops need to be defined"
                raise HypoDDException(msg)
            layers = kwargs.get("layer_tops")
            if len(layers) > 30:
                msg = "Model must have <= 30 layers"
                raise HypoDDException(msg)
            depths = [str(_i[0]) for _i in layers]
            velocities = [str(_i[1]) for _i in layers]
            ratios = [str(_i[2]) for _i in layers]
            depths.append('-9')
            velocities.append('-9')
            ratios.append('-9')
            # Use imod 5 which allows for negative station elevations by using
            # straight rays.
            forward_model = [
                "1",  # IMOD
                " ".join(depths),
                " ".join(velocities),
                " ".join(ratios)
            ]
            self.forward_model_string = "\n".join(forward_model)
        else:
            msg = "Model type {model_type} unknown."
            msg.format(model_type=model_type)
            raise HypoDDException(msg)

    def _get_forward_model_string(self):
        """
        Returns the forward model specification for hypoDD.inp.
        """
        if not hasattr(self, "forward_model_string"):
            msg = (
                "Velocity model could not be found. Did you run the "
                + "setup_velocity_model() method?"
            )
            raise HypoDDException(msg)
        return self.forward_model_string

    def _create_output_event_file(self):
        """
        Write the final output file in QuakeML format.
        """
        self.log("Writing final output file...")
        hypodd_reloc = os.path.join(
            os.path.join(self.working_dir, "output_files", "hypoDD.reloc")
        )

        cat = Catalog()
        self.output_catalog = cat
        for filename in self.event_files:
            cat += read_events(filename)

        with open(hypodd_reloc, "r") as open_file:
            for line in open_file:
                (
                    event_id,
                    lat,
                    lon,
                    depth,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    year,
                    month,
                    day,
                    hour,
                    minute,
                    second,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    cluster_id,
                ) = line.split()
                event_id = self.event_map[int(event_id)]
                cluster_id = int(cluster_id)
                res_id = ResourceIdentifier(event_id)
                lat, lon, depth = list(map(float, [lat, lon, depth]))
                # Convert back to meters.
                depth *= 1000.0
                # event = res_id.getReferredObject()
                event = res_id.get_referred_object()
                # Create new origin.
                new_origin = Origin()
                sec = int(float(second))
                # Correct for a bug in hypoDD which can write 60 seconds...
                add_minute = False
                if sec >= 60:
                    sec = 0
                    add_minute = True
                new_origin.time = UTCDateTime(
                    int(year),
                    int(month),
                    int(day),
                    int(hour),
                    int(minute),
                    sec,
                    int((float(second) % 1.0) * 1e6),
                )
                if add_minute is True:
                    new_origin.time = new_origin.time + 60.0
                new_origin.latitude = lat
                new_origin.longitude = lon
                new_origin.depth = depth
                new_origin.method_id = "HypoDD"
                # Put the cluster id in the comments to be able to use it later
                # on.
                new_origin.comments.append(
                    Comment(text="HypoDD cluster id: %i" % cluster_id)
                )
                event.origins.append(new_origin)
        cat.write(self.output_event_file, format="quakeml")

        self.log("Finished! Final output file: %s" % self.output_event_file)

    def _create_plots(self, marker_size=2):
        """
        Creates some plots of the relocated event Catalog.
        """
        import matplotlib.pylab as plt
        from matplotlib.cm import get_cmap
        from matplotlib.colors import ColorConverter

        catalog = self.output_catalog
        # Generate the output plot filenames.
        original_filename = os.path.join(
            self.paths["output_files"], "original_event_location.pdf"
        )
        relocated_filename = os.path.join(
            self.paths["output_files"], "relocated_event_location.pdf"
        )

        # Some lists to store everything in.
        original_latitudes = []
        original_longitudes = []
        original_depths = []
        relocated_latitudes = []
        relocated_longitudes = []
        relocated_depths = []
        # The colors will be used to distinguish between different event types.
        # grey: event will/have not been relocated.
        # red: relocated with cluster id 1
        # green: relocated with cluster id 2
        # blue: relocated with cluster id 3
        # yellow: relocated with cluster id 4
        # orange: relocated with cluster id 5
        # cyan: relocated with cluster id 6
        # magenta: relocated with cluster id 7
        # brown: relocated with cluster id 8
        # lime: relocated with cluster id >8 or undetermined cluster_id
        color_invalid = ColorConverter().to_rgba("grey")
        cmap = get_cmap("Paired", 12)

        colors = []
        magnitudes = []

        for event in catalog:
            # If the last origin was not created by hypoDD then both the
            # original and the relocated origin is the origin designated
            # as preferred.
            if (
                event.origins[-1].method_id is None
                or "HYPODD" not in str(event.origins[-1].method_id).upper()
            ):
                original_origin = event.preferred_origin()
                relocated_origin = event.preferred_origin()
            # If the last origin was created by hypoDD then the original
            # origin is the origin designated as preferred, and the
            # relocated origin is the last origin.
            else:
                original_origin = event.preferred_origin()
                relocated_origin = event.origins[-1]
            original_latitudes.append(original_origin.latitude)
            original_longitudes.append(original_origin.longitude)
            original_depths.append(original_origin.depth / 1000.0)
            relocated_latitudes.append(relocated_origin.latitude)
            relocated_longitudes.append(relocated_origin.longitude)
            relocated_depths.append(relocated_origin.depth / 1000.0)
            if event.preferred_magnitude() is not None:
                magnitudes.append(event.preferred_magnitude())
            else:
                magnitudes.append(Magnitude(mag=0.0))

            # Use color to Code the different events. Colorcode by event
            # cluster or indicate if an event did not get relocated.
            if (
                relocated_origin.method_id is None
                or "HYPODD" not in str(relocated_origin.method_id).upper()
            ):
                colors.append(color_invalid)
            # Otherwise get the cluster id, stored in the comments.
            else:
                for comment in relocated_origin.comments:
                    comment = comment.text
                    if comment and "HypoDD cluster id" in comment:
                        cluster_id = int(comment.split(":")[-1])
                        break
                else:
                    cluster_id = 0
                colors.append(cmap(int(cluster_id)))

        # Plot the original event location.
        plt.subplot(221)
        plt.scatter(
            original_latitudes, original_depths, s=marker_size ** 2, c=colors
        )
        plt.xlabel("Latitude")
        plt.ylabel("Depth in km")
        # Invert the depth axis.
        plt.ylim(plt.ylim()[::-1])
        plot1_xlim = plt.xlim()
        plot1_ylim = plt.ylim()
        plt.subplot(222)
        plt.scatter(
            original_longitudes, original_depths, s=marker_size ** 2, c=colors
        )
        plt.xlabel("Longitude")
        plt.ylabel("Depth in km")
        plt.ylim(plt.ylim()[::-1])
        # Invert the depth axis.
        plot2_xlim = plt.xlim()
        plot2_ylim = plt.ylim()
        plt.subplot(212)
        plt.scatter(
            original_longitudes,
            original_latitudes,
            s=marker_size ** 2,
            c=colors,
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plot3_xlim = plt.xlim()
        plot3_ylim = plt.ylim()
        plt.savefig(original_filename)
        self.log("Output figure: %s" % original_filename)

        # Plot the relocated event locations.
        plt.clf()
        plt.subplot(221)
        plt.scatter(
            relocated_latitudes, relocated_depths, s=marker_size ** 2, c=colors
        )
        plt.xlabel("Latitude")
        plt.ylabel("Depth in km")
        plt.xlim(plot1_xlim)
        plt.ylim(plot1_ylim)
        plt.subplot(222)
        plt.scatter(
            relocated_longitudes,
            relocated_depths,
            s=marker_size ** 2,
            c=colors,
        )
        plt.xlabel("Longitude")
        plt.ylabel("Depth in km")
        plt.xlim(plot2_xlim)
        plt.ylim(plot2_ylim)
        plt.subplot(212)
        plt.scatter(
            relocated_longitudes,
            relocated_latitudes,
            s=marker_size ** 2,
            c=colors,
        )
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.xlim(plot3_xlim)
        plt.ylim(plot3_ylim)
        plt.savefig(relocated_filename)
        self.log("Output figure: %s" % relocated_filename)
