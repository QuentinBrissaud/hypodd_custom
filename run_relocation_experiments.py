#!/usr/bin/env python
"""
Example batch runner for HypoDDPy relocation parameter experiments.

Edit the USER SETTINGS block, then run:

    python run_relocation_experiments.py

The script writes one folder per setup under OUTPUT_ROOT.
"""

from pathlib import Path
import glob

from relocation_experiment_runner import (
    RelocationExperiment,
    run_relocation_experiments,
)


# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------

BASE_CONFIG_FILE = "example_hypodd_config.ini"
OUTPUT_ROOT = "experiment_runs"

EVENT_FILES = ["events.xml"]
STATION_FILES = ["stations.xml"]

# Set USE_WAVEFORMS to True only for cross-correlation experiments.
USE_WAVEFORMS = False
WAVEFORM_GLOB = (
    "/projects/restricted/REMWAR/REMWAR_Event_Analysis/"
    "waveforms_from_picks/*.mseed"
)

VELOCITY_MODEL = {
    "model_type": "layered_variable_vp_vs_ratio",
    "layer_tops": [
        (-10.00, 5.80, 5.80 / 1.73),
        (100.90, 8.32, 8.32 / 4.61),
    ],
}

# ---------------------------------------------------------------------------
# EXPERIMENT DEFINITIONS
# ---------------------------------------------------------------------------


EXPERIMENTS = [
    RelocationExperiment(
        name="maxsep_5_obs12_damp30",
        description="Baseline catalog-only run with 5 km pairing.",
        overrides={
            "relocator.use_cross_correlation": False,
            "ph2dt.MAXSEP": 5,
            "ph2dt.MAXNGH": 10,
            "ph2dt.MINLNK": 12,
            "ph2dt.MINOBS": 12,
            "ph2dt.MAXOBS": 50,
            "hypodd.OBSCT": 12,
            "hypodd.ITERATIONS": [
                "100 0 0 -999 -999 0.1 0.05 6 -999 30",
            ],
        },
    ),
    RelocationExperiment(
        name="maxsep_10_obs12_damp30",
        description="Same observation threshold with broader 10 km pairing.",
        overrides={
            "relocator.use_cross_correlation": False,
            "ph2dt.MAXSEP": 10,
            "ph2dt.MAXNGH": 10,
            "ph2dt.MINLNK": 12,
            "ph2dt.MINOBS": 12,
            "ph2dt.MAXOBS": 50,
            "hypodd.OBSCT": 12,
            "hypodd.ITERATIONS": [
                "100 0 0 -999 -999 0.1 0.05 6 -999 30",
            ],
        },
    ),
    RelocationExperiment(
        name="maxsep_5_obs20_damp30",
        description="Stricter link threshold at 5 km pairing.",
        overrides={
            "relocator.use_cross_correlation": False,
            "ph2dt.MAXSEP": 5,
            "ph2dt.MAXNGH": 10,
            "ph2dt.MINLNK": 20,
            "ph2dt.MINOBS": 20,
            "ph2dt.MAXOBS": 50,
            "hypodd.OBSCT": 20,
            "hypodd.ITERATIONS": [
                "100 0 0 -999 -999 0.1 0.05 6 -999 30",
            ],
        },
    ),
    RelocationExperiment(
        name="maxsep_5_obs12_damp10",
        description="Lower damping, allowing larger model updates.",
        overrides={
            "relocator.use_cross_correlation": False,
            "ph2dt.MAXSEP": 5,
            "ph2dt.MAXNGH": 10,
            "ph2dt.MINLNK": 12,
            "ph2dt.MINOBS": 12,
            "ph2dt.MAXOBS": 50,
            "hypodd.OBSCT": 12,
            "hypodd.ITERATIONS": [
                "100 0 0 -999 -999 0.1 0.05 6 -999 10",
            ],
        },
    ),
    RelocationExperiment(
        name="maxsep_5_obs12_mean_shift_w1",
        description="Zero mean-shift constraint with weight 1.",
        overrides={
            "relocator.use_cross_correlation": False,
            "relocator.enforce_mean_shift_constraint": True,
            "relocator.mean_shift_constraint_weight": 1.0,
            "ph2dt.MAXSEP": 5,
            "ph2dt.MAXNGH": 10,
            "ph2dt.MINLNK": 12,
            "ph2dt.MINOBS": 12,
            "ph2dt.MAXOBS": 50,
            "hypodd.OBSCT": 12,
            "hypodd.ITERATIONS": [
                "100 0 0 -999 -999 0.1 0.05 6 -999 30",
            ],
        },
    ),
]


def main():
    waveform_files = None
    if USE_WAVEFORMS:
        waveform_files = sorted(glob.glob(WAVEFORM_GLOB))
        if not waveform_files:
            raise FileNotFoundError("No waveform files matched %s" % WAVEFORM_GLOB)

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    summary = run_relocation_experiments(
        experiments=EXPERIMENTS,
        base_config_file=BASE_CONFIG_FILE,
        output_root=OUTPUT_ROOT,
        event_files=EVENT_FILES,
        station_files=STATION_FILES,
        waveform_files=waveform_files,
        velocity_model=VELOCITY_MODEL,
        overwrite=False,
        stop_on_failure=False,
    )

    print("\nExperiment summary")
    print("------------------")
    for row in summary:
        print(
            "{name}: {status} ({working_dir})".format(
                name=row["name"],
                status=row["status"],
                working_dir=row["working_dir"],
            )
        )
        if row["error"]:
            print("  error:", row["error"])


if __name__ == "__main__":
    main()

