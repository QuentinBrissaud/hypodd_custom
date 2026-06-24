# HypoDDPy Configuration Reference

This file describes the parameters accepted by `HypoDDRelocator.load_configuration_file()`.

There are two example configuration files in this repository:

- `default_hypodd_config.ini`: preserves the wrapper's default behavior as closely as possible.
- `example_hypodd_config.ini`: an arrival-time-only example intended for catalog picks without waveform cross-correlation.

Blank values mean "leave unset and let the wrapper choose its default".

## `[relocator]`

### `use_cross_correlation`

Boolean. If `true`, HypoDDPy uses waveform cross-correlation differential times (`dt.cc`) together with catalog differential times (`dt.ct`). This corresponds to HypoDD `IDAT=3`.

If `false`, HypoDDPy skips waveform cross-correlation and uses catalog arrival-time picks only. This corresponds to HypoDD `IDAT=2`.

### `event_fix`

Optional integer written into `event.sel` as HypoDD's `ev_fix` flag.

Common values are:

- blank: do not add a fixed-parameter flag.
- `1`: fix depth.

Other `ev_fix` values are interpreted by the underlying HypoDD Fortran code.

### `fixed_depth_km`

Optional float in kilometers. If set, all selected event depths are replaced with this value before relocation.

For surface-only relocation, use a small positive depth such as `0.01` km instead of exactly `0.0` km, because HypoDD's layered travel-time code can behave poorly for exactly zero-depth sources.

### `enforce_mean_shift_constraint`

Boolean. If `true`, HypoDDPy patches the LSQR solver before compiling HypoDD to add a zero-mean model-update constraint per cluster:

```text
sum(dx) = 0
sum(dy) = 0
sum(dz) = 0
sum(dt) = 0
```

This is the LSQR equivalent of the mean-shift constraint used to keep each cluster centered. Changing this option forces HypoDD to be recompiled.

Default: `false`, which preserves upstream HypoDD behavior.

### `mean_shift_constraint_weight`

Float. Weight applied to the added LSQR mean-shift constraint rows when `enforce_mean_shift_constraint = true`.

This parameter has no effect when `enforce_mean_shift_constraint = false`.

### `lsqr_constraint_weight`

Float. General LSQR fixed-parameter constraint weight used by the patched HypoDD source when `event_fix` is active.

Default: `100.0`, matching the original hardcoded HypoDD value.

### `lsqr_xyz_constraint_weight`

Float. LSQR fixed-location weight used by the patched HypoDD source for the lower-weight x/y/z fixed-parameter constraints.

Default: `10.0`, matching the original hardcoded HypoDD value.

### `lsqr_time_constraint_weight`

Float. LSQR fixed-origin-time weight used by the patched HypoDD source for time-only fixed-parameter constraints.

Default: `1000.0`, matching the original hardcoded HypoDD value.

## `[cross_correlation]`

These parameters control waveform cross-correlation. They are used only when:

```ini
[relocator]
use_cross_correlation = true
```

### `cc_time_before`

Float in seconds. Start the cross-correlation window this many seconds before the pick time.

### `cc_time_after`

Float in seconds. End the cross-correlation window this many seconds after the pick time.

### `cc_maxlag`

Float in seconds. Maximum lag searched when correcting the second pick relative to the first pick.

### `cc_filter`

String. Filter applied before cross-correlation.

Allowed values:

- `bandpass`: apply the configured bandpass filter.
- `none`: do not apply a filter inside `xcorr_pick_correction`.

### `cc_filter_min_freq`

Float in Hz. Lower corner frequency of the bandpass filter applied before cross-correlation.

### `cc_filter_max_freq`

Float in Hz. Upper corner frequency of the bandpass filter applied before cross-correlation.

### `cc_min_allowed_cross_corr_coeff`

Float. Minimum accepted cross-correlation coefficient. Correlations below this value are discarded.

### `cc_p_phase_weighting`

Comma-separated component weights used for P-pick cross-correlation.

Example:

```ini
cc_p_phase_weighting = Z:1.0,E:0.0,N:0.0
```

The component is matched as the final channel character, so `Z` matches channels such as `BHZ` or `HHZ`.

### `cc_s_phase_weighting`

Comma-separated component weights used for S-pick cross-correlation.

Example:

```ini
cc_s_phase_weighting = Z:1.0,E:1.0,N:1.0
```

If you encode artificial phase-specific waveform channels, for example channels ending in `P` and `S`, use matching component keys:

```ini
cc_p_phase_weighting = P:1.0
cc_s_phase_weighting = S:1.0
```

## `[ph2dt]`

These parameters control `ph2dt`, the program that builds catalog differential-time pairs from event picks.

### `MINWGHT`

Float. Minimum pick weight accepted by `ph2dt`.

Picks with weights below this value are ignored.

### `MAXDIST`

Optional float in kilometers. Maximum event-station distance considered by `ph2dt`.

If unset, HypoDDPy computes a value large enough to include all event-station pairs in the loaded catalog and station inventory.

### `MAXSEP`

Optional float in kilometers. Maximum event-event separation for candidate differential-time pairs.

If unset, HypoDDPy computes this automatically from the event geometry. The current automatic value is based on the lower part of the event-pair distance distribution, so it scales with the spatial extent of the catalog.

Smaller values make tighter, more local clusters. Larger values allow more distant event pairs and can merge clusters.

### `MAXTIMESEP_DAYS`

Float in days. Wrapper-only event-pair time filter applied to `dt.ct` after `ph2dt` has created catalog differential-time pairs.

Use `-999` to disable. For example, `MAXTIMESEP_DAYS = 30` keeps only event-pair blocks whose origin times are separated by 30 days or less.

This does not change `ph2dt` itself; it removes event-pair blocks from `dt.ct` before HypoDD runs. If you change this value in an existing working directory, delete the existing `input_files/dt.ct` or use a fresh working directory to regenerate the unfiltered `dt.ct`.

### `MAXNGH`

Integer. Maximum number of neighboring events considered for each event.

Smaller values make the event graph sparser. Larger values create more links and can merge clusters.

### `MINLNK`

Integer. Minimum number of shared observations required for strong event-pair links.

Increasing this value makes the graph stricter and usually creates smaller or more isolated clusters.

### `MINOBS`

Integer. Minimum number of observations required for an event pair to be selected.

Increasing this value rejects weaker event pairs.

### `MAXOBS`

Integer. Maximum number of observations kept per event pair.

This limits how many station-phase observations can contribute to a single event pair.

## `[hypodd]`

These parameters control the HypoDD relocation itself.

### `IPHA`

Integer phase selector.

Common values are:

- `1`: P phases only.
- `2`: S phases only.
- `3`: P and S phases.

### `OBSCC`

Integer. Minimum number of cross-correlation observations required by HypoDD for clustering/relocation.

In catalog-only mode, this is normally `0`.

### `OBSCT`

Integer. Minimum number of catalog differential-time observations required by HypoDD for clustering/relocation.

In catalog-only mode, this is the main HypoDD cluster threshold.

### `MINDS`

Float in kilometers. Minimum event-station distance filter used inside HypoDD.

Use `-999` to disable this filter.

### `MAXDS`

Float in kilometers. Maximum event-station distance filter used inside HypoDD.

Use `-999` to disable this filter.

### `MAXGAP`

Float in degrees. Maximum azimuthal gap filter used inside HypoDD.

Use `-999` to disable this filter.

### `ISTART`

Integer. Starting model selector for HypoDD.

The wrapper default is `2`, which starts from catalog locations.

### `ISOLV`

Integer. Inversion solver selector.

The wrapper default is `2`, which uses LSQR.

### `IAQ`

Integer. Air-quake handling option.

The wrapper default is `2`, which removes air quakes.

### `CID`

Integer. Cluster selector.

Use `0` to relocate all clusters.

### `ID`

Optional integer. Event selector.

Leave blank to relocate all events.

### `ITERATIONS`

One or more HypoDD iteration rows. Each row has ten values:

```text
NITER WTCCP WTCCS MAXRCC MAXDCC WTCTP WTCTS MAXRCT MAXDCT DAMP
```

Where:

- `NITER`: number of iterations for this row.
- `WTCCP`: P-phase cross-correlation weight.
- `WTCCS`: S-phase cross-correlation weight.
- `MAXRCC`: maximum allowed cross-correlation residual.
- `MAXDCC`: maximum allowed cross-correlation event-pair separation.
- `WTCTP`: P-phase catalog differential-time weight.
- `WTCTS`: S-phase catalog differential-time weight.
- `MAXRCT`: maximum allowed catalog differential-time residual.
- `MAXDCT`: maximum allowed catalog event-pair separation.
- `DAMP`: LSQR damping value.

Use `-999` for residual or distance cutoffs that should be disabled.

In arrival-time-only mode, `WTCCP` and `WTCCS` are normally `0`.

## `[compiler]`

HypoDD uses static Fortran array sizes. These parameters control the array dimensions written to `hypoDD.inc` before compilation.

### `MAXEVE0`

Integer. Maximum number of events for selected internal HypoDD arrays.

### `MAXDATA`

Integer. Maximum number of differential-time observations.

Increase this if HypoDD reports that data arrays are too small.

### `MAXDATA0`

Integer. Maximum number of additional input observations used by HypoDD.

Increase this if the Fortran code reports an input-data array limit.

### `MAXCL`

Integer. Maximum number of clusters.

Increase this if HypoDD stops with:

```text
STOP >>> Increase MAXCL in hypoDD.inc.
```

Tighter `ph2dt` clustering parameters, especially smaller `MAXSEP` or larger `MINLNK`/`MINOBS`, can create more clusters and therefore require a larger `MAXCL`.

## Notes On Defaults

The LSQR fixed-parameter weights:

```ini
lsqr_constraint_weight = 100.0
lsqr_xyz_constraint_weight = 10.0
lsqr_time_constraint_weight = 1000.0
```

match the original hardcoded HypoDD values.

The mean-shift constraint is a HypoDDPy patch. It is disabled by default and only affects the compiled Fortran source when `enforce_mean_shift_constraint = true`.
