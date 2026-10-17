# Baseline, climatology and threshold

[← Pipeline](PIPELINE.md) · [Index](README.md) · [Objects →](OBJECTS.md)

## What fitting produces

`fit_baseline` receives daily temperatures, a checked grid, a `DefinitionSpec`
and a `BaselineSpec`. It returns a `BaselineResult`, **not detected events**.

| Field | Meaning | Layout |
|---|---|---|
| `reference` | Mean historical seasonal reference, or fitted trend + seasonality | Fixed: day-of-year; dynamic: time |
| `threshold` | Absolute temperature used by the detector | Same reference layout |
| `sample_count` | Number of observed samples supporting the seasonal quantile | Day-of-year sampling diagnostics |
| `invalid_reason` | Outside-domain, undefined baseline or insufficient sample flags | Sampling/result layout; inspect variable dimensions |
| `trend`, `seasonal` | Dynamic decomposition terms | Time and spatial dimensions |
| `residual_threshold` | Seasonal percentile of the fitted residual | Day-of-year and spatial dimensions |
| `*_by_year` | Optional thresholds/counts/flags for within-reference correction | Additional `reference_year` dimension |

Inspect `.dataset`, dimensions and attributes; dynamic and fixed results have
different layouts. Use `.check_invariants()` and pass the fitted object to `detect`
with the same scientific specs. `detect` aligns the baseline to the daily dates.
The reference is used for intensity; it is not interchangeable with the threshold.

## Seven methods at a glance

| `method` | Reference layout | Engines (`params` key `engine`) | Install |
|---|---|---|---|
| `fixed_climatology` | Day of year | Core NumPy | Core |
| `detrended` | Time | Core least squares | Core |
| `shifting` | Time | Core | Core |
| `stl` | Time | `vectorized` (default, non-robust) or `statsmodels` (also robust) | Core; `statsmodels` engine needs `.[dynamic]` |
| `state_space` | Time | statsmodels Kalman smoother (no alternative) | `.[dynamic]` |
| `gam` | Time | `vectorized` (default) or `pygam` (reference engine, cell by cell) | Core; `pygam` engine needs `.[dynamic]` |
| `ssa` | Time | Core (fixed-seed randomized decomposition) | Core |

Every method returns a `BaselineResult` with `reference` and `threshold`; `detect`
accepts any of them with the same `DefinitionSpec`. Each dynamic method is selected
by `replace` from a fixed spec, with the parameters it accepts:

```python
from dataclasses import replace

detrended = replace(baseline, method="detrended",
                    params=(("trend_degree", 1), ("n_harmonics", 3)))
shifting = replace(baseline, method="shifting",
                   params=(("window_years", 30), ("min_years", 10)))
stl = replace(baseline, method="stl", params=(("engine", "vectorized"),))
robust_stl = replace(baseline, method="stl",
                     params=(("engine", "statsmodels"), ("robust", 1)))
state_space = replace(baseline, method="state_space",
                      params=(("trend_timescale_days", 5600),))
gam = replace(baseline, method="gam", params=(("engine", "pygam"),))
ssa = replace(baseline, method="ssa",
              params=(("window_days", 5479), ("n_components", 12)))
```

The reference engines run cell by cell and can be much slower. On a test machine,
robust `statsmodels` STL took about two minutes per cell for a 30-year daily series,
while the vectorized engine fitted four cells in about 12 seconds. Try a small
subset before fitting a full grid.

The specs are checked when built; their `params` are checked against each method's
registry when `fit_baseline` runs. The engines are independent implementations of the
same model: [validation](VALIDATION.md) records how closely they agree.

## Fixed historical reference

The synthetic Peru tutorial uses 1991–2020 (the reference of the real OISST case) to answer: *which days in 1991–2025 exceed the
historical seasonal P90 or fall below P10?* Both temperatures and thresholds stay
on the same scale. Marine settings are 11-day pooling, 31-day circular smoothing,
Hyndman–Fan type 7 and `hobday_366_interp`. Only samples within the declared reference
period enter the window. February 29 is interpolated between February 28 and March 1
before smoothing; it remains a date in detection. `min_samples=264` prevents poorly
sampled windows from producing valid thresholds.

AHW uses the configured 15-day pooling, type 8 and no smoothing in these tutorials.
Changing the estimator, smoothing or leap policy can change membership at the
boundary; record them instead of calling every P90 configuration equivalent.

## Six dynamic methods

The implemented decomposition is:

```text
temperature = trend + seasonal + residual
reference(t) = trend(t) + seasonal(t)
threshold(t) = reference(t) + percentile_of_residual(day_of_year)
```

| Method | Implemented model and important effective defaults | Requirements and interpretation |
|---|---|---|
| `detrended` | Degree-1 polynomial trend; 3 annual harmonics; `min_fit_years=3` | Alternative seasonal climatology supported; asks about anomalies relative to fitted trend |
| `shifting` | Seasonal mean over preceding complete years; `window_years=30`, `min_years=10` | Early dates can lack history; adjust reference sampling to available history |
| `stl` | STL on a 366-position calendar grid; trend window 7,305 days, periodic seasonality, vectorized non-robust engine | Complete decomposition series; robust fitting needs `engine="statsmodels"` |
| `state_space` | Local linear trend + 3 trigonometric harmonics; fixed relative variances; trend timescale 5,600 days | Optional statsmodels required; model variances are explicit, not freely estimated |
| `gam` | Penalized spline trend, 1.5 trend splines/decade, 20 seasonal splines; both penalties 0.6 | Vectorized engine in core; optional `engine="pygam"`; extrapolation end effects need review |
| `ssa` | Window 5,479 days, 12 components, trend-period cutoff 10,958 days, 3 harmonics | Complete decomposition series; potentially expensive matrices; short histories cannot resolve all scales |

These are algorithm defaults in 0.0.1, not universal optimal scientific choices.
Override via the immutable `params` tuple. Unknown method parameters raise
`ValidationError` when fitting. Fixed climatology requires no dynamic parameter
registry. The [API](API.md#baselinespec) and [scientific contract](SCIENTIFIC_CONTRACT.md)
give the input contract; [validation](VALIDATION.md) documents measured comparisons.

Starting from the tutorial's objects:

```python
from dataclasses import replace

dynamic = replace(baseline, method="detrended",
                  params=(("trend_degree", 1), ("n_harmonics", 3)))
fitted_dynamic = xhw.fit_baseline(prepared, variable="sst", grid=grid,
                                  baseline=dynamic, definition=definition)
```

Filled observations are excluded from residual quantiles. STL and SSA need a
complete decomposition series; an explicit interpolation decision can therefore
affect model fitting even though filled values do not become quantile samples.

![Fixed versus detrended membership](assets/synthetic_peru/baseline_comparison.png)

*Executed synthetic Peru reconstruction, 1991–2025, reference 1991–2020, identical marine definition.
Color is the percentage of **valid** days belonging to events at each cell; each
panel has its own color scale. Different masks answer different reference questions.
Their unweighted membership Jaccard on jointly valid cell-days is about 0.919,
not a claim that one method is scientifically better. The field is synthetic: the figure
shows how the two references behave, not a climate result.*

## Retrospective and causal fitting

Retrospective fitting can use later information to explain earlier temperatures.
For a causal result, declare `fit_mode="causal"`, `available_until`, `lag_days`,
history and extrapolation. The reference period must end by availability.

```python
causal = replace(dynamic, fit_mode="causal", available_until="2020-12-31",
                 params=(("causal_scheme", "cutoff"), ("lag_days", 0),
                         ("max_extrapolation_days", 30)))
```

This config fits once at the cutoff and limits forecasts to the configured horizon;
it does not make all 2021–2025 dates valid. `expanding` refits using earlier data,
with method-specific behavior; the default refit interval is 30 days. For that
scheme, `lag_days + refit_every_days <= max_extrapolation_days` is required.
An available causal method is not evidence of accurate extrapolation. In particular,
the recorded GAM end-effect discrepancy reaches approximately 0.43 °C in one
validation configuration; consult that evidence before use.

Availability must also constrain preprocessing. Fitting in causal mode cannot
undo a temperature previously interpolated using observations beyond the cutoff,
or a domain/quality decision inferred from unavailable future data. Restrict those
inputs deliberately when designing a causal workflow. The executed tutorials
here use retrospective fitting.

## Correction within the reference period

The default is `in_base_correction="none"`, including the
tutorials. Those years contributed to their own threshold. An artificial
in-base/out-of-base change is possible and must be considered in rate comparisons.

| Choice | Output | Cost and consequence |
|---|---|---|
| `none` | One seasonal threshold | Uses one common threshold for all reference years |
| `leave_one_year_out` | Daily binary masks using target-year thresholds from other years | One fewer reference year can violate `min_samples`; stores year-specific fields |
| `mean_threshold` | Mean of thresholds from year-substitution replicas, then binary detection | Many more fits; differs from averaging detection rates |
| `in_base_rates` | Annual fractions and valid-day counts from bootstrap replicas | No single daily mask; cannot substitute directly for `detect` |

```python
corrected = replace(baseline, in_base_correction="leave_one_year_out")
```

For dynamic methods, correction changes the residual quantile, not trend or
seasonality. The reference climatology itself is unchanged. Detection near the
reference boundary can consequently change just beyond it as well.

## Memory and fitting cost

Exact temporal quantiles need all reference dates for each processed spatial
chunk. Dynamic baselines also store trend, seasonality, reference and threshold
over time: estimate at least `4 * time * cells * 8` bytes just for these float64
fields, then add input, masks, workspaces and validation. Per-year correction can
multiply quantile storage by the number of reference years. A Dask graph delays
work; it does not remove these requirements. The synthetic Peru tutorial compares only
fixed and detrended to keep a complete, reviewable run affordable.

## STL attribution

Bonino et al. (2023) apply STL to marine heatwaves, using trend and seasonal windows
different from xHW defaults. See [references and the exact scope](REFERENCES.md#attribution-and-configuration-limits).
The citation does not change xHW defaults or imply reproduction of their macroevents.
