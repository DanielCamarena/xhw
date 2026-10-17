# Scientific contract and limits

## Data and identity

Inputs are xarray datasets with explicit variable names, temperature units, calendar,
daily clock, grid topology, domain mask and physical cell measures. Detection operates
on the supplied native analysis grid. Days must be ordered, unique and regularly
spaced; missing observations and missing timestamps have different meanings.
Supported calendars are `standard`, `gregorian` and `proleptic_gregorian`, represented
by `datetime64`, for dates on or after 1582-10-15. Temperature aliases cover Celsius
and Kelvin; users must maintain consistent units across components and thresholds.

The public API is `xhw.__all__`. Specs are immutable metadata contracts. Their
`spec_id` is a full SHA-256 of canonical JSON including entity type and schema major
version. Designated optional fields are omitted from the identity payload when they hold
their default values, including in nested Specs. Package and entity-schema versions are
independent. Serialized products store complete Specs, units, flags and provenance.
Readers recalculate identifiers and reject incompatible schema majors.

`CouplingSpec` validates declarative metadata and catalog identities; it does not
compute associations. Presence of metadata fields alone is not an executed analysis.

## Temporal extremes

MHW and MCS use the upper and lower thermal tails of a common detector. The
`hobday2016` configuration uses P90/P10, an 11-day pooling window, 31-day circular
smoothing, Hyndman-Fan type 7, strict comparison, minimum duration 5 days and gaps
up to 2 days. Qualify runs before joining them: two subminimum runs cannot become
an event merely by joining. Baseline samples stay strictly inside the declared
reference period. The `hobday_366_interp` policy interpolates 29 February between
28 February and 1 March without discarding that day from detection.

`interpolate_gaps` is an explicit preprocessing operation: only interior gaps up
to its declared limit are interpolated, and filled days are flagged. A series
exceeding the declared filled fraction is invalid. Baseline estimation and detection
record their preprocessing separately. `missing_policy="break"` breaks a run;
`"as_non_exceedance"` can include a missing day as a joined gap between qualified
runs. Such a day remains invalid and is excluded from intensity averages.

`XMask` preserves `exceedance`, `event_membership`, `valid`, `joined_gap`,
`invalid_reason` and, when present, `filled`. Invalid-reason bits distinguish outside
domain, missing data, undefined baseline, insufficient samples and low data quality.
A joined gap is event membership without exceedance, never a new thermal observation.
Intensity is temperature minus reference; threshold-relative excess is separate.
Cold anomalies remain signed. Heat categories use `hobday2018`; cold categories
use the explicitly named `schlegel2021` convention.

AHW profiles `perkins2013_tx90` and `perkins2013_tn90` use a 15-day window, type 8,
no threshold smoothing, strict P90 exceedance, minimum duration 3 days and gap 0.
TX90/TN90 runs differ from annual percentage indices. Joint `and_daily` detection
combines daily exceedances before applying joint persistence. Each DailyComponent
has its own variable, statistic, percentile and lag (0 or 1 day). Invalid components
make the combined day invalid; a shifted final day is censored by missing context.

`aggregate_daily` declares statistic, instantaneous or period-ending sampling,
input clock and target day. UTC or a fixed `offset:±HH:MM` is supported. The default
requires complete sampling per day, outputs NaN for incomplete days and records
`<name>_n_samples`. A period ending at midnight belongs to the previous interval.
Downstream checks reject conflicting aggregation metadata; absent metadata is a
declared, unverified convention. ERA5 air temperature over ocean remains an
atmospheric variable. Its analysis `t2m` and forecast-derived `mx2t`/`mn2t` need
not bound each other; sampled hourly extrema are not continuous true extrema.

## Reference-period correction and dynamic baselines

`in_base_correction="none"` is the default. Years contributing to
their own threshold can produce an artificial in-base/out-of-base step.
`"leave_one_year_out"` removes the target year's dates from each sample window;
`"mean_threshold"` substitutes other years and averages their thresholds.
These produce binary masks with year-specific thresholds. `in_base_rates` implements
the original Zhang bootstrap by averaging annual rates across replicas, and returns
fractions and valid-day counts. It does not produce a daily mask. LOO has one fewer
reference year and can violate `min_samples`; per-year thresholds multiply storage
by the number of reference years. The reference climatology itself is not corrected.
Events crossing the reference boundary can change membership just beyond it.

Dynamic methods are `detrended`, `shifting`, `stl`, `state_space`, `gam` and `ssa`:
`X = trend + seasonal + residual`, `reference = trend + seasonal`, and
`threshold = reference + residual_percentile(day_of_year)`. Percentiles exclude
filled days. Correction applies to residual thresholds; trend and seasonality still
use the configured fitting data. Seasonal phase follows the calendar date, including
leap years. A dynamic baseline answers a different scientific question from a fixed
historical reference.

Retrospective decomposition may use later information. Causal mode records
`available_until`, `lag_days`, history and extrapolation limits. `cutoff` fits once;
`expanding` uses earlier data, with method-specific refitting. The reference period
must end no later than availability. STL and SSA need complete decomposition series;
other methods tolerate observed gaps subject to sampling rules. `shifting` requires
sufficient prior complete years, so residual sampling must reflect available history.
Use GAM causal extrapolation with care: observed end effects are documented in
[validation](VALIDATION.md). The state-space implementation uses fixed relative
variances and an explicit trend timescale; it does not estimate an arbitrary new model.

## Objects, tracking and genealogy

`build_objects` executes an ordered SpatialRecipe over `exceedance` or
`event_membership`, restricted to valid cells. It preserves the original XMask and
records `processed_mask`, `added` and `removed`. Operations do not commute.
The default is `exceedance → segment → persist`, face-connected daily patches,
minimum size 5% of the declared physical domain and the detector's persistence rule.
The 5% is a configurable choice, not a measured property. Large domains can lose
nearly all patches; a warning reports removal above 90%. Use an explicit MinArea
appropriate to the question.

Structured lines `(x)`, surfaces `(y,x)` and volumes `(z,y,x)` are supported, including
curvilinear horizontal coordinates and static sigma geometry. Measures are length,
area and volume with their declared units. Volume connectivity includes 6/18/26
neighbours. `vertical_gap_m` links across layers only if intervening thickness fits
the limit in both columns. Time-dependent geometry and unstructured topology are
rejected by this implementation. Longitude wraps only with `periodic_x`; regional
borders and invalid cells are barriers. Binary morphology uses integer operations
and validity-aware boundaries. Physical horizontal disks do not imply a 3-D radius.

Tracking overlaps the same cells on consecutive days. Link measures are shared
measure, source fraction, target fraction, intersection-over-union and overlap
relative to the smaller object. The default uses the smaller-object denominator
with inclusive `>=`; zero threshold still needs positive shared cells. Gap links
choose a single candidate under declared overlap and tie rules. A monitoring area
percentile requires a separate frozen calibration sample.

Retrospective track IDs identify connected components. Monitoring assigns IDs
forward without rewriting earlier IDs and records parents and merges; identities
can fragment at splits and merges. Repeated calls are complete-window analyses,
not persistent incremental state. Edge and window censoring is explicit; unqualified
border candidates are included only in monitoring catalogs.

`build_lineage` uses daily patches as nodes, stores positive-overlap edges and the
chosen gap bridges, and records which edges satisfy LinkPolicy. Splits terminate a
parent branch and create child branches; merges terminate parents and create one
branch. Dominance annotates inheritance without deleting secondary parents.
Spacetime events are weakly connected components in retrospective mode. Monitoring
unions create a new event by default (`new_event`) or retain one (`survivor`), with
`merged_into` and provisional metrics. `partition_merges` optionally returns a new
ObjectField assigning merged cells to nearby parents; it preserves the input.

Metrics distinguish calendar duration, occupied days, bridged days, instantaneous
size, footprint union, area/volume-time, path length and net displacement. Volumetric
paths separate horizontal and vertical distance. Intensity is optional and can be
weighted by physical cell measure. Graph series and maps are computed on demand.
Genealogy describes geometric connection; it establishes no physical attribution.

An optional intensity field must preserve the object's time and physical cell
coordinates; dimension transposition is allowed, implicit cell reordering is not.
Only finite observations enter intensity denominators. A day with missing field
can still be occupied; it contributes to frequency but not an intensity average.
All-missing intensity is NaN, including cumulative intensity. LineageGraph schema
0.1 stores valid count and measure per node so these reductions survive serialization.
See [intensity and observation support](METRICS.md#intensity-and-observation-support).
Maximum is algebraic maximum of the supplied field; positive cold severity must
be supplied and labelled explicitly.

## Computation and memory

Container construction is structural and lazy where possible; explicit invariant
checks evaluate values. Baseline fitting requires a whole time series per spatial
chunk and does not silently rechunk it. Detection uses bounded temporal halos.
Catalog construction, object processing and lineage tables are eager: budget dense
time-space masks, labels and tabular storage before processing large domains.
SSA limits spectral batches but still has method-specific temporary arrays. Dynamic
baselines store several dense time-space fields, unlike the smaller fixed DOY fields.
An expanding decomposition can require a fit per cutoff. Dask input alone does not
guarantee bounded memory or distributed object tracking.
