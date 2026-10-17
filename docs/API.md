# Public API reference

[Index](README.md) · [Processing guide](PIPELINE.md) · [Examples](TUTORIALS.md)

This reference covers all **44** exports in `xhw.__all__`, inspected against the installed publication **0.0.1**. Signatures below are real signatures, including defaults and keyword-only arguments. Private storage constructors are shown for completeness; create scientific products using factories or `from_dataset`.

## Shared conventions

Examples assume `import xhw` (and `import inspect` for signature inspection) and the named inputs from the synthetic tutorial, unless the example is a self-contained metadata constructor. Run [the API walkthrough](../examples/api_walkthrough.py) for a compact executable inspection/serialization example. Snippets using `hourly` refer to the ERA5 tutorial. No hidden automatic profile configuration, downloading or plotting is implied.

Metadata specs are frozen dataclasses. All provide `to_dict()`, `to_json()`, `from_dict(data)`, `from_json(text)` and `spec_id`; `spec_id` hashes canonical scientific configuration. Keyword constructor errors raise `ValidationError` or a more specific subclass. `Spec.from_json(Spec.to_json())` preserves identity.

Scientific products provide `.dataset`, `.to_dataset()` and `.from_dataset(...)`; use the public `.check_invariants()` / `.check_against(...)` methods available below. Structural validation does not substitute for full numerical checking. `.dataset` is a shallow view with read-only NumPy arrays. Modifications require a new product; serialized specs, identities, flags and provenance stay attached.

For NetCDF persistence use `product.to_dataset().to_netcdf(...)`, then read the Dataset with xarray and use the matching `from_dataset`. Entity schema and package versions are independent. A newer schema minor may warn, whereas an incompatible major is rejected. Field-dependent graph maps after reading can use an explicitly supplied aligned field; see `LineageGraph.maps` below.

Known limitation: tracking duration filtering does not delete short daily patch labels; graph-derived catalog export rejects components below the mask minimum. Graph analysis remains available; see [the observed export constraint](OBJECTS.md#catalog-export-qualification).

## Subdaily preparation

### aggregate_daily

```python
xhw.aggregate_daily(ds: 'xr.Dataset', *, variable: 'str', statistic: 'str', sampling: 'str', input_clock: 'str', day_definition: 'str', time_dim: 'str' = 'time', min_samples_per_day: 'int | None' = None, name: 'str | None' = None) -> 'xr.Dataset'
```

Aggregate a regularly sampled variable to max/min/mean on a declared UTC or fixed-offset day. Returns a Dataset with daily temperature, valid sample counts and aggregation metadata; incomplete days are NaN. Period-ending stamps belong to the preceding interval. Units are preserved. Raises ValidationError for irregular sampling, absent variables, bad clocks or incompatible options.

```python
daily = xhw.aggregate_daily(hourly, variable="mx2t", statistic="max",
    sampling="period_ending", input_clock="UTC", day_definition="offset:-05:00", name="tx")
```

### interpolate_gaps

```python
xhw.interpolate_gaps(ds: 'xr.Dataset', *, variable: 'str', grid: 'GridSpec', max_days: 'int', max_filled_fraction: 'float') -> 'xr.Dataset'
```

Fill only interior temperature NaN runs up to max_days. Returns a new Dataset with filled evidence and low-data-quality flags when max_filled_fraction is exceeded. These are preprocessing limits, not detector event gaps. Observed-only baseline quantiles exclude filled values. Each temporal series is materialized for interpolation.

```python
prepared = xhw.interpolate_gaps(ds, variable="sst", grid=grid,
    max_days=2, max_filled_fraction=0.05)
```

## Baseline and detection

### fit_baseline

```python
xhw.fit_baseline(ds: 'xr.Dataset', *, variable: 'str | tuple[str, ...]', grid: 'GridSpec', baseline: 'BaselineSpec', definition: 'DefinitionSpec', provenance: 'Provenance | None' = None) -> 'BaselineResult'
```

Fit a fixed seasonal or one of six dynamic references and thresholds on explicit scientific specs. Returns BaselineResult with reference, threshold, sample diagnostics and decomposition/per-year fields where configured. Raises UnitsError, CalendarNotSupportedError, ValidationError or NotSupportedError for incompatible data/specs/methods; optional engines can require extra dependencies. Exact quantiles require complete temporal chunks; dynamic decomposition and validation can be eager. See BASELINES.md for all methods and defaults.

```python
fitted = xhw.fit_baseline(prepared, variable="sst", grid=grid,
    baseline=baseline, definition=definition)
fitted.check_invariants()
```

### detect

```python
xhw.detect(ds: 'xr.Dataset', *, variable: 'str | tuple[str, ...]', grid: 'GridSpec', definition: 'DefinitionSpec', baseline_result: 'BaselineResult', provenance: 'Provenance | None' = None) -> 'XMask'
```

Compare daily temperature with a compatible fitted baseline, apply component lags/AND if present, qualify runs then join gaps under the declared rule. Returns XMask preserving exceedance, membership, joined gaps, validity, invalid_reason and optional filled. Identity, unit, grid and aggregation mismatches raise validation errors. Detection needs complete temporal series per processed spatial chunk.

```python
mask = xhw.detect(prepared, variable="sst", grid=grid,
    definition=definition, baseline_result=fitted)
mask.check_invariants()
```

### build_catalog

```python
xhw.build_catalog(mask: 'XMask', *, analysis_mode: 'str', provenance: 'Provenance | None' = None) -> 'EventCatalog'
```

Enumerate temporal events independently at each cell. Returns EventCatalog with dates, counts, intensities, categories if configured, location indices and boundary censoring. analysis_mode declares retrospective or monitoring; context can provide additional mask context. The event table is materialized. It is not a distinct spatial-event census.

```python
catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
catalog.check_against(mask)
```

### in_base_rates

```python
xhw.in_base_rates(ds: 'xr.Dataset', *, variable: 'str | tuple[str, ...]', grid: 'GridSpec', baseline_result: 'BaselineResult', provenance: 'Provenance | None' = None) -> 'xr.Dataset'
```

Average annual exceedance/event-day rates across year-substitution bootstrap replicas. Returns xarray Dataset with exceedance_frequency, event_day_frequency and valid_days on a year axis. It needs a BaselineResult from these data and a compatible percentile definition; ValidationError rejects a mismatched baseline or unsupported configuration. It is not a daily binary mask and can require many complete fits.

```python
rates = xhw.in_base_rates(prepared, variable="sst", grid=grid,
    baseline_result=fitted)
```

## Objects and genealogy

### build_objects

```python
xhw.build_objects(mask: 'XMask', *, recipe: 'SpatialRecipe | None' = None, analysis_mode: 'str' = 'retrospective', calibration=None, provenance: 'Provenance | None' = None) -> 'ObjectField'
```

Execute an ordered SpatialRecipe on the selected XMask layer restricted to valid cells. Returns ObjectField with processed_mask/added/removed, daily labels/table when segmented, and tracker IDs only with an explicit track step. Default is exceedance → segment → persist, without tracking. Eager; point support, unsupported geometry and unfrozen monitoring percentile calibration are rejected.

```python
objects = xhw.build_objects(mask, recipe=recipe, analysis_mode="retrospective")
objects.check_invariants()
objects.check_against(mask)
```

### build_lineage

```python
xhw.build_lineage(objects: 'ObjectField', *, policy: 'LinkPolicy | None' = None, analysis_mode: 'str | None' = None, merge_rule: 'str' = 'new_event', field: 'xr.DataArray | None' = None, weighted: 'bool' = True, provenance: 'Provenance | None' = None) -> 'LineageGraph'
```

Create daily-node transitions, branches and spacetime-event metrics from an ObjectField containing labels. policy defaults to the track step or any-positive-overlap policy. Every shared candidate remains in transitions; linked selects the policy-qualified edges. field is optional and must align to the object grid/time; weighted uses cell measures for spatial means. Eager. Returns LineageGraph; it is a complete-window analysis, not persistent incremental state.

```python
graph = xhw.build_lineage(objects, field=mask.intensity(), weighted=True)
graph.check_against(objects)
```

### neighbor_fraction

```python
xhw.neighbor_fraction(mask: 'XMask', *, footprint: 'Footprint', layer: 'str' = 'exceedance') -> 'xr.DataArray'
```

Return the fraction of valid neighbours in the requested mask layer under the footprint. The denominator is valid neighbours, not all geometric neighbours; inspect missing/no-neighbour output before thresholding. Returns DataArray; supported line/surface/volume grids only. It does not train a classifier or change XMask.

```python
fractions = xhw.neighbor_fraction(mask, footprint=xhw.Footprint.connected(1),
    layer="event_membership")
```

### ingest_objects

```python
xhw.ingest_objects(labels: 'xr.DataArray', *, mask: 'XMask', kind: 'str', source: 'SourceRef', method: 'str', source_layer: 'str' = 'exceedance', analysis_mode: 'str' = 'retrospective', provenance: 'Provenance | None' = None, connectivity: 'Footprint | None' = None, on_violation: 'str' = 'error') -> 'ObjectField'
```

Read external daily patches or tracks as a new ObjectField with declared source provenance. Data must align to the mask grid/time, use nonnegative integer labels and respect validity. kind selects patches/tracks; background is zero. It does not run upstream algorithms or rewrite original detection. Raises ValidationError for malformed labels and NotSupportedError for unsupported grids.

```python
external = xhw.ingest_objects(objects.dataset.label, mask=mask, kind="patches",
    method="tutorial-copy", source_layer="event_membership",
    source=xhw.SourceRef(dataset_id="example:external-labels", variable="label",
        accessed="2026-10-10", product_status="unknown"))
```

### compare_objects

```python
xhw.compare_objects(a: 'ObjectField', b: 'ObjectField') -> 'xr.Dataset'
```

Compare compatible ObjectFields derived from the same mask. Returns a Dataset of processed-mask overlap and object-comparison metrics. It does not infer which method is physically better. Identity/geometry incompatibilities raise ValidationError; comparisons are eager.

```python
comparison = xhw.compare_objects(objects, external)
```

### partition_merges

```python
xhw.partition_merges(objects: 'ObjectField', mask: 'XMask', *, policy: 'LinkPolicy | None' = None, provenance: 'Provenance | None' = None) -> 'ObjectField'
```

Optionally partition merge geometry into branches using the mask and a declared LinkPolicy. Returns a new ObjectField, preserving the source field. This is a different geometry analysis from retaining all parents in genealogy. Requires compatible labels, mask identity and supported geometry; validation errors explain failures. Eager.

```python
partitioned = xhw.partition_merges(objects, mask)
partitioned.check_against(mask)
```

## Metadata contracts

### SourceRef

```python
xhw.SourceRef(*, dataset_id: 'str', variable: 'str', accessed: 'str', product_status: "Literal['final', 'preliminary', 'unknown']", version: 'str | None' = None, query: 'str | None' = None, checksum: 'str | None' = None) -> None
```

Dataset identity, variable, accessed date, final/preliminary/unknown status, optional product version/query/checksum. Validates metadata shape, not remote existence or scientific quality. A hash identifies bytes; retain the physical selection too.

```python
source = xhw.SourceRef(dataset_id="synthetic:demo", variable="sst",
    accessed="2026-10-10", product_status="final")
```

Public fields: `dataset_id`, `variable`, `accessed`, `product_status`, `version`, `query`, `checksum`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### ProcessingStep

```python
xhw.ProcessingStep(*, name: 'str', params: 'tuple[tuple[str, str | int | float | bool | None], ...]' = ()) -> None
```

Named serializable operation with scalar parameter pairs. Used in provenance and aggregation metadata. Parameters cannot contain arbitrary Python objects or secrets.

```python
step = xhw.ProcessingStep(name="subset", params=(("latitude_min", -12.0),))
```

Public fields: `name`, `params`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### Provenance

```python
xhw.Provenance(*, run_id: 'str', created_utc: 'str', software: 'tuple[tuple[str, str], ...]', inputs: 'tuple[SourceRef, ...]' = (), parents: 'tuple[str, ...]' = (), steps: 'tuple[ProcessingStep, ...]' = ()) -> None
```

Run identity, UTC creation time, software versions, input SourceRefs, parent runs and ordered ProcessingSteps. new() captures versions and generates a run UUID. Execution provenance is separate from scientific review status.

```python
provenance = xhw.Provenance.new(inputs=(source,), steps=(step,))
```

Public fields: `run_id`, `created_utc`, `software`, `inputs`, `parents`, `steps`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
new(*, inputs: 'tuple[SourceRef, ...]' = (), parents: 'tuple[str, ...]' = (), steps: 'tuple[ProcessingStep, ...]' = ()) -> 'Provenance'
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### CoordRoles

```python
xhw.CoordRoles(*, lat: 'str | None' = None, lon: 'str | None' = None, depth: 'str | None' = None, along: 'str | None' = None) -> None
```

Names of latitude/longitude/depth/along coordinates, including supported 2-D horizontal coordinates. Names refer to variables in the Dataset; this does not create coordinates.

```python
coords = xhw.CoordRoles(lat="lat", lon="lon")
```

Public fields: `lat`, `lon`, `depth`, `along`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### CellMeasure

```python
xhw.CellMeasure(*, kind: "Literal['none', 'unit', 'length', 'area', 'volume']", var: 'str | None' = None, units: 'str | None' = None) -> None
```

Declared none/unit/length/area/volume measure; physical weights name a variable and units. unit is an explicit cell-count approximation. A volume can be constructed from area times VerticalSpec thickness. grid.check verifies compatibility.

```python
measure = xhw.CellMeasure(kind="area", var="cell_area", units="m2")
```

Public fields: `kind`, `var`, `units`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### VerticalSpec

```python
xhw.VerticalSpec(*, coordinate: 'str', thickness: 'str', thickness_units: 'str', time_dependent: 'bool') -> None
```

Vertical coordinate, thickness variable/units, geometry and optional bounds/static conventions. Time-dependent geometry is representable as metadata but rejected by current object processing. Physical vertical gaps depend on these thicknesses.

```python
vertical = xhw.VerticalSpec(coordinate="depth", thickness="dz",
    thickness_units="m", time_dependent=False)
```

Public fields: `coordinate`, `thickness`, `thickness_units`, `time_dependent`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### NativeGrid

```python
xhw.NativeGrid(*, product: 'str', resolution: 'str', is_native: 'bool', regridded_from: 'SourceRef | None' = None) -> None
```

Product, resolution, native-grid declaration and optional metadata. A native declaration does not erase a documented selection or decimation; the GLORYS case in xhw-benchmark records both.

```python
native = xhw.NativeGrid(product="my product", resolution="0.25 degree", is_native=True)
```

Public fields: `product`, `resolution`, `is_native`, `regridded_from`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### GridSpec

```python
xhw.GridSpec(*, support: "Literal['point', 'line', 'surface', 'volume']", topology: "Literal['none', 'transect', 'structured', 'unstructured']", time_dim: 'str', spatial_dims: 'tuple[str, ...]', coords: 'CoordRoles', calendar: 'str', frequency: "Literal['1D']", time_axes: "Literal['single', 'forecast']", periodic_x: 'bool', cell_measure: 'CellMeasure', vertical: 'VerticalSpec | None', domain: "Literal['ocean', 'land', 'air_over_ocean', 'air_over_land']", domain_mask: 'str', domain_fraction: 'str | None', native: 'NativeGrid') -> None
```

Temperature support/topology, dimension names, coordinate roles, daily frequency/calendar, time axes, periodic longitude, physical weights and domain. check(ds) validates names, axes, geometry and relevant metadata; no mandatory regridding is performed. See _data.py for complete surface and volume constructors.

```python
grid.check(ds)
roundtrip = xhw.GridSpec.from_json(grid.to_json())
assert roundtrip.spec_id == grid.spec_id
```

Public fields: `support`, `topology`, `time_dim`, `spatial_dims`, `coords`, `calendar`, `frequency`, `time_axes`, `periodic_x`, `cell_measure`, `vertical`, `domain`, `domain_mask`, `domain_fraction`, `native`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
check(self, ds: 'xr.Dataset') -> 'None'
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### DailyComponent

```python
xhw.DailyComponent(*, variable: 'str', daily_statistic: "Literal['native', 'mean', 'max', 'min']", percentile: 'float', lag_days: 'int') -> None
```

One daily variable/statistic/percentile/lag in an AND AHW definition. Lag 1 means component temperature on D+1 is tested relative to reference day D. Each component has its own fitted threshold.

```python
component = xhw.DailyComponent(variable="tn_daily", daily_statistic="min",
    percentile=90.0, lag_days=1)
```

Public fields: `variable`, `daily_statistic`, `percentile`, `lag_days`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### DefinitionSpec

```python
xhw.DefinitionSpec(*, profile: 'str', phenomenon: "Literal['mhw', 'mcs', 'ahw']", variable: 'str', combination: "Literal['single', 'and_daily']", daily_statistic: "Literal['native', 'mean', 'max', 'min']", day_definition: 'str', tail: "Literal['upper', 'lower']", comparator: "Literal['>', '>=', '<', '<=']", threshold_kind: "Literal['percentile', 'absolute']", percentile: 'float | None', absolute_threshold: 'tuple[float, str] | None', min_duration_days: 'int', max_gap_days: 'int', join_rule: "Literal['qualify_then_join', 'join_then_qualify']", missing_policy: "Literal['break', 'as_non_exceedance']", season_months: 'tuple[int, ...] | None', category_scheme: "Literal['hobday2018', 'schlegel2021'] | None", components: 'tuple[DailyComponent, ...] | None' = None) -> None
```

The complete detection definition: phenomenon, variable meaning, tail/comparator, percentile or absolute threshold, daily clock/statistic, duration/gaps/join order, missing policy, categories and optional joint components. Profile strings label rather than auto-configure this contract. Bad combinations raise ValidationError. See the executed tutorials for complete constructors.

```python
print(definition.to_dict())
assert xhw.DefinitionSpec.from_json(definition.to_json()).spec_id == definition.spec_id
```

Public fields: `profile`, `phenomenon`, `variable`, `combination`, `daily_statistic`, `day_definition`, `tail`, `comparator`, `threshold_kind`, `percentile`, `absolute_threshold`, `min_duration_days`, `max_gap_days`, `join_rule`, `missing_policy`, `season_months`, `category_scheme`, `components`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### BaselineSpec

```python
xhw.BaselineSpec(*, method: "Literal['fixed_climatology', 'shifting', 'detrended', 'stl', 'gam', 'ssa', 'state_space']", computed_on: "Literal['temperature', 'anomaly']", reference_period: 'tuple[str, str]', source: 'SourceRef', doy_window_days: 'int | None', smoothing_window_days: 'int | None', percentile_estimator: 'str', leap_day_policy: "Literal['hobday_366_interp', 'calendar_doy', 'not_applicable']", min_samples: 'int', fit_mode: "Literal['retrospective', 'causal']", available_until: 'str | None', params: 'tuple[tuple[str, str | int | float], ...]' = (), in_base_correction: "Literal['none', 'mean_threshold', 'leave_one_year_out']" = 'none') -> None
```

Baseline method, observed reference period, source, seasonal windows/estimator/leap policy, sample minimum, retrospective/causal availability, parameters and in-base correction. Scientific defaults of dynamic method parameters are validated during fit. reference_years exposes the declared year set. See BASELINES.md.

```python
from dataclasses import replace
dynamic = replace(baseline, method="detrended", params=(("trend_degree", 1),))
```

Public fields: `method`, `computed_on`, `reference_period`, `source`, `doy_window_days`, `smoothing_window_days`, `percentile_estimator`, `leap_day_policy`, `min_samples`, `fit_mode`, `available_until`, `params`, `in_base_correction`.

Inspection properties: `reference_years`, `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
available_before(self, instant: 'np.datetime64') -> 'bool'
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### CouplingSpec

```python
xhw.CouplingSpec(*, strategy: "Literal['event_first', 'mask_first']", phenomena: 'tuple[str, str]', spatial_rule: "Literal['same_support', 'coupling_grid', 'coastal_pairs', 'column_projection']", spatial_params: 'tuple[tuple[str, str | float], ...]', lag_window_days: 'tuple[int, int]', membership_basis: "Literal['exceedance', 'event_membership']", min_phase_days: 'int', matching: "Literal['many_to_many']") -> None
```

DECLARATIVE ONLY. Validates association metadata, spatial rule, lag window, membership basis and participating catalog identities. check_catalogs checks contracts; it does not compute associations, compound detections or causal attribution. Do not present schema fields as an executed capability.

```python
print(inspect.signature(xhw.CouplingSpec))  # construction contract, not an analysis engine
```

Public fields: `strategy`, `phenomena`, `spatial_rule`, `spatial_params`, `lag_window_days`, `membership_basis`, `min_phase_days`, `matching`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
check_catalogs(self, atmospheric, marine) -> 'None'
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

## Recipe contracts

### Footprint

```python
xhw.Footprint(*, kind: "Literal['connectivity', 'radius', 'offsets']", connectivity: 'int | None' = None, radius_km: 'float | None' = None, offsets: 'tuple[tuple[int, ...], ...] | None' = None, vertical_gap_m: 'float | None' = None) -> None
```

Neighbourhood: connected rank 1/2/3, physical surface disk or symmetric explicit index offsets. Optional vertical_gap_m uses intervening thickness on volumes; adjacent layers touch at gap zero. Different footprints change patches, never original evidence.

```python
faces = xhw.Footprint.connected(1)
diagonals = xhw.Footprint.connected(2)
physical = xhw.Footprint.connected(1, vertical_gap_m=5.0)
```

Public fields: `kind`, `connectivity`, `radius_km`, `offsets`, `vertical_gap_m`.

Inspection properties: `physical_vertical`, `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
connected(rank: 'int' = 1, vertical_gap_m: 'float | None' = None) -> 'Footprint'
disc(radius_km: 'float') -> 'Footprint'
explicit(offsets) -> 'Footprint'
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### MinArea

```python
xhw.MinArea(*, kind: "Literal['none', 'percentile', 'domain_fraction', 'absolute_km', 'absolute_km2', 'absolute_km3']", value: 'float | None' = None, floor_km2: 'float | None' = None) -> None
```

Minimum physical patch size: none, domain_fraction in percent, retrospective/frozen-calibration percentile, or absolute km/km2/km3. default() is 5% of domain, not 0.05%. percentile optional floor_km2 applies only to the supported area convention.

```python
cut = xhw.MinArea.default()
assert cut.value == 5.0
no_cut = xhw.MinArea.none()
```

Public fields: `kind`, `value`, `floor_km2`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
default() -> 'MinArea'
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
none() -> 'MinArea'
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### RecipeStep

```python
xhw.RecipeStep(*, op: "Literal['persist', 'open', 'close', 'segment', 'track']", footprint: 'Footprint | None' = None, min_area: 'MinArea | None' = None, min_duration_days: 'int | None' = None, max_gap_days: 'int | None' = None, join_rule: "Literal['qualify_then_join', 'join_then_qualify'] | None" = None, min_overlap: 'float | None' = None, link_measure: "Literal['overlap_smaller', 'iou', 'source_fraction', 'target_fraction'] | None" = None, dominance_measure: "Literal['overlap_smaller', 'iou', 'source_fraction', 'target_fraction', 'shared'] | None" = None, tie_rule: "Literal['lowest_id', 'largest_area'] | None" = None, branch_rule: "Literal['inherit', 'terminate_and_create'] | None" = None) -> None
```

One validated recipe operation. Factories persist(), morph("open"/"close", footprint), segment(), track(). Track must directly follow segment and be last. rule(name) exposes an effective link/dominance/tie/branch default when stored value is None. Temporal persistence, morphology and tracker gap limits have separate meanings.

```python
tracking = xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=0,
    min_duration_days=5, link_measure="overlap_smaller")
closing = xhw.RecipeStep.morph("close", xhw.Footprint.connected(1))
```

Public fields: `op`, `footprint`, `min_area`, `min_duration_days`, `max_gap_days`, `join_rule`, `min_overlap`, `link_measure`, `dominance_measure`, `tie_rule`, `branch_rule`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
morph(op: 'str', footprint: 'Footprint') -> 'RecipeStep'
persist(*, min_duration_days: 'int | None' = None, max_gap_days: 'int | None' = None, join_rule: 'str | None' = None) -> 'RecipeStep'
rule(self, name: 'str') -> 'str'
segment(*, footprint: 'Footprint | None' = None, min_area: 'MinArea | None' = None) -> 'RecipeStep'
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
track(*, min_overlap: 'float' = 0.0, max_gap_days: 'int' = 0, min_duration_days: 'int' = 1, link_measure: 'str | None' = None, dominance_measure: 'str | None' = None, tie_rule: 'str | None' = None, branch_rule: 'str | None' = None) -> 'RecipeStep'
```

### SpatialRecipe

```python
xhw.SpatialRecipe(*, input_layer: "Literal['exceedance', 'event_membership']", steps: 'tuple[RecipeStep, ...]') -> None
```

Chosen original layer plus ordered steps. default() is exceedance → segment → persist, without tracking. has_segment/has_track are inspection properties. Identity includes ordering because operations do not commute.

```python
recipe = xhw.SpatialRecipe(input_layer="event_membership", steps=(
    xhw.RecipeStep.segment(min_area=xhw.MinArea.none()),
    xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=0, min_duration_days=5)))
```

Public fields: `input_layer`, `steps`.

Inspection properties: `has_segment`, `has_track`, `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
default() -> 'SpatialRecipe'
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

### LinkPolicy

```python
xhw.LinkPolicy(*, measure: "Literal['overlap_smaller', 'iou', 'source_fraction', 'target_fraction']" = 'overlap_smaller', threshold: 'float' = 0.0, max_gap_days: 'int' = 0, dominance_measure: "Literal['overlap_smaller', 'iou', 'source_fraction', 'target_fraction', 'shared']" = 'overlap_smaller', tie_rule: "Literal['lowest_id', 'largest_area']" = 'lowest_id') -> None
```

Genealogy overlap denominator, threshold, permitted empty object days, dominance and tie rules. Inclusive >= requires positive shared measure even at zero threshold. from_track() makes an explicit policy from the recipe step; changing policy can change graph components.

```python
policy = xhw.LinkPolicy.from_track(recipe.steps[-1])
assert policy.measure == "overlap_smaller"
```

Public fields: `measure`, `threshold`, `max_gap_days`, `dominance_measure`, `tie_rule`.

Inspection properties: `short_id`, `spec_id`.

Public methods (inherited serialization methods included):

```python
from_dict(data: 'dict[str, Any]')
from_json(text: 'str')
from_track(step) -> 'LinkPolicy'
to_dict(self) -> 'dict[str, Any]'
to_json(self) -> 'str'
```

## Scientific products

### BaselineResult

```python
xhw.BaselineResult(_dataset: 'xr.Dataset', grid: 'GridSpec', baseline: 'BaselineSpec', definition: 'DefinitionSpec', provenance: 'Provenance') -> None
```

Fit output. Use fit_baseline or from_dataset, not the dataclass storage constructor. dataset is a shallow immutable-data view; to_dataset serializes specs/IDs. Contains compatible grid/baseline/definition/provenance. Fixed versus dynamic layouts differ. check_invariants computes data; check_against validates association with supplied contracts/data according to its signature.

```python
restored = xhw.BaselineResult.from_dataset(fitted.to_dataset())
restored.check_invariants()
```

Public fields: `grid`, `baseline`, `definition`, `provenance`.

Inspection properties: `baseline_id`, `dataset`, `definition_id`, `doy_mode`.

Public methods (inherited serialization methods included):

```python
check_invariants(self) -> 'None'
expand(self, times) -> 'xr.Dataset'
from_dataset(ds: 'xr.Dataset', *, grid: 'GridSpec | None' = None, baseline: 'BaselineSpec | None' = None, definition: 'DefinitionSpec | None' = None, provenance: 'Provenance | None' = None, copy: 'bool' = False) -> 'BaselineResult'
to_dataset(self) -> 'xr.Dataset'
```

### XMask

```python
xhw.XMask(_dataset: 'xr.Dataset', grid: 'GridSpec', definition: 'DefinitionSpec', baseline: 'BaselineSpec', provenance: 'Provenance') -> None
```

Detection evidence: separate exceedance, event_membership, joined_gap, valid and invalid_reason, plus optional filled. intensity() returns signed temperature minus reference, with difference-unit metadata. dataset uses read-only arrays; from_dataset validates structure/serialized identity; check_invariants materializes numerical checks.

```python
mask.check_invariants()
intensity = mask.intensity()
restored_mask = xhw.XMask.from_dataset(mask.to_dataset())
```

Public fields: `grid`, `definition`, `baseline`, `provenance`.

Inspection properties: `baseline_id`, `dataset`, `definition_id`.

Public methods (inherited serialization methods included):

```python
check_invariants(self) -> 'None'
from_dataset(ds: 'xr.Dataset', *, grid: 'GridSpec | None' = None, definition: 'DefinitionSpec | None' = None, baseline: 'BaselineSpec | None' = None, provenance: 'Provenance | None' = None, copy: 'bool' = False) -> 'XMask'
intensity(self) -> 'xr.DataArray'
to_dataset(self) -> 'xr.Dataset'
```

### EventCatalog

```python
xhw.EventCatalog(_dataset: 'xr.Dataset', definition: 'DefinitionSpec', baseline: 'BaselineSpec', provenance: 'Provenance', granularity: 'str', analysis_mode: 'str', location_dims: 'tuple[str, ...]', source_mask_run_id: 'str', window_start: 'str', window_end: 'str') -> None
```

Temporal per-cell or graph-derived object_tracked rows with specs, source mask identity, window and censoring. Inspect granularity and location_dims. from_dataset handles structure and invariants; check_against(mask) checks correspondence. For object_tracked graph catalogs use graph.check_catalog. The graph export can reject short qualified components; see OBJECTS.md.

```python
catalog.check_against(mask)
restored_catalog = xhw.EventCatalog.from_dataset(catalog.to_dataset())
```

Public fields: `definition`, `baseline`, `provenance`, `granularity`, `analysis_mode`, `location_dims`, `source_mask_run_id`, `window_start`, `window_end`.

Inspection properties: `baseline_id`, `dataset`, `definition_id`, `phenomenon`.

Public methods (inherited serialization methods included):

```python
check_against(self, mask: 'XMask') -> 'None'
check_invariants(self) -> 'None'
from_dataset(ds: 'xr.Dataset', *, mask: 'XMask | None' = None, granularity: 'str | None' = None, analysis_mode: 'str | None' = None, provenance: 'Provenance | None' = None, copy: 'bool' = False) -> 'EventCatalog'
to_dataset(self) -> 'xr.Dataset'
```

### ObjectField

```python
xhw.ObjectField(_dataset: 'xr.Dataset', grid: 'GridSpec', recipe: 'SpatialRecipe | None', analysis_mode: 'str', provenance: 'Provenance', definition_id: 'str', baseline_id: 'str', source_mask_run_id: 'str', origin: 'str', final_after_days: 'int', source_layer: 'str') -> None
```

Processed geometry, original-layer deltas, daily patch labels/table and optional tracking. Inspect has_label/has_track, recipe, analysis_mode and origin. check_against(mask) verifies provenance and original evidence. Track duration filtering can leave short patch labels with zero track IDs. to_dataset/from_dataset retain serializable configuration and identities.

```python
objects.check_against(mask)
restored_objects = xhw.ObjectField.from_dataset(objects.to_dataset())
```

Public fields: `grid`, `recipe`, `analysis_mode`, `provenance`, `definition_id`, `baseline_id`, `source_mask_run_id`, `origin`, `final_after_days`, `source_layer`.

Inspection properties: `dataset`, `has_label`, `has_track`.

Public methods (inherited serialization methods included):

```python
check_against(self, mask: 'XMask') -> 'None'
check_invariants(self) -> 'None'
from_dataset(ds: 'xr.Dataset', *, copy: 'bool' = False) -> 'ObjectField'
to_dataset(self) -> 'xr.Dataset'
```

### LineageGraph

```python
xhw.LineageGraph(_dataset: 'xr.Dataset', grid: 'GridSpec', policy: 'LinkPolicy', analysis_mode: 'str', merge_rule: 'str', provenance: 'Provenance', definition_id: 'str', baseline_id: 'str', source_objects_run_id: 'str', source_mask_run_id: 'str', final_after_days: 'int', window_start: 'str', window_end: 'str') -> None
```

Transitions, node types/flags, branch DAG, spacetime groups and physical metrics. members returns row indices; transitions returns Dataset; series/maps accept exactly one event/branch selector. maps requires the original ObjectField; optional field intensity survives in-memory graph inspection but field-dependent maps after serialization require attention to the listed signature. to_event_catalog declares qualified graph rows and rejects short components; do not silently requalify them. check_catalog validates a successful export.

```python
event = int(graph.dataset.event_number.values[0])
indices = graph.members(spacetime_event=event)
series = graph.series(spacetime_event=event)
maps = graph.maps(objects, spacetime_event=event)
graph.check_against(objects)
```

Public fields: `grid`, `policy`, `analysis_mode`, `merge_rule`, `provenance`, `definition_id`, `baseline_id`, `source_objects_run_id`, `source_mask_run_id`, `final_after_days`, `window_start`, `window_end`.

Inspection properties: `dataset`.

Public methods (inherited serialization methods included):

```python
check_against(self, objects: 'ObjectField') -> 'None'
check_catalog(self, catalog) -> 'None'
check_invariants(self) -> 'None'
from_dataset(ds: 'xr.Dataset', *, copy: 'bool' = False) -> 'LineageGraph'
maps(self, objects: 'ObjectField', *, spacetime_event: 'int | None' = None, branch: 'int | None' = None, field: 'xr.DataArray | None' = None) -> 'xr.Dataset'
members(self, *, spacetime_event: 'int | None' = None, branch: 'int | None' = None) -> 'np.ndarray'
node_type_names(self) -> 'np.ndarray'
series(self, *, spacetime_event: 'int | None' = None, branch: 'int | None' = None) -> 'xr.Dataset'
to_dataset(self) -> 'xr.Dataset'
to_event_catalog(self, mask, *, provenance: 'Provenance | None' = None)
transitions(self) -> 'xr.Dataset'
```

## Version and diagnostics

### __version__

```python
xhw.__version__ = '0.0.1'
```

Installed distribution version, distinct from the schema versions of each serialized entity. This reference was inspected against publication 0.0.1.

```python
print(xhw.__version__)
```

### XHWError

```python
xhw.XHWError(message)
```

Base class for xHW errors; catch only when a workflow can respond meaningfully.

```python
try:
    grid.check(ds)
except xhw.XHWError as error:
    print(str(error))
```

### ValidationError

```python
xhw.ValidationError(message)
```

Invalid specs, missing names, irregular frequency, incompatible clocks or unsupported combinations.

```python
try:
    grid.check(ds)
except xhw.ValidationError as error:
    print(str(error))
```

### SchemaError

```python
xhw.SchemaError(message)
```

Malformed serialized entity, incompatible schema major or identity mismatch.

```python
try:
    grid.check(ds)
except xhw.SchemaError as error:
    print(str(error))
```

### UnitsError

```python
xhw.UnitsError(message)
```

Missing/incompatible temperature or measure units. xHW does not perform implicit conversions.

```python
try:
    grid.check(ds)
except xhw.UnitsError as error:
    print(str(error))
```

### CalendarNotSupportedError

```python
xhw.CalendarNotSupportedError(message)
```

Calendar or date representation is outside the supported daily contract.

```python
try:
    grid.check(ds)
except xhw.CalendarNotSupportedError as error:
    print(str(error))
```

### NotSupportedError

```python
xhw.NotSupportedError(message)
```

Metadata can describe a topology/geometry/operation that the current implementation rejects.

```python
try:
    grid.check(ds)
except xhw.NotSupportedError as error:
    print(str(error))
```

### InvariantError

```python
xhw.InvariantError(message)
```

A numerical product violates its scientific invariants. Do not suppress it as successful validation.

```python
try:
    grid.check(ds)
except xhw.InvariantError as error:
    print(str(error))
```

### XHWSchemaWarning

```python
xhw.XHWSchemaWarning(message)
```

Newer compatible schema minor or other schema-reading advisory; inspect its context.

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error", xhw.XHWSchemaWarning)
    grid.check(ds)
```

### XHWDataQualityWarning

```python
xhw.XHWDataQualityWarning(message)
```

Insufficient reference/history, interpolation quality or overaggressive size cuts; investigate rather than hiding it.

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("error", xhw.XHWDataQualityWarning)
    grid.check(ds)
```

## Error handling

Treat missing data, units and frequency failures as input repairs. Do not catch an `InvariantError` and continue as if validation passed. Tutorial error checks cover irregular sampling, invalid units and an absent plotting dependency; checks of absent or modified registered files belong to xhw-benchmark. Acquisition client errors remain separate from the xHW scientific contract.

## Intensity fields and observation support

`build_lineage` and `graph.maps` require matching time, dimension sizes and physical
coordinates. Dimension transposition is allowed; reordering cells requires explicit
preparation. Coordinates are interpreted in the declared `GridSpec` system; coordinate
attributes do not trigger unit conversion.

`LineageGraph` schema `0.1` stores `object_intensity_valid_cells` and
`object_intensity_valid_measure` with its optional intensity attributes. Both support
variables are required when intensity is present and survive Dataset/NetCDF round trips.
Missing or inconsistent support raises `SchemaError` or `InvariantError`.

`check_against(objects)` checks geometry and topology; it cannot validate an original
intensity field that is not stored. Invariants check stored support consistency.
See [intensity metrics](METRICS.md#intensity-and-observation-support) for the reductions,
missing-value rules and cold-severity convention.

Missing required serialized Spec fields raise `SchemaError` naming the entity and
nested field path. Optional defaults and direct Python constructor errors retain
their normal meaning.
