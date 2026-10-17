# Metrics, tables and figures

[← Objects](OBJECTS.md) · [Index](README.md) · [Plotting →](PLOTTING.md)

## Choose the population first

A cell-level `EventCatalog` has one row per event and cell, with spatial indices.
A graph describes daily patches, branches and spacetime events across cells.
Counting all per-cell events is not counting distinct spatial heatwaves.
State which population an analysis uses and how boundary-censored rows are handled.

| Metric | Meaning | Units / denominator |
|---|---|---|
| Calendar duration | `end − start + 1`, including bridged days | days |
| Active days | Days with at least one occupied patch | days; differs from duration when object gaps are bridged |
| Patch size | Sum of declared cell measures | m for lines, m² for surfaces, m³ for volumes |
| Peak size | Largest daily occupied size of a branch/event | Same physical units; peak date is separate |
| Size-time | Sum of daily occupied sizes | m·day, m²·day or m³·day |
| Footprint | Union of occupied cells over the selected branch/event | Physical measure, not sum across dates |
| Displacement / path | Net centroid displacement / sum of successive centroid distances | Physical distance; depends on grid geometry and weighting |
| Intensity | Temperature minus reference, signed | Temperature difference; a negative MCS intensity remains negative |
| Threshold excess | Temperature relative to threshold | Distinct from intensity; calculate and label explicitly |
| Temporal frequency | Occupied valid cell-days / valid cell-days | Per-cell percentage; excludes invalid support |
| Daily area/volume fraction | Occupied physical measure / declared domain measure | Fraction; report valid support alongside it when missing is present |

`event_active_days` and `branch_active_days` are graph fields; temporal catalogs
use `n_exceedance_days` and `n_joined_gap_days`. In a graph-derived catalog those
names describe occupied and bridged object days, not temperature exceedance at
a single cell. Follow the product's granularity rather than reading the name alone.

## Inspect a temporal catalog

```python
table = catalog.dataset[["start", "end", "duration_days", "intensity_max",
                          "left_censored", "right_censored"]]
longest = table.sortby("duration_days", ascending=False).isel(event=slice(0, 5))
```

In joint TX+TN detection, intensity fields have a `component` dimension. Select
the intended component before ranking or comparing intensity. An AND profile
is not the intersection of two independently persistent event catalogs: daily
component exceedances are combined before persistence.

For a frequency map retaining validity:

```python
valid_days = mask.dataset.valid.sum("time")
frequency_pct = (100 * (mask.dataset.event_membership & mask.dataset.valid).sum("time")
                 / valid_days.where(valid_days > 0))
```

This restricts both numerator and denominator to valid support, including when
invalid missing days belong to allowed joined gaps.
Never let an invalid domain cell produce a misleading zero-frequency observation.

## Branch and spacetime-event analysis

```python
event = int(graph.dataset.event_number.values[0])
series = graph.series(spacetime_event=event)
maps = graph.maps(objects, spacetime_event=event)
branch_series = graph.series(branch=1)
```

Give exactly one selector: `spacetime_event` or `branch`. The `.members()` method
returns object-row indices; `.transitions()` returns the edge table. `.maps()`
requires the ObjectField from which the graph was built. For intensity metrics,
build the graph with `field=mask.intensity()` and specify `weighted=True` or
`False`. The default weighted mean uses area/volume/length, whereas a simple
cell mean can differ on latitude-varying or uneven-depth cells.

Graph tables have event/branch duration, active and bridged days, peak size,
size-time, footprint, displacement and censoring fields. Volumes also include
depth extrema, vertical displacement and horizontal footprint. Inspect each
variable's units and `cell_methods` before exporting a scientific table.

![Area, intensity and occupied-day map of a synthetic spacetime event](assets/synthetic_peru/metrics.png)

*Executed synthetic Peru: largest area-time component, event 32, fixed 1991–2020
reference, P90, temporal minimum 5/gap 2, face patches ≥5% of ocean area, any
positive shared-cell linking, track minimum 5/object gap 0, retrospective graph.
Left: occupied area in km². Middle: area-weighted signed mean and maximum
intensity in K. Right: occupied days per cell, not an annual climatological
frequency. The event spans 392 calendar days. The field is a seeded reconstruction: it
illustrates the metrics and is not a climate event.*

## Plot with xarray and Matplotlib

xHW does not supply a native graphics API. Its xarray datasets work with familiar
plotting tools; see [xarray plotting documentation](https://docs.xarray.dev/en/stable/user-guide/plotting.html).
The [plotting guide](PLOTTING.md) has more recipes, the tutorial figure functions in
[`examples/plotting.py`](../examples/plotting.py) and a gallery of every saved figure.

```python
import matplotlib.pyplot as plt

series.area.plot()
plt.ylabel(f"Occupied size ({series.area.attrs['units']})")
plt.show()

maps.frequency.where(objects.dataset[objects.grid.domain_mask]).plot()
plt.title(f"Event {event}: occupied days per cell")
plt.show()
```

Label the date selection, depth slice, unit conversion, reference and definition.
If showing one level of a volume, state that it is a slice. A palette of tracker IDs
can repeat for large ID sets: numerical annotations, not colors alone, identify
objects. Figures and map sequences are static in saved notebooks so GitHub readers
can inspect results without running a widget or downloading a multi-gigabyte dataset.

## Intensity and observation support

Only finite field observations contribute to intensity. Weighted spatial means divide
by finite cell measure; unweighted means divide by finite cell count. Map means use
finite observed occupied days. Geometric occupancy contributes to frequency even when
its intensity is missing.

All-missing means, maxima and cumulative intensity are `NaN`; a measured zero remains
zero. Partial cumulative intensity sums observed support and does not imply complete
coverage. The graph stores valid cell count and valid physical measure per patch;
see [the serialization contract](API.md#intensity-fields-and-observation-support).

Maximum is the algebraic maximum of the supplied scalar field. For signed cold
anomalies `[-2, -6]`, it is `-2`. To measure greatest cold severity, supply a positive
severity field explicitly, preserving its temperature-difference units:

```python
signed = mask.intensity()
cold_severity = (-signed).assign_attrs(signed.attrs, long_name="positive cold severity")
cold_graph = xhw.build_lineage(objects, field=cold_severity)
```

Label that convention: peaks of signed anomaly and positive cold severity describe
different quantities. The sign choice does not change the detector.
