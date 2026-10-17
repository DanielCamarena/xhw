# Two worked tutorials

[← Plotting](PLOTTING.md) · [Index](README.md) · [API](API.md)

Both tutorials generate their input, so they need no data files, credentials or network. Their
notebooks contain **executed static figures and tables**, so they can be read on GitHub without
Jupyter. Every notebook shows the principal xHW calls by stage; its paired script performs the
same analysis. Shared local helpers handle explicit specs and synthetic inputs
([_data.py](../examples/_data.py)), reports ([_report.py](../examples/_report.py)) and Matplotlib
figures ([plotting.py](../examples/plotting.py), documented in the [plotting guide](PLOTTING.md)).
The real-data tutorials (OISST over Peru and the equatorial Pacific, OSTIA, ERA5 and GLORYS) are in the companion project
[xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark/blob/main/docs/TUTORIALS.md).

Install xHW from the repository root, then:

```bash
python -m pip install -r examples/requirements-interactive.txt
```

| Case | Executed notebook | Paired script | Result |
|---|---|---|---|
| Synthetic | [synthetic.ipynb](../examples/synthetic.ipynb) | [synthetic.py](../examples/synthetic.py) | Masks, geometry, IDs, splits/merges and metrics with manual expectations |
| Synthetic Peru | [synthetic_peru.ipynb](../examples/synthetic_peru.ipynb) | [synthetic_peru.py](../examples/synthetic_peru.py) | MHW/MCS, fixed versus detrended baselines, temporal and spatial events on a realistic-sized field |

The results below come from these executions on publication 0.0.1. Exact run records include
specs, checksum, versions and review state; see [reproducibility](REPRODUCIBILITY.md). The core's
`check_invariants`, `check_against` and applicable graph/catalog checks run in both formats.
Plotting is done with xarray/Matplotlib; xHW has no native plotting API.

## Synthetic

```bash
python examples/synthetic.py --output-dir outputs/synthetic
```

This is an intentionally constructed, noise-free field with 7 × 9 cells, daily
1973-01-01 through 2003-01-31. The 30-year reference is constant 20 °C, so mean
and P90 are exactly 20 °C. Categories are disabled because their denominator
would be degenerate. The example illustrates representations, not realistic
climate variability or category calibration.

During January 2003, two five-cell hot bars occupy days 1–15. Three connector
cells are hot on days 1–5 and 11–15; their five-day gap is too long for temporal
joining. Therefore the central object is connected, splits into two daily patches
on day 6, and merges on day 11. Additional isolated demonstrations show a cool
joined day, a missing interpolated hot day and a diagonal-only pair.

| Hand-checkable expectation | Executed result |
|---|---|
| Bar membership 10 × 15 + connector 3 × 10 + two demonstrations 13 + 13 + diagonal pair 2 × 5 | 216 member cell-days |
| One cool member day is not an exceedance | 215 exceedance cell-days; 1 joined gap |
| Exactly one missing observation is filled | 1 filled cell-day; original raw temperature remains NaN |
| Central genealogy 5 + 2 × 5 + 5 daily nodes | 20 daily patches, 4 branches, 1 split, 1 merge, duration 15 days |
| Separate demonstration events remain separate under face connectivity | 5 retained tracks/spacetime events in the full field |

The detector uses strict P90, five-day minimum, two-day temporal gaps and
qualify-before-join. Objects use event membership, face neighbours and no area
cut; tracking uses smaller-object overlap ≥0.1, zero object gaps and minimum five
days. The separate closing example changes geometry and records added/removed
cells; it is not the tracked field. Eight figures cover temperatures/missing,
morphology, connectivity, labels/tracks, branch genealogy, size/intensity and frequency.

![Executed split and merge](assets/synthetic/genealogy.png)

*Central event: teal segments are branches; amber arrows link them. Two outgoing
arrows identify the split and two incoming identify the merge. Daily labels and
track IDs are shown separately in the notebook.*

## Synthetic Peru

```bash
python examples/synthetic_peru.py --output-dir outputs/synthetic_peru
```

The input is a seeded **reconstruction** of the OISST case: `synthetic_peru()` generates, in about
a second, a daily SST field on a 32 × 20 grid at 0.25° (12°S–4°S, 82°W–77°W), 1991–2025, with the
coast, seasonal cycle, trend and anomaly statistics of that case. Nothing is copied from an
observation; [Data](DATA.md#the-synthetic-stand-in-for-oisst) compares its statistics with OISST's.
The whole tutorial takes about half a minute. Reference 1991–2020 and the marine definition are
those of the real case: fixed climatology with 11-day pooling and 31-day smoothing, strict P90 (MHW)
or P10 (MCS), five-day minimum, two-day gaps, qualify before join. There are 349 ocean cells and no
missing ocean day.

| Output | Per-cell events | Member cell-days | Maximum cell duration |
|---|---:|---:|---:|
| Fixed MHW, strict upper P90 | 14 797 | 375 893 | 391 days |
| Fixed MCS, strict lower P10 | 22 978 | 336 518 | 119 days |
| Detrended MHW, same definition | 14 854 | 385 752 | 390 days |

The detrended method (degree-1 trend, three harmonics, fitted retrospectively) answers a different
reference question: its membership overlaps the fixed one with a Jaccard index of 0.919 on jointly
valid cell-days, without area weighting. This is a comparison of definitions, not a ranking.

Spatial recipe: event membership → face patches of at least 5% of the ocean area → tracking with any
positive smaller-object overlap, no object gaps and a five-day minimum. There are 2 623 daily
patches in 112 retained tracks; the genealogy has 292 branches and 140 spacetime components, with
32 split nodes and 47 merge nodes. The largest by area-time is event 32, lasting 392 calendar days
(it is the 1997–98 pulse placed in the generator). The graph holds 28 components shorter than five
days whose patch labels are kept but whose track IDs were filtered, so the qualified
`EventCatalog` export rejects them: the known 0.0.1 constraint described in
[Objects](OBJECTS.md#catalog-export-qualification).

![Frequency of MHW and MCS in the synthetic Peru field](assets/synthetic_peru/frequency.png)

*Percentage of valid days inside an event, per cell, 1991–2025; each panel has its own colour scale.
The field is synthetic: patterns show how the stages behave, not where heatwaves occur.*

![Fixed versus detrended membership](assets/synthetic_peru/baseline_comparison.png)

*The same MHW definition with two references. Colour is the percentage of **valid** days belonging
to events at each cell.*

![Size and intensity through the largest spacetime event](assets/synthetic_peru/metrics.png)

*Occupied area in km², area-weighted signed intensity in K relative to the reference, and occupied
days per cell for event 32. The generator's maximum intensity here (about 9.5 K) is higher than the
7.3 K of the corresponding OISST event; its extremes are wider than the real ones.*

## Re-run notebooks and compare formats

```bash
python examples/execute_notebooks.py --output-dir outputs/notebooks
```

The runner uses its own Python installation for a **fresh kernel per notebook**, validates nbformat,
executes all cells with nbclient, writes actual outputs in place and records timings. It overwrites
saved notebook outputs; work in a checkout. `--case synthetic|synthetic_peru` selects one. Scripts
and notebooks agree on the numerical metrics recorded in their `report.json` files.
