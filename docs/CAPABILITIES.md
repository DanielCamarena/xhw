# Capability, public API and evidence

[Index](README.md) · [Validation](VALIDATION.md) · [Tutorials](TUTORIALS.md)

The installed xHW package defines the available calls. The
following matrix separates implemented capability, evidence and public teaching
material. Mathematical and numerical equivalence tests support the listed contracts.
Real-data examples live in [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark);
execution alone does not grant scientific approval.

| Capability | Public API | Evidence in this distribution | Example |
|---|---|---|---|
| Data identity and input checks | SourceRef, GridSpec.check | [grid tests](../tests/test_grid.py), [spec tests](../tests/test_specs.py) | All tutorials; real inputs in [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark) |
| Declared hourly aggregation | aggregate_daily | [aggregation tests](../tests/test_aggregate.py); exact registered hourly/daily agreement in the ERA5 case | ERA5 case in [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark) |
| Limited temperature interpolation | interpolate_gaps | [baseline tests](../tests/test_baseline.py), [tutorial expectations](../tests/test_tutorial_contracts.py) | Synthetic missing versus temporal gap |
| Fixed seasonal baseline and temporal marine detection | fit_baseline, detect, build_catalog | [marine equivalence](../tests/test_equivalence_mhw.py), [detection fixtures](../tests/test_detect.py) | Synthetic, synthetic Peru; real cases in [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark) |
| AHW and daily AND/lag profiles | DailyComponent, DefinitionSpec, detect | [AHW tests](../tests/test_ahw.py), [climdex comparisons](../tests/test_equivalence_climdex.py) | ERA5 four profiles over two domains, in [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark) |
| Within-reference correction and annual rates | BaselineSpec, in_base_rates | [correction](../tests/test_in_base_correction.py), [annual rates](../tests/test_in_base.py) | [baseline guide](BASELINES.md), [API walkthrough](../examples/api_walkthrough.py) |
| Six dynamic methods | BaselineSpec, fit_baseline | [dynamic tests](../tests/test_dynamic.py), [STL](../tests/test_equivalence_stl.py), [SSA](../tests/test_equivalence_ssa.py) | Synthetic Peru fixed/detrended; all six described in the baseline guide |
| Geometry and daily segmentation | SpatialRecipe, build_objects | [morphology](../tests/test_morph.py), [objects](../tests/test_objects.py), [structured grids](../tests/test_structured.py) | Synthetic, synthetic Peru |
| Volume and physical vertical connectivity | VerticalSpec, Footprint | [volumes](../tests/test_volumes.py), [vertical](../tests/test_vertical.py) | GLORYS retained levels, in [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark) |
| Explicit overlap tracking | RecipeStep.track | [tracking](../tests/test_tracking.py), [track rules](../tests/test_track_rules.py) | Synthetic, synthetic Peru |
| Genealogy, branches and spacetime groups | build_lineage, LineageGraph | [genealogy](../tests/test_lineage.py), [external comparison](../tests/test_equivalence_lineage.py) | Manually checked split/merge; synthetic Peru graph; real graphs in [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark) |
| Weighted metrics and map/series inspection | graph.series/maps, mask.intensity | [metrics](../tests/test_lineage_metrics.py), [supports](../tests/test_lineage_supports.py) | Size/intensity curves, occupied-day maps |
| External label ingestion/comparison | ingest_objects, compare_objects, neighbor_fraction | [ingestion](../tests/test_ingest.py), [object references](../tests/test_equivalence_objects.py) | API walkthrough and complete signatures |
| Optional merge partition | partition_merges | [partition tests](../tests/test_partition.py) | API walkthrough; new geometry output |
| Serialization and identity | to_dataset/from_dataset; spec JSON | [round trips](../tests/test_roundtrip.py), [public API snapshot](../tests/test_api_snapshot.py) | API walkthrough |
| Figures (example code, not API) | none; [plotting.py](../examples/plotting.py) over xarray/Matplotlib | [documented code test](../tests/test_documentation_code.py) runs the README and plotting guide | [Plotting guide](PLOTTING.md); both tutorials |
| Declarative association metadata | CouplingSpec | [metadata tests](../tests/test_specs.py); no association engine | Explicitly labelled declarative in API |

## Review states

| Material | Execution state | Scientific review state |
|---|---|---|
| Small synthetic event geometry | Executed; manual numerical expectations and invariants checked | Constructed illustration, not a climate-data conclusion |
| Synthetic Peru reconstruction | Executed; generator, tests and run record agree (the record's source checksum is reproduced by the generator) | Illustration of the API, not a climate result |
| Real-data cases (OISST, OSTIA, ERA5, GLORYS) | Executed and reproduced in [xhw-benchmark](https://github.com/DanielCamarena/xhw-benchmark) | OISST temporal reference approved; every other summary pending scientific review |

[Catalog export qualification](OBJECTS.md#catalog-export-qualification) requires
components that meet the mask definition's duration rule. A valid graph can retain
shorter components for analysis through its dataset, series and maps.
The tests cover input errors, hand-calculated tutorial expectations and deterministic
synthetic generation.
