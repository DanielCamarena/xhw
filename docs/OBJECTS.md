# Connected objects, tracking and genealogy

[← Baselines](BASELINES.md) · [Index](README.md) · [Metrics →](METRICS.md)

## A spatial recipe changes the representation

`build_objects(mask, recipe=...)` receives an immutable `XMask`. It starts from
`exceedance` or `event_membership`, restricted to valid cells, and applies an
**ordered** `SpatialRecipe`. The result is an `ObjectField`, with:

- `processed_mask`, `added`, `removed` and `valid`: geometry and its relation to
  the chosen original layer;
- `label`: consecutive daily patch labels, zero for background;
- an `object` table: day, daily label, size, cell count and centroid;
- `track_id` and a `track` table only when the recipe includes tracking.

The original mask keeps its evidence. An added spatial cell does not become an
original thermal exceedance. Missing cells remain barriers. `added`/`removed`
refer to the chosen input layer, not a freely selected comparison layer.

```python
default = xhw.SpatialRecipe.default()
# exceedance -> segment (faces, size >= 5% of physical domain) -> persist
patches = xhw.build_objects(mask, recipe=default)
assert patches.has_label and not patches.has_track
```

The default has **no track step**. `persist` applies per-cell persistence to the
working geometry and inherits the mask's duration/gap rules. Spatial filtering
before persistence can differ from spatial filtering of already detected temporal
events. Order is part of the scientific configuration.

## Connectivity and filters

| Support | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| Line | Adjacent cells | Same two neighbours | Same two neighbours |
| Surface | 4 face neighbours | 8 face/diagonal neighbours | 8 neighbours |
| Volume | 6 face neighbours | 18 face/edge neighbours | 26 neighbours |

Use `Footprint.connected(rank)`, a surface physical disk via
`Footprint.disc(radius_km)`, or symmetric explicit index offsets. A physical
horizontal disk is not a general 3-D distance neighbourhood.

`MinArea.default()` means **5 percent**, represented by `value=5.0`, of the grid's
declared physical domain measure. It is a configurable choice. A regional
absolute minimum or `MinArea.none()` may better fit the question. A percentile
minimum is calibrated from patch sizes; monitoring requires a separate frozen
calibration sample. A warning reports when more than 90% of patches are discarded.
Line sizes are lengths, surface sizes areas and volume sizes volumes; choose
`absolute_km`, `absolute_km2` or `absolute_km3` accordingly.

![The same mask under two neighbourhood definitions](assets/synthetic/connectivity.png)

*Executed synthetic field on January 7. The two upper-right hot cells touch only
diagonally: rank 1 keeps them separate, rank 2 joins them. The two large bars remain
separate. No temperature or temporal evidence changes.*

![Spatial closing preserves the original layer](assets/synthetic/morphology.png)

*January 7, `event_membership → close(rank 1) → segment(no size cut)`. Closing
bridges part of the spatial gap; the removal panel is empty in this example. Yellow means true.
The first panel remains the original membership, while changes are recorded in
`added` and `removed`. The recipe is illustrative, not a recommended morphology
for every ocean region.*

For volumes, `vertical_gap_m` connects cells if the intervening layer thickness
fits the physical limit in both columns. Adjacent layers touch at gap zero.
Decimated levels and approximate thickness can change which voxels connect;
see the [GLORYS tutorial](https://github.com/DanielCamarena/xhw-benchmark/blob/main/docs/TUTORIALS.md#glorys) in xhw-benchmark. `periodic_x=True` permits longitude wrapping;
regional borders do not wrap by default.

## Add tracking explicitly

The complete synthetic tutorial uses:

```python
recipe = xhw.SpatialRecipe(input_layer="event_membership", steps=(
    xhw.RecipeStep.segment(footprint=xhw.Footprint.connected(1),
                          min_area=xhw.MinArea.none()),
    xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=0, min_duration_days=5),
))
objects = xhw.build_objects(mask, recipe=recipe, analysis_mode="retrospective")
objects.check_invariants()
objects.check_against(mask)
```

The track step must be last and directly follow segmentation. Overlap uses the
**same grid cells**; it does not advect a patch or link by distance alone.

| `link_measure` | Formula, with shared measure I and source/target sizes A/B |
|---|---|
| `overlap_smaller` (default) | I / min(A, B) |
| `iou` | I / (A + B − I) |
| `source_fraction` | I / A |
| `target_fraction` | I / B |

The comparator is inclusive `>=`. Threshold zero still requires positive shared
measure. `max_gap_days` permits object disappearance and reappearance; this is
independent of the detector's temporal gap at a cell. State the track duration
filter separately from temporal persistence. Retrospective IDs use the complete
analysis window; monitoring assigns IDs forward under dominance/tie/branch rules.
Successive calls are complete-window computations, not a persistent tracker service.

![Daily labels and retrospective track identifiers](assets/synthetic/objects.png)

*Face connectivity, no size cut, smaller-object overlap ≥0.1, no object gap,
minimum track duration 5 days. Numbers are actual IDs; palettes may repeat for
large ID sets. Daily labels restart each day. The central split and merged pieces
share one retrospective connected-component track ID.*

## Genealogy retains the transitions

```python
graph = xhw.build_lineage(objects, field=mask.intensity(), weighted=True)
graph.check_invariants()
graph.check_against(objects)
edges = graph.transitions()
nodes = graph.members(spacetime_event=3)  # object-row indices, not a Dataset
```

The event number in this snippet refers to the constructed tutorial; in another
case obtain it from `graph.dataset.event_number`. The default `LinkPolicy` is
inferred from the recipe's track step, or uses any-positive-overlap linking for
objects without tracking. Explicitly changing policy can change the genealogy.

| Element | Meaning |
|---|---|
| Node | One daily patch in the object table |
| Transition | Candidate shared-cell pair; `linked` identifies edges passing policy; below-threshold edges remain inspectable |
| Branch | Maximal chain of 1-to-1 links; split/merge ends incoming branches and creates outgoing ones |
| Retrospective spacetime event | Connected component of qualifying links, including its branches |
| Monitoring event | Forward identity under `merge_rule="new_event"` or `"survivor"`; provisional/window information remains explicit |

```mermaid
flowchart LR
    A[Branch A: days 1-5] -->|split| B[Branch B: days 6-10]
    A -->|split| C[Branch C: days 6-10]
    B -->|merge| D[Branch D: days 11-15]
    C -->|merge| D
```

*Conceptual diagram of the executed central synthetic event: four branches,
20 daily patches, one split and one merge in a single 15-day spacetime event.*

Dominance is an annotation, not permission to delete the other parent or child.
Tracker identity, a branch and a spacetime event are different concepts, even
when some IDs coincide in a particular retrospective configuration. Genealogy
records association/precedence under overlap rules; it does not attribute causes.
Optional `partition_merges` creates a new partitioned ObjectField rather than
rewriting the source labels; it answers a different geometry question.

## Catalog export qualification

The track duration filter suppresses IDs of short components but retains their
daily **patch labels**. `build_lineage` uses all daily labels, so its valid graph
can contain short components that have no retained tracker ID. This is visible
in the synthetic Peru tutorial (140 spacetime components versus 112 retained tracks) and in
the real OISST case (136 versus 118).

`graph.to_event_catalog(mask)` declares all graph rows qualified. It raises
`InvariantError: qualified events shorter than min_duration_days` when a component
is below the mask definition's minimum. The synthetic Peru tutorial records 28 such
components (18 in the real OISST case) and uses `graph.dataset`, `.series()` and `.maps()` for analysis.
It does not silently discard or relabel them. The synthetic export is valid and
checked with `graph.check_catalog(...)`. Review qualification explicitly before using the export.
