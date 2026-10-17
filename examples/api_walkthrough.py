# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Executable public-API inspection, ingestion, comparison and serialization without files.

This uses the deterministic synthetic temperatures. It illustrates contracts,
not statistical realism. Plotting and notebook dependencies are unnecessary.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

from _data import synthetic_inputs

import xhw


def main() -> None:
    ds, grid, baseline, definition = synthetic_inputs()
    prepared = xhw.interpolate_gaps(
        ds, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
    )
    fitted = xhw.fit_baseline(
        prepared, variable="sst", grid=grid, baseline=baseline, definition=definition
    )
    mask = xhw.detect(
        prepared, variable="sst", grid=grid, definition=definition, baseline_result=fitted
    )
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    recipe = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.segment(min_area=xhw.MinArea.none()),
            xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=0, min_duration_days=5),
        ),
    )
    objects = xhw.build_objects(mask, recipe=recipe)
    graph = xhw.build_lineage(objects, field=mask.intensity())
    external = xhw.ingest_objects(
        objects.dataset.label,
        mask=mask,
        kind="patches",
        method="tutorial-copy",
        source_layer="event_membership",
        source=xhw.SourceRef(
            dataset_id="synthetic:external-label-example",
            variable="label",
            accessed="2026-10-10",
            product_status="unknown",
        ),
    )
    external.check_against(mask)
    comparison = xhw.compare_objects(objects, external)
    print("Same processed geometry, with explicitly different origins:")
    print(comparison)
    neighbours = xhw.neighbor_fraction(
        mask, footprint=xhw.Footprint.connected(1), layer="event_membership"
    )
    print("Neighbour fraction dimensions:", neighbours.dims)
    partitioned = xhw.partition_merges(objects, mask)
    partitioned.check_invariants()
    partitioned.check_against(mask)
    for product in (fitted, mask, catalog, objects, graph):
        restored = type(product).from_dataset(product.to_dataset())
        restored.check_invariants()
        print(type(product).__name__, "serialized Dataset round-trip checked")
    graph.check_catalog(graph.to_event_catalog(mask))
    restored_graph = xhw.LineageGraph.from_dataset(graph.to_dataset())
    restored_graph.maps(objects, spacetime_event=3, field=mask.intensity())
    assert xhw.GridSpec.from_json(grid.to_json()).spec_id == grid.spec_id

    # A small point fixture makes bootstrap inspection cheap. Three years are illustrative,
    # below the 30-year marine recommendation; the expected advisory is shown explicitly.
    point = prepared.isel(lat=0, lon=0).sel(time=slice("2000-01-01", "2003-01-31"))
    point_grid = replace(
        grid,
        support="point",
        topology="none",
        spatial_dims=(),
        cell_measure=xhw.CellMeasure(kind="none"),
    )
    short_baseline = replace(
        baseline, reference_period=("2000-01-01", "2002-12-31"), min_samples=20
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always", xhw.XHWDataQualityWarning)
        point_fit = xhw.fit_baseline(
            point, variable="sst", grid=point_grid, baseline=short_baseline, definition=definition
        )
    rates = xhw.in_base_rates(
        point, variable="sst", grid=point_grid, baseline_result=point_fit
    )
    print("Illustrative annual bootstrap rates (not a daily mask):")
    print(rates[["exceedance_frequency", "event_day_frequency", "valid_days"]])


if __name__ == "__main__":
    main()
