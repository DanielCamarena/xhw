# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""A complete, hand-checkable synthetic pipeline; no input data or network needed."""

from pathlib import Path

import numpy as np
from _data import synthetic_inputs
from _report import arguments, spatial_summary, temporal_summary, write_report
from plotting import (
    connectivity_figure,
    frequency_figure,
    genealogy_figure,
    largest_event,
    mask_figure,
    metrics_figure,
    sequence_figure,
    series_figure,
)

import xhw


def run(data_dir: Path, output_dir: Path) -> dict:
    # %% Prepare a deterministic field and declare its scientific meaning
    raw, grid, baseline, definition = synthetic_inputs()
    grid.check(raw)
    print("Synthetic surface: constant reference 1973–2002; injected event in January 2003.")

    # %% Interpolate one missing temperature; keep its evidence flag
    prepared = xhw.interpolate_gaps(
        raw, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
    )
    assert int(prepared.filled.sum()) == 1

    # %% Fit the baseline and detect per-cell events
    fitted = xhw.fit_baseline(
        prepared, variable="sst", grid=grid, baseline=baseline, definition=definition
    )
    fitted.check_invariants()
    mask = xhw.detect(
        prepared, variable="sst", grid=grid, definition=definition, baseline_result=fitted
    )
    mask.check_invariants()
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    catalog.check_against(mask)
    assert int(mask.dataset.joined_gap.sum()) == 1  # cool day; interpolated hot day exceeds
    figure = series_figure(
        raw,
        mask,
        variable="sst",
        selection={"lat": 0, "lon": 0},
        period=("2003-01-01", "2003-01-20"),
        output_dir=output_dir,
    )
    figure = series_figure(
        raw,
        mask,
        variable="sst",
        selection={"lat": 0, "lon": 8},
        period=("2003-01-01", "2003-01-20"),
        output_dir=output_dir,
        name="missing",
    )

    # %% Compare geometry transformations without rewriting temporal evidence
    morphology = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.morph("close", footprint=xhw.Footprint.connected(1)),
            xhw.RecipeStep.segment(min_area=xhw.MinArea.none()),
        ),
    )
    processed = xhw.build_objects(mask, recipe=morphology)
    processed.check_invariants()
    processed.check_against(mask)
    figure = mask_figure(mask, processed, "2003-01-07", output_dir)
    figure = connectivity_figure(mask, "2003-01-07", output_dir)

    # %% Segment daily patches and explicitly track them
    recipe = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.segment(
                footprint=xhw.Footprint.connected(1), min_area=xhw.MinArea.none()
            ),
            xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=0, min_duration_days=5),
        ),
    )
    objects = xhw.build_objects(mask, recipe=recipe, analysis_mode="retrospective")
    objects.check_invariants()
    objects.check_against(mask)
    dates = ["2003-01-05", "2003-01-06", "2003-01-10", "2003-01-11", "2003-01-15"]
    figure = sequence_figure(objects, dates, output_dir)

    # %% Genealogy retains all qualifying parents and children
    graph = xhw.build_lineage(objects, field=mask.intensity(), weighted=True)
    graph.check_invariants()
    graph.check_against(objects)
    tracked_catalog = graph.to_event_catalog(mask)
    graph.check_catalog(tracked_catalog)
    event = largest_event(graph)
    assert int(graph.dataset.is_split.sum()) == 1
    assert int(graph.dataset.is_merge.sum()) == 1
    assert graph.members(spacetime_event=event).size == 20  # 5 + 2*5 + 5 daily patches
    assert (
        np.unique(graph.dataset.object_branch.values[graph.members(spacetime_event=event)]).size
        == 4
    )
    figure = genealogy_figure(graph, event, output_dir)
    figure = metrics_figure(graph, objects, event, output_dir)
    frequency_figure({"Synthetic membership": mask}, output_dir)
    del figure

    from exploration import objects_gif

    objects_gif(objects, output_dir, dates=mask.dataset.time.sel(
        time=slice("2003-01-01", "2003-01-20")).values)

    # %% Record real execution results and specifications
    metrics = {
        "temporal": temporal_summary(mask, catalog),
        "spatial": spatial_summary(objects, graph),
        "main_event_daily_patches": int(graph.members(spacetime_event=event).size),
        "morphology_added_cell_days": int(processed.dataset.added.sum()),
    }
    report = write_report(
        output_dir,
        "synthetic",
        metrics,
        grid=grid,
        baseline=baseline,
        definition=definition,
        recipe=recipe,
        graph=graph,
        review="executed; deterministic expectations checked",
        extra={
            "selection": "1973-01-01 to 2003-01-31; 7 x 9 surface",
            "expected_main_event": "20 patches, 4 branches, 1 split, 1 merge",
        },
    )
    return report


if __name__ == "__main__":
    args = arguments(__doc__)
    run(args.data_dir, args.output_dir)
