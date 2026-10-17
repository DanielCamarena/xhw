# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Synthetic Peru case: a seeded reconstruction of OISST for MHW/MCS, baselines and objects.

No data files and no network: the field is generated in about a second. It reproduces the grid,
coast, seasonal cycle, trend and anomaly statistics of the OISST case without copying any value,
so every stage can be run in a minute. Results are illustrations, not climate findings.
"""

from dataclasses import replace
from pathlib import Path

from _data import marine_definition, synthetic_peru_inputs
from _report import (
    arguments,
    lineage_catalog_status,
    spatial_summary,
    temporal_summary,
    write_report,
)
from plotting import (
    frequency_figure,
    genealogy_figure,
    largest_event,
    metrics_figure,
    sequence_figure,
    series_figure,
)

import xhw


def run(data_dir: Path, output_dir: Path) -> dict:
    # %% Generate the reconstruction (seed 7) and declare it like any other input
    raw, grid, baseline, definition = synthetic_peru_inputs()
    grid.check(raw)
    print("Synthetic Peru reconstruction, 1991–2025; reference 1991–2020; seed 7.")
    prepared = xhw.interpolate_gaps(
        raw, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
    )

    # %% Marine heatwaves and cold spells use distinct percentile tails
    masks, catalogs, metrics = {}, {}, {}
    for name, detector in (("MHW", definition), ("MCS", marine_definition(cold=True))):
        fitted = xhw.fit_baseline(
            prepared, variable="sst", grid=grid, baseline=baseline, definition=detector
        )
        fitted.check_invariants()
        masks[name] = xhw.detect(
            prepared, variable="sst", grid=grid, definition=detector, baseline_result=fitted
        )
        masks[name].check_invariants()
        catalogs[name] = xhw.build_catalog(masks[name], analysis_mode="retrospective")
        catalogs[name].check_against(masks[name])
        metrics[name] = temporal_summary(masks[name], catalogs[name])
    mask = masks["MHW"]
    figure = frequency_figure(masks, output_dir)
    figure = series_figure(
        raw,
        mask,
        variable="sst",
        selection={"lat": 15, "lon": 10},
        period=("2023-02-01", "2023-06-01"),
        output_dir=output_dir,
    )

    # %% Compare the same definition with a retrospective detrended baseline
    dynamic = replace(baseline, method="detrended")
    fitted_dynamic = xhw.fit_baseline(
        prepared, variable="sst", grid=grid, baseline=dynamic, definition=definition
    )
    fitted_dynamic.check_invariants()
    dynamic_mask = xhw.detect(
        prepared, variable="sst", grid=grid, definition=definition, baseline_result=fitted_dynamic
    )
    dynamic_mask.check_invariants()
    dynamic_catalog = xhw.build_catalog(dynamic_mask, analysis_mode="retrospective")
    dynamic_catalog.check_against(dynamic_mask)
    metrics["detrended"] = temporal_summary(dynamic_mask, dynamic_catalog)
    common_valid = mask.dataset.valid & dynamic_mask.dataset.valid
    a, b = mask.dataset.event_membership, dynamic_mask.dataset.event_membership
    union = int(((a | b) & common_valid).sum())
    metrics["fixed_detrended_jaccard"] = (
        float(((a & b) & common_valid).sum()) / union if union else 1.0
    )
    figure = frequency_figure(
        {"Fixed MHW": mask, "Detrended MHW": dynamic_mask}, output_dir, name="baseline_comparison"
    )

    # %% Track event-membership patches; the default recipe would not track
    recipe = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.segment(
                footprint=xhw.Footprint.connected(1), min_area=xhw.MinArea.default()
            ),
            xhw.RecipeStep.track(min_overlap=0.0, max_gap_days=0, min_duration_days=5),
        ),
    )
    objects = xhw.build_objects(mask, recipe=recipe, analysis_mode="retrospective")
    objects.check_invariants()
    objects.check_against(mask)
    figure = sequence_figure(objects, ["2023-03-01", "2023-03-15", "2023-04-01"], output_dir)

    # %% Derive branches and spacetime events with area-weighted intensity
    graph = xhw.build_lineage(objects, field=mask.intensity(), weighted=True)
    graph.check_invariants()
    graph.check_against(objects)
    metrics["lineage_catalog_export"] = lineage_catalog_status(graph, mask)
    event = largest_event(graph)
    figure = genealogy_figure(graph, event, output_dir)
    metrics_figure(graph, objects, event, output_dir)
    del figure
    metrics["spatial"] = spatial_summary(objects, graph)
    metrics["largest_size_time_event"] = int(event)

    # %% Preserve the distinction between reproduced output and scientific approval
    report = write_report(
        output_dir,
        "synthetic_peru",
        metrics,
        grid=grid,
        baseline=baseline,
        definition=definition,
        recipe=recipe,
        graph=graph,
        review="executed; synthetic reconstruction, an illustration and not a climate result",
        extra={
            "selection": "synthetic: 32 x 20 cells, 0.25 degree, lat -12 to -4, lon -82 to -77",
            "generator": "examples/_data.py::synthetic_peru, seed 7, rounded to 0.01 degC",
            "dynamic_baseline": dynamic.to_dict(),
            "comparison_denominator": "union of membership on jointly valid cell-days",
            "common_valid_cell_days": int(common_valid.sum()),
            "oisst_registered_reference": {
                "per_cell_events": 16702,
                "tracks": 118,
                "spacetime_events": 136,
                "daily_patches": 2956,
                "note": "the reconstruction is calibrated to these figures only roughly",
            },
        },
    )
    return report


if __name__ == "__main__":
    args = arguments(__doc__)
    run(args.data_dir, args.output_dir)
