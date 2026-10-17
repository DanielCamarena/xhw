# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Portable run reports for the public tutorials; figures live in plotting.py."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

# Re-exported so tutorial code can keep importing figures from one place.
from plotting import (  # noqa: F401
    connectivity_figure,
    frequency_figure,
    genealogy_figure,
    largest_event,
    map_panel,
    mask_figure,
    metrics_figure,
    pyplot,
    save_figure,
    sequence_figure,
    series_figure,
)
from provenance import code_fingerprint

import xhw


def arguments(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs") / parser.prog.removesuffix(".py")
    )
    return parser.parse_args()


def temporal_summary(mask, catalog) -> dict:
    ds = mask.dataset
    return {
        "events": int(catalog.dataset.sizes["event"]),
        "exceedance_cell_days": int(ds["exceedance"].sum()),
        "event_cell_days": int(ds["event_membership"].sum()),
        "joined_cell_days": int(ds["joined_gap"].sum()),
        "valid_cell_days": int(ds["valid"].sum()),
        "filled_cell_days": int(ds["filled"].sum()) if "filled" in ds else 0,
        "max_duration_days": int(catalog.dataset["duration_days"].max())
        if catalog.dataset.sizes["event"]
        else 0,
    }


def spatial_summary(objects, graph) -> dict:
    ds, g = objects.dataset, graph.dataset
    return {
        "patches": int(ds.sizes["object"]),
        "tracks": int(ds.sizes.get("track", 0)),
        "branches": int(g.sizes["branch"]),
        "spacetime_events": int(g.sizes["spacetime_event"]),
        "splits": int(g["is_split"].sum()),
        "merges": int(g["is_merge"].sum()),
        "max_size": float(ds["object_area"].max()),
        "size_units": ds["object_area"].attrs["units"],
        "max_size_time": float(g["event_area_days"].max()),
        "max_duration_days": int(g["event_duration_days"].max()),
    }


def lineage_catalog_status(graph, mask) -> dict:
    """Inspect catalog qualification; never suppress other validation errors.

    Tracking's duration filter suppresses track IDs, not the daily patch labels.
    A graph over those labels can contain short components. Its EventCatalog export
    declares all rows qualified, and therefore rejects such components.
    """
    short = int((graph.dataset.event_duration_days < mask.definition.min_duration_days).sum())
    if short:
        try:
            graph.to_event_catalog(mask)
        except xhw.InvariantError as exc:
            if str(exc) != "qualified events shorter than min_duration_days":
                raise
            print(
                f"Catalog export constraint: {short} graph components are shorter than "
                f"{mask.definition.min_duration_days} days. Use graph.dataset for these components."
            )
            return {
                "status": "not exportable as qualified EventCatalog",
                "short_components": short,
                "reason": str(exc),
            }
        raise AssertionError("Expected the documented catalog qualification constraint")
    catalog = graph.to_event_catalog(mask)
    graph.check_catalog(catalog)
    return {"status": "exported and checked", "short_components": 0}


def write_report(
    output_dir: Path,
    case: str,
    metrics: dict,
    *,
    grid,
    baseline,
    definition,
    recipe=None,
    graph=None,
    review: str,
    extra: dict | None = None,
) -> dict:
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "xhw",
            "numpy",
            "xarray",
            "scipy",
            "matplotlib",
            "h5netcdf",
            "nbformat",
            "nbclient",
        )
    }
    report = {
        "case": case,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "review_status": review,
        "platform": platform.system(),
        "python": platform.python_version(),
        "versions": versions,
        "code": code_fingerprint(),
        "source": baseline.source.to_dict(),
        "grid": grid.to_dict(),
        "baseline": baseline.to_dict(),
        "definition": definition.to_dict(),
        "recipe": recipe.to_dict() if recipe else None,
        "link_policy": graph.policy.to_dict() if graph else None,
        "extra": extra or {},
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    # Only portable scientific results are printed; no workstation paths or credentials.
    print(json.dumps(metrics, indent=2))
    return report
