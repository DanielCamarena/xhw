# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Hand-calculated tutorial expectations and actionable input errors, without real data."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
import pytest

import xhw

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))


@pytest.fixture(scope="module")
def tutorial():
    from _data import synthetic_inputs

    raw, grid, baseline, definition = synthetic_inputs()
    prepared = xhw.interpolate_gaps(
        raw, variable="sst", grid=grid, max_days=2, max_filled_fraction=0.05
    )
    fitted = xhw.fit_baseline(
        prepared, variable="sst", grid=grid, baseline=baseline, definition=definition
    )
    mask = xhw.detect(
        prepared, variable="sst", grid=grid, definition=definition, baseline_result=fitted
    )
    return raw, grid, mask


def test_constructed_temporal_counts_and_gap_semantics(tutorial):
    raw, _, mask = tutorial
    # Two bars: 10*15; connector: 3*10; two demonstration cells: 13+13;
    # diagonal pair: 2*5. The cool demonstration day is membership without exceedance.
    assert int(mask.dataset.event_membership.sum()) == 216
    assert int(mask.dataset.exceedance.sum()) == 215
    day = mask.dataset.sel(time="2003-01-07")
    assert bool(day.joined_gap.isel(lat=0, lon=0))
    assert not bool(day.exceedance.isel(lat=0, lon=0))
    assert bool(day.filled.isel(lat=0, lon=8))
    assert bool(day.exceedance.isel(lat=0, lon=8))
    assert not bool(day.joined_gap.isel(lat=0, lon=8))
    assert np.isnan(raw.sst.sel(time="2003-01-07").isel(lat=0, lon=8))
    mask.check_invariants()
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    assert catalog.dataset.sizes["event"] == 20  # 10 bars + 6 connector + 2 demos + 2 diagonal
    catalog.check_against(mask)


def test_constructed_split_merge_and_explicit_tracking(tutorial):
    _, _, mask = tutorial
    recipe = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.segment(min_area=xhw.MinArea.none()),
            xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=0, min_duration_days=5),
        ),
    )
    objects = xhw.build_objects(mask, recipe=recipe)
    objects.check_invariants()
    objects.check_against(mask)
    graph = xhw.build_lineage(objects, field=mask.intensity())
    graph.check_invariants()
    graph.check_against(objects)
    assert int(graph.dataset.is_split.sum()) == 1
    assert int(graph.dataset.is_merge.sum()) == 1
    index = int(np.argmax(graph.dataset.event_area_days.values))
    event = int(graph.dataset.event_number.values[index])
    assert graph.members(spacetime_event=event).size == 20  # 5 + 2*5 + 5 daily nodes
    assert int(graph.dataset.event_n_branches.values[index]) == 4
    assert int(graph.dataset.event_duration_days.values[index]) == 15
    graph.check_catalog(graph.to_event_catalog(mask))
    default = xhw.build_objects(mask)
    assert default.has_label and not default.has_track


def test_irregular_frequency_and_incompatible_units(tutorial):
    raw, grid, _ = tutorial
    with pytest.raises(xhw.ValidationError, match="regular|daily|frequency|1D"):
        grid.check(raw.isel(time=[0, 1, 3]))
    from _data import synthetic_inputs

    _, _, baseline, definition = synthetic_inputs()
    incompatible = raw.copy(deep=True)
    incompatible.sst.attrs["units"] = "m/s"
    with pytest.raises(xhw.UnitsError, match="units|temperature"):
        xhw.fit_baseline(
            incompatible, variable="sst", grid=grid, baseline=baseline, definition=definition
        )


def test_matplotlib_missing_has_an_actionable_error(monkeypatch):
    original = builtins.__import__

    def without_matplotlib(name, *args, **kwargs):
        if name.split(".")[0] == "matplotlib":
            raise ImportError("simulated absent optional requirement")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_matplotlib)
    from _report import pyplot

    with pytest.raises(ImportError, match="examples/requirements.txt"):
        pyplot()
