# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""The seeded Peru reconstruction: reproducible, shaped like OISST, and usable end to end."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import xhw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))


@pytest.fixture(scope="module")
def inputs():
    from _data import synthetic_peru_inputs

    return synthetic_peru_inputs()


def digest(ds) -> str:
    return hashlib.sha256(ds["sst"].values.tobytes()).hexdigest()


def test_same_seed_same_field_other_seed_other_field(inputs):
    from _data import synthetic_peru

    ds = inputs[0]
    assert digest(synthetic_peru(7)) == digest(ds)
    assert digest(synthetic_peru(8)) != digest(ds)


def test_the_recorded_run_was_made_with_this_generator(inputs):
    record = json.loads((ROOT / "docs/run_records/synthetic_peru.json").read_text("utf-8"))
    assert record["source"]["fields"]["checksum"] == "sha256:" + digest(inputs[0])


def test_grid_coast_and_values_follow_the_oisst_case(inputs):
    ds, grid, _, _ = inputs
    grid.check(ds)
    assert dict(ds.sizes) == {"time": 12784, "lat": 32, "lon": 20}
    assert (
        str(ds.time.values[0])[:10] == "1991-01-01" and str(ds.time.values[-1])[:10] == "2025-12-31"
    )
    assert int(ds.domain_mask.sum()) == 349  # the ocean cells of the registered OISST case
    sst = ds["sst"]
    assert bool(sst.where(ds.domain_mask).notnull().sum() == 349 * 12784)
    assert bool(sst.where(~ds.domain_mask).isnull().all())  # land is missing on every day
    assert np.allclose(
        sst.values[~np.isnan(sst.values)] * 100, np.round(sst.values[~np.isnan(sst.values)] * 100)
    )


def test_statistics_stay_near_the_oisst_values_they_were_fitted_to(inputs):
    sst = inputs[0]["sst"].where(inputs[0].domain_mask)
    days = (sst.time.values - sst.time.values[0]).astype("timedelta64[D]").astype(float)
    domain = sst.mean(["lat", "lon"]).values
    seasonal = np.column_stack(
        [np.ones_like(days), days / 365.25]
        + [f(2 * np.pi * k * days / 365.25) for k in (1, 2) for f in (np.cos, np.sin)]
    )
    coef, *_ = np.linalg.lstsq(seasonal, domain, rcond=None)
    anomaly = domain - seasonal @ coef
    assert -0.015 < coef[1] < -0.003  # OISST: -0.0093 degC per year
    assert 1.0 < anomaly.std() < 1.6  # OISST: 1.26
    assert 20.0 < float(sst.mean()) < 22.5  # OISST: 21.1


def test_the_whole_chain_runs_and_finds_splits_and_merges(inputs):
    ds, grid, baseline, definition = inputs
    fitted = xhw.fit_baseline(
        ds, variable="sst", grid=grid, baseline=baseline, definition=definition
    )
    mask = xhw.detect(ds, variable="sst", grid=grid, definition=definition, baseline_result=fitted)
    mask.check_invariants()
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    catalog.check_against(mask)
    share = float(mask.dataset.event_membership.sum() / mask.dataset.valid.sum())
    assert 0.06 < share < 0.13  # OISST: 9.5% of valid days in an MHW; the reconstruction 8.4%
    recipe = xhw.SpatialRecipe(
        input_layer="event_membership",
        steps=(
            xhw.RecipeStep.segment(min_area=xhw.MinArea.default()),
            xhw.RecipeStep.track(min_overlap=0.0, max_gap_days=0, min_duration_days=5),
        ),
    )
    objects = xhw.build_objects(mask, recipe=recipe)
    objects.check_against(mask)
    graph = xhw.build_lineage(objects, field=mask.intensity(), weighted=True)
    graph.check_against(objects)
    assert int(graph.dataset.is_split.sum()) >= 1 and int(graph.dataset.is_merge.sum()) >= 1
