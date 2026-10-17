# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""EventCatalog: fixture catalogs, analysis modes, censoring and coupling checks."""

import numpy as np
import pytest
import xarray as xr
from _fixtures import (
    ALL,
    EXPECTED_LAG_START_DAYS,
    F4_CANDIDATE,
    baseline,
    catalog,
    f1_point_mhw,
    f4_line,
    f7_coastal_pair,
    hobday,
    point_mask,
    provenance,
)

import xhw


@pytest.mark.parametrize("name", ALL)
def test_fixture_catalogs_match_their_masks(name):
    fixture = ALL[name]()
    fixture.catalog.check_invariants()
    fixture.catalog.check_against(fixture.mask)


def test_f4_monitoring_records_the_edge_candidate():
    fixture = f4_line("monitoring")
    fixture.catalog.check_against(fixture.mask)
    qualified = fixture.catalog.dataset["qualified"].values
    assert qualified.tolist() == [True, True, False]


def test_retrospective_catalog_rejects_candidates():
    mask = f4_line().mask
    rows = [((0,), 1, 5, 5, 5, 0, False, False, 0, True, False),
            ((1,), 1, 5, 5, 5, 0, False, True, 4, True, False), F4_CANDIDATE]
    with pytest.raises(xhw.InvariantError, match="retrospective"):
        catalog(mask, rows)


def test_monitoring_catalog_missing_candidate_fails_against_mask():
    mask = f4_line().mask
    rows = [((0,), 1, 5, 5, 5, 0, False, False, 0, True, False),
            ((1,), 1, 5, 5, 5, 0, False, True, 4, True, False)]
    with pytest.raises(xhw.InvariantError, match="missing rows"):
        catalog(mask, rows, mode="monitoring").check_against(mask)


def test_interior_short_run_is_never_recorded_even_in_monitoring():
    fixture = ALL["F3"]()                        # days 5-6: interior short run
    monitoring = catalog(fixture.mask, [((), 1, 3, 3, 3, 0, False, False, 0, True, False)],
                         mode="monitoring")
    monitoring.check_against(fixture.mask)


def test_wrong_censoring_is_detected_against_mask():
    mask = f4_line().mask
    rows = [((0,), 1, 5, 5, 5, 0, False, False, 0, True, False),
            ((1,), 1, 5, 5, 5, 0, False, False, 0, True, False)]    # missing-data censoring
    with pytest.raises(xhw.InvariantError, match="wrong counts/censoring"):
        catalog(mask, rows).check_against(mask)


@pytest.mark.parametrize("row,message", [
    (((), 1, 12, 11, 10, 2, False, False, 0, True, False), "duration_days"),
    (((), 0, 12, 13, 11, 2, False, False, 0, True, False), "window start not censored"),
    (((), 1, 13, 13, 11, 2, False, False, 0, True, False), "window end not censored"),
    (((), 1, 12, 12, 10, 2, True, False, 0, True, False), "censored flags"),
    (((), 1, 12, 12, 10, 2, False, True, 16, True, False), "unknown censor_reason"),
    (((), 1, 3, 3, 3, 0, False, False, 0, True, False), "shorter than min_duration"),
])
def test_row_rules(row, message):
    mask = f1_point_mhw().mask
    with pytest.raises(xhw.InvariantError, match=message):
        catalog(mask, [row])


def test_overlapping_events_rejected():
    mask = f1_point_mhw().mask
    rows = [((), 1, 7, 7, 5, 2, False, False, 0, True, False),
            ((), 5, 12, 8, 8, 0, False, False, 0, True, False)]
    with pytest.raises(xhw.InvariantError, match="overlapping"):
        catalog(mask, rows)


def test_monitoring_requires_baseline_available_before_window():
    late = baseline(reference_period=("1990-01-01", "2005-12-31"))
    temp = [10, 13, 13, 13, 13, 13, 10]
    exc = np.array([False, True, True, True, True, True, False])
    mask = point_mask(temp, exc, exc, np.zeros(7, bool), definition=hobday(), base=late)
    with pytest.raises(xhw.ValidationError, match="available before"):
        catalog(mask, [((), 1, 5, 5, 5, 0, False, False, 0, True, False)], mode="monitoring")
    causal = baseline(reference_period=("1970-01-01", "1998-12-31"), fit_mode="causal",
                      available_until="1999-06-30")
    mask = point_mask(temp, exc, exc, np.zeros(7, bool), definition=hobday(), base=causal)
    catalog(mask, [((), 1, 5, 5, 5, 0, False, False, 0, True, False)], mode="monitoring")


def test_catalog_needs_mask_as_parent_and_known_granularity():
    fixture = f1_point_mhw()
    ds = fixture.catalog.dataset
    with pytest.raises(xhw.ValidationError, match="parent"):
        xhw.EventCatalog.from_dataset(ds, mask=fixture.mask, granularity="point_temporal",
                                      analysis_mode="retrospective",
                                      provenance=provenance("orphan"))
    parent = provenance("c", parents=("F1",))
    with pytest.raises(xhw.NotSupportedError, match="line, surface or volume"):   # a point
        xhw.EventCatalog.from_dataset(ds, mask=fixture.mask, granularity="object_tracked",
                                      analysis_mode="retrospective", provenance=parent)
    with pytest.raises(xhw.ValidationError):
        xhw.EventCatalog.from_dataset(ds, mask=fixture.mask, granularity="point_temporal",
                                      analysis_mode="realtime", provenance=parent)


def test_catalog_variables_must_be_one_dimensional_and_typed():
    fixture = f1_point_mhw()
    ds = fixture.catalog.dataset.copy(deep=True)
    ds["duration_days"] = ("event", ds["duration_days"].values.astype(np.int32))
    with pytest.raises(xhw.ValidationError, match="int64"):
        xhw.EventCatalog.from_dataset(ds, mask=fixture.mask, granularity="point_temporal",
                                      analysis_mode="retrospective",
                                      provenance=provenance("c", parents=("F1",)))


def test_catalog_values_are_read_only():
    with pytest.raises(ValueError):
        f1_point_mhw().catalog.dataset["qualified"].values[0] = False


# --- F7: coupling is declared, not computed ----------------------------------------------

def test_f7_coupling_spec_matches_catalogs_and_documented_lag():
    pair = f7_coastal_pair()
    for fixture in (pair.ahw, pair.mhw):
        fixture.catalog.check_against(fixture.mask)
    pair.coupling.check_catalogs(pair.ahw.catalog, pair.mhw.catalog)
    with pytest.raises(xhw.ValidationError):
        pair.coupling.check_catalogs(pair.mhw.catalog, pair.ahw.catalog)
    start_m = pair.mhw.catalog.dataset["start"].values[0]
    start_a = pair.ahw.catalog.dataset["start"].values[0]
    assert (start_m - start_a) // np.timedelta64(1, "D") == EXPECTED_LAG_START_DAYS


def test_global_event_key_separates_phenomena():
    pair = f7_coastal_pair()
    keys = {(c.phenomenon, c.provenance.run_id, int(i))
            for c in (pair.ahw.catalog, pair.mhw.catalog)
            for i in c.dataset["event_id"].values}
    assert len(keys) == 2


def test_non_dataset_catalog_input_rejected():
    with pytest.raises(xhw.ValidationError):
        xhw.EventCatalog.from_dataset(xr.DataArray([1]))
