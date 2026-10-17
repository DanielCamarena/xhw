# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""build_catalog (temporal detection ): rows reproduce the hand-written catalogs; metrics by
hand.
"""

import numpy as np
import pytest
import xarray as xr
from _fixtures import (
    ALL,
    constant_baseline,
    f4_line,
    f7_coastal_pair,
    hobday,
    point_mask,
)

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
ROW_VARS = ("start", "end", "duration_days", "n_exceedance_days", "n_joined_gap_days",
            "left_censored", "right_censored", "censor_reason", "qualified", "contains_forecast")
BASES = {"F1": (12.0, 10.0), "F2": (8.0, 10.0), "F3": (30.0, 25.0), "F4": (12.0, 10.0),
         "F5": (12.0, 10.0), "F6": (12.0, 10.0), "F8": (12.0, 10.0), "F9": (12.0, 10.0)}


def detected(fixture, q, c, definition=None, ds=None):
    base = constant_baseline(fixture.mask, threshold=q, reference=c, definition=definition)
    return xhw.detect(ds if ds is not None else fixture.mask.dataset, variable="temperature",
                      grid=fixture.mask.grid, definition=definition or fixture.mask.definition,
                      baseline_result=base)


def assert_same_rows(built, expected):
    for name in (*ROW_VARS, *(f"index_{d}" for d in expected.location_dims)):
        np.testing.assert_array_equal(built.dataset[name].values, expected.dataset[name].values,
                                      err_msg=name)


@pytest.mark.parametrize("name", BASES)
def test_rows_reproduce_hand_written_catalogs(name):
    fixture = ALL[name]()
    mask = detected(fixture, *BASES[name])
    built = xhw.build_catalog(mask, analysis_mode=fixture.catalog.analysis_mode)
    assert_same_rows(built, fixture.catalog)
    built.check_against(mask)


def test_monitoring_candidate_is_reproduced():
    fixture = f4_line("monitoring")
    built = xhw.build_catalog(detected(fixture, 12.0, 10.0), analysis_mode="monitoring")
    assert_same_rows(built, fixture.catalog)


def test_coupling_pair_catalogs_are_reproduced():
    pair = f7_coastal_pair()
    for fixture, base in ((pair.mhw, (12.0, 10.0)), (pair.ahw, (30.0, 25.0))):
        assert_same_rows(xhw.build_catalog(detected(fixture, *base),
                                           analysis_mode="retrospective"), fixture.catalog)


# --- metrics computed by hand -----------------------------------------------------------------

def f1_metrics():
    return xhw.build_catalog(detected(ALL["F1"](), 12.0, 10.0),
                             analysis_mode="retrospective").dataset.isel(event=0)


def test_f1_intensities_include_joined_gap_days():
    m = f1_metrics()     # anomaly over days 1-12: 3,3,3,3,3,0,0,3,3,3,3,3
    assert m["intensity_max"] == 3.0
    assert m["intensity_mean"] == pytest.approx(2.5)
    assert m["intensity_std"] == pytest.approx(np.sqrt(1.25))
    assert m["intensity_cumulative"] == pytest.approx(30.0)
    assert m["intensity_max_rel_threshold"] == 1.0
    assert m["intensity_mean_rel_threshold"] == pytest.approx(0.5)
    assert m["intensity_cumulative_rel_threshold"] == pytest.approx(6.0)
    assert m["temperature_max"] == 13.0 and m["temperature_mean"] == pytest.approx(12.5)
    assert m["peak_date"].values == np.datetime64("2000-01-02")
    assert m["intensity_cumulative"].attrs["units"] == "degC d"
    assert m["intensity_mean"].attrs["units_metadata"] == "temperature: difference"


def test_f1_rates_follow_the_half_day_convention():
    m = f1_metrics()
    assert m["rate_onset"] == pytest.approx((3 - 0.5 * (3 + 0)) / 0.5)       # 3.0
    assert m["rate_decline"] == pytest.approx((3 - 0.5 * (3 + 0)) / 11.5)


def test_event_at_window_start_uses_the_edge_rule():
    m = xhw.build_catalog(detected(ALL["F5"](), 12.0, 10.0),
                          analysis_mode="retrospective").dataset.isel(event=0)
    assert m["rate_onset"] == 0.0                                            # peak on day 0
    assert m["rate_decline"] == pytest.approx(1.5 / 4.5)


def heat_series(values):
    temp = [10.0, *values, 10.0]
    exc = np.array([t > 12 for t in temp])
    return point_mask(temp, exc, exc, np.zeros(len(temp), bool), definition=hobday())


def test_heat_categories_follow_hobday2018():
    fixture = type("F", (), {"mask": heat_series([13, 14, 16, 18, 13])})
    m = xhw.build_catalog(detected(fixture, 12.0, 10.0),
                          analysis_mode="retrospective").dataset.isel(event=0)
    assert int(m["category_max"]) == 4                                       # 18 -> IV
    assert [int(m[f"days_category_{k}"]) for k in (1, 2, 3, 4)] == [2, 1, 1, 1]
    assert m["category_max"].attrs["category_scheme"] == "hobday2018"


def test_cold_spells_have_negative_intensities_and_schlegel_categories():
    definition = hobday(phenomenon="mcs", tail="lower", comparator="<", percentile=10.0,
                        category_scheme="schlegel2021")
    temp = [10.0, 7.0, 7.0, 4.0, 7.0, 7.0, 10.0]
    exc = np.array([t < 8 for t in temp])
    fixture = type("F", (), {"mask": point_mask(temp, exc, exc, np.zeros(7, bool),
                                                definition=definition, threshold=8.0)})
    m = xhw.build_catalog(detected(fixture, 8.0, 10.0, definition=definition),
                          analysis_mode="retrospective").dataset.isel(event=0)
    assert m["intensity_max"] == -6.0 and m["temperature_min"] == 4.0
    assert int(m["category_max"]) == 3          # r = (4 - 10) / (8 - 10) = 3 -> III
    assert m["rate_onset"] > 0                  # rates describe the oriented anomaly


def test_filled_and_missing_days_are_counted():
    fixture = f4_line()
    profile = hobday(missing_policy="as_non_exceedance")
    ds = xhw.interpolate_gaps(fixture.mask.dataset, variable="temperature",
                              grid=fixture.mask.grid, max_days=2, max_filled_fraction=0.2)
    filled = xhw.build_catalog(detected(fixture, 12.0, 10.0, profile, ds),
                               analysis_mode="retrospective").dataset
    assert filled["n_filled_days"].values.tolist() == [0, 1]
    temp = [10, 13, 13, 13, 13, 13, np.nan, 13, 13, 13, 13, 13, 10]
    exc = np.array([t == 13 for t in temp])
    reason = np.where(np.isnan(temp), 2, 0).astype(np.uint8)
    gap_fixture = type("F", (), {"mask": point_mask(temp, exc, exc, np.zeros(13, bool),
                                                    definition=hobday(), invalid_reason=reason)})
    m = xhw.build_catalog(detected(gap_fixture, 12.0, 10.0, profile),
                          analysis_mode="retrospective").dataset.isel(event=0)
    assert int(m["n_missing_days"]) == 1 and int(m["duration_days"]) == 11
    assert m["intensity_mean"] == pytest.approx(3.0)       # missing day excluded


def test_monitoring_requires_an_available_baseline_and_empty_catalogs_work(tmp_path):
    fixture = ALL["F1"]()
    base = constant_baseline(fixture.mask, threshold=99.0, reference=10.0)   # no exceedance
    mask = xhw.detect(fixture.mask.dataset, variable="temperature", grid=fixture.mask.grid,
                      definition=fixture.mask.definition, baseline_result=base)
    empty = xhw.build_catalog(mask, analysis_mode="retrospective")
    assert empty.dataset.sizes["event"] == 0
    with pytest.raises(xhw.ValidationError):
        xhw.build_catalog(mask, analysis_mode="realtime")


def test_catalog_with_metrics_round_trips(tmp_path):
    built = xhw.build_catalog(detected(ALL["F1"](), 12.0, 10.0), analysis_mode="retrospective")
    path = tmp_path / "catalog.nc"
    built.to_dataset().to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        again = xhw.EventCatalog.from_dataset(opened.load())
    for name in built.dataset.data_vars:
        np.testing.assert_array_equal(again.dataset[name].values, built.dataset[name].values)
    assert again.provenance.steps[0].name == "build_catalog"
