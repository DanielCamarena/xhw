# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""fit_baseline / BaselineResult (temporal detection ). Expected values computed by hand."""

import numpy as np
import pytest
import xarray as xr
from _fixtures import baseline, f5_surface, grid, hobday, surface_grid

import xhw
from xhw._baseline import doy_positions

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")


def point_ds(values, start, *, units="degC", domain=True, filled=None):
    first = np.datetime64(start, "D")
    times = np.arange(first, first + np.timedelta64(len(values), "D")).astype("datetime64[ns]")
    ds = xr.Dataset(
        {"sst": ("time", np.asarray(values, dtype=float), {"units": units}),
         "domain_mask": ((), domain)},
        coords={"time": times, "lat": -12.0, "lon": -77.5},
    )
    if filled is not None:
        ds["filled"] = ("time", np.asarray(filled, dtype=bool))
    return ds


def by_year(start, end, value_of):
    first, last = np.datetime64(start, "D"), np.datetime64(end, "D")
    times = np.arange(first, last + np.timedelta64(1, "D")).astype("datetime64[ns]")
    return times, np.array([value_of(t) for t in times.astype("datetime64[D]")])


def fit(ds, *, base=None, definition=None, spec=None):
    return xhw.fit_baseline(ds, variable="sst", grid=spec or grid(),
                            baseline=base or baseline(), definition=definition or hobday())


def three_years(**overrides):
    params = dict(reference_period=("2000-01-01", "2002-12-31"), doy_window_days=1,
                  smoothing_window_days=None, min_samples=3)
    params.update(overrides)
    return baseline(**params)


# --- day-of-year positions -------------------------------------------------------

def test_doy_positions_hobday_and_calendar():
    dates = np.array(["2023-02-28", "2023-03-01", "2024-02-29", "2024-03-01", "2023-12-31"],
                     dtype="datetime64[D]")
    assert doy_positions(dates, "hobday_366_interp").tolist() == [59, 61, 60, 61, 366]
    assert doy_positions(dates, "calendar_doy").tolist() == [59, 60, 60, 61, 365]


# --- percentile estimator and pooling ------------------------------------------------------

def year_valued():
    times, values = by_year("2000-01-01", "2002-12-31",
                            lambda t: float(t.astype(object).year - 1999))  # 1, 2, 3
    return point_ds(values, "2000-01-01")


def test_type7_percentile_of_known_sample():
    result = fit(year_valued(), base=three_years())
    thr = result.dataset["threshold"].values
    ref = result.dataset["reference"].values
    np.testing.assert_allclose(thr[180], 2.8)          # P90 type 7 of [1, 2, 3]
    np.testing.assert_allclose(ref[180], 2.0)
    np.testing.assert_allclose(thr[59], 2.8)           # 29 Feb from its neighbours


def test_estimator_mapping_changes_threshold():
    result = fit(year_valued(), base=three_years(percentile_estimator="hyndman_fan_1"))
    np.testing.assert_allclose(result.dataset["threshold"].values[180], 3.0)  # inverted CDF


def test_window_is_strict_to_reference_period():
    ds = point_ds(np.full(1119, 10.0), "1999-12-20")    # data beyond both ends of the period
    result = fit(ds, base=three_years(doy_window_days=11, min_samples=1))
    count = result.dataset["sample_count"].values
    assert count[0] == 6 + 11 + 11                     # 1 Jan: 27-31 Dec 1999 excluded
    assert count[365] == 11 + 11 + 6                   # 31 Dec: 1-5 Jan 2003 excluded
    assert count[179] == 33
    assert count[59] == min(count[58], count[60])      # 29 Feb


def test_feb29_is_interpolated_from_neighbours_before_smoothing():
    times, values = by_year("2000-01-01", "2002-12-31",
                            lambda t: 10.0 if str(t)[5:] == "02-28" else
                            20.0 if str(t)[5:] == "03-01" else 15.0)
    result = fit(point_ds(values, "2000-01-01"), base=three_years())
    thr = result.dataset["threshold"].values
    assert thr[58] == 10.0 and thr[60] == 20.0 and thr[59] == 15.0


# --- smoothing (strict, circular) ----------------------------------------------------------

def test_smoothing_keeps_constant_fields():
    times, values = by_year("2000-01-01", "2002-12-31", lambda t: 12.5)
    result = fit(point_ds(values, "2000-01-01"), base=three_years(smoothing_window_days=31))
    np.testing.assert_allclose(result.dataset["threshold"].values, 12.5)


def test_invalid_day_invalidates_its_smoothing_window_with_wraparound():
    times, values = by_year("2000-01-01", "2002-12-31", lambda t: 12.5)
    values[np.flatnonzero(times == np.datetime64("2001-01-01"))] = np.nan  # Jan 1: 2 samples
    result = fit(point_ds(values, "2000-01-01"), base=three_years(smoothing_window_days=31))
    reason = result.dataset["invalid_reason"].values
    invalid = np.flatnonzero(reason != 0)
    expected = sorted([*range(0, 16), *range(366 - 15, 366)])     # Jan 1 +/- 15, circular
    assert invalid.tolist() == expected
    assert set(reason[invalid]) == {8}


# --- tails, absolute thresholds, domain, filled days ---------------------------------------

def test_cold_tail_percentile_is_symmetric_to_heat_on_negated_data():
    rng = np.random.default_rng(0)
    values = rng.normal(15.0, 2.0, 1096)
    cold = fit(point_ds(values, "2000-01-01"), base=three_years(doy_window_days=11),
               definition=hobday(phenomenon="mcs", tail="lower", comparator="<",
                                 percentile=10.0, category_scheme=None))
    heat = fit(point_ds(-values, "2000-01-01"), base=three_years(doy_window_days=11))
    np.testing.assert_allclose(cold.dataset["threshold"].values,
                               -heat.dataset["threshold"].values)


def test_absolute_threshold_is_converted_to_data_units():
    definition = hobday(threshold_kind="absolute", percentile=None,
                        absolute_threshold=(25.0, "degC"))
    result = fit(point_ds(np.full(1096, 290.0), "2000-01-01", units="K"), base=three_years(),
                 definition=definition)
    np.testing.assert_allclose(result.dataset["threshold"].values, 298.15)
    np.testing.assert_allclose(result.dataset["reference"].values, 290.0)


def test_outside_domain_cells_are_invalid():
    result = fit(point_ds(np.full(1096, 10.0), "2000-01-01", domain=False), base=three_years())
    assert set(np.unique(result.dataset["invalid_reason"].values)) == {1}
    assert np.all(np.isnan(result.dataset["threshold"].values))
    result.check_invariants()


def test_filled_days_are_excluded_from_the_baseline():
    values = np.full(1096, 10.0)
    values[181] = 99.0                                  # an interpolated (filled) day
    filled = np.zeros(1096, bool)
    filled[181] = True
    result = fit(point_ds(values, "2000-01-01", filled=filled), base=three_years(min_samples=2))
    doy = doy_positions(np.array(["2000-06-30"], dtype="datetime64[D]"), "hobday_366_interp")[0]
    assert result.dataset["sample_count"].values[doy - 1] == 2
    assert result.dataset["threshold"].values[doy - 1] == 10.0


# --- not_applicable (constant threshold) ---------------------------------------------------

def test_constant_threshold_without_day_of_year_axis():
    base = three_years(doy_window_days=None, leap_day_policy="not_applicable", min_samples=10)
    result = fit(year_valued(), base=base)
    assert "dayofyear" not in result.dataset.dims
    expected = np.percentile(np.repeat([1.0, 2.0, 3.0], [366, 365, 365]), 90)
    np.testing.assert_allclose(result.dataset["threshold"].values, expected)
    expanded = result.expand(np.array(["2010-05-01", "2010-05-02"], dtype="datetime64[D]"))
    assert expanded.sizes["time"] == 2


# --- expand ----------------------------------------------------------------------------------

def test_expand_maps_dates_to_positions():
    times, values = by_year("2000-01-01", "2002-12-31",
                            lambda t: 10.0 if str(t)[5:] == "02-28" else
                            20.0 if str(t)[5:] == "03-01" else 15.0)
    result = fit(point_ds(values, "2000-01-01"), base=three_years())
    out = result.expand(np.array(["2023-02-28", "2023-03-01", "2024-02-29"],
                                 dtype="datetime64[D]"))
    assert out["threshold"].values.tolist() == [10.0, 20.0, 15.0]


# --- warnings ------------------------------------------------------------------------

def test_short_period_and_low_min_samples_warn_with_statistics():
    with pytest.warns(xhw.XHWDataQualityWarning, match="at least 30.*expected exceedances"):
        fit(year_valued(), base=three_years(doy_window_days=11))


def test_min_samples_above_possible_warns_and_invalidates():
    with pytest.warns(xhw.XHWDataQualityWarning, match="exceeds"):
        result = fit(year_valued(), base=three_years(min_samples=10))
    assert np.all(result.dataset["invalid_reason"].values == 8)


def test_thirty_complete_years_do_not_warn():
    ds = point_ds(np.full(10958, 10.0), "1991-01-01")  # 1991-01-01.. 2020-12-31
    base = baseline(reference_period=("1991-01-01", "2020-12-31"), min_samples=264)
    with warnings_as_errors():
        fit(ds, base=base)


class warnings_as_errors:
    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", xhw.XHWDataQualityWarning)

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)


# --- scope, chunking, serialization --------------------------------------------------------

@pytest.mark.parametrize("overrides", [{"computed_on": "anomaly"}])
def test_unimplemented_baselines_are_rejected(overrides):
    with pytest.raises(xhw.NotSupportedError):
        fit(year_valued(), base=three_years(**overrides))


def surface_series():
    rng = np.random.default_rng(1)
    ds = f5_surface().mask.dataset[["temperature", "domain_mask", "domain_fraction",
                                    "cell_area"]].copy(deep=True)
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2003-01-01")).astype(
        "datetime64[ns]")
    field = rng.normal(15.0, 1.0, (times.size, 7, 7))
    ds = ds.drop_vars("temperature").drop_dims("time").assign_coords(time=times)
    ds["sst"] = (("time", "y", "x"), field, {"units": "degC"})
    return ds


def test_time_chunked_input_is_rejected_with_suggestion():
    pytest.importorskip("dask")
    ds = surface_series()
    ds["sst"] = ds["sst"].chunk({"time": 400})
    with pytest.raises(xhw.ValidationError, match="Suggested: ds = ds.chunk"):
        fit(ds, base=three_years(doy_window_days=11), spec=surface_grid())


def test_spatial_chunks_are_lazy_and_equal_to_eager():
    dask = pytest.importorskip("dask")
    ds = surface_series()
    eager = fit(ds, base=three_years(doy_window_days=11), spec=surface_grid())

    def forbid(*args, **kwargs):
        raise AssertionError("computed during fit_baseline")

    lazy_ds = ds.copy()  # only the (time x space) variable is dask; statics may be read
    lazy_ds["sst"] = ds["sst"].chunk({"time": -1, "y": 3, "x": 4})
    with dask.config.set(scheduler=forbid):
        lazy = fit(lazy_ds, base=three_years(doy_window_days=11), spec=surface_grid())
    assert lazy.dataset["threshold"].chunks is not None
    for name in ("threshold", "reference", "sample_count", "invalid_reason"):
        np.testing.assert_array_equal(lazy.dataset[name].values, eager.dataset[name].values)


def test_netcdf_round_trip_and_invariants(tmp_path):
    result = fit(year_valued(), base=three_years())
    result.check_invariants()
    path = tmp_path / "baseline.nc"
    result.to_dataset().to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        again = xhw.BaselineResult.from_dataset(opened.load())
    assert (again.baseline, again.definition, again.grid) == (
        result.baseline, result.definition, result.grid)
    for name in ("threshold", "reference", "sample_count", "invalid_reason"):
        np.testing.assert_array_equal(again.dataset[name].values, result.dataset[name].values)
