# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""aggregate_daily and the day/statistic consistency checks."""

import numpy as np
import pytest
import xarray as xr
from _fixtures import ahw_baseline, grid, perkins2013

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")


def hourly(values, start="2000-01-01T00:00", name="mx2t", step="h"):
    first = np.datetime64(start, "m")
    times = first + np.arange(len(values)) * np.timedelta64(1, step)
    return xr.Dataset({name: ("time", np.asarray(values, dtype=float), {"units": "K"}),
                       "domain_mask": ((), True)},
                      coords={"time": times.astype("datetime64[ns]"), "lat": -12.0,
                              "lon": -77.5})


def agg(ds, **kw):
    params = dict(variable="mx2t", statistic="max", sampling="period_ending",
                  input_clock="UTC", day_definition="UTC", name="tx")
    params.update(kw)
    return xhw.aggregate_daily(ds, **params)


def day(out, date):
    return out.sel(time=np.datetime64(date, "ns"))


# --- which stamps belong to which day ----------------------------------------------------

def test_period_ending_midnight_stamp_belongs_to_the_previous_day():
    values = np.full(24 * 3, 280.0)
    values[24] = 300.0                    # stamp 2000-01-02 00:00: hour 23-24 of 1 January
    out = agg(hourly(values))
    assert day(out, "2000-01-01")["tx"].item() == 300.0
    assert day(out, "2000-01-02")["tx"].item() == 280.0


def test_instantaneous_midnight_stamp_opens_its_day():
    values = np.full(24 * 3, 280.0)
    values[24] = 300.0
    out = agg(hourly(values, name="t2m"), variable="t2m", sampling="instantaneous")
    assert day(out, "2000-01-02")["tx"].item() == 300.0
    assert day(out, "2000-01-01")["tx"].item() == 280.0


def test_offset_day_puts_0400_utc_in_the_previous_local_day():
    values = np.full(24 * 4, 280.0)
    values[24 + 4] = 300.0                # 2000-01-02 04:00 UTC = 2000-01-01 23:00 at -05:00
    out = agg(hourly(values, name="t2m"), variable="t2m", sampling="instantaneous",
              day_definition="offset:-05:00")
    assert day(out, "2000-01-01")["tx"].item() == 300.0
    assert day(out, "2000-01-02")["tx"].item() == 280.0


def test_input_clock_is_taken_into_account():
    values = np.full(24 * 3, 280.0)
    values[24 + 4] = 300.0                # stamps already on local time -05:00
    out = agg(hourly(values, name="t2m"), variable="t2m", sampling="instantaneous",
              input_clock="offset:-05:00", day_definition="offset:-05:00")
    assert day(out, "2000-01-02")["tx"].item() == 300.0


def test_edge_day_with_5_of_24_hours_is_nan():
    values = np.arange(24 * 3, dtype=float) + 270.0   # 2000-01-01 00:00.. 01-03 23:00 UTC
    out = agg(hourly(values, name="t2m"), variable="t2m", sampling="instantaneous",
              day_definition="offset:-05:00")
    first = day(out, "1999-12-31")        # local 19:00-23:00 = UTC 00:00-04:00: 5 samples
    assert int(first["tx_n_samples"]) == 5 and np.isnan(first["tx"].item())
    assert int(day(out, "2000-01-01")["tx_n_samples"]) == 24
    assert day(out, "2000-01-01")["tx"].item() == 270.0 + 28   # UTC 05:00.. next 04:00


def test_missing_hours_and_reduced_minimum():
    values = np.full(48, 280.0)
    values[3] = np.nan
    strict = agg(hourly(values))
    assert np.isnan(day(strict, "2000-01-01")["tx"].item())    # stamps 01..24 of 1 January
    relaxed = agg(hourly(values), min_samples_per_day=20)
    assert day(relaxed, "2000-01-01")["tx"].item() == 280.0
    assert relaxed["tx_n_samples"].attrs["min_samples_per_day"] == 20


def test_mean_and_min_statistics():
    values = np.concatenate([np.arange(24.0), np.arange(24.0)])
    out = agg(hourly(values, name="t2m"), variable="t2m", sampling="instantaneous",
              statistic="mean")
    assert day(out, "2000-01-01")["tx"].item() == pytest.approx(11.5)
    assert day(out, "2000-01-01")["tx"].attrs["cell_methods"] == "time: mean"
    low = agg(hourly(values, name="t2m"), variable="t2m", sampling="instantaneous",
              statistic="min")
    assert day(low, "2000-01-02")["tx"].item() == 0.0


def test_sampled_t2m_underestimates_the_true_maximum():
    """: an hourly instantaneous sample misses a peak between hours; mx2t does not."""
    minutes = np.arange(0, 3 * 24 * 60 + 1)                     # 1-minute "true" signal
    signal = 290 + 8 * np.sin(2 * np.pi * (minutes / 1440.0 - 0.3) + 0.013)
    hours = minutes[::60]
    t2m = signal[::60]
    mx2t = np.array([signal[max(0, h - 59):h + 1].max() for h in hours])
    ds = hourly(t2m, name="t2m").assign(mx2t=("time", mx2t, {"units": "K"}))
    from_t2m, from_mx2t = xr.align(agg(ds, variable="t2m", sampling="instantaneous")["tx"],
                                   agg(ds, variable="mx2t")["tx"])
    both = (from_t2m.notnull() & from_mx2t.notnull()).values
    assert both.sum() >= 2
    assert np.all(from_t2m.values[both] < from_mx2t.values[both])


# --- validation ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"day_definition": "native"}, {"input_clock": "native"}, {"statistic": "median"},
    {"sampling": "accumulated"}, {"min_samples_per_day": 25}, {"min_samples_per_day": 0},
])
def test_invalid_arguments(kw):
    with pytest.raises(xhw.ValidationError):
        agg(hourly(np.full(48, 280.0)), **kw)


def test_irregular_axis_is_rejected():
    ds = hourly(np.full(48, 280.0)).isel(time=[i for i in range(48) if i != 10])
    with pytest.raises(xhw.ValidationError, match="not regular"):
        agg(ds)


def test_units_are_never_converted_and_statics_are_kept():
    out = agg(hourly(np.full(48, 280.0)))
    assert out["tx"].attrs["units"] == "K" and "domain_mask" in out
    assert out["tx"].attrs["ancillary_variables"] == "tx_n_samples"
    assert out["tx_n_samples"].attrs["xhw_step"] == "aggregate_daily"


def test_chunked_equals_eager():
    pytest.importorskip("dask")
    rng = np.random.default_rng(0)
    ds = hourly(280 + rng.normal(0, 3, 24 * 20))
    ds["mx2t"][rng.random(24 * 20) < 0.02] = np.nan
    eager = agg(ds, day_definition="offset:-05:00")
    lazy = agg(ds.chunk({"time": 37}), day_definition="offset:-05:00")
    assert lazy["tx"].chunks is not None
    xr.testing.assert_identical(lazy.compute(), eager)


# --- consistency with the DefinitionSpec (fit_baseline / detect) --------------------------

def daily_case(day_definition="offset:-05:00", statistic="max"):
    """Six years of hourly data aggregated to TX, ready for the TX90 profile."""
    hours = 24 * (365 * 6 + 2) + 5
    rng = np.random.default_rng(3)
    ds = hourly(290 + rng.normal(0, 2, hours), start="1999-12-31T05:00")
    return agg(ds, statistic=statistic, day_definition=day_definition).isel(time=slice(1, -1))


def fit_and_detect(daily, definition):
    base = ahw_baseline(reference_period=("2000-01-01", "2005-12-31"), min_samples=10)
    g = grid(domain="land")
    result = xhw.fit_baseline(daily, variable="tx", grid=g, baseline=base,
                              definition=definition)
    return result, xhw.detect(daily, variable="tx", grid=g, definition=definition,
                              baseline_result=result)


def test_matching_step_is_recorded_in_provenance():
    result, mask = fit_and_detect(daily_case(), perkins2013(variable="tx"))
    for product in (result, mask):
        step = product.provenance.steps[0]
        params = dict(step.params)
        assert step.name == "aggregate_daily"
        assert (params["variable"], params["source_variable"], params["sampling"]) == (
            "tx", "mx2t", "period_ending")
        assert params["day_definition"] == "offset:-05:00" and params["samples_per_day"] == 24
    mask.check_invariants()


@pytest.mark.parametrize("definition", [
    perkins2013(variable="tx", day_definition="UTC"),
    perkins2013("min", variable="tx"),
])
def test_mismatching_step_is_an_error(definition):
    with pytest.raises(xhw.ValidationError, match="was aggregated with"):
        fit_and_detect(daily_case(), definition)


def test_missing_step_is_accepted_and_recorded_as_unverified():
    daily = daily_case().drop_vars("tx_n_samples")
    result, mask = fit_and_detect(daily, perkins2013(variable="tx"))
    step = mask.provenance.steps[0]
    assert step.name == "daily_input_declared" and dict(step.params)["verified"] is False


def test_step_survives_unit_conversion_and_netcdf(tmp_path):
    daily = daily_case()
    daily["tx"] = (daily["tx"] - 273.15).assign_attrs(units="degC")   # step not on tx
    path = tmp_path / "daily.nc"
    daily.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        again = opened.load()
    result, mask = fit_and_detect(again, perkins2013(variable="tx"))
    assert mask.provenance.steps[0].name == "aggregate_daily"
    mask.provenance.to_json()                  # numpy attribute values stored as plain JSON
