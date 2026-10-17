# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""GridSpec.check: time axis, calendar, coordinates, periodicity, domain and measures."""

import numpy as np
import pytest
from _fixtures import f4_line, f5_surface, f6_volume, grid, surface_grid

import xhw


def surface_ds():
    return f5_surface().mask.dataset.copy(deep=True)


def shifted_time(ds, times):
    return ds.assign_coords(time=np.asarray(times, dtype="datetime64[ns]"))


def test_fixture_datasets_pass():
    for fixture in (f4_line(), f5_surface(), f6_volume()):
        fixture.mask.grid.check(fixture.mask.dataset)


def test_duplicate_day_rejected():
    ds = surface_ds()
    times = ds.time.values.copy()
    times[3] = times[2]
    with pytest.raises(xhw.ValidationError, match="duplicate"):
        surface_grid().check(shifted_time(ds, times))


def test_missing_day_rejected():
    ds = surface_ds()
    times = ds.time.values + np.array([0, 0, 0, 1, 1, 1]) * np.timedelta64(1, "D")
    with pytest.raises(xhw.ValidationError, match="regular daily"):
        surface_grid().check(shifted_time(ds, times))


def test_decreasing_time_rejected():
    ds = surface_ds()
    with pytest.raises(xhw.ValidationError, match="increasing"):
        surface_grid().check(shifted_time(ds, ds.time.values[::-1]))


def test_non_daily_samples_are_not_days():
    ds = surface_ds()
    times = ds.time.values[0] + np.arange(6) * np.timedelta64(6, "h")
    with pytest.raises(xhw.ValidationError, match="regular daily"):
        surface_grid().check(shifted_time(ds, times))


def test_unsupported_encoded_calendar_rejected():
    ds = surface_ds()
    ds.time.encoding["calendar"] = "noleap"
    with pytest.raises(xhw.CalendarNotSupportedError):
        surface_grid().check(ds)


def test_non_datetime_time_rejected():
    ds = surface_ds().assign_coords(time=np.arange(6))
    with pytest.raises(xhw.CalendarNotSupportedError):
        surface_grid().check(ds)


def test_dates_before_gregorian_reform_rejected():
    ds = surface_ds()
    times = np.arange(np.datetime64("1500-01-01"), np.datetime64("1500-01-07")).astype(
        "datetime64[s]")
    with pytest.raises(xhw.ValidationError, match="1582"):
        surface_grid().check(ds.assign_coords(time=times))


def test_latitude_out_of_range_rejected():
    ds = surface_ds().assign_coords(lat=("y", np.linspace(80.0, 95.0, 7)))
    with pytest.raises(xhw.ValidationError, match="latitude"):
        surface_grid().check(ds)


def test_periodic_x_requires_global_longitude():
    periodic = xhw.GridSpec.from_dict({**surface_grid().to_dict(),
                                       "fields": {**surface_grid().to_dict()["fields"],
                                                  "periodic_x": True}})
    ds = surface_ds()
    with pytest.raises(xhw.ValidationError, match="360"):
        periodic.check(ds)                                   # regional longitude
    global_lon = ds.assign_coords(lon=("x", np.arange(7) * (360.0 / 7)))
    periodic.check(global_lon)


def test_domain_mask_must_be_static_boolean():
    ds = surface_ds()
    ds["domain_mask"] = ds["domain_mask"].astype(float)
    with pytest.raises(xhw.ValidationError, match="boolean"):
        surface_grid().check(ds)
    ds = surface_ds()
    ds["domain_mask"] = ds["valid"]
    with pytest.raises(xhw.ValidationError, match="static"):
        surface_grid().check(ds)


def test_domain_fraction_in_unit_interval():
    ds = surface_ds()
    ds["domain_fraction"] = ds["domain_fraction"] * 2.0
    with pytest.raises(xhw.ValidationError, match=r"\[0, 1\]"):
        surface_grid().check(ds)


def test_cell_measure_must_be_positive():
    ds = surface_ds()
    ds["cell_area"] = -ds["cell_area"]
    with pytest.raises(xhw.ValidationError, match="> 0"):
        surface_grid().check(ds)


def test_along_coordinate_must_be_monotonic():
    fixture = f4_line()
    ds = fixture.mask.dataset.assign_coords(along=("along", [0.0, 5000.0, 2000.0]))
    with pytest.raises(xhw.ValidationError, match="monotonic"):
        fixture.mask.grid.check(ds)


def test_volume_thickness_must_be_positive():
    fixture = f6_volume()
    ds = fixture.mask.dataset.copy(deep=True)
    ds["dz"] = ("z", [10.0, 0.0], {"units": "m"})
    with pytest.raises(xhw.ValidationError, match="thickness"):
        fixture.mask.grid.check(ds)


def test_missing_dimension_rejected():
    with pytest.raises(xhw.ValidationError, match="dimension"):
        grid(spatial_dims=("station",)).check(f5_surface().mask.dataset)
