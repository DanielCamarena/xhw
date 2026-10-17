# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Synthetic atmospheric series and configuration for stored climdex comparisons."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import xhw

REF = Path(__file__).parent / "reference" / "climdex"
INPUTS = REF / "inputs.csv"
START, END = "1981-01-01", "2010-12-31"
SEED = 20260924
YEAR_EDGE = set(range(1, 8)) | set(range(360, 367))
FEB29_WINDOWS = set(range(53, 68))
DIVERGENT = YEAR_EDGE | FEB29_WINDOWS

def make_inputs() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(START, END, freq="D")
    phase = 2 * np.pi * (dates.dayofyear.to_numpy() - 15) / 365.25
    noise = np.zeros((dates.size, 2))
    for i in range(1, dates.size):                    # AR(1), phi = 0.7
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, [2.0, 1.5])
    tmax = 26 + 5 * np.cos(phase) + noise[:, 0]
    tmin = 17 + 4 * np.cos(phase) + noise[:, 1]
    for series in (tmax, tmin):
        series[rng.random(dates.size) < 0.005] = np.nan
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "tmax": np.round(tmax, 2),
                         "tmin": np.round(tmin, 2)})

def read_inputs() -> xr.Dataset:
    frame = pd.read_csv(INPUTS, na_values="NA")
    times = pd.to_datetime(frame["date"]).to_numpy().astype("datetime64[ns]")
    return xr.Dataset({name: ("time", frame[name].to_numpy(float), {"units": "degC"})
                       for name in ("tmax", "tmin")} | {"domain_mask": ((), True)},
                      coords={"time": times, "lat": -12.0, "lon": -77.5})

def specs(statistic: str):
    source = xhw.SourceRef(dataset_id="synthetic:climdex-benchmark", variable="tmax",
                           accessed="2026-09-24", product_status="final")
    grid = xhw.GridSpec(support="point", topology="none", time_dim="time", spatial_dims=(),
                        coords=xhw.CoordRoles(lat="lat", lon="lon"), calendar="standard",
                        frequency="1D", time_axes="single", periodic_x=False,
                        cell_measure=xhw.CellMeasure(kind="none"), vertical=None,
                        domain="land", domain_mask="domain_mask", domain_fraction=None,
                        native=xhw.NativeGrid(product="synthetic", resolution="n/a",
                                              is_native=True))
    base = xhw.BaselineSpec(method="fixed_climatology", computed_on="temperature",
                            reference_period=(START, END), source=source, doy_window_days=15,
                            smoothing_window_days=None, percentile_estimator="hyndman_fan_8",
                            leap_day_policy="hobday_366_interp", min_samples=1,
                            fit_mode="retrospective", available_until=None)
    kind = "tx" if statistic == "max" else "tn"
    definition = xhw.DefinitionSpec(
        profile=f"perkins2013_{kind}90", phenomenon="ahw", variable=f"{kind}_daily",
        combination="single", daily_statistic=statistic, day_definition="UTC", tail="upper",
        comparator=">", threshold_kind="percentile", percentile=90.0, absolute_threshold=None,
        min_duration_days=3, max_gap_days=0, join_rule="qualify_then_join",
        missing_policy="as_non_exceedance", season_months=None, category_scheme=None)
    return grid, base, definition

def xhw_run(ds: xr.Dataset, variable: str, statistic: str):
    grid, base, definition = specs(statistic)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xhw.XHWDataQualityWarning)
        result = xhw.fit_baseline(ds, variable=variable, grid=grid, baseline=base,
                                  definition=definition)
    mask = xhw.detect(ds, variable=variable, grid=grid, definition=definition,
                      baseline_result=result)
    return result, mask

def month_day_positions() -> np.ndarray:
    """hobday_366 position of each climdex day (month-day of a 365-day year)."""
    days = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    offsets = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    return offsets[days.month.to_numpy() - 1] + days.day.to_numpy()
