# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Detect a known synthetic marine heatwave without files or optional dependencies."""

import numpy as np
import xarray as xr

import xhw


def main() -> None:
    times = np.arange(np.datetime64("1982-01-01"), np.datetime64("2015-01-01"))
    phase = (times - times.astype("datetime64[Y]")).astype(int)
    temperature = 18 + 3 * np.sin(2 * np.pi * phase / 365.25)
    temperature += np.random.default_rng(23).normal(0, 0.4, times.size)
    temperature[(times >= np.datetime64("2012-06-01"))
                & (times <= np.datetime64("2012-06-10"))] += 6
    ds = xr.Dataset({"sst": ("time", temperature, {"units": "degC"}),
                     "domain_mask": ((), True)},
                    coords={"time": times.astype("datetime64[ns]"), "lat": -12.0, "lon": -77.5})
    source = xhw.SourceRef(dataset_id="synthetic:quickstart", variable="sst",
                           accessed="2026-10-07", product_status="final")
    grid = xhw.GridSpec(
        support="point", topology="none", time_dim="time", spatial_dims=(),
        coords=xhw.CoordRoles(lat="lat", lon="lon"), calendar="standard", frequency="1D",
        time_axes="single", periodic_x=False, cell_measure=xhw.CellMeasure(kind="none"),
        vertical=None, domain="ocean", domain_mask="domain_mask", domain_fraction=None,
        native=xhw.NativeGrid(product="synthetic", resolution="point", is_native=True))
    definition = xhw.DefinitionSpec(
        profile="hobday2016", phenomenon="mhw", variable="sea_surface_temperature",
        combination="single", daily_statistic="native", day_definition="native", tail="upper",
        comparator=">", threshold_kind="percentile", percentile=90.0, absolute_threshold=None,
        min_duration_days=5, max_gap_days=2, join_rule="qualify_then_join",
        missing_policy="as_non_exceedance", season_months=None, category_scheme="hobday2018")
    baseline = xhw.BaselineSpec(
        method="fixed_climatology", computed_on="temperature",
        reference_period=("1982-01-01", "2011-12-31"), source=source, doy_window_days=11,
        smoothing_window_days=31, percentile_estimator="hyndman_fan_7",
        leap_day_policy="hobday_366_interp", min_samples=264,
        fit_mode="retrospective", available_until=None)
    result = xhw.fit_baseline(ds, variable="sst", grid=grid, baseline=baseline,
                              definition=definition)
    mask = xhw.detect(ds, variable="sst", grid=grid, definition=definition, baseline_result=result)
    mask.check_invariants()
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    catalog.check_against(mask)
    known = mask.dataset["event_membership"].sel(time=slice("2012-06-01", "2012-06-10"))
    assert bool(known.all())
    print(f"xHW {xhw.__version__}: known 10-day synthetic event detected")
    print(catalog.dataset[["start", "end", "duration_days", "intensity_max"]])


if __name__ == "__main__":
    main()
