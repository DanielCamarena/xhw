# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Offline marineHeatWaves equivalence and quantified reference-window/missing-data differences.

Independent inputs and outputs are stored with integrity hashes; the external module is not
imported.
"""

import warnings

import numpy as np
import pytest
import xarray as xr
from _reference import load_npz
from _sst_series import sst_series

import xhw

SOURCE = xhw.SourceRef(dataset_id="synthetic:benchmark", variable="sst", accessed="2026-09-23",
                       product_status="final")
GRID = xhw.GridSpec(support="point", topology="none", time_dim="time", spatial_dims=(),
                    coords=xhw.CoordRoles(lat="lat", lon="lon"), calendar="standard",
                    frequency="1D", time_axes="single", periodic_x=False,
                    cell_measure=xhw.CellMeasure(kind="none"), vertical=None, domain="ocean",
                    domain_mask="domain_mask", domain_fraction=None,
                    native=xhw.NativeGrid(product="synthetic", resolution="n/a", is_native=True))
CATEGORY = {"Moderate": 1, "Strong": 2, "Severe": 3, "Extreme": 4}


def definition(cold: bool = False) -> xhw.DefinitionSpec:
    return xhw.DefinitionSpec(
        profile="hobday2016", phenomenon="mcs" if cold else "mhw",
        variable="sea_surface_temperature", combination="single", daily_statistic="native",
        day_definition="native", tail="lower" if cold else "upper",
        comparator="<" if cold else ">", threshold_kind="percentile",
        percentile=10.0 if cold else 90.0, absolute_threshold=None, min_duration_days=5,
        max_gap_days=2, join_rule="qualify_then_join", missing_policy="as_non_exceedance",
        season_months=None, category_scheme="schlegel2021" if cold else "hobday2018")


def baseline(period=("1982-01-01", "2021-12-31")) -> xhw.BaselineSpec:
    return xhw.BaselineSpec(
        method="fixed_climatology", computed_on="temperature", reference_period=period,
        source=SOURCE, doy_window_days=11, smoothing_window_days=31,
        percentile_estimator="hyndman_fan_7", leap_day_policy="hobday_366_interp",
        min_samples=1, fit_mode="retrospective", available_until=None)


def run_xhw(times, values, *, cold=False, period=("1982-01-01", "2021-12-31"), fill=False):
    ds = xr.Dataset({"sst": ("time", values, {"units": "degC"}), "domain_mask": ((), True)},
                    coords={"time": times.astype("datetime64[ns]"), "lat": 0.0, "lon": 0.0})
    if fill:
        ds = xhw.interpolate_gaps(ds, variable="sst", grid=GRID, max_days=2,
                                  max_filled_fraction=0.05)
    d = definition(cold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xhw.XHWDataQualityWarning)
        result = xhw.fit_baseline(ds, variable="sst", grid=GRID, baseline=baseline(period),
                                  definition=d)
    mask = xhw.detect(ds, variable="sst", grid=GRID, definition=d, baseline_result=result)
    return mask, xhw.build_catalog(mask, analysis_mode="retrospective").dataset


def run_oliver(ordinals, values, *, cold=False, period=(1982, 2021)):
    """Read independent stored outputs for exactly the registered input and settings."""
    ref = load_npz("marine", "outputs.npz")
    for seed in (1, 2, 3, 4, 5):
        key = f"seed{seed}_{'cold' if cold else 'hot'}"
        if f"{key}_input" not in ref:
            continue
        registered = (1991, 2020) if seed == 4 else (1982, 2021)
        if period != registered:
            continue
        stored = ref[f"{key}_input"]
        if not np.allclose(stored, values, rtol=0, atol=1e-12, equal_nan=True):
            continue
        from datetime import date
        actual_ordinals = np.array([date.fromisoformat(str(t)).toordinal()
                                   for t in ref[f"{key}_times"]])
        np.testing.assert_array_equal(ordinals, actual_ordinals)
        prefix = f"{key}_event_"
        events = {name[len(prefix):]: ref[name] for name in ref.files if name.startswith(prefix)}
        events["n_events"] = int(events["n_events"])
        return events, {name: ref[f"{key}_clim_{name}"] for name in ("thresh", "seas")}
    pytest.fail("No independent reference matches the input and configuration")


@pytest.mark.parametrize("cold", [False, True], ids=["mhw", "mcs"])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_equivalence_on_gap_free_series_with_full_record_reference(seed, cold):
    times, ordinals, values = sst_series(seed)
    events, clim = run_oliver(ordinals, values, cold=cold)
    mask, cat = run_xhw(times, values, cold=cold)

    np.testing.assert_allclose(mask.dataset["threshold"].values, clim["thresh"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(mask.dataset["reference"].values, clim["seas"], rtol=0, atol=1e-9)

    starts = np.array(events["date_start"], dtype="datetime64[ns]")
    ends = np.array(events["date_end"], dtype="datetime64[ns]")
    np.testing.assert_array_equal(cat["start"].values, starts)
    np.testing.assert_array_equal(cat["end"].values, ends)
    assert cat.sizes["event"] == events["n_events"] > 5
    np.testing.assert_array_equal(cat["duration_days"].values, events["duration"])

    pairs = [("intensity_max", "intensity_max"), ("intensity_mean", "intensity_mean"),
             ("intensity_std", "intensity_var"),                 # Oliver's "var" is a std
             ("intensity_cumulative", "intensity_cumulative")]
    if not cold:  # Oliver re-signs only seasonal-relative intensities for cold spells
        pairs += [("intensity_max_rel_threshold", "intensity_max_relThresh"),
                  ("intensity_mean_rel_threshold", "intensity_mean_relThresh"),
                  ("intensity_cumulative_rel_threshold", "intensity_cumulative_relThresh")]
    for ours, theirs in pairs:
        np.testing.assert_allclose(cat[ours].values, events[theirs],
                                   rtol=0, atol=1e-9, err_msg=ours)
    np.testing.assert_array_equal(cat["category_max"].values,
                                  [CATEGORY[c] for c in events["category"]])
    np.testing.assert_allclose(cat["rate_onset"].values, events["rate_onset"], rtol=0, atol=1e-9)
    last = times.size - 1
    first = times[0].astype("datetime64[ns]")
    peak_idx = (cat["peak_date"].values - first) // np.timedelta64(1, "D")
    comparable = ~((ends == times[last].astype("datetime64[ns]")) & (peak_idx == last))
    np.testing.assert_allclose(cat["rate_decline"].values[comparable],
                               np.array(events["rate_decline"])[comparable], rtol=0, atol=1e-9)


def test_divergence_reference_window_only_near_year_boundary():
    """A3: Oliver's window spills 5 days outside an interior reference period."""
    times, ordinals, values = sst_series(4)
    _, clim = run_oliver(ordinals, values, period=(1991, 2020))
    mask, _ = run_xhw(times, values, period=("1991-01-01", "2020-12-31"))
    ours = mask.dataset["threshold"].values
    doy = (times - times.astype("datetime64[Y]")).astype(int) + 1
    interior = (doy >= 40) & (doy <= 325)
    np.testing.assert_allclose(ours[interior], clim["thresh"][interior], rtol=0, atol=1e-9)
    diff = np.abs(ours - clim["thresh"])[~interior]
    assert diff.max() > 1e-6                      # the divergence exists...
    assert diff.max() < 0.5                       #... and is small (degC)


def test_divergence_single_missing_day_inside_an_event():
    """A8: same event dates; Oliver averages a 0 anomaly, xHW excludes or interpolates it."""
    times, ordinals, values = sst_series(5)
    values = values.copy()
    k = int(np.flatnonzero(times == np.datetime64("2010-06-15"))[0])
    values[k - 10:k + 11] += 6.0                  # an unmistakable 21-day event
    values[k] = np.nan
    events, _ = run_oliver(ordinals, values)
    _, plain = run_xhw(times, values)
    _, filled = run_xhw(times, values, fill=True)
    target = np.datetime64("2010-06-15", "ns")

    def pick(start, end):
        return int(np.flatnonzero((start <= target) & (end >= target))[0])

    o = pick(np.array(events["date_start"], dtype="datetime64[ns]"),
             np.array(events["date_end"], dtype="datetime64[ns]"))
    for cat in (plain, filled):
        i = pick(cat["start"].values, cat["end"].values)
        assert cat["start"].values[i] == np.datetime64(events["date_start"][o], "ns")
        assert cat["end"].values[i] == np.datetime64(events["date_end"][o], "ns")
        assert cat["intensity_mean"].values[i] > events["intensity_mean"][o]
    assert int(plain["n_missing_days"].values[pick(plain["start"].values,
                                                    plain["end"].values)]) == 1
    assert int(filled["n_filled_days"].values[pick(filled["start"].values,
                                                    filled["end"].values)]) == 1
