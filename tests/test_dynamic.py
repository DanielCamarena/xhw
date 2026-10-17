# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Dynamic baselines,: `detrended` and `shifting`.

reference = trend + seasonal, threshold = reference + Q_R(d). Expected values come from exact
synthetic series (no noise), from the temporal detection machinery applied to the residuals,
or from closed forms (moving windows over a linear trend).
"""

import numpy as np
import pytest
import xarray as xr
from _fixtures import baseline, grid, hobday, provenance, txtn90

import xhw
from xhw._baseline import doy_positions

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
REF = ("1991-01-01", "2010-12-31")


def daily_times(start="1990-01-01", end="2015-12-31"):
    return np.arange(np.datetime64(start), np.datetime64(end) + np.timedelta64(1, "D")).astype(
        "datetime64[ns]")


def phase(times):
    return (doy_positions(times, "hobday_366_interp") - 0.5) / 366


def years_since(times, origin="2000-01-01"):
    return (times - np.datetime64(origin)) / np.timedelta64(1, "D") / 365.25


def point(values, times, filled=None):
    ds = xr.Dataset({"sst": ("time", np.asarray(values, float), {"units": "degC"}),
                     "domain_mask": ((), True)},
                    coords={"time": times, "lat": -12.0, "lon": -77.5})
    if filled is not None:
        ds["filled"] = ("time", np.asarray(filled, bool))
    return ds


def exact_series(times):
    """Linear trend + two harmonics on the calendar phase, no noise."""
    ph = phase(times)
    return (15 + 0.4 * years_since(times) + 3 * np.cos(2 * np.pi * ph)
            + 1.5 * np.sin(4 * np.pi * ph))


def dyn(method="detrended", params=(), **overrides):
    values = dict(method=method, reference_period=REF, doy_window_days=5,
                  smoothing_window_days=None, min_samples=1, params=params)
    values.update(overrides)
    return baseline(**values)


def fit(ds, spec, definition=None, g=None):
    return xhw.fit_baseline(ds, variable="sst", grid=g or grid(), baseline=spec,
                            definition=definition or hobday())


def noisy(times, seed=0, sd=0.5):
    return exact_series(times) + np.random.default_rng(seed).normal(0, sd, times.size)


# --- spec: parameters are validated when fitting -----------------------

def test_specs_valid_before_m1c_stay_valid():
    spec = baseline(method="shifting", computed_on="anomaly", smoothing_window_days=None,
                    leap_day_policy="calendar_doy", params=(("window_years", 15),))
    assert spec.params == (("window_years", 15),)
    assert xhw.BaselineSpec.from_json(spec.to_json()) == spec


@pytest.mark.parametrize("overrides,error", [
    ({"params": (("trend_dgree", 1),)}, xhw.ValidationError),           # unknown name
    ({"params": (("trend_degree", 7),)}, xhw.ValidationError),          # out of range
    ({"params": (("lag_days", 1),)}, xhw.ValidationError),              # causal-only parameter
    ({"computed_on": "anomaly"}, xhw.NotSupportedError),
    ({"fit_mode": "causal", "available_until": "2010-12-31",
      "params": (("causal_scheme", "expanding"), ("refit_every_days", 60))},
     xhw.ValidationError),                                                # 60 > 30 days horizon
])
def test_invalid_dynamic_configurations_fail_at_fit(overrides, error):
    times = daily_times()
    with pytest.raises(error):
        fit(point(exact_series(times), times), dyn(**overrides))


def test_absolute_threshold_is_rejected():
    times = daily_times()
    definition = hobday(threshold_kind="absolute", percentile=None,
                        absolute_threshold=(25.0, "degC"))
    with pytest.raises(xhw.ValidationError, match="percentile"):
        fit(point(exact_series(times), times), dyn(), definition=definition)


# --- level 1 ------------------------------------------------------------------------------

def test_exact_series_is_reproduced_without_residual():
    times = daily_times()
    x = exact_series(times)
    ds = fit(point(x, times), dyn(params=(("n_harmonics", 2),))).dataset
    assert ds["threshold"].dims == ("time",) and ds["residual_threshold"].dims == ("dayofyear",)
    np.testing.assert_allclose(ds["reference"].values, x, atol=1e-8)
    np.testing.assert_allclose(ds["trend"].values + ds["seasonal"].values, x, atol=1e-8)
    np.testing.assert_allclose(ds["residual_threshold"].values, 0.0, atol=1e-8)
    np.testing.assert_allclose(ds["threshold"].values, x, atol=1e-8)


@pytest.mark.parametrize("policy", ["hobday_366_interp", "calendar_doy"])
@pytest.mark.parametrize("method", ["detrended", "shifting"])
def test_residual_percentile_is_the_m1a_threshold_of_the_residuals(policy, method):
    times = daily_times()
    params = (("window_years", 5), ("min_years", 3)) if method == "shifting" else ()
    spec = dyn(method, params, leap_day_policy=policy, smoothing_window_days=31)
    result = fit(point(noisy(times), times), spec).dataset
    residual = point(noisy(times) - result["reference"].values, times)
    fixed = fit(residual, dyn("fixed_climatology", leap_day_policy=policy,
                              smoothing_window_days=31)).dataset
    np.testing.assert_allclose(result["residual_threshold"].values, fixed["threshold"].values,
                               atol=1e-10)
    np.testing.assert_array_equal(result["sample_count"].values, fixed["sample_count"].values)


@pytest.mark.parametrize("method,params", [
    ("detrended", ()), ("detrended", (("seasonal", "climatology"),)),
    ("shifting", (("window_years", 5), ("min_years", 3)))])
def test_calendar_phase_does_not_drift_and_ignores_the_leap_policy(method, params):
    times = daily_times("1970-01-01", "2010-12-31")       # 41 years, 10 leap years
    ph = phase(times)
    x = 20 + 4 * np.cos(2 * np.pi * (ph - phase(np.array(["2001-07-15"],
                                                          dtype="datetime64[ns]"))[0]))
    results = {policy: fit(point(x, times), dyn(method, params, leap_day_policy=policy,
                                                reference_period=("1971-01-01", "2000-12-31")))
               for policy in ("hobday_366_interp", "calendar_doy")}
    np.testing.assert_array_equal(results["hobday_366_interp"].dataset["seasonal"].values,
                                  results["calendar_doy"].dataset["seasonal"].values)
    seasonal = results["calendar_doy"].dataset["seasonal"].to_series().dropna()
    peaks = seasonal.groupby(seasonal.index.year).idxmax()
    assert set(peaks.dt.strftime("%m-%d")) == {"07-15"}


def test_filled_days_are_ignored_by_trend_seasonal_and_percentile():
    times = daily_times()
    x = noisy(times)
    filled = np.zeros(times.size, bool)
    filled[[400, 401, 3000]] = True
    with_filled = x.copy()
    with_filled[filled] = 1e3
    as_nan = x.copy()
    as_nan[filled] = np.nan
    a = fit(point(with_filled, times, filled), dyn()).dataset
    b = fit(point(as_nan, times), dyn()).dataset
    for name in ("trend", "seasonal", "residual_threshold", "sample_count"):
        np.testing.assert_allclose(a[name].values, b[name].values, err_msg=name)


def test_shifting_windows_over_a_linear_trend():
    """x = b * years: the centred window has no lag; the 5 previous years lag 3 years."""
    times = daily_times("1980-01-01", "2010-12-31")
    slope = 0.1
    x = slope * years_since(times)
    retro = fit(point(x, times), dyn("shifting", (("window_years", 5), ("min_years", 5)),
                                     doy_window_days=1)).dataset
    inner = (times >= np.datetime64("1983-01-01")) & (times < np.datetime64("2008-01-01"))
    np.testing.assert_allclose((x - retro["reference"].values)[inner], 0.0, atol=2e-3)
    causal = fit(point(x, times), dyn("shifting", (("window_years", 5), ("min_years", 5)),
                                      doy_window_days=1, fit_mode="causal",
                                      available_until="2010-12-31")).dataset
    defined = causal["reference"].notnull().values
    assert not defined[times < np.datetime64("1985-01-01")].any()       # no 5 previous years
    np.testing.assert_allclose((x - causal["reference"].values)[defined], 3 * slope, atol=2e-3)


def test_cutoff_extrapolates_a_bounded_horizon():
    times = daily_times()
    spec = dyn(fit_mode="causal", available_until="2010-12-31",
               params=(("max_extrapolation_days", 10),))
    ds = fit(point(noisy(times), times), spec).dataset
    last = np.datetime64("2010-12-31") + np.timedelta64(10, "D")
    covered = times <= last
    assert (ds["invalid_reason"].values[covered] == 0).all()
    assert (ds["invalid_reason"].values[~covered] == 4).all()
    assert np.isnan(ds["threshold"].values[~covered]).all()


def test_cutoff_and_expanding_never_use_later_data():
    times = daily_times()
    x = noisy(times)
    later = x.copy()
    later[times > np.datetime64("2010-12-31")] += 5.0
    cutoff = dyn(fit_mode="causal", available_until="2010-12-31")
    a, b = (fit(point(v, times), cutoff).dataset for v in (x, later))
    for name in ("trend", "seasonal", "residual_threshold"):
        np.testing.assert_array_equal(a[name].values, b[name].values, err_msg=name)
    expanding = dyn(fit_mode="causal", available_until="2010-12-31",
                    params=(("causal_scheme", "expanding"), ("refit_every_days", 20)))
    inside = x.copy()
    inside[times >= np.datetime64("2004-06-01")] -= 3.0
    a, b = (fit(point(v, times), expanding).dataset for v in (x, inside))
    before = times <= np.datetime64("2004-06-01")
    for name in ("trend", "seasonal"):
        np.testing.assert_array_equal(a[name].values[before], b[name].values[before])
        assert not np.allclose(a[name].values[~before], b[name].values[~before],
                               equal_nan=True)
    history = times < np.datetime64("1993-01-01")                      # min_fit_years = 3
    assert (a["invalid_reason"].values[history] == 4).all()


def test_in_base_correction_applies_to_the_residual_percentile():
    times = daily_times()
    ds = point(noisy(times), times)
    spec = dyn(in_base_correction="leave_one_year_out", min_samples=5)
    result = fit(ds, spec)
    data = result.dataset
    assert data["residual_threshold_by_year"].dims == ("reference_year", "dayofyear")
    residual = noisy(times) - fit(ds, dyn(min_samples=5)).dataset["reference"].values
    for year in (1991, 2005, 2010):
        without = point(np.where(times.astype("datetime64[Y]").astype(int) + 1970 == year,
                                 np.nan, residual), times)
        plain = fit(without, dyn("fixed_climatology", min_samples=5)).dataset
        np.testing.assert_allclose(data["residual_threshold_by_year"].sel(
            reference_year=year).values, plain["threshold"].values, atol=1e-10)
    expanded = result.expand(times)
    in_2005 = (times >= np.datetime64("2005-01-01")) & (times < np.datetime64("2006-01-01"))
    pos = doy_positions(times[in_2005], "hobday_366_interp") - 1
    q = data["residual_threshold_by_year"].sel(reference_year=2005).values[pos]
    np.testing.assert_allclose(expanded["threshold"].values[in_2005],
                               data["reference"].values[in_2005] + q, atol=1e-10)
    outside = times >= np.datetime64("2011-01-01")
    np.testing.assert_array_equal(expanded["threshold"].values[outside],
                                  data["threshold"].values[outside])
    result.check_invariants()


# --- detect, containers, laziness ---------------------------------------------------------

def test_detect_uses_the_dated_threshold_and_flags_uncovered_dates():
    times = daily_times()
    ds = point(noisy(times), times)
    result = fit(ds, dyn(fit_mode="causal", available_until="2010-12-31"))
    mask = xhw.detect(ds, variable="sst", grid=grid(), definition=hobday(),
                      baseline_result=result).dataset
    np.testing.assert_array_equal(mask["threshold"].values, result.dataset["threshold"].values)
    valid = mask["valid"].values
    np.testing.assert_array_equal(mask["exceedance"].values[valid],
                                  (ds["sst"].values > mask["threshold"].values)[valid])
    assert (mask["invalid_reason"].values[~valid] == 4).all()
    longer = point(np.r_[noisy(times), 20.0], np.r_[times, times[-1] + np.timedelta64(1, "D")])
    tail = xhw.detect(longer, variable="sst", grid=grid(), definition=hobday(),
                      baseline_result=result).dataset
    assert tail["invalid_reason"].values[-1] == 4


def test_and_daily_has_one_decomposition_per_component():
    times = daily_times("1995-01-01", "2012-12-31")
    tx = noisy(times, 1) + 10
    ds = xr.Dataset({"tx": ("time", tx, {"units": "degC"}),
                     "tn": ("time", noisy(times, 2), {"units": "degC"}),
                     "domain_mask": ((), True)},
                    coords={"time": times, "lat": -12.0, "lon": -77.5})
    definition = txtn90()
    spec = dyn(reference_period=("1996-01-01", "2010-12-31"))
    g = grid(domain="land")
    result = xhw.fit_baseline(ds, variable=("tx", "tn"), grid=g, baseline=spec,
                              definition=definition)
    data = result.dataset
    assert data["reference"].dims == ("component", "time")
    assert data["residual_threshold"].dims == ("component", "dayofyear")
    assert float((data["reference"].sel(component="tx_daily")
                  - data["reference"].sel(component="tn_daily")).mean()) == pytest.approx(
        10, abs=0.1)
    mask = xhw.detect(ds, variable=("tx", "tn"), grid=g, definition=definition,
                      baseline_result=result)
    mask.check_invariants()
    result.check_invariants()


def test_netcdf_round_trip_and_structure(tmp_path):
    times = daily_times()
    result = fit(point(noisy(times), times), dyn(in_base_correction="mean_threshold"))
    result.check_invariants()
    path = tmp_path / "dynamic.nc"
    result.to_dataset().to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        again = xhw.BaselineResult.from_dataset(opened.load())
    assert again.baseline == result.baseline
    for name in result.dataset.data_vars:
        np.testing.assert_array_equal(again.dataset[name].values, result.dataset[name].values)
    again.check_invariants()
    with pytest.raises(xhw.ValidationError, match="dynamic"):
        xhw.BaselineResult.from_dataset(result.dataset, grid=grid(),
                                        baseline=dyn("fixed_climatology"),
                                        definition=hobday(), provenance=provenance("p"))


def test_spatial_chunks_are_lazy_and_equal_to_eager():
    dask = pytest.importorskip("dask")
    times = daily_times("1995-01-01", "2012-12-31")
    rng = np.random.default_rng(7)
    cols = 5
    values = exact_series(times)[:, None] + rng.normal(0, 0.5, (times.size, cols))
    ds = xr.Dataset({"sst": (("time", "along"), values, {"units": "degC"}),
                     "domain_mask": ("along", np.arange(cols) != 1),
                     "length": ("along", np.ones(cols), {"units": "m"})},
                    coords={"time": times, "along": ("along", np.arange(cols) * 1.0,
                                                     {"units": "m"})})
    g = grid(support="line", topology="transect", spatial_dims=("along",),
             coords=xhw.CoordRoles(along="along"),
             cell_measure=xhw.CellMeasure(kind="length", var="length", units="m"))
    spec = dyn(reference_period=("1996-01-01", "2010-12-31"),
               in_base_correction="leave_one_year_out")
    eager = xhw.fit_baseline(ds, variable="sst", grid=g, baseline=spec, definition=hobday())

    def forbid(*args, **kwargs):
        raise AssertionError("computed during fit_baseline")

    lazy_ds = ds.copy()
    lazy_ds["sst"] = ds["sst"].chunk({"time": -1, "along": 2})
    with dask.config.set(scheduler=forbid):
        lazy = xhw.fit_baseline(lazy_ds, variable="sst", grid=g, baseline=spec,
                                definition=hobday())
    assert lazy.dataset["threshold"].chunks is not None
    for name, var in eager.dataset.data_vars.items():
        if var.dtype.kind == "f":  # least squares on different column batches: rounding only
            np.testing.assert_allclose(lazy.dataset[name].values, var.values, rtol=1e-12,
                                       atol=1e-12, err_msg=name)
        else:
            np.testing.assert_array_equal(lazy.dataset[name].values, var.values, err_msg=name)
    eager.check_invariants()


# --- level 2: a warming climate -----------------------------------------------------------

def test_detrended_baseline_removes_the_spurious_trend_of_a_warming_climate():
    """Linear warming of 0.5 degC/decade: with the fixed baseline the exceedance rate grows; with
    the detrended baseline it stays near the nominal 10 % in every period.
    """
    times = daily_times("1985-01-01", "2014-12-31")
    rng = np.random.default_rng(11)
    cols = 30
    noise = np.zeros((times.size, cols))
    for i in range(1, times.size):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(0, 0.7, cols)
    warming = 0.05 * years_since(times)[:, None]
    values = exact_series(times)[:, None] - 0.4 * years_since(times)[:, None] + warming + noise
    ds = xr.Dataset({"sst": (("time", "along"), values, {"units": "degC"}),
                     "domain_mask": ("along", np.ones(cols, bool)),
                     "length": ("along", np.ones(cols), {"units": "m"})},
                    coords={"time": times, "along": ("along", np.arange(cols) * 1.0,
                                                     {"units": "m"})})
    g = grid(support="line", topology="transect", spatial_dims=("along",),
             coords=xhw.CoordRoles(along="along"),
             cell_measure=xhw.CellMeasure(kind="length", var="length", units="m"))
    definition = hobday(min_duration_days=1, max_gap_days=0)
    rates = {}
    for method in ("fixed_climatology", "detrended"):
        result = xhw.fit_baseline(ds, variable="sst", grid=g, definition=definition,
                                  baseline=dyn(method, reference_period=("1985-01-01",
                                                                        "2014-12-31")))
        mask = xhw.detect(ds, variable="sst", grid=g, definition=definition,
                          baseline_result=result).dataset
        rates[method] = mask["exceedance"].groupby("time.year").mean().mean("along").values
    early, late = slice(0, 5), slice(-5, None)
    assert rates["fixed_climatology"][late].mean() - rates["fixed_climatology"][early].mean() \
        > 0.10
    assert abs(rates["detrended"][late].mean() - rates["detrended"][early].mean()) < 0.03
    assert rates["detrended"].mean() == pytest.approx(0.10, abs=0.01)


def test_in_base_rates_accepts_dynamic_baselines():
    times = daily_times("1995-01-01", "2006-12-31")
    ds = point(noisy(times), times)
    definition = hobday(min_duration_days=1, max_gap_days=0)
    spec = dyn(reference_period=("1996-01-01", "2000-12-31"))
    result = fit(ds, spec, definition=definition)
    rates = xhw.in_base_rates(ds, variable="sst", grid=grid(), baseline_result=result)
    assert rates["year"].values.tolist() == list(range(1995, 2007))
    assert np.isfinite(rates["exceedance_frequency"].values).all()


@pytest.mark.parametrize("method,mode,expected", [
    ("detrended", "retrospective", 0.95), ("shifting", "retrospective", 0.95),
    ("shifting", "causal", 0.999)])
def test_defaults_keep_most_of_a_one_year_event(method, mode, expected):
    """Rigidity criterion of the defaults: a +1.5 degC anomaly lasting one year stays in the
    residual (>= 90 %; measured 0.97 / 0.97 / 1.0).
    """
    times = daily_times("1980-01-01", "2019-12-31")
    base = exact_series(times)
    event = (times >= np.datetime64("2000-01-01")) & (times < np.datetime64("2001-01-01"))
    spec = dyn(method, fit_mode=mode, reference_period=("1981-01-01", "2010-12-31"),
               available_until="2019-12-31" if mode == "causal" else None)
    residuals = [x - fit(point(x, times), spec).dataset["reference"].values
                 for x in (base + 1.5 * event, base)]
    kept = np.nanmean((residuals[0] - residuals[1])[event]) / 1.5
    assert kept >= expected


# ---: stl and state_space (statsmodels engines) -----------------------------------

needs_statsmodels = pytest.mark.skipif(
    __import__("importlib").util.find_spec("statsmodels") is None,
    reason="statsmodels (extra 'dynamic') not installed")


@needs_statsmodels
@pytest.mark.parametrize("method", ["stl", "state_space"])
def test_engines_keep_the_calendar_phase(method):
    times = daily_times("1980-01-01", "2010-12-31")
    ph = phase(times)
    x = 20 + 4 * np.cos(2 * np.pi * (ph - phase(np.array(["2001-07-15"],
                                                          dtype="datetime64[ns]"))[0]))
    results = {policy: fit(point(x, times), dyn(method, leap_day_policy=policy,
                                                reference_period=("1981-01-01", "2000-12-31")))
               for policy in ("hobday_366_interp", "calendar_doy")}
    np.testing.assert_array_equal(results["hobday_366_interp"].dataset["seasonal"].values,
                                  results["calendar_doy"].dataset["seasonal"].values)
    seasonal = results["calendar_doy"].dataset["seasonal"].to_series().dropna()
    peaks = seasonal.groupby(seasonal.index.year).idxmax()
    assert set(peaks.dt.strftime("%m-%d")) == {"07-15"}
    results["calendar_doy"].check_invariants()


@needs_statsmodels
def test_stl_needs_a_complete_series_and_decomposes_the_filled_one():
    times = daily_times("1995-01-01", "2012-12-31")
    x = noisy(times)
    spec = dyn("stl", reference_period=("1996-01-01", "2010-12-31"))
    gap = x.copy()
    gap[2000] = np.nan
    undefined = fit(point(gap, times), spec).dataset
    assert ((undefined["invalid_reason"].values & 4) != 0).all()   # baseline_undefined
    filled = np.zeros(times.size, bool)
    filled[2000] = True
    a = fit(point(np.where(filled, 25.0, x), times, filled), spec).dataset
    b = fit(point(np.where(filled, 15.0, x), times, filled), spec).dataset
    assert not np.allclose(a["trend"].values, b["trend"].values)       # the fill is used
    plain = fit(point(x, times), spec).dataset
    assert (a["sample_count"].values <= plain["sample_count"].values).all()
    assert a["sample_count"].values.sum() < plain["sample_count"].values.sum()  # not counted


@needs_statsmodels
def test_state_space_tolerates_gaps():
    times = daily_times("1995-01-01", "2012-12-31")
    x = noisy(times)
    x[[100, 2000, 2001, 5000]] = np.nan
    ds = fit(point(x, times), dyn("state_space",
                                  reference_period=("1996-01-01", "2010-12-31"))).dataset
    assert np.isfinite(ds["reference"].values).all()


@needs_statsmodels
@pytest.mark.parametrize("method", ["stl", "state_space"])
def test_cutoff_engines_never_use_later_data(method):
    times = daily_times("1995-01-01", "2012-12-31")
    x = noisy(times)
    later = x.copy()
    later[times > np.datetime64("2010-12-31")] += 5.0
    spec = dyn(method, fit_mode="causal", available_until="2010-12-31",
               reference_period=("1996-01-01", "2010-12-31"),
               params=(("max_extrapolation_days", 20),))
    a, b = (fit(point(v, times), spec).dataset for v in (x, later))
    for name in ("trend", "seasonal", "residual_threshold"):
        np.testing.assert_array_equal(a[name].values, b[name].values, err_msg=name)
    covered = times <= np.datetime64("2010-12-31") + np.timedelta64(20, "D")
    assert (a["invalid_reason"].values[~covered] == 4).all()
    assert (a["invalid_reason"].values[covered] == 0).all()


@needs_statsmodels
def test_state_space_expanding_is_causal_with_lag():
    times = daily_times("1995-01-01", "2012-12-31")
    x = noisy(times)
    spec = dyn("state_space", fit_mode="causal", available_until="2010-12-31",
               reference_period=("1996-01-01", "2010-12-31"),
               params=(("causal_scheme", "expanding"), ("lag_days", 3),
                       ("refit_every_days", 1)))
    changed = x.copy()
    start = np.datetime64("2004-06-01")
    changed[times >= start] += 4.0
    a, b = (fit(point(v, times), spec).dataset for v in (x, changed))
    same = times <= start + np.timedelta64(3, "D")
    for name in ("trend", "seasonal"):
        np.testing.assert_array_equal(a[name].values[same], b[name].values[same])
    assert not np.allclose(a["trend"].values[~same], b["trend"].values[~same],
                           equal_nan=True)


@needs_statsmodels
@pytest.mark.parametrize("method,mode,expected", [
    ("stl", "retrospective", 0.9), ("state_space", "retrospective", 0.95),
    ("state_space", "causal", 0.9)])
def test_engine_defaults_keep_most_of_a_one_year_event(method, mode, expected):
    """Measured stl 0.914; state_space 0.972 (smoother), 0.906 (one-pass filter)."""
    times = daily_times("1980-01-01", "2019-12-31")
    base = exact_series(times)
    event = (times >= np.datetime64("2000-01-01")) & (times < np.datetime64("2001-01-01"))
    params = ((("causal_scheme", "expanding"), ("refit_every_days", 1))
              if mode == "causal" else ())
    spec = dyn(method, fit_mode=mode, reference_period=("1981-01-01", "2010-12-31"),
               available_until="2019-12-31" if mode == "causal" else None, params=params)
    residuals = [x - fit(point(x, times), spec).dataset["reference"].values
                 for x in (base + 1.5 * event, base)]
    kept = np.nanmean((residuals[0] - residuals[1])[event]) / 1.5
    assert kept >= expected


@needs_statsmodels
@pytest.mark.parametrize("seasonal_window", ["periodic", 7])
def test_vectorized_stl_equals_statsmodels(seasonal_window):
    """The vectorised engine (all cells at once) reproduces statsmodels cell by cell."""
    times = daily_times("1996-01-01", "2010-12-31")
    rng = np.random.default_rng(3)
    cols = 3
    values = exact_series(times)[:, None] + rng.normal(0, 0.5, (times.size, cols))
    ds = xr.Dataset({"sst": (("time", "along"), values, {"units": "degC"}),
                     "domain_mask": ("along", np.ones(cols, bool)),
                     "length": ("along", np.ones(cols), {"units": "m"})},
                    coords={"time": times, "along": ("along", np.arange(cols) * 1.0,
                                                     {"units": "m"})})
    g = grid(support="line", topology="transect", spatial_dims=("along",),
             coords=xhw.CoordRoles(along="along"),
             cell_measure=xhw.CellMeasure(kind="length", var="length", units="m"))
    out = {}
    for engine in ("vectorized", "statsmodels"):
        spec = dyn("stl", reference_period=("1997-01-01", "2009-12-31"),
                   params=(("trend_window_days", 1827), ("seasonal_window", seasonal_window),
                           ("engine", engine)))
        out[engine] = xhw.fit_baseline(ds, variable="sst", grid=g, baseline=spec,
                                       definition=hobday()).dataset
    for name in ("trend", "seasonal", "threshold"):
        np.testing.assert_allclose(out["vectorized"][name].values,
                                   out["statsmodels"][name].values, rtol=0, atol=1e-10,
                                   err_msg=name)


def test_robust_stl_needs_the_statsmodels_engine():
    times = daily_times()
    with pytest.raises(xhw.ValidationError, match="robust"):
        fit(point(noisy(times), times), dyn("stl", params=(("robust", 1),)))


# ---: gam (penalised B-splines) --------------------------------------------------

needs_pygam = pytest.mark.skipif(__import__("importlib").util.find_spec("pygam") is None,
                                 reason="pygam (extra 'dynamic') not installed")


def test_gam_reproduces_a_linear_trend_and_its_own_spline_cycle():
    """Without penalty on the time term and with the cycle inside the spline space, the GAM
    reproduces the series (linear trend: null space of the second difference).
    """
    from xhw._gam import cyclic_basis

    times = daily_times("1995-01-01", "2012-12-31")
    ph = phase(times)
    cycle = cyclic_basis(ph, 20) @ np.sin(np.arange(20) * 2 * np.pi / 20)
    x = 12 + 0.3 * years_since(times) + 2 * cycle
    spec = dyn("gam", reference_period=("1996-01-01", "2010-12-31"),
               params=(("trend_lam", 0.0), ("seasonal_lam", 0.0)))
    ds = fit(point(x, times), spec).dataset
    np.testing.assert_allclose(ds["reference"].values, x, atol=1e-6)
    np.testing.assert_allclose(ds["residual_threshold"].values, 0.0, atol=1e-6)


@needs_pygam
@pytest.mark.parametrize("seasonal_lam", [0.0, 0.6])
def test_vectorized_gam_against_pygam(seasonal_lam):
    """Same basis and solver as pygam's LinearGAM: equal when the seasonal term is not penalised;
    with its penalty, pygam's periodic penalty (period n - 2) differs from the cyclic second
    difference used here by a few thousandths of a degree.
    """
    times = daily_times("1995-01-01", "2012-12-31")
    x = noisy(times, 4)
    x[[50, 3000, 3001]] = np.nan
    out = {}
    for engine in ("vectorized", "pygam"):
        spec = dyn("gam", reference_period=("1996-01-01", "2010-12-31"),
                   params=(("seasonal_lam", seasonal_lam), ("engine", engine)))
        out[engine] = fit(point(x, times), spec).dataset
    gap = np.nanmax(np.abs(out["vectorized"]["reference"].values
                           - out["pygam"]["reference"].values))
    assert gap < (1e-6 if seasonal_lam == 0 else 0.02)


def test_gam_tolerates_gaps_and_keeps_the_calendar_phase():
    times = daily_times("1980-01-01", "2010-12-31")
    ph = phase(times)
    x = 20 + 4 * np.cos(2 * np.pi * (ph - phase(np.array(["2001-07-15"],
                                                          dtype="datetime64[ns]"))[0]))
    x[[100, 5000, 5001]] = np.nan
    results = {policy: fit(point(x, times), dyn("gam", leap_day_policy=policy,
                                                reference_period=("1981-01-01", "2000-12-31")))
               for policy in ("hobday_366_interp", "calendar_doy")}
    np.testing.assert_array_equal(results["hobday_366_interp"].dataset["seasonal"].values,
                                  results["calendar_doy"].dataset["seasonal"].values)
    seasonal = results["calendar_doy"].dataset["seasonal"].to_series().dropna()
    peaks = seasonal.groupby(seasonal.index.year).idxmax()
    assert set(peaks.dt.strftime("%m-%d")) == {"07-15"}
    assert np.isfinite(results["calendar_doy"].dataset["reference"].values).all()


def test_gam_cutoff_never_uses_later_data():
    times = daily_times("1995-01-01", "2012-12-31")
    x = noisy(times)
    later = x.copy()
    later[times > np.datetime64("2010-12-31")] += 5.0
    spec = dyn("gam", fit_mode="causal", available_until="2010-12-31",
               reference_period=("1996-01-01", "2010-12-31"))
    a, b = (fit(point(v, times), spec).dataset for v in (x, later))
    for name in ("trend", "seasonal", "residual_threshold"):
        np.testing.assert_array_equal(a[name].values, b[name].values, err_msg=name)


def test_gam_default_keeps_most_of_a_one_year_event():
    times = daily_times("1980-01-01", "2019-12-31")
    base = exact_series(times)
    event = (times >= np.datetime64("2000-01-01")) & (times < np.datetime64("2001-01-01"))
    spec = dyn("gam", reference_period=("1981-01-01", "2010-12-31"))
    residuals = [x - fit(point(x, times), spec).dataset["reference"].values
                 for x in (base + 1.5 * event, base)]
    assert np.nanmean((residuals[0] - residuals[1])[event]) / 1.5 >= 0.9


# ---: ssa (singular spectrum analysis) --------------------------------------------

def test_ssa_leading_triples_equal_the_exact_svd():
    """The randomised SVD (FFT products, never forming the Hankel matrix) reproduces the
    reconstructions of the signal eigentriples of numpy's exact SVD.
    """
    from xhw import _ssa

    rng = np.random.default_rng(1)
    n, window, rank = 366 * 8, 732, 6
    t = np.arange(n)
    x = 15 + 0.001 * t + 3 * np.cos(2 * np.pi * t / 366) + rng.normal(0, 0.4, n)
    traj = np.lib.stride_tricks.sliding_window_view(x, window).T
    u, s, vt = np.linalg.svd(traj, full_matrices=False)
    sigma, uu, vv = _ssa.leading_triples(x[None], window, rank)
    for k in range(3):        # trend + the annual pair (singular values 20896, 1898, 1896;
                              # the next ones, ~34, are noise)
        pick = (np.arange(rank) == k)[None].astype(float)
        exact = _ssa.reconstruct(s[None, :rank], u[None, :, :rank], vt.T[None, :, :rank],
                                 pick, n)
        np.testing.assert_allclose(_ssa.reconstruct(sigma, uu, vv, pick, n), exact, atol=1e-8)


def test_ssa_hankel_products_and_diagonal_averaging_by_hand():
    from xhw import _ssa

    x = np.arange(1.0, 8.0)                      # 1..7, window 3: trajectory 3 x 5
    traj = np.lib.stride_tricks.sliding_window_view(x, 3).T
    hankel = _ssa._Hankel(x[None], 3)
    v = np.arange(5.0)[None, :, None]
    np.testing.assert_allclose(hankel.times(v)[0, :, 0], traj @ v[0, :, 0])
    u = np.array([1.0, -1.0, 2.0])[None, :, None]
    np.testing.assert_allclose(hankel.transposed_times(u)[0, :, 0], traj.T @ u[0, :, 0])
    rank1 = np.outer(u[0, :, 0], v[0, :, 0])
    expected = [np.mean(np.diag(rank1[::-1], k)) for k in range(-2, 5)]
    got = _ssa.reconstruct(np.ones((1, 1)), u, v, np.ones((1, 1)), 7)[0]
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_ssa_keeps_the_calendar_phase_and_needs_complete_series():
    times = daily_times("1980-01-01", "2010-12-31")
    ph = phase(times)
    x = 20 + 4 * np.cos(2 * np.pi * (ph - phase(np.array(["2001-07-15"],
                                                          dtype="datetime64[ns]"))[0]))
    results = {policy: fit(point(x, times), dyn("ssa", leap_day_policy=policy,
                                                reference_period=("1981-01-01", "2000-12-31")))
               for policy in ("hobday_366_interp", "calendar_doy")}
    np.testing.assert_array_equal(results["hobday_366_interp"].dataset["seasonal"].values,
                                  results["calendar_doy"].dataset["seasonal"].values)
    seasonal = results["calendar_doy"].dataset["seasonal"].to_series().dropna()
    peaks = seasonal.groupby(seasonal.index.year).idxmax()
    assert set(peaks.dt.strftime("%m-%d")) == {"07-15"}
    gap = x.copy()
    gap[4000] = np.nan
    undefined = fit(point(gap, times), dyn("ssa",
                                           reference_period=("1981-01-01", "2000-12-31")))
    assert ((undefined.dataset["invalid_reason"].values & 4) != 0).all()


def test_ssa_cutoff_never_uses_later_data():
    times = daily_times("1990-01-01", "2012-12-31")
    x = noisy(times)
    later = x.copy()
    later[times > np.datetime64("2010-12-31")] += 5.0
    spec = dyn("ssa", fit_mode="causal", available_until="2010-12-31",
               reference_period=("1991-01-01", "2010-12-31"))
    a, b = (fit(point(v, times), spec).dataset for v in (x, later))
    for name in ("trend", "seasonal", "residual_threshold"):
        np.testing.assert_array_equal(a[name].values, b[name].values, err_msg=name)


def test_ssa_default_keeps_most_of_a_one_year_event():
    times = daily_times("1980-01-01", "2019-12-31")
    base = exact_series(times)
    event = (times >= np.datetime64("2000-01-01")) & (times < np.datetime64("2001-01-01"))
    spec = dyn("ssa", reference_period=("1981-01-01", "2010-12-31"))
    residuals = [x - fit(point(x, times), spec).dataset["reference"].values
                 for x in (base + 1.5 * event, base)]
    assert np.nanmean((residuals[0] - residuals[1])[event]) / 1.5 >= 0.9


# --- (real grids): memory and domain ---------------------------------------------------

def test_ssa_blocks_of_cells_give_the_same_decomposition():
    from xhw import _ssa

    rng = np.random.default_rng(3)
    t = np.arange(366 * 8)
    x = (10 + 0.001 * t + 2 * np.cos(2 * np.pi * t / 366))[None] + rng.normal(0, 0.3, (5, t.size))
    options = dict(window=730, rank=8, period=366, trend_period=1500, n_harmonics=2)
    whole = _ssa.decompose(x, cells_per_block=5, **options)
    for block in (1, 2):
        for got, expected in zip(_ssa.decompose(x, cells_per_block=block, **options), whole):
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)
    assert 1 <= _ssa.block_cells(12810, 5490, 12) < 64     # ERA5 box: blocks, not 693 at once


@pytest.mark.parametrize("method", ["detrended", "ssa"])
def test_cells_outside_the_domain_are_not_decomposed(method, monkeypatch):
    from xhw import _dynamic

    times = daily_times("1985-01-01", "2012-12-31")
    rng = np.random.default_rng(11)
    values = exact_series(times)[:, None] + rng.normal(0, 0.4, (times.size, 4))
    inside = np.array([True, False, True, False])
    ds = xr.Dataset({"sst": (("time", "along"), values, {"units": "degC"}),
                     "domain_mask": ("along", inside),
                     "length": ("along", np.ones(4), {"units": "m"})},
                    coords={"time": times, "along": ("along", np.arange(4.0), {"units": "m"})})
    g = grid(support="line", topology="transect", spatial_dims=("along",),
             coords=xhw.CoordRoles(along="along"),
             cell_measure=xhw.CellMeasure(kind="length", var="length", units="m"))
    seen = []
    original = _dynamic._METHODS[method]

    def spy(source, plan, rows, targets):
        seen.append(source.shape[0])
        return original(source, plan, rows, targets)

    monkeypatch.setitem(_dynamic._METHODS, method, spy)
    spec = dyn(method, reference_period=("1991-01-01", "2010-12-31"))
    masked = xhw.fit_baseline(ds, variable="sst", grid=g, baseline=spec, definition=hobday())
    assert seen and set(seen) == {2}
    alone = xhw.fit_baseline(ds.isel(along=inside), variable="sst", grid=g, baseline=spec,
                             definition=hobday())
    for name in ("trend", "seasonal", "threshold"):
        got = masked.dataset[name].isel(along=inside).values
        np.testing.assert_allclose(got, alone.dataset[name].values, rtol=0, atol=1e-9,
                                   err_msg=name)
        assert np.isnan(masked.dataset[name].isel(along=~inside).values).all()
    masked.check_invariants()
