# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""In-base correction of the reference years.

Level 1 values are written by hand (declared numerical rules §): three reference years, a
one-day window and type 7, so every threshold is a percentile of a few known numbers.
"""

import json

import numpy as np
import pytest
import xarray as xr
from _fixtures import ahw_baseline, baseline, grid, hobday, perkins2013, provenance, txtn90

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
YEAR_VALUE = {2000: 1.0, 2001: 2.0, 2002: 4.0, 2003: 3.7}
#: P90 type 7 of the reference years without (LOO) or replacing (mean) each year
NORMAL = 3.6                                    # [1, 2, 4]
LOO = {2000: 3.8, 2001: 3.7, 2002: 1.9}         # [2, 4], [1, 4], [1, 2]
MEAN = {2000: (3.6 + 4.0) / 2, 2001: (3.4 + 4.0) / 2, 2002: (1.8 + 2.0) / 2}


def point_ds(values, start="2000-01-01"):
    first = np.datetime64(start, "D")
    times = np.arange(first, first + np.timedelta64(len(values), "D")).astype("datetime64[ns]")
    return xr.Dataset({"sst": ("time", np.asarray(values, dtype=float), {"units": "degC"}),
                       "domain_mask": ((), True)},
                      coords={"time": times, "lat": -12.0, "lon": -77.5})


def year_valued(last="2002-12-31"):
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64(last) + np.timedelta64(1, "D"))
    years = times.astype("datetime64[Y]").astype(int) + 1970
    return point_ds([YEAR_VALUE[y] for y in years])


def three_years(**overrides):
    params = dict(reference_period=("2000-01-01", "2002-12-31"), doy_window_days=1,
                  smoothing_window_days=None, min_samples=2)
    params.update(overrides)
    return baseline(**params)


def fit(ds, correction, definition=None, **overrides):
    return xhw.fit_baseline(ds, variable="sst", grid=grid(),
                            baseline=three_years(in_base_correction=correction, **overrides),
                            definition=definition or hobday())


# --- spec ---------------------------------------------------------------------------------

def test_default_is_omitted_from_spec_id_and_old_json_reads():
    spec = baseline()
    assert spec.in_base_correction == "none"
    data = spec.to_dict()
    del data["fields"]["in_base_correction"]
    data["schema_version"] = "0.1"
    old = xhw.BaselineSpec.from_dict(data)
    assert old == spec and old.spec_id == spec.spec_id
    corrected = baseline(in_base_correction="leave_one_year_out")
    assert corrected.spec_id != spec.spec_id
    assert xhw.BaselineSpec.from_json(corrected.to_json()) == corrected


@pytest.mark.parametrize("overrides", [
    {"in_base_correction": "bootstrap"},
    {"in_base_correction": "mean_threshold", "reference_period": ("2000-01-01", "2000-12-31")},
])
def test_invalid_corrections_are_rejected(overrides):
    with pytest.raises(xhw.ValidationError):
        baseline(**overrides)


def test_reference_years_cover_partial_years():
    spec = baseline(reference_period=("1999-07-01", "2001-03-31"))
    assert spec.reference_years == (1999, 2000, 2001)


def test_absolute_threshold_cannot_be_corrected():
    definition = hobday(threshold_kind="absolute", percentile=None,
                        absolute_threshold=(25.0, "degC"))
    with pytest.raises(xhw.ValidationError, match="percentile"):
        fit(year_valued(), "leave_one_year_out", definition=definition)


# --- level 1: thresholds by hand ----------------------------------------------------------

def test_leave_one_year_out_by_hand():
    result = fit(year_valued(), "leave_one_year_out")
    ds = result.dataset
    assert ds["threshold_by_year"].dims == ("reference_year", "dayofyear")
    assert ds["reference_year"].values.tolist() == [2000, 2001, 2002]
    np.testing.assert_allclose(ds["threshold"].values, NORMAL)
    for i, year in enumerate((2000, 2001, 2002)):
        np.testing.assert_allclose(ds["threshold_by_year"].values[i], LOO[year], err_msg=year)
    assert (ds["sample_count_by_year"].values[:, 180] == 2).all()
    assert (ds["invalid_reason_by_year"].values == 0).all()
    result.check_invariants()


def test_mean_threshold_by_hand():
    ds = fit(year_valued(), "mean_threshold").dataset
    for i, year in enumerate((2000, 2001, 2002)):
        np.testing.assert_allclose(ds["threshold_by_year"].values[i], MEAN[year], err_msg=year)
    assert (ds["sample_count_by_year"].values[:, 180] == 3).all()


def test_too_few_samples_without_the_year_invalidate_that_year_only():
    ds = fit(year_valued(), "leave_one_year_out", min_samples=3).dataset
    assert (ds["invalid_reason"].values == 0).all()
    assert (ds["invalid_reason_by_year"].values == 8).all()          # insufficient_samples
    assert np.isnan(ds["threshold_by_year"].values).all()


def test_window_crossing_new_year_removes_dates_not_positions():
    """: a January window of 2002 keeps 31 Dec 2001; year 2001 loses its 31 Dec."""
    n = 3 * 365 + 1
    ds = point_ds(np.arange(n, dtype=float) ** 1.1)
    values = ds["sst"].to_series()

    def at(*dates):
        return values[list(dates)].to_numpy()

    result = fit(ds, "leave_one_year_out", doy_window_days=3).dataset
    sample = at("2000-01-01", "2000-01-02", "2000-12-31", "2002-01-01", "2002-01-02")
    np.testing.assert_allclose(result["threshold_by_year"].sel(reference_year=2001).values[0],
                               np.percentile(sample, 90, method="linear"))
    assert result["sample_count_by_year"].sel(reference_year=2001).values[0] == 5

    mean = fit(ds, "mean_threshold", doy_window_days=3).dataset
    with_2000 = np.concatenate([sample, at("2000-01-01", "2000-01-02", "2000-12-31")])
    with_2002 = np.concatenate([sample, at("2002-01-01", "2002-01-02")])
    expected = np.mean([np.percentile(s, 90, method="linear") for s in (with_2000, with_2002)])
    np.testing.assert_allclose(mean["threshold_by_year"].sel(reference_year=2001).values[0],
                               expected)
    assert mean["sample_count_by_year"].sel(reference_year=2001).values[0] == 7


def test_leave_one_year_out_equals_a_fit_without_that_year():
    """29 Feb, smoothing and sample counts apply to each replica as to the ordinary fit."""
    rng = np.random.default_rng(3)
    ds = point_ds(rng.normal(20, 1, 3 * 365 + 1))
    options = dict(doy_window_days=5, smoothing_window_days=11, min_samples=1)
    corrected = fit(ds, "leave_one_year_out", **options).dataset
    for year in (2000, 2001, 2002):
        without = ds.copy()
        without["sst"] = ds["sst"].where(ds["time"].dt.year != year)
        plain = fit(without, "none", **options).dataset
        mine = corrected.sel(reference_year=year)
        np.testing.assert_array_equal(mine["threshold_by_year"].values, plain["threshold"].values)
        np.testing.assert_array_equal(mine["sample_count_by_year"].values,
                                      plain["sample_count"].values)


def test_replica_of_a_year_ignores_that_year():
    rng = np.random.default_rng(4)
    values = rng.normal(20, 1, 3 * 365 + 1)
    changed = values.copy()
    changed[366:731] += rng.normal(0, 5, 365)                        # all of 2001
    for correction in ("leave_one_year_out", "mean_threshold"):
        a, b = (fit(point_ds(v), correction, doy_window_days=11,
                    smoothing_window_days=31).dataset for v in (values, changed))
        np.testing.assert_array_equal(a["threshold_by_year"].sel(reference_year=2001).values,
                                      b["threshold_by_year"].sel(reference_year=2001).values)
        assert not np.allclose(a["threshold"].values, b["threshold"].values)


def test_without_day_of_year_axis():
    ds = fit(year_valued(), "leave_one_year_out", doy_window_days=None,
             leap_day_policy="not_applicable", min_samples=1).dataset
    assert ds["threshold_by_year"].dims == ("reference_year",)
    np.testing.assert_allclose(ds["threshold_by_year"].values,
                               [np.percentile(v, 90, method="linear") for v in
                                ([2.0] * 365 + [4.0] * 365, [1.0] * 366 + [4.0] * 365,
                                 [1.0] * 366 + [2.0] * 365)])


# --- expand and detect --------------------------------------------------------------------

@pytest.mark.parametrize("correction,expected", [("leave_one_year_out", LOO),
                                                 ("mean_threshold", MEAN)])
def test_expand_and_detect_use_the_threshold_of_each_year(correction, expected):
    ds = year_valued(last="2003-12-31")
    result = fit(ds, correction)
    mask = xhw.detect(ds, variable="sst", grid=grid(), definition=hobday(),
                      baseline_result=result).dataset
    years = mask["time"].dt.year.values
    want = np.array([expected.get(y, NORMAL) for y in years])
    np.testing.assert_allclose(mask["threshold"].values, want)
    np.testing.assert_allclose(result.expand(mask["time"].values)["threshold"].values, want)
    np.testing.assert_array_equal(mask["exceedance"].values, ds["sst"].values > want)
    np.testing.assert_allclose(mask["reference"].values, 7.0 / 3.0)   # never corrected
    uncorrected = xhw.detect(ds, variable="sst", grid=grid(), definition=hobday(),
                             baseline_result=fit(ds, "none")).dataset
    outside = years == 2003
    for name in ("threshold", "exceedance", "event_membership"):
        np.testing.assert_array_equal(mask[name].values[outside],
                                      uncorrected[name].values[outside], err_msg=name)


def test_in_base_invalid_year_invalidates_its_days_only():
    ds = year_valued(last="2003-12-31")
    mask = xhw.detect(ds, variable="sst", grid=grid(), definition=hobday(),
                      baseline_result=fit(ds, "leave_one_year_out", min_samples=3)).dataset
    inside = mask["time"].dt.year.values <= 2002
    assert (mask["invalid_reason"].values[inside] == 8).all()
    assert (mask["invalid_reason"].values[~inside] == 0).all()


def test_and_daily_has_per_component_thresholds_by_year():
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2004-01-01"))
    years = times.astype("datetime64[Y]").astype(int) + 1970
    tx = np.array([YEAR_VALUE[y] for y in years]) + 20
    ds = xr.Dataset({"tx": ("time", tx, {"units": "degC"}),
                     "tn": ("time", tx - 10, {"units": "degC"}), "domain_mask": ((), True)},
                    coords={"time": times.astype("datetime64[ns]"), "lat": -12.0, "lon": -77.5})
    definition = txtn90()
    base = ahw_baseline(reference_period=("2000-01-01", "2002-12-31"), doy_window_days=1,
                        percentile_estimator="hyndman_fan_7", min_samples=2,
                        in_base_correction="leave_one_year_out")
    g = grid(domain="land")
    result = xhw.fit_baseline(ds, variable=("tx", "tn"), grid=g, baseline=base,
                              definition=definition)
    by_year = result.dataset["threshold_by_year"]
    assert by_year.dims == ("reference_year", "component", "dayofyear")
    np.testing.assert_allclose(by_year.sel(component="tx_daily").values[:, 0],
                               [LOO[y] + 20 for y in (2000, 2001, 2002)])
    np.testing.assert_allclose(by_year.sel(component="tn_daily").values[:, 0],
                               [LOO[y] + 10 for y in (2000, 2001, 2002)])
    mask = xhw.detect(ds, variable=("tx", "tn"), grid=g, definition=definition,
                      baseline_result=result)
    thr = mask.dataset["threshold"].sel(component="tx_daily").values
    np.testing.assert_allclose(thr[:366], LOO[2000] + 20)
    np.testing.assert_allclose(thr[-365:], NORMAL + 20)
    mask.check_invariants()


# --- containers ---------------------------------------------------------------------------

def test_netcdf_round_trip_keeps_the_years(tmp_path):
    result = fit(year_valued(), "mean_threshold")
    path = tmp_path / "baseline.nc"
    result.to_dataset().to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        again = xhw.BaselineResult.from_dataset(opened.load())
    assert again.baseline == result.baseline
    assert json.loads(opened.attrs["xhw_baseline"])["schema_version"] == "0.1"
    for name in ("threshold_by_year", "sample_count_by_year", "invalid_reason_by_year"):
        np.testing.assert_array_equal(again.dataset[name].values, result.dataset[name].values)
    again.check_invariants()


def test_structure_requires_years_iff_corrected():
    corrected = fit(year_valued(), "leave_one_year_out")
    with pytest.raises(xhw.ValidationError, match="threshold_by_year"):
        xhw.BaselineResult.from_dataset(
            corrected.dataset.drop_vars("threshold_by_year"), grid=grid(),
            baseline=corrected.baseline, definition=corrected.definition,
            provenance=provenance("p"))
    with pytest.raises(xhw.ValidationError, match="in_base_correction"):
        xhw.BaselineResult.from_dataset(
            corrected.dataset, grid=grid(), baseline=three_years(),
            definition=corrected.definition, provenance=provenance("p"))


def test_spatial_chunks_are_lazy_and_equal_to_eager():
    dask = pytest.importorskip("dask")
    rng = np.random.default_rng(5)
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2004-01-01")).astype(
        "datetime64[ns]")
    cols = 6
    ds = xr.Dataset({"sst": (("time", "along"), rng.normal(20, 1, (times.size, cols)),
                             {"units": "degC"}),
                     "domain_mask": ("along", np.arange(cols) != 2),
                     "segment_length": ("along", np.full(cols, 1.0), {"units": "m"})},
                    coords={"time": times, "along": ("along", np.arange(cols) * 1.0,
                                                     {"units": "m"})})
    g = grid(support="line", topology="transect", spatial_dims=("along",),
             coords=xhw.CoordRoles(along="along"),
             cell_measure=xhw.CellMeasure(kind="length", var="segment_length", units="m"))
    base = three_years(doy_window_days=5, in_base_correction="leave_one_year_out")
    eager = xhw.fit_baseline(ds, variable="sst", grid=g, baseline=base, definition=hobday())

    def forbid(*args, **kwargs):
        raise AssertionError("computed during fit_baseline")

    lazy_ds = ds.copy()
    lazy_ds["sst"] = ds["sst"].chunk({"time": -1, "along": 4})
    with dask.config.set(scheduler=forbid):
        lazy = xhw.fit_baseline(lazy_ds, variable="sst", grid=g, baseline=base,
                                definition=hobday())
    assert lazy.dataset["threshold_by_year"].chunks is not None
    for name in ("threshold_by_year", "sample_count_by_year", "invalid_reason_by_year"):
        np.testing.assert_array_equal(lazy.dataset[name].values, eager.dataset[name].values)
    masks = [xhw.detect(d, variable="sst", grid=g, definition=hobday(), baseline_result=r)
             for d, r in ((ds, eager), (lazy_ds.chunk({"time": 100}), lazy))]
    np.testing.assert_array_equal(masks[0].dataset["threshold"].values,
                                  masks[1].dataset["threshold"].values)
    eager.check_invariants()
    masks[0].check_invariants()


# --- in_base_rates (Zhang 2005 rates) -----------------------------------------------------

def test_in_base_rates_by_hand():
    """Replicas of 2001: j=2000 -> P90 3.4, j=2002 -> 4.0; TX 2.0 never exceeds either."""
    ds = year_valued(last="2003-12-31")
    definition = perkins2013(variable="sst", min_duration_days=1)
    result = xhw.fit_baseline(ds, variable="sst", grid=grid(domain="land"),
                              baseline=three_years(), definition=definition)
    rates = xhw.in_base_rates(ds, variable="sst", grid=grid(domain="land"),
                              baseline_result=result)
    assert rates["exceedance_frequency"].dims == ("year",)
    assert rates["year"].values.tolist() == [2000, 2001, 2002, 2003]
    # 2000 (1.0): replicas 3.6 and 4.0 -> 0; 2002 (4.0): replicas 1.8 and 2.0 -> 1;
    # 2003 (3.7) outside: ordinary 3.6 in both replicas -> 1
    np.testing.assert_allclose(rates["exceedance_frequency"].values, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(rates["event_day_frequency"].values, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(rates["valid_days"].values, [366, 365, 365, 365])
    assert rates.attrs["baseline_id"] == result.baseline_id


def test_in_base_rates_average_partial_exceedances():
    """2001 at 3.7 exceeds replica j=2000 (3.4) but not j=2002 (4.0): rate 1/2."""
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2003-01-01"))
    years = times.astype("datetime64[Y]").astype(int) + 1970
    values = {2000: 1.0, 2001: 3.7, 2002: 4.0}
    ds = point_ds([values[y] for y in years])
    definition = perkins2013(variable="sst", min_duration_days=1)
    result = xhw.fit_baseline(ds, variable="sst", grid=grid(domain="land"),
                              baseline=three_years(), definition=definition)
    rates = xhw.in_base_rates(ds, variable="sst", grid=grid(domain="land"),
                              baseline_result=result)
    # replicas of 2001: [1, 4, 1] -> 3.4 and [1, 4, 4] -> 4.0
    assert rates["exceedance_frequency"].sel(year=2001).item() == pytest.approx(0.5)


def test_in_base_rates_rejects_other_data():
    ds = year_valued(last="2003-12-31")
    definition = perkins2013(variable="sst", min_duration_days=1)
    result = xhw.fit_baseline(ds, variable="sst", grid=grid(domain="land"),
                              baseline=three_years(), definition=definition)
    other = ds.copy()
    other["sst"] = ds["sst"] + 1
    with pytest.raises(xhw.ValidationError, match="not fitted on these data"):
        xhw.in_base_rates(other, variable="sst", grid=grid(domain="land"),
                          baseline_result=result)
