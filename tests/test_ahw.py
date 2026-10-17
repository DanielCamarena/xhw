# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""AHW profiles (atmospheric detection): perkins2013_tx90/tn90 and and_daily TX+TN with lag.

Expected values are written by hand from docs/VALIDATION.md and declared numerical rules §D-E.
"""

import hashlib

import numpy as np
import pytest
import xarray as xr
from _fixtures import ahw_baseline, grid, hobday, perkins2013, provenance, txtn90

import xhw
from xhw._schema import canonical_json

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
LAND = grid(domain="land")


def daily(start="2000-01-01", **series):
    n = len(next(iter(series.values())))
    first = np.datetime64(start, "D")
    times = np.arange(first, first + np.timedelta64(n, "D")).astype("datetime64[ns]")
    data = {name: ("time", np.asarray(v, dtype=float), {"units": "degC"})
            for name, v in series.items()}
    return xr.Dataset({**data, "domain_mask": ((), True)},
                      coords={"time": times, "lat": -12.0, "lon": -77.5})


def constant_dataset(definition, thresholds, references):
    """Baseline fields written by hand: constant Q and C per component (or single)."""
    attrs = {"units": "degC", "units_metadata": "temperature: on_scale"}
    comps = definition.components
    shape = (len(comps), 366) if comps else (366,)
    dims = ("component", "dayofyear") if comps else ("dayofyear",)
    q = np.broadcast_to(np.asarray(thresholds, float).reshape(-1, 1), (len(thresholds), 366))
    c = np.broadcast_to(np.asarray(references, float).reshape(-1, 1), (len(references), 366))
    coords = {"dayofyear": np.arange(1, 367)}
    if comps:
        coords["component"] = [x.variable for x in comps]
    ds = xr.Dataset({"threshold": (dims, q.reshape(shape).copy(), attrs),
                     "reference": (dims, c.reshape(shape).copy(), attrs),
                     "sample_count": (dims, np.full(shape, 999, np.int32)),
                     "invalid_reason": (dims, np.zeros(shape, np.uint8)),
                     "domain_mask": ((), True)}, coords=coords)
    return ds


def constant_result(definition, thresholds, references, g=LAND, data=None):
    ds = constant_dataset(definition, thresholds, references) if data is None else data
    return xhw.BaselineResult.from_dataset(ds, grid=g, baseline=ahw_baseline(),
                                           definition=definition,
                                           provenance=provenance("hand-baseline"))


def run(ds, definition, result, variable):
    return xhw.detect(ds, variable=variable, grid=LAND, definition=definition,
                      baseline_result=result)


def membership(mask):
    return np.flatnonzero(mask.dataset["event_membership"].values).tolist()


# --- perkins2013_tx90 / tn90: threshold estimation ------------------------------------------

def yearly(years, value_of, start_year=2000):
    first = np.datetime64(f"{start_year}-01-01", "D")
    last = np.datetime64(f"{start_year + years}-01-01", "D")
    times = np.arange(first, last)
    values = [value_of(t) for t in times]
    return daily(start=str(first), tx=values)


def test_type8_percentile_by_hand():
    ds = yearly(10, lambda t: float(t.astype(object).year - 1999))      # 1..10
    base = ahw_baseline(reference_period=("2000-01-01", "2009-12-31"), doy_window_days=1,
                        min_samples=10)
    result = xhw.fit_baseline(ds, variable="tx", grid=LAND, baseline=base,
                              definition=perkins2013(variable="tx"))
    # Hyndman-Fan type 8, n=10, p=0.9: np + (p+1)/3 = 9.6333 -> 9 + 0.6333 (type 7: 9.1)
    np.testing.assert_allclose(result.dataset["threshold"].values[180], 9 + 1.9 / 3)


def test_fifteen_day_window_pools_15_samples_per_year_strictly_inside_the_period():
    ds = yearly(10, lambda t: 20.0)
    base = ahw_baseline(reference_period=("2000-01-01", "2009-12-31"), min_samples=1)
    result = xhw.fit_baseline(ds, variable="tx", grid=LAND, baseline=base,
                              definition=perkins2013(variable="tx"))
    count = result.dataset["sample_count"].values
    assert count[180] == 15 * 10
    assert count[0] == 8 + 15 * 9                    # 1 Jan 2000: no December 1999
    assert np.all(np.isfinite(result.dataset["threshold"].values))   # no smoothing needed


# --- perkins2013 persistence rules (VALIDATION TX90/TN90, Climpact C3-C6) --------------------

@pytest.mark.parametrize("values,expected", [
    ([26, 31, 31, 31, 26], [1, 2, 3]),                   # three TX > Q90 -> one event
    ([26, 31, 31, 26, 31, 31, 26], []),                  # runs of 2 are not events
    ([26, 30, 30, 30, 26], []),                          # equality is not an exceedance
    ([26, 31, 31, np.nan, 31, 31, 26], []),              # missing = non-exceedance, gap 0
    ([31, 31, 31, np.nan, 31, 31, 31], [0, 1, 2, 4, 5, 6]),   # two events, never joined
])
def test_tx90_runs(values, expected):
    definition = perkins2013(variable="tx")
    mask = run(daily(tx=values), definition, constant_result(definition, [30.0], [25.0]), "tx")
    assert membership(mask) == expected
    mask.check_invariants()


def test_tn90_is_nocturnal_and_independent_of_tx():
    definition = perkins2013("min", variable="tn")
    ds = daily(tn=[18, 21, 21, 21, 18], tx=[25, 25, 25, 25, 25])   # TX never extreme
    mask = run(ds, definition, constant_result(definition, [20.0], [16.0]), "tn")
    assert membership(mask) == [1, 2, 3]


def test_feb29_is_a_real_day():
    """27-28 Feb and 1 Mar hot, 29 Feb not: no event in xHW (Climpact drops 29 Feb: 3 days)."""
    definition = perkins2013(variable="tx")
    ds = daily(start="2004-02-26", tx=[26, 31, 31, 26, 31, 26])
    mask = run(ds, definition, constant_result(definition, [30.0], [25.0]), "tx")
    assert membership(mask) == []


def test_fit_then_detect_with_the_profile():
    rng = np.random.default_rng(5)
    ds = yearly(6, lambda t: 0.0)
    ds["tx"] = ds["tx"] + 28 + rng.normal(0, 2, ds.sizes["time"])
    base = ahw_baseline(reference_period=("2000-01-01", "2005-12-31"), min_samples=60)
    definition = perkins2013(variable="tx")
    result = xhw.fit_baseline(ds, variable="tx", grid=LAND, baseline=base, definition=definition)
    mask = run(ds, definition, result, "tx")
    mask.check_invariants()
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    catalog.check_against(mask)
    assert (catalog.dataset["duration_days"] >= 3).all()
    assert "category_max" not in catalog.dataset


# --- and_daily: specification ---------------------------------------------------------------

def comp(variable, lag=0, statistic="max"):
    return xhw.DailyComponent(variable=variable, daily_statistic=statistic, percentile=90.0,
                              lag_days=lag)


@pytest.mark.parametrize("overrides", [
    {"components": None},
    {"components": (comp("tx_daily"),)},                            # needs >= 2
    {"components": (comp("tx_daily", lag=1), comp("tn_daily"))},     # reference lag 0
    {"components": (comp("tx_daily"), comp("tx_daily"))},           # unique variables
    {"percentile": 90.0},
    {"daily_statistic": "max"},
    {"threshold_kind": "absolute"},
])
def test_and_daily_rules(overrides):
    with pytest.raises(xhw.ValidationError):
        txtn90(**overrides)


def test_component_rules_and_single_without_components():
    with pytest.raises(xhw.ValidationError):
        comp("tn_daily", lag=2)
    with pytest.raises(xhw.ValidationError):
        comp("tn_daily", statistic="median")
    with pytest.raises(xhw.ValidationError):
        hobday(components=(comp("a"), comp("b")))


def test_components_default_keeps_existing_spec_ids():
    """`components=None` is omitted from the canonical identifier payload."""
    spec = hobday()
    fields = spec.to_dict()["fields"]
    fields.pop("components")
    payload = {"type": "DefinitionSpec", "schema_major": 0, "fields": fields}
    old = "sha256:" + hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()
    assert spec.spec_id == old
    assert txtn90().spec_id != txtn90(tn_lag=1).spec_id              # lag_days is identity


def test_and_daily_json_round_trip():
    spec = txtn90(tn_lag=1)
    again = xhw.DefinitionSpec.from_json(spec.to_json())
    assert again == spec and again.components[1].lag_days == 1
    old_payload = hobday().to_dict()
    old_payload["schema_version"] = "0.1"
    old_payload["fields"].pop("components")
    assert xhw.DefinitionSpec.from_dict(old_payload) == hobday()


# --- and_daily: detection ---------------------------------------------------------------------

TX_Q, TN_Q = 30.0, 20.0


def combined(tx, tn, lag=0, start="2000-01-01"):
    definition = txtn90(tn_lag=lag)
    result = constant_result(definition, [TX_Q, TN_Q], [25.0, 15.0])
    mask = run(daily(start=start, tx=tx, tn=tn), definition, result, ("tx", "tn"))
    mask.check_invariants()
    return mask


def test_alternating_hot_days_and_nights_are_not_a_joint_event():
    tx = [26, 31, 26, 31, 26, 31, 26]
    tn = [18, 18, 21, 18, 21, 18, 21]
    mask = combined(tx, tn)
    assert not mask.dataset["exceedance"].values.any() and membership(mask) == []


def test_three_coincident_days_are_one_event():
    mask = combined([26, 31, 31, 31, 26], [18, 21, 21, 21, 18])
    assert membership(mask) == [1, 2, 3]
    ds = mask.dataset
    assert ds["temperature"].dims == ("component", "time")
    assert ds["component"].values.tolist() == ["tx_daily", "tn_daily"]
    assert ds["lag_days"].values.tolist() == [0, 0]


def test_persistence_is_applied_after_the_daily_and():
    # TX hot on days 1-4 and TN hot on days 3-6: each component has a 4-day run, but only
    # days 3-4 are joint -> no event (a per-variable AND of memberships would give 3-4 too,
    # while an OR would give an event; ).
    mask = combined([26, 31, 31, 31, 31, 26, 26, 26], [18, 18, 18, 21, 21, 21, 21, 18])
    assert mask.dataset["exceedance"].values.tolist() == [False] * 3 + [True] * 2 + [False] * 3
    assert membership(mask) == []


def test_missing_component_value_invalidates_the_joint_day():
    mask = combined([26, 31, 31, 31, 26], [18, 21, np.nan, 21, 18])
    ds = mask.dataset
    assert not ds["valid"].values[2] and ds["invalid_reason"].values[2] == 2
    assert membership(mask) == []


def test_lag_pairs_the_hot_day_with_the_following_night():
    tx = [26, 31, 31, 31, 26, 26]
    tn = [18, 18, 21, 21, 21, 18]            # nights after days 1-3 are hot (TN of days 2-4)
    assert membership(combined(tx, tn, lag=0)) == []        # same day: only days 2-3 joint
    lagged = combined(tx, tn, lag=1)
    assert membership(lagged) == [1, 2, 3]
    assert lagged.dataset["lag_days"].values.tolist() == [0, 1]


def test_lag_makes_the_last_day_invalid_and_censors_events_reaching_it():
    mask = combined([26, 26, 31, 31, 31], [18, 18, 18, 21, 21], lag=1)
    ds = mask.dataset
    assert not ds["valid"].values[-1] and ds["invalid_reason"].values[-1] == 2
    assert membership(mask) == []                  # days 2-3 joint, day 4 lacks TN(D+1)
    mask = combined([26, 31, 31, 31, 26], [18, 18, 21, 21, 21], lag=1)
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective").dataset
    assert catalog["end"].values[0] == np.datetime64("2000-01-04")
    assert bool(catalog["right_censored"].values[0]) and catalog["censor_reason"].values[0] == 4


def test_per_component_metrics_on_their_own_dates():
    mask = combined([26, 31, 33, 32, 26, 26], [18, 18, 21, 24, 22, 18], lag=1)
    cat = xhw.build_catalog(mask, analysis_mode="retrospective")
    ds = cat.dataset
    assert ds["intensity_max"].dims == ("event", "component")
    assert ds["n_filled_days"].dims == ("event",)
    event = ds.isel(event=0)
    tx, tn = event.sel(component="tx_daily"), event.sel(component="tn_daily")
    assert float(tx["intensity_max"]) == 33 - 25 and float(tn["intensity_max"]) == 24 - 15
    assert tx["peak_date"].values == np.datetime64("2000-01-03")
    assert tn["peak_date"].values == np.datetime64("2000-01-03")    # TN of 4 Jan -> day 3
    assert float(tn["temperature_mean"]) == pytest.approx((21 + 24 + 22) / 3)
    assert event["lag_days"].values.tolist() == [0, 1]
    cat.check_against(mask)


def test_fit_baseline_with_components_equals_single_fits():
    rng = np.random.default_rng(2)
    n = 365 * 4 + 1
    ds = daily(tx=28 + rng.normal(0, 2, n), tn=18 + rng.normal(0, 2, n))
    base = ahw_baseline(reference_period=("2000-01-01", "2003-12-31"), min_samples=40)
    both = xhw.fit_baseline(ds, variable=("tx", "tn"), grid=LAND, baseline=base,
                            definition=txtn90())
    assert both.dataset["threshold"].dims == ("component", "dayofyear")
    for i, (name, statistic) in enumerate((("tx", "max"), ("tn", "min"))):
        single = xhw.fit_baseline(ds, variable=name, grid=LAND, baseline=base,
                                  definition=perkins2013(statistic, variable=name))
        np.testing.assert_array_equal(both.dataset["threshold"].values[i],
                                      single.dataset["threshold"].values)
    with pytest.raises(xhw.ValidationError, match="tuple of 2"):
        xhw.fit_baseline(ds, variable="tx", grid=LAND, baseline=base, definition=txtn90())
    ds["tn"] = (ds["tn"] + 273.15).assign_attrs(units="K")
    with pytest.raises(xhw.UnitsError):
        xhw.fit_baseline(ds, variable=("tx", "tn"), grid=LAND, baseline=base,
                         definition=txtn90())


def random_pair(seed=4, n=240, cols=12):
    rng = np.random.default_rng(seed)
    tx = np.where(rng.random((n, cols)) < 0.6, 31.0, 26.0)
    tn = np.where(rng.random((n, cols)) < 0.6, 21.0, 18.0)
    tn[rng.random((n, cols)) < 0.03] = np.nan
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-01")
                      + np.timedelta64(n, "D")).astype("datetime64[ns]")
    ds = xr.Dataset({"tx": (("time", "along"), tx, {"units": "degC"}),
                     "tn": (("time", "along"), tn, {"units": "degC"}),
                     "domain_mask": ("along", np.ones(cols, bool)),
                     "segment_length": ("along", np.full(cols, 5000.0), {"units": "m"})},
                    coords={"time": times, "along": ("along", np.arange(cols) * 5000.0,
                                                     {"units": "m"})})
    g = grid(support="line", topology="transect", spatial_dims=("along",), domain="land",
             coords=xhw.CoordRoles(along="along"),
             cell_measure=xhw.CellMeasure(kind="length", var="segment_length", units="m"))
    return ds, g


def line_result(definition, ds, g):
    data = constant_dataset(definition, [TX_Q, TN_Q], [25.0, 15.0]).drop_vars("domain_mask")
    data = data.expand_dims(along=ds.sizes["along"]).assign_coords(along=ds["along"])
    data["domain_mask"] = ds["domain_mask"]
    return constant_result(definition, None, None, g=g, data=data)


@pytest.mark.parametrize("chunk", [7, 50])
def test_and_daily_chunked_equals_eager(chunk):
    pytest.importorskip("dask")
    ds, g = random_pair()
    definition = txtn90(tn_lag=1)
    result = line_result(definition, ds, g)
    eager = xhw.detect(ds, variable=("tx", "tn"), grid=g, definition=definition,
                       baseline_result=result)
    lazy_ds = ds.copy()
    for name in ("tx", "tn"):
        lazy_ds[name] = ds[name].chunk({"time": chunk, "along": 5})
    lazy = xhw.detect(lazy_ds, variable=("tx", "tn"), grid=g, definition=definition,
                      baseline_result=result)
    assert lazy.dataset["event_membership"].chunks is not None
    for name in ("exceedance", "event_membership", "joined_gap", "valid", "invalid_reason",
                 "temperature", "threshold"):
        np.testing.assert_array_equal(lazy.dataset[name].values, eager.dataset[name].values,
                                      err_msg=name)
    eager.check_invariants()


def test_and_daily_netcdf_round_trip(tmp_path):
    ds, g = random_pair()
    definition = txtn90(tn_lag=1)
    mask = xhw.detect(ds, variable=("tx", "tn"), grid=g, definition=definition,
                      baseline_result=line_result(definition, ds, g))
    catalog = xhw.build_catalog(mask, analysis_mode="retrospective")
    assert catalog.dataset.sizes["event"] > 0
    for product, cls, name in ((mask, xhw.XMask, "m.nc"),
                               (catalog, xhw.EventCatalog, "c.nc")):
        path = tmp_path / name
        product.to_dataset().to_netcdf(path, engine="h5netcdf")
        with xr.open_dataset(path, engine="h5netcdf") as opened:
            again = cls.from_dataset(opened.load())
        assert again.definition == definition
        for var in product.dataset.data_vars:
            np.testing.assert_array_equal(again.dataset[var].values,
                                          product.dataset[var].values, err_msg=var)
    again_mask = xhw.XMask.from_dataset(
        xr.open_dataset(tmp_path / "m.nc", engine="h5netcdf").load())
    again_mask.check_invariants()
    xhw.EventCatalog.from_dataset(
        xr.open_dataset(tmp_path / "c.nc", engine="h5netcdf").load()).check_against(again_mask)


def test_tampered_combined_exceedance_breaks_the_invariant():
    mask = combined([26, 31, 31, 31, 26], [18, 21, 21, 21, 18])
    ds = mask.dataset
    ds["exceedance"] = ds["exceedance"].copy(data=np.zeros(5, bool))
    ds["event_membership"] = ds["exceedance"]
    broken = xhw.XMask.from_dataset(ds, grid=mask.grid, definition=mask.definition,
                                          baseline=mask.baseline, provenance=mask.provenance)
    with pytest.raises(xhw.InvariantError, match="exceedance !="):
        broken.check_invariants()


def test_and_daily_detect_does_not_compute():
    dask = pytest.importorskip("dask")
    ds, g = random_pair()
    definition = txtn90(tn_lag=1)
    result = line_result(definition, ds, g)
    lazy_ds = ds.copy()
    for name in ("tx", "tn"):
        lazy_ds[name] = ds[name].chunk({"time": 60})

    def forbid(*args, **kwargs):
        raise AssertionError("computed during detect")

    with dask.config.set(scheduler=forbid):
        xhw.detect(lazy_ds, variable=("tx", "tn"), grid=g, definition=definition,
                   baseline_result=result)
