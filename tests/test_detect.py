# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""interpolate_gaps / detect (temporal detection ). Expected masks are the hand-written data
contracts fixtures.
"""

import numpy as np
import pytest
import xarray as xr
from _fixtures import (
    ALL,
    baseline,
    constant_baseline,
    f4_line,
    f7_coastal_pair,
    grid,
    hobday,
    point_mask,
)

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
MASK_VARS = ("exceedance", "event_membership", "joined_gap", "valid", "invalid_reason")


FIXTURE_BASELINES = {"F1": (12.0, 10.0), "F2": (8.0, 10.0), "F3": (30.0, 25.0),
                     "F4": (12.0, 10.0), "F5": (12.0, 10.0), "F6": (12.0, 10.0),
                     "F8": (12.0, 10.0), "F9": (12.0, 10.0)}


def run_detect(mask, base, *, definition=None, ds=None, variable="temperature"):
    return xhw.detect(ds if ds is not None else mask.dataset, variable=variable, grid=mask.grid,
                      definition=definition or mask.definition, baseline_result=base)


def assert_same_mask(actual, expected):
    for name in MASK_VARS:
        a = actual.dataset[name].transpose(*expected.dataset[name].dims).values
        np.testing.assert_array_equal(a, expected.dataset[name].values, err_msg=name)


@pytest.mark.parametrize("name", FIXTURE_BASELINES)
def test_detect_reproduces_hand_written_fixture_masks(name):
    fixture = ALL[name]()
    q, c = FIXTURE_BASELINES[name]
    detected = run_detect(fixture.mask, constant_baseline(fixture.mask, threshold=q, reference=c))
    assert_same_mask(detected, fixture.mask)
    detected.check_invariants()


def test_detect_reproduces_coupling_pair_masks():
    pair = f7_coastal_pair()
    for fixture, (q, c) in ((pair.mhw, (12.0, 10.0)), (pair.ahw, (30.0, 25.0))):
        base = constant_baseline(fixture.mask, threshold=q, reference=c)
        assert_same_mask(run_detect(fixture.mask, base), fixture.mask)


# --- missing data policies ---------------------------------------------------------

def gap_series(policy):
    temp = [10, 13, 13, 13, 13, 13, np.nan, 13, 13, 13, 13, 13, 10]
    definition = hobday(missing_policy=policy)
    n = len(temp)
    reason = np.zeros(n, np.uint8)
    reason[6] = 2
    exc = np.array([13.0 == t for t in temp])
    mask = point_mask(temp, exc, exc, np.zeros(n, bool), definition=hobday(),
                      invalid_reason=reason)
    return mask, definition


def test_break_policy_keeps_two_events_around_a_missing_day():
    mask, definition = gap_series("break")
    base = constant_baseline(mask, threshold=12.0, reference=10.0, definition=definition)
    detected = run_detect(mask, base, definition=definition)
    membership = detected.dataset["event_membership"].values
    assert membership.tolist() == [False] + [True] * 5 + [False] + [True] * 5 + [False]


def test_oliver_policy_joins_across_a_missing_day():
    mask, definition = gap_series("as_non_exceedance")
    base = constant_baseline(mask, threshold=12.0, reference=10.0, definition=definition)
    detected = run_detect(mask, base, definition=definition)
    ds = detected.dataset
    assert ds["event_membership"].values.tolist() == [False] + [True] * 11 + [False]
    assert ds["joined_gap"].values.tolist() == [False] * 6 + [True] + [False] * 6
    assert not ds["valid"].values[6] and ds["invalid_reason"].values[6] == 2
    detected.check_invariants()                     # relaxed invariant


def test_rule_order_matters_for_short_runs():
    temp = [10, 13, 13, 13, 10, 13, 13, 13, 10]
    exc = np.array([t == 13 for t in temp])
    for rule, expected in (("qualify_then_join", [False] * 9),
                           ("join_then_qualify", [False] + [True] * 7 + [False])):
        definition = hobday(join_rule=rule)
        mask = point_mask(temp, exc, np.zeros(9, bool), np.zeros(9, bool), definition=hobday())
        base = constant_baseline(mask, threshold=12.0, reference=10.0, definition=definition)
        detected = run_detect(mask, base, definition=definition)
        assert detected.dataset["event_membership"].values.tolist() == expected, rule


# --- interpolate_gaps -------------------------------------------------------------------------

def test_interpolation_fills_short_interior_gaps_only():
    values = [np.nan, 1, np.nan, 3, np.nan, np.nan, 6, np.nan, np.nan, np.nan, 10, np.nan]
    first = np.datetime64("2000-01-01", "D")
    times = np.arange(first, first + np.timedelta64(12, "D")).astype("datetime64[ns]")
    ds = xr.Dataset({"sst": ("time", np.array(values), {"units": "degC"}),
                     "domain_mask": ((), True)},
                    coords={"time": times, "lat": -12.0, "lon": -77.5})
    out = xhw.interpolate_gaps(ds, variable="sst", grid=grid(), max_days=2,
                               max_filled_fraction=0.5)
    np.testing.assert_array_equal(out["sst"].values, [np.nan, 1, 2, 3, 4, 5, 6, np.nan, np.nan,
                                                      np.nan, 10, np.nan])
    assert np.flatnonzero(out["filled"].values).tolist() == [2, 4, 5]
    assert out["filled_fraction"].item() == pytest.approx(3 / 12)
    assert not out["low_data_quality"].item()
    assert out["filled"].attrs["max_days"] == 2


def f4_with_profile(max_filled_fraction):
    fixture = f4_line()
    profile = hobday(missing_policy="as_non_exceedance")
    ds = xhw.interpolate_gaps(fixture.mask.dataset, variable="temperature",
                              grid=fixture.mask.grid, max_days=2,
                              max_filled_fraction=max_filled_fraction)
    base = constant_baseline(fixture.mask, threshold=12.0, reference=10.0, definition=profile)
    return run_detect(fixture.mask, base, definition=profile, ds=ds)


def test_f4_with_hobday2016_profile_is_one_event_with_a_filled_day():
    detected = f4_with_profile(0.2)                      # 1 of 10 days filled = 10 %
    ds = detected.dataset
    assert ds["event_membership"].values[:, 1].tolist() == [False] + [True] * 9
    assert ds["filled"].values[:, 1].tolist() == [False] * 6 + [True] + [False] * 3
    detected.check_invariants()
    steps = [s.name for s in detected.provenance.steps]
    assert steps == ["interpolate_gaps", "detect"]


def test_too_many_filled_days_invalidate_the_series():
    ds = f4_with_profile(0.05).dataset                   # 10 % > 5 %
    assert set(np.unique(ds["invalid_reason"].values[:, 1])) == {16}
    assert not ds["event_membership"].values[:, 1].any()
    assert ds["event_membership"].values[:, 0].any()     # other series unaffected


# --- time chunking with bounded overlap -----------------------------------------------

def random_line(seed=3, n_time=200, n_cols=40):
    rng = np.random.default_rng(seed)
    temp = np.where(rng.random((n_time, n_cols)) < 0.65, 13.0, 10.0)
    temp[rng.random((n_time, n_cols)) < 0.03] = np.nan   # scattered missing days
    temp[40:141, 0] = 13.0                               # an event longer than any chunk
    first = np.datetime64("2000-01-01", "D")
    times = np.arange(first, first + np.timedelta64(n_time, "D")).astype("datetime64[ns]")
    ds = xr.Dataset({"sst": (("time", "along"), temp, {"units": "degC"}),
                     "domain_mask": ("along", np.ones(n_cols, bool)),
                     "segment_length": ("along", np.full(n_cols, 5000.0), {"units": "m"})},
                    coords={"time": times, "along": ("along", np.arange(n_cols) * 5000.0,
                                                     {"units": "m"})})
    return ds, f4_line().mask.grid


@pytest.mark.parametrize("policy", ["break", "as_non_exceedance"])
@pytest.mark.parametrize("chunk", [13, 30, 77])
def test_time_chunked_detection_equals_eager(policy, chunk):
    pytest.importorskip("dask")
    ds, g = random_line()
    definition = hobday(missing_policy=policy)
    template = f4_line().mask
    base = constant_baseline(template, threshold=12.0, reference=10.0, definition=definition,
                             grid_spec=g)
    base = xhw.BaselineResult.from_dataset(
        base.dataset.isel(along=[0] * 40).assign_coords(along=ds["along"]), grid=g,
        baseline=base.baseline, definition=definition, provenance=base.provenance)
    eager = xhw.detect(ds, variable="sst", grid=g, definition=definition, baseline_result=base)
    lazy_ds = ds.copy()
    lazy_ds["sst"] = ds["sst"].chunk({"time": chunk, "along": 16})
    lazy = xhw.detect(lazy_ds, variable="sst", grid=g, definition=definition,
                      baseline_result=base)
    assert lazy.dataset["event_membership"].chunks is not None
    for name in MASK_VARS:
        np.testing.assert_array_equal(lazy.dataset[name].values, eager.dataset[name].values,
                                      err_msg=name)
    assert eager.dataset["event_membership"].values[40:141, 0].all()


def test_time_chunked_interpolation_equals_eager():
    pytest.importorskip("dask")
    ds, g = random_line(seed=5)
    eager = xhw.interpolate_gaps(ds, variable="sst", grid=g, max_days=2,
                                 max_filled_fraction=0.05)
    lazy_ds = ds.copy()
    lazy_ds["sst"] = ds["sst"].chunk({"time": 17})
    lazy = xhw.interpolate_gaps(lazy_ds, variable="sst", grid=g, max_days=2,
                                max_filled_fraction=0.05)
    for name in ("sst", "filled", "low_data_quality"):
        np.testing.assert_array_equal(lazy[name].values, eager[name].values, err_msg=name)


def test_detect_does_not_compute():
    dask = pytest.importorskip("dask")
    ds, g = random_line()
    template = f4_line().mask
    base = constant_baseline(template, threshold=12.0, reference=10.0, grid_spec=g)
    base = xhw.BaselineResult.from_dataset(
        base.dataset.isel(along=[0] * 40).assign_coords(along=ds["along"]), grid=g,
        baseline=base.baseline, definition=base.definition, provenance=base.provenance)
    lazy_ds = ds.copy()
    lazy_ds["sst"] = ds["sst"].chunk({"time": 50})

    def forbid(*args, **kwargs):
        raise AssertionError("computed during detect")

    with dask.config.set(scheduler=forbid):
        xhw.detect(lazy_ds, variable="sst", grid=g, definition=base.definition,
                   baseline_result=base)


# --- consistency checks -----------------------------------------------------------------------

def test_baseline_must_match_definition_and_units():
    fixture = ALL["F1"]()
    base = constant_baseline(fixture.mask, threshold=12.0, reference=10.0)
    with pytest.raises(xhw.ValidationError, match="different DefinitionSpec"):
        run_detect(fixture.mask, base, definition=hobday(percentile=95.0))
    ds = fixture.mask.dataset.copy()
    ds["temperature"] = ds["temperature"] + 273.15
    ds["temperature"].attrs = {"units": "K"}
    with pytest.raises(xhw.UnitsError):
        run_detect(fixture.mask, base, ds=ds)


def test_fit_then_detect_end_to_end():
    rng = np.random.default_rng(11)
    first = np.datetime64("2000-01-01", "D")
    times = np.arange(first, first + np.timedelta64(1096, "D")).astype("datetime64[ns]")
    ds = xr.Dataset({"sst": ("time", 15 + rng.normal(0, 1, 1096), {"units": "degC"}),
                     "domain_mask": ((), True)},
                    coords={"time": times, "lat": -12.0, "lon": -77.5})
    spec = baseline(reference_period=("2000-01-01", "2002-12-31"), min_samples=20)
    result = xhw.fit_baseline(ds, variable="sst", grid=grid(), baseline=spec, definition=hobday())
    mask = xhw.detect(ds, variable="sst", grid=grid(), definition=hobday(),
                      baseline_result=result)
    mask.check_invariants()
    assert mask.provenance.parents == (result.provenance.run_id,)
