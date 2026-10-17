# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Atmospheric thresholds, warm runs and bootstrap rates against stored climdex outputs.

Interior thresholds are exact; year-edge and leap-day convention differences are explicit.
"""

import dataclasses
import hashlib
import json
import warnings

import _climdex_case as case
import numpy as np
import pandas as pd
import pytest

import xhw
from xhw._baseline import doy_positions

REF = case.REF


def reference(name):
    path = REF / name
    if not (REF / "manifest.json").exists() or not path.exists():
        pytest.fail("missing climdex reference files: reinstall the complete source distribution "
                    "")
    manifest = json.loads((REF / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name], name
    return path


def test_inputs_are_the_registered_synthetic_series():
    reference("inputs.csv")
    expected = case.make_inputs()
    stored = pd.read_csv(case.INPUTS, na_values="NA")
    assert stored["date"].tolist() == expected["date"].tolist()
    for name in ("tmax", "tmin"):
        np.testing.assert_allclose(stored[name].to_numpy(), expected[name].to_numpy())


@pytest.fixture(scope="module")
def runs():
    ds = case.read_inputs()
    return ds, {"tmax": case.xhw_run(ds, "tmax", "max"), "tmin": case.xhw_run(ds, "tmin", "min")}


@pytest.mark.parametrize("variable,column", [("tmax", "tx90"), ("tmin", "tn90")])
def test_thresholds_equal_on_interior_days(runs, variable, column):
    climdex = pd.read_csv(reference("thresholds.csv"))[column].to_numpy(float)
    result = runs[1][variable][0]
    positions = case.month_day_positions()
    ours = result.dataset["threshold"].values[positions - 1]
    interior = ~np.isin(positions, sorted(case.DIVERGENT))
    np.testing.assert_allclose(ours[interior], climdex[interior], rtol=0, atol=1e-9)
    diff = np.abs(ours - climdex)[~interior]
    print(f"{variable}: {interior.sum()} equal days; divergent days max |diff| = "
          f"{diff.max():.3f} degC, mean {diff.mean():.3f} degC (A5: year edge, 29 Feb)")


@pytest.mark.parametrize("variable,column", [("tmax", "tx_hw"), ("tmin", "tn_hw")])
def test_heatwave_days_differ_only_near_divergent_days(runs, variable, column):
    ds, results = runs
    climpact = pd.read_csv(reference("climpact_hw.csv"))
    ours = pd.Series(results[variable][1].dataset["event_membership"].values,
                     index=pd.DatetimeIndex(ds["time"].values).strftime("%Y-%m-%d"))
    theirs = climpact.set_index("date")[column].astype(bool)
    common = ours.reindex(theirs.index)
    differ = pd.DatetimeIndex(theirs.index[(common != theirs).to_numpy()])
    times = pd.DatetimeIndex(ds["time"].values)
    positions = doy_positions(times.values, "hobday_366_interp")
    flagged = times[np.isin(positions, sorted(case.DIVERGENT | {60}))]
    near = {date + pd.Timedelta(days=k) for date in flagged for k in range(-2, 3)}
    unexplained = [d for d in differ if d not in near]
    print(f"{variable}: {len(differ)} of {len(theirs)} days differ, all near A5 days")
    assert unexplained == []


def test_tx90p_bootstrap_matches_climdex_up_to_a5():
    """B.5 level 3: `in_base_rates` against climdex TX90p with the Zhang bootstrap (5-day window,
    type 8, every year in base). Not exact: the A5 windows (year edge, 29 Feb) differ by
    construction; the bound is the measured one ( max 0.120 pp).
    """
    tx90p = pd.read_csv(reference("tx90p_annual.csv")).set_index("year")["tx90p"]
    ds = case.read_inputs()
    grid, base, definition = case.specs("max")
    base = dataclasses.replace(base, doy_window_days=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xhw.XHWDataQualityWarning)
        result = xhw.fit_baseline(ds, variable="tmax", grid=grid, baseline=base,
                                  definition=definition)
    rates = xhw.in_base_rates(ds, variable="tmax", grid=grid, baseline_result=result)
    ours = rates["exceedance_frequency"].to_series() * 100
    assert ours.index.tolist() == tx90p.index.tolist()
    diff = ours.to_numpy() - tx90p.to_numpy()
    print(f"TX90p: max |diff| {np.abs(diff).max():.3f} pp, mean {diff.mean():+.3f} pp, "
          f"{int((np.abs(diff) < 1e-9).sum())} of {diff.size} years equal")
    assert np.abs(diff).max() < 0.15
    assert abs(diff.mean()) < 0.06
