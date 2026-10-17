# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Level 2 of B.5: the in-base step disappears with correction.

Stationary synthetic data with strong interannual persistence (one random offset per year and
site, sd 1, plus daily noise, sd 0.5): 10 reference years and 10 later years, 300 independent
sites, fixed seed. Without correction the reference years exceed P90 exactly 10 % of the time
and later years more often (the self-censoring of docs/science/in_base_inhomogeneity.md §2-3);
each correction must remove that step within its noise. Runtime about a minute.
"""

import warnings

import numpy as np
import pytest
import xarray as xr
from _fixtures import ahw_baseline, grid, perkins2013  # noqa: E402

import xhw  # noqa: E402

SITES, SEED, BASE_YEARS = 300, 1, 10


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(SEED)
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2020-01-01")).astype(
        "datetime64[ns]")
    years = times.astype("datetime64[Y]").astype(int) + 1970
    offset = rng.normal(0, 1.0, (20, SITES))
    tx = offset[years - 2000] + rng.normal(0, 0.5, (times.size, SITES))
    ds = xr.Dataset({"tx": (("time", "site"), tx, {"units": "degC"}),
                     "domain_mask": ("site", np.ones(SITES, bool)),
                     "length": ("site", np.ones(SITES), {"units": "m"})},
                    coords={"time": times, "site": ("site", np.arange(SITES, dtype=float),
                                                    {"units": "m"})})
    g = grid(support="line", topology="transect", spatial_dims=("site",), domain="land",
             coords=xhw.CoordRoles(along="site"),
             cell_measure=xhw.CellMeasure(kind="length", var="length", units="m"))
    return ds, g, perkins2013(variable="tx", min_duration_days=1)


def fit(case, correction):
    ds, g, definition = case
    base = ahw_baseline(reference_period=("2000-01-01", f"{1999 + BASE_YEARS}-12-31"),
                        doy_window_days=5, min_samples=1, in_base_correction=correction)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xhw.XHWDataQualityWarning)
        return xhw.fit_baseline(ds, variable="tx", grid=g, baseline=base, definition=definition)


def step(rates):
    """Out-of-base minus in-base mean annual rate (pp) per site -> mean and standard error."""
    rates = np.asarray(rates) * 100
    per_site = rates[BASE_YEARS:].mean(0) - rates[:BASE_YEARS].mean(0)
    return per_site.mean(), per_site.std(ddof=1) / np.sqrt(per_site.size)


def mask_rates(case, correction):
    ds, g, definition = case
    mask = xhw.detect(ds, variable="tx", grid=g, definition=definition,
                      baseline_result=fit(case, correction)).dataset
    return mask["exceedance"].groupby("time.year").mean("time").transpose("year", "site")


def test_uncorrected_step_is_detected(case):
    rates = mask_rates(case, "none")
    assert float(rates[:BASE_YEARS].mean()) == pytest.approx(0.10, abs=0.002)  # §2: nominal
    value, se = step(rates)
    print(f"none: step {value:+.2f} pp (SE {se:.2f})")
    assert value > 3 * se


@pytest.mark.parametrize("correction", ["leave_one_year_out", "mean_threshold"])
def test_binary_corrections_remove_the_step(case, correction):
    value, se = step(mask_rates(case, correction))
    print(f"{correction}: step {value:+.2f} pp (SE {se:.2f})")
    assert abs(value) < 2.5 * se


def test_zhang_rates_remove_the_step(case):
    ds, g, _ = case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xhw.XHWDataQualityWarning)
        rates = xhw.in_base_rates(ds, variable="tx", grid=g, baseline_result=fit(case, "none"))
    value, se = step(rates["exceedance_frequency"].transpose("year", "site"))
    print(f"zhang rates: step {value:+.2f} pp (SE {se:.2f})")
    assert abs(value) < 2.5 * se
