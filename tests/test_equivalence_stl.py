# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""STL against stored R stats::stl components on the 366-position grid.

Maximum error is 1e-9; stlplus is a contrast with a different algorithm.
"""

import hashlib
import json
import warnings

import _stl_case as case
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from _fixtures import baseline, grid, hobday  # noqa: E402

import xhw

pytest.importorskip("statsmodels")


def reference(name):
    path = case.REF / name
    if not (case.REF / "manifest.json").exists() or not path.exists():
        pytest.fail("missing STL reference files: reinstall the complete source distribution")
    manifest = json.loads((case.REF / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name], name
    return pd.read_csv(path)


def test_inputs_are_the_registered_grid():
    np.testing.assert_allclose(reference("inputs.csv")["x"].to_numpy(), case.on_grid(),
                               rtol=0, atol=1e-12)


ENGINES = [(name, engine) for name in case.CASES for engine in ("statsmodels", "vectorized")
           if not (case.CASES[name][1] and engine == "vectorized")]  # robust: statsmodels only


@pytest.mark.parametrize("name,engine", ENGINES)
def test_stl_equals_stats_stl(name, engine):
    s_window, robust = case.CASES[name]
    ref = reference(f"{name}.csv")
    series = case.series()
    ds = xr.Dataset({"sst": ("time", series.to_numpy(), {"units": "degC"}),
                     "domain_mask": ((), True)},
                    coords={"time": series.index.to_numpy(), "lat": 0.0, "lon": 0.0})
    spec = baseline(method="stl", reference_period=("2006-01-01", "2015-12-31"),
                    smoothing_window_days=None, min_samples=1,
                    params=(("trend_window_days", case.TREND_WINDOW_DAYS),
                            ("seasonal_window", s_window), ("robust", robust),
                            ("engine", engine)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xhw.XHWDataQualityWarning)
        result = xhw.fit_baseline(ds, variable="sst", grid=grid(), baseline=spec,
                                  definition=hobday()).dataset
    slots = case.grid_index(series.index)
    for component in ("trend", "seasonal"):
        ours = result[component].to_numpy()
        diff = np.abs(ours - ref[f"stl_{component}"].to_numpy()[slots])
        contrast = np.abs(ours - ref[f"stlplus_{component}"].to_numpy()[slots])
        print(f"{name} {engine} {component}: max |xHW - stats::stl| {diff.max():.1e}; "
              f"vs stlplus {contrast.max():.1e}")
        assert diff.max() < 1e-9
