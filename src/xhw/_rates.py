# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""in_base_rates: annual frequencies with the bootstrap of Zhang et al. (2005).

Zhang's original correction does not produce a binary mask: for each reference year it
averages the exceedance rate over m - 1 replicas whose threshold replaces that year by each
other reference year. The result is a plain Dataset of annual frequencies, not a catalog
entity.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import xarray as xr

from ._baseline import BaselineResult, _fit
from ._detect import detect
from ._errors import XHWDataQualityWarning
from ._schema import CONVENTIONS, require
from ._specs import GridSpec, ProcessingStep, Provenance

YEAR_DIM = "year"


def in_base_rates(ds: xr.Dataset, *, variable: str | tuple[str, ...], grid: GridSpec,
                  baseline_result: BaselineResult,
                  provenance: Provenance | None = None) -> xr.Dataset:
    """Annual exceedance and event-day frequencies per location, Zhang et al. (2005).

    For every calendar year of `ds`, ``exceedance_frequency`` and ``event_day_frequency`` are
    the fractions of valid days that exceed the threshold or belong to an event (only valid
    days are counted). Each is the mean over the m - 1 replicas in which every reference year
    takes the threshold fitted with that year replaced by another reference year; years
    outside the reference keep the ordinary threshold (they differ between replicas only
    through events that cross into a reference year). `ds` must be the data `baseline_result`
    was fitted on: the ordinary threshold is refitted and compared. Cost: about m^2 threshold
    fits and m - 1 runs of `detect`.
    """
    require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
    require(isinstance(baseline_result, BaselineResult), "baseline_result must be a BaselineResult")
    baseline, definition = baseline_result.baseline, baseline_result.definition
    require(baseline.method in ("fixed_climatology", "detrended", "shifting", "stl",
                                "state_space", "gam", "ssa")
            and definition.threshold_kind == "percentile",
            "in_base_rates needs a fixed or implemented dynamic percentile baseline ")
    zhang = dataclasses.replace(baseline, in_base_correction="mean_threshold")
    replicas = len(baseline.reference_years) - 1
    tdim = grid.time_dim
    totals = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XHWDataQualityWarning)  # already raised by fit_baseline
        for k in range(replicas):
            fit = _fit(ds, variable=variable, grid=grid, baseline=zhang, definition=definition,
                       provenance=baseline_result.provenance, replica=k)
            if k == 0:
                same = np.array_equal(fit.dataset["threshold"].values,
                                      baseline_result.dataset["threshold"].values, equal_nan=True)
                require(same, "baseline_result was not fitted on these data (threshold differs)")
            mask = detect(ds, variable=variable, grid=grid, definition=definition,
                          baseline_result=fit, provenance=fit.provenance).dataset
            year = mask[tdim].dt.year.rename(YEAR_DIM)
            valid = mask["valid"]
            n_valid = valid.groupby(year).sum(tdim)
            parts = (n_valid,
                     mask["exceedance"].groupby(year).sum(tdim) / n_valid,
                     (mask["event_membership"] & valid).groupby(year).sum(tdim) / n_valid)
            totals = parts if totals is None else tuple(a + b for a, b in zip(totals, parts))
    n_valid, exceed, event = (a / replicas for a in totals)
    dims = (YEAR_DIM, *grid.spatial_dims)
    out = xr.Dataset({
        "exceedance_frequency": exceed.transpose(*dims).assign_attrs(
            long_name="fraction of valid days above the threshold", units="1"),
        "event_day_frequency": event.transpose(*dims).assign_attrs(
            long_name="fraction of valid days inside an event", units="1"),
        "valid_days": n_valid.transpose(*dims).assign_attrs(
            long_name="valid days per year (mean over replicas)"),
    })
    if provenance is None:
        provenance = Provenance.new(
            parents=(baseline_result.provenance.run_id,),
            steps=(ProcessingStep(name="in_base_rates",
                                  params=(("method", "zhang2005"), ("replicas", replicas))),))
    out.attrs.update({
        "Conventions": CONVENTIONS,
        "xhw_baseline": baseline.to_json(),
        "xhw_definition": definition.to_json(),
        "xhw_provenance": provenance.to_json(),
        "baseline_id": baseline.spec_id,
        "definition_id": definition.spec_id,
        "in_base_method": "zhang2005",
    })
    return out
