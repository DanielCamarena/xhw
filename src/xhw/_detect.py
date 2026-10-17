# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""interpolate_gaps and detect: detection masks from a fitted baseline.

Mask construction is local in time: event membership and joined gaps of a day depend only on
days within `min_duration_days + max_gap_days`, so dask inputs may be chunked in time and are
processed with a bounded overlap. Baselines, in contrast, need whole series.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from ._aggregate import daily_steps
from ._baseline import (
    BaselineResult,
    check_input,
    daily_statistics,
    input_variables,
)
from ._errors import UnitsError
from ._mask import (
    BASELINE_UNDEFINED,
    COMPONENT,
    INSUFFICIENT_SAMPLES,
    LAG_COORD,
    LOW_DATA_QUALITY,
    MISSING_DATA,
    ON_SCALE,
    OUTSIDE_DOMAIN,
    XMask,
    runs,
)
from ._schema import is_int, is_number, require, temperature_units
from ._specs import DefinitionSpec, GridSpec, ProcessingStep, Provenance

_COMPARE = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal}
_STEP_ATTR = "xhw_step"


# --- time-local kernels (per series, vectorized over locations) ---------------------------


def _interp_core(values: np.ndarray, *, max_days: int) -> np.ndarray:
    """Linear interpolation of interior NaN blocks of length <= max_days; never at the edges."""
    out = np.array(values, dtype=float, copy=True)
    flat = out.reshape(-1, out.shape[-1])
    last = flat.shape[1] - 1
    for series in flat:
        for s, e in runs(np.isnan(series)):
            n = e - s + 1
            if s == 0 or e == last or n > max_days:
                continue
            a, b = series[s - 1], series[e + 1]
            series[s:e + 1] = a + (b - a) * np.arange(1, n + 1) / (n + 1)
    return out


def _membership_core(exc: np.ndarray, eligible: np.ndarray, *, min_days: int, max_gap: int,
                     rule: str) -> np.ndarray:
    """Pack event membership (bit 0) and joined gaps (bit 1) per series."""
    e = np.asarray(exc, dtype=bool).reshape(-1, exc.shape[-1])
    ok = np.asarray(eligible, dtype=bool).reshape(-1, exc.shape[-1])
    packed = np.zeros(e.shape, dtype=np.uint8)
    for i in range(e.shape[0]):
        pieces = runs(e[i])
        if rule == "qualify_then_join":
            kept = [(s, t) for s, t in pieces if t - s + 1 >= min_days]
            for s, t in kept:
                packed[i, s:t + 1] |= 1
            for (_, end), (start, _) in zip(kept, kept[1:]):
                width = start - end - 1
                if 0 < width <= max_gap and ok[i, end + 1:start].all():
                    packed[i, end + 1:start] |= 3
        else:  # join_then_qualify
            merged: list[list[int]] = []
            for s, t in pieces:
                if (merged and 0 < s - merged[-1][1] - 1 <= max_gap
                        and ok[i, merged[-1][1] + 1:s].all()):
                    merged[-1][1] = t
                else:
                    merged.append([s, t])
            for s, t in merged:
                if t - s + 1 >= min_days:
                    packed[i, s:t + 1] |= 1
                    packed[i, s:t + 1] |= np.where(e[i, s:t + 1], 0, 2).astype(np.uint8)
    return packed.reshape(exc.shape)


def persist_array(field: np.ndarray, eligible: np.ndarray, *, min_days: int, max_gap: int,
                  rule: str) -> tuple[np.ndarray, np.ndarray]:
    """The temporal operator T (minimum duration and gap joining) on any boolean field with time
    on the first axis: ``(membership, joined_gap)`` of the same shape.

    `detect` applies it to the exceedance it computes; object tracking recipes apply it to an
    already processed field (``exceedance -> S -> T``). `eligible` marks the days that may be
    bridged.
    """
    moved = np.ascontiguousarray(np.moveaxis(np.asarray(field, bool), 0, -1))
    ok = np.ascontiguousarray(np.moveaxis(np.asarray(eligible, bool), 0, -1))
    packed = np.moveaxis(_membership_core(moved, ok, min_days=min_days, max_gap=max_gap,
                                          rule=rule), -1, 0)
    return (packed & 1) != 0, (packed & 2) != 0


def _along_time(func, arrays: list[xr.DataArray], time_dim: str, depth: int, dtype):
    """Apply a time-local kernel; dask inputs use map_overlap with `depth` days of halo."""
    template = arrays[0]
    dims = [d for d in template.dims if d != time_dim] + [time_dim]
    aligned = [a.transpose(*dims) for a in arrays]
    lazy = any(a.chunks is not None for a in aligned)
    if lazy:
        import dask.array as dsa

        chunks = next(a.data.chunks for a in aligned if a.chunks is not None)
        data = [a.data.rechunk(chunks) if a.chunks is not None
                else dsa.from_array(np.asarray(a.data), chunks=chunks) for a in aligned]
        if depth:
            result = dsa.map_overlap(func, *data, depth={len(dims) - 1: depth},
                                     boundary="none", dtype=dtype, trim=True)
        else:
            result = dsa.map_blocks(func, *data, dtype=dtype)
    else:
        result = func(*[np.asarray(a.data) for a in aligned])
    return xr.DataArray(result, dims=dims, coords=aligned[0].coords).transpose(*template.dims)


# --- public functions ------------------------------------------------------------------------


def interpolate_gaps(ds: xr.Dataset, *, variable: str, grid: GridSpec, max_days: int,
                     max_filled_fraction: float) -> xr.Dataset:
    """Fill interior gaps of at most `max_days` days by linear interpolation.

    Series edges are never extrapolated. Adds `filled` (bool), `filled_fraction` and
    `low_data_quality` (series whose filled fraction exceeds `max_filled_fraction`); `detect`
    invalidates those series (bit 16) and excludes nothing else. Baselines ignore filled days.
    """
    require(isinstance(ds, xr.Dataset) and isinstance(grid, GridSpec),
            "expected an xarray.Dataset and a GridSpec")
    require(is_int(max_days) and max_days >= 1, "max_days must be an int >= 1")
    require(is_number(max_filled_fraction) and 0.0 <= max_filled_fraction <= 1.0,
            "max_filled_fraction must be in [0, 1]")
    require("filled" not in ds, "dataset already has a 'filled' variable (fill only once)")
    grid.check(ds)
    require(variable in ds.data_vars, f"variable {variable!r} not found")
    original = ds[variable]
    require(set(original.dims) == {grid.time_dim, *grid.spatial_dims},
            f"{variable!r} dims {original.dims} do not match the grid")
    filled_values = _along_time(
        lambda v: _interp_core(v, max_days=max_days), [original], grid.time_dim,
        max_days + 1, float)
    filled = original.isnull() & filled_values.notnull()
    fraction = filled.mean(grid.time_dim)
    domain = ds[grid.domain_mask]
    step = {_STEP_ATTR: "interpolate_gaps", "method": "linear", "max_days": int(max_days),
            "max_filled_fraction": float(max_filled_fraction)}
    out = ds.copy()
    out[variable] = filled_values.assign_attrs(original.attrs)
    out["filled"] = filled.assign_attrs(step, long_name="interpolated day")
    out["filled_fraction"] = fraction.assign_attrs(long_name="fraction of interpolated days")
    out["low_data_quality"] = ((fraction > max_filled_fraction) & domain).assign_attrs(
        long_name="series with too many interpolated days")
    return out


def detect(ds: xr.Dataset, *, variable: str | tuple[str, ...], grid: GridSpec,
           definition: DefinitionSpec, baseline_result: BaselineResult,
           provenance: Provenance | None = None) -> XMask:
    """Build the XMask of `definition` using a fitted `baseline_result`.

    For ``combination="and_daily"``, `variable` is a tuple aligned with the components: each
    component is compared with its own threshold on its own date, its exceedance is shifted by
    ``lag_days`` to the reference day, the daily exceedances are combined with AND and only
    then persistence and gaps are applied. With a lag, the last day is invalid.
    """
    require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
    require(isinstance(baseline_result, BaselineResult), "baseline_result must be a BaselineResult")
    require(baseline_result.definition_id == definition.spec_id,
            "baseline_result was fitted for a different DefinitionSpec")
    require(baseline_result.grid.spatial_dims == grid.spatial_dims,
            "baseline_result and grid have different spatial dims")
    grid.check(ds)
    names = input_variables(variable, definition)
    units = check_input(ds, names, grid)
    daily = daily_steps(ds, names, daily_statistics(definition), definition.day_definition)
    base = baseline_result.dataset
    for dim in grid.spatial_dims:
        require(base.sizes[dim] == ds.sizes[dim], f"baseline and data differ in size along {dim}")
    if units != temperature_units("threshold", base["threshold"].attrs.get("units")):
        raise UnitsError("data and baseline temperature units differ; convert explicitly")

    combined = definition.combination == "and_daily"
    temps = [ds[name] for name in names]
    tdim = grid.time_dim
    chunks = None
    if temps[0].chunks is not None:  # keep the expansion lazy
        chunks = {d: temps[0].chunksizes[d] for d in grid.spatial_dims}
    # in-base years take their own threshold when the baseline is corrected
    onto = baseline_result._onto(ds[tdim].values, chunks).assign_coords({tdim: ds[tdim]})

    domain = ds[grid.domain_mask]
    common = xr.where(~domain, OUTSIDE_DOMAIN, 0)
    if "low_data_quality" in ds:
        common = common | xr.where(ds["low_data_quality"], LOW_DATA_QUALITY, 0)
    lags = [c.lag_days for c in definition.components] if combined else [0]
    reason, exceedance = None, None
    for i, (temp, lag) in enumerate(zip(temps, lags)):
        part = onto.isel({COMPONENT: i}) if combined else onto
        r = common | xr.where(domain & temp.isnull(), MISSING_DATA, 0)
        r = (r | (part["invalid_reason"] & (BASELINE_UNDEFINED | INSUFFICIENT_SAMPLES)))
        r = r.transpose(*temp.dims).astype(np.uint8)
        e = xr.apply_ufunc(_COMPARE[definition.comparator], temp, part["threshold"],
                           dask="allowed").transpose(*temp.dims) & (r == 0)
        if lag:  # value of day D+lag decides day D; the last `lag` days lack it
            e = e.shift({tdim: -lag}, fill_value=False)
            r = r.shift({tdim: -lag}, fill_value=MISSING_DATA).astype(np.uint8)
        reason = r if reason is None else reason | r
        exceedance = e if exceedance is None else exceedance & e
    if combined:
        reason = xr.where(domain, reason, OUTSIDE_DOMAIN)
    reason = reason.transpose(*temps[0].dims).astype(np.uint8)
    valid = reason == 0
    exceedance = exceedance & valid
    eligible = valid | ((reason == MISSING_DATA)
                        & (definition.missing_policy == "as_non_exceedance"))
    min_days, max_gap = definition.min_duration_days, definition.max_gap_days
    if min_days == 1 and max_gap == 0:
        membership, joined = exceedance, xr.zeros_like(exceedance)
    else:
        packed = _along_time(
            lambda e, ok: _membership_core(e, ok, min_days=min_days, max_gap=max_gap,
                                           rule=definition.join_rule),
            [exceedance, eligible], tdim, min_days + max_gap, np.uint8)
        membership, joined = (packed & 1) != 0, (packed & 2) != 0

    dims = temps[0].dims
    temp_attrs = {"units": temps[0].attrs["units"], "units_metadata": ON_SCALE}
    if combined:
        labels = [c.variable for c in definition.components]
        temperature = xr.concat(temps, dim=COMPONENT, combine_attrs="drop").assign_coords(
            {COMPONENT: labels}).assign_attrs(temp_attrs)
        comp_dims = (COMPONENT, *dims)
        temperature = temperature.transpose(*comp_dims)
        threshold = onto["threshold"].transpose(*comp_dims)
        reference = onto["reference"].transpose(*comp_dims)
    else:
        temperature = temps[0].assign_attrs(temp_attrs, standard_name=definition.variable)
        threshold = onto["threshold"].transpose(*dims)
        reference = onto["reference"].transpose(*dims)
    out = xr.Dataset({
        "temperature": temperature,  # shallow copy: caller attrs untouched
        "threshold": threshold,
        "reference": reference,
        "exceedance": exceedance.transpose(*dims),
        "event_membership": membership.transpose(*dims),
        "joined_gap": joined.transpose(*dims),
        "valid": valid,
        "invalid_reason": reason,
    })
    if combined:
        out = out.assign_coords({LAG_COORD: (COMPONENT, np.array(lags, dtype=np.int64))})
    statics = [grid.domain_mask, grid.domain_fraction, grid.cell_measure.var,
               grid.vertical.thickness if grid.vertical else None]
    for name in (*statics, "filled", "segment"):
        if name is not None and name in ds.data_vars:
            out[name] = ds[name]
    if provenance is None:
        steps = list(daily)
        if "filled" in ds and _STEP_ATTR in ds["filled"].attrs:
            attrs = ds["filled"].attrs
            steps.append(ProcessingStep(name=attrs[_STEP_ATTR], params=tuple(
                (k, attrs[k]) for k in ("method", "max_days", "max_filled_fraction"))))
        steps.append(ProcessingStep(name="detect", params=(("variable", ",".join(names)),)))
        provenance = Provenance.new(parents=(baseline_result.provenance.run_id,),
                                    steps=tuple(steps))
    return XMask.from_dataset(out, grid=grid, definition=definition,
                                    baseline=baseline_result.baseline, provenance=provenance)
