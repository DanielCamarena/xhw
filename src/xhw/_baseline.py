# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""BaselineResult and fit_baseline: fixed climatology and percentile threshold (temporal
detection).

Baseline fitting and mask detection are different computations: the baseline uses observed
data only (days marked ``filled`` are excluded), a window strictly inside the reference
period, and needs the whole time series per cell (exact percentiles, ). With
``BaselineSpec.in_base_correction`` each year of the reference period also gets its own
threshold, estimated without that year's dates.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr

from ._aggregate import daily_steps
from ._errors import (
    InvariantError,
    NotSupportedError,
    UnitsError,
    ValidationError,
    XHWDataQualityWarning,
)
from ._mask import (
    COMPONENT,
    ON_SCALE,
    check_component_coords,
    check_entity,
    check_stored_id,
    load_attr_json,
    read_only,
    strip_xhw_attrs,
)
from ._schema import (
    CONVENTIONS,
    SCHEMA_VERSIONS,
    check_serializable_attrs,
    parse_iso_utc,
    require,
    temperature_units,
)
from ._specs import (
    NUMPY_PERCENTILE_METHOD,
    BaselineSpec,
    DefinitionSpec,
    GridSpec,
    ProcessingStep,
    Provenance,
)

DOY = "dayofyear"
#: dimension of the in-base thresholds
YEAR = "reference_year"
BY_YEAR = ("threshold_by_year", "sample_count_by_year", "invalid_reason_by_year")
#: dynamic baselines correct the residual percentile, not the absolute threshold
BY_YEAR_DYNAMIC = ("residual_threshold_by_year", "sample_count_by_year",
                   "invalid_reason_by_year")


def time_layout(ds: xr.Dataset) -> bool:
    """True for a baseline on the time axis, recognised by its ``trend`` variable. Day-of-year
    results with a dynamic method name stay valid.
    """
    return "trend" in ds.data_vars


def by_year_names(ds: xr.Dataset) -> tuple[str, str, str]:
    return BY_YEAR_DYNAMIC if time_layout(ds) else BY_YEAR
N_DOY = 366
FEB29 = 60
#: invalid_reason bits used by baselines.
OUTSIDE_DOMAIN, BASELINE_UNDEFINED, INSUFFICIENT_SAMPLES = 1, 4, 8
BASELINE_REASON_MASKS = np.array([1, 4, 8], dtype=np.uint8)
BASELINE_REASON_MEANINGS = "outside_domain baseline_undefined insufficient_samples"
#: day offsets of each month in a leap year: 1 March is always position 61.
_LEAP_MONTH_OFFSETS = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
_KELVIN = 273.15
_TARGET_CHUNK_MB = 200.0
_BY_YEAR_WARN_BYTES = 1e9


# --- day-of-year positions ------------------------------------------------------------------


def doy_positions(times, policy: str) -> np.ndarray:
    """Day-of-year position (1..366) of each time under a leap-day policy."""
    index = pd.DatetimeIndex(np.asarray(times, dtype="datetime64[ns]"))
    if policy == "hobday_366_interp":
        return _LEAP_MONTH_OFFSETS[index.month.to_numpy() - 1] + index.day.to_numpy()
    if policy == "calendar_doy":
        return index.dayofyear.to_numpy()
    raise ValidationError(f"leap_day_policy {policy!r} has no day-of-year positions")


def _window_indices(pos: np.ndarray, in_ref: np.ndarray, half: int) -> dict[int, np.ndarray]:
    """Time indices pooled for each day of year: +/- `half` positions, strictly in the period."""
    offsets = np.arange(-half, half + 1)
    sets = {}
    for d in range(1, N_DOY + 1):
        base = np.flatnonzero((pos == d) & in_ref)
        cand = (base[:, None] + offsets).ravel()
        cand = cand[(cand >= 0) & (cand < pos.size)]
        sets[d] = cand[in_ref[cand]]
    return sets


# --- numerical core (per series, vectorized over locations) --------------------------------


def _reasons(count: np.ndarray, min_samples: int) -> np.ndarray:
    reason = np.where(count < min_samples, INSUFFICIENT_SAMPLES, 0)
    reason = np.where(count == 0, reason | BASELINE_UNDEFINED, reason)
    return reason.astype(np.uint8)


def _stats(sample: np.ndarray, pct: float, method: str):
    count = np.isfinite(sample).sum(axis=0).astype(np.int32)
    if sample.shape[0] == 0:
        nan = np.full(sample.shape[1], np.nan)
        return nan, nan.copy(), count
    complete = count == sample.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> NaN, flagged
        if complete.all():  # same values; nanpercentile loops over columns in Python
            thr = np.percentile(sample, pct, axis=0, method=method)
        else:
            thr = np.empty(sample.shape[1])
            thr[complete] = np.percentile(sample[:, complete], pct, axis=0, method=method)
            thr[~complete] = np.nanpercentile(sample[:, ~complete], pct, axis=0, method=method)
        ref = np.nanmean(sample, axis=0)
    return thr, ref, count


def _circular_window(values: np.ndarray, width: int) -> np.ndarray:
    """Sum over a centred circular window along axis 0 (odd width)."""
    half = width // 2
    padded = np.concatenate([values[-half:], values, values[:half]], axis=0) if half else values
    zero = np.zeros((1,) + values.shape[1:], dtype=padded.dtype)
    csum = np.cumsum(np.concatenate([zero, padded], axis=0), axis=0)
    return csum[width:] - csum[:-width]


def _smooth(thr, ref, reason, width):
    """Strict circular moving average: a position is valid only if all inputs are valid."""
    valid = reason == 0
    n_valid = _circular_window(valid.astype(np.int64), width)
    smooth = [np.where(n_valid == width,
                       _circular_window(np.where(valid, a, 0.0), width) / width, np.nan)
              for a in (thr, ref)]
    out_reason = np.zeros_like(reason)
    for bit in (BASELINE_UNDEFINED, INSUFFICIENT_SAMPLES):
        present = _circular_window(((reason & bit) != 0).astype(np.int64), width) > 0
        out_reason = np.where(present, out_reason | bit, out_reason)
    return smooth[0], smooth[1], out_reason.astype(np.uint8)


def _fit_core(values, *, sets, doy_mode, feb29, pct, method, smooth, min_samples, absolute):
    lead = values.shape[:-1]
    series = np.asarray(values, dtype=float).reshape(-1, values.shape[-1]).T  # (T, N)
    if doy_mode:
        n = series.shape[1]
        thr = np.full((N_DOY, n), np.nan)
        ref = np.full((N_DOY, n), np.nan)
        count = np.zeros((N_DOY, n), dtype=np.int32)
        for d, idx in sets.items():
            if feb29 and d == FEB29:
                continue
            thr[d - 1], ref[d - 1], count[d - 1] = _stats(series[idx], pct, method)
        reason = _reasons(count, min_samples)
        if feb29:  # interpolated from 28 Feb and 1 Mar before smoothing
            a, b = FEB29 - 2, FEB29
            thr[FEB29 - 1] = 0.5 * (thr[a] + thr[b])
            ref[FEB29 - 1] = 0.5 * (ref[a] + ref[b])
            count[FEB29 - 1] = np.minimum(count[a], count[b])
            reason[FEB29 - 1] = reason[a] | reason[b]
        thr[reason != 0] = np.nan
        ref[reason != 0] = np.nan
        if smooth:
            thr, ref, reason = _smooth(thr, ref, reason, smooth)
        shape = lead + (N_DOY,)
        outs = [a.T.reshape(shape) for a in (thr, ref, count, reason)]
    else:
        thr, ref, count = _stats(series[sets[0]], pct, method)
        reason = _reasons(count, min_samples)
        outs = [a.reshape(lead) for a in (thr, ref, count, reason)]
    thr_out, ref_out, count_out, reason_out = outs
    if absolute is not None:
        thr_out = np.where(reason_out == 0, absolute, np.nan)
    ref_out = np.where(reason_out == 0, ref_out, np.nan)
    thr_out = np.where(reason_out == 0, thr_out, np.nan)
    return thr_out, ref_out, count_out.astype(np.int32), reason_out.astype(np.uint8)


def _year_sets(sets, year_of, years, correction, replica=None):
    """Sample sets of the replicas of each reference year.

    The dates of year y are removed (``leave_one_year_out``: one replica) or replaced by the
    dates of each other year j (``mean_threshold``: m - 1 replicas, Zhang et al. 2005). Dates
    are matched by calendar year, so a January window keeps the December of the year before.
    With `replica` = k only the k-th other year is used.
    """
    out = []
    for y in years:
        kept = {d: idx[year_of[idx] != y] for d, idx in sets.items()}
        if correction == "leave_one_year_out":
            out.append([kept])
            continue
        others = [j for j in years if j != y]
        if replica is not None:
            others = [others[replica]]
        out.append([{d: np.concatenate([kept[d], idx[year_of[idx] == j]])
                     for d, idx in sets.items()} for j in others])
    return out


def _fit_years_core(values, *, year_sets, **kwargs):
    """Threshold, sample count and invalid_reason per reference year (axis before DOY).

    With several replicas the threshold is their mean; the count is the smallest and the
    reasons are combined, so a year is invalid wherever one of its replicas is.
    """
    outs = []
    for replicas in year_sets:
        thr = count = reason = None
        for sets in replicas:
            t, _, c, r = _fit_core(values, sets=sets, **kwargs)
            thr = t if thr is None else thr + t
            count = c if count is None else np.minimum(count, c)
            reason = r if reason is None else reason | r
        outs.append((thr / len(replicas), count, reason))
    axis = -2 if kwargs["doy_mode"] else -1
    return tuple(np.stack([o[k] for o in outs], axis=axis) for k in range(3))


# --- helpers shared with detection ---------------------------------------------------------


def require_single_time_chunk(da: xr.DataArray, time_dim: str, what: str) -> None:
    """Raise a clear error (with a suggested chunking) if `time_dim` is split in chunks."""
    if da.chunks is None:
        return
    time_chunks = da.chunksizes.get(time_dim, ())
    if len(time_chunks) <= 1:
        return
    spatial = [d for d in da.dims if d != time_dim]
    per_cell = da.sizes[time_dim] * da.dtype.itemsize
    cells = max(1, int(_TARGET_CHUNK_MB * 1e6 // per_cell))
    side = max(1, int(cells ** (1.0 / len(spatial)))) if spatial else 1
    suggestion = {time_dim: -1, **{d: min(side, da.sizes[d]) for d in spatial}}
    mb = per_cell * math.prod(suggestion[d] for d in spatial) / 1e6 if spatial else per_cell / 1e6
    raise ValidationError(
        f"{what} needs the whole time series per cell, but {time_dim!r} has "
        f"{len(time_chunks)} chunks. Suggested: ds = ds.chunk({suggestion}) "
        f"(~{mb:.0f} MB per chunk). Not rechunked automatically."
    )


def _reference_mask(times: np.ndarray, baseline: BaselineSpec) -> np.ndarray:
    start = np.datetime64(parse_iso_utc("reference start", baseline.reference_period[0]), "ns")
    end = parse_iso_utc("reference end", baseline.reference_period[1]) + timedelta(days=1)
    return (times >= start) & (times < np.datetime64(end, "ns"))


def _to_units(value: float, units: str, target: str) -> float:
    if units == target:
        return value
    return value + _KELVIN if target == "K" else value - _KELVIN


# --- quality warnings ----------------------------------------------------------------


def _quality_warnings(baseline, definition, pct, in_ref, doy_mode, count, domain) -> None:
    start = parse_iso_utc("start", baseline.reference_period[0])
    end = parse_iso_utc("end", baseline.reference_period[1])
    years = ((end - start).days + 1) / 365.25
    possible = round(years) * baseline.doy_window_days if doy_mode else int(in_ref.sum())
    recommended = math.ceil(0.8 * possible)
    notes = []
    if doy_mode and years < 29.5:
        notes.append(f"reference period covers {years:.1f} years; Hobday et al. (2016) "
                     "recommend at least 30")
    if baseline.min_samples > possible:
        notes.append(f"min_samples={baseline.min_samples} exceeds the {possible} possible samples:"
                     " every threshold will be invalid")
    elif baseline.min_samples < recommended:
        notes.append(f"min_samples={baseline.min_samples} is below the recommended {recommended} "
                     f"(80% of {possible} possible samples; completeness criterion to be verified "
                     "against WMO-No. 1203)")
    if count is not None:
        inside = np.broadcast_to(domain, count.shape)
        low = (count < recommended) & inside
        if low.any():
            notes.append(f"{100 * low.sum() / inside.sum():.1f}% of in-domain thresholds rest on "
                         f"fewer than {recommended} samples")
    if not notes:
        return
    if pct is not None:
        q = (100.0 - pct) / 100.0 if definition.tail == "upper" else pct / 100.0
        for n in sorted({max(baseline.min_samples, 1), recommended}):
            rel_se = 100 * math.sqrt((1 - q) / (n * q))
            notes.append(f"with n={n} samples: ~{n * q:.0f} expected exceedances, relative "
                         f"standard error of their frequency ~{rel_se:.0f}%")
    warnings.warn("baseline sampling: " + "; ".join(notes), XHWDataQualityWarning, stacklevel=3)


# --- container -------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BaselineResult:
    """Fitted reference climatology and threshold on a day-of-year axis (1..366) x space.

    Stores the full DefinitionSpec because percentile and tail live there; `detect` requires
    the same definition. `sample_count` is the raw number of observed samples per day of year
    (before smoothing). Build with `fit_baseline` or read with `from_dataset(ds)`.
    """

    _dataset: xr.Dataset
    grid: GridSpec
    baseline: BaselineSpec
    definition: DefinitionSpec
    provenance: Provenance

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, *, grid: GridSpec | None = None,
                     baseline: BaselineSpec | None = None,
                     definition: DefinitionSpec | None = None,
                     provenance: Provenance | None = None, copy: bool = False) -> BaselineResult:
        require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
        given = [grid, baseline, definition, provenance]
        if all(spec is None for spec in given):
            check_entity(ds, "BaselineResult")
            grid = GridSpec.from_dict(load_attr_json(ds, "xhw_grid"))
            baseline = BaselineSpec.from_dict(load_attr_json(ds, "xhw_baseline"))
            definition = DefinitionSpec.from_dict(load_attr_json(ds, "xhw_definition"))
            provenance = Provenance.from_dict(load_attr_json(ds, "xhw_provenance"))
            check_stored_id(ds, "baseline_id", baseline)
            check_stored_id(ds, "definition_id", definition)
        elif any(spec is None for spec in given):
            raise ValidationError("pass all of grid, baseline, definition and provenance, or none")
        ds = strip_xhw_attrs(ds.copy(deep=True) if copy else ds)
        _check_structure(ds, grid, baseline, definition)
        return cls(read_only(ds), grid, baseline, definition, provenance)

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset.copy(deep=False)

    @property
    def baseline_id(self) -> str:
        return self.baseline.spec_id

    @property
    def definition_id(self) -> str:
        return self.definition.spec_id

    @property
    def doy_mode(self) -> bool:
        return self.baseline.leap_day_policy != "not_applicable"

    def expand(self, times) -> xr.Dataset:
        """Threshold, reference and invalid_reason on a time axis (lazy for dask data).

        With an in-base correction, dates inside the reference period take the threshold and
        invalid_reason of their own year; the reference climatology is not corrected.
        """
        return self._onto(times)

    def _onto(self, times, chunks: dict | None = None) -> xr.Dataset:
        times = np.asarray(times, dtype="datetime64[ns]")
        if time_layout(self._dataset):
            return self._onto_dynamic(times, chunks)
        tdim = self.grid.time_dim
        fields = self._dataset[["threshold", "reference", "invalid_reason"]]
        corrected = self.baseline.in_base_correction != "none"
        by_year = self._dataset[["threshold_by_year", "invalid_reason_by_year"]] \
            if corrected else xr.Dataset()
        if chunks:
            fields = fields.chunk(chunks)
            by_year = by_year.chunk(chunks) if corrected else by_year
        index = {}
        if self.doy_mode:
            pos = doy_positions(times, self.baseline.leap_day_policy)
            index[DOY] = xr.DataArray(pos - 1, dims=tdim)
            out = fields.isel(index).drop_vars(DOY)
        else:
            out = fields.expand_dims({tdim: times.size})
        if corrected:
            inside = _reference_mask(times, self.baseline)
            year = pd.DatetimeIndex(times).year.to_numpy()
            first = self.baseline.reference_years[0]
            index[YEAR] = xr.DataArray(np.where(inside, year - first, 0), dims=tdim)
            own = by_year.isel(index).drop_vars(list(index))
            inside = xr.DataArray(inside, dims=tdim)
            for name in ("threshold", "invalid_reason"):
                field = out[name]
                value = xr.where(inside, own[f"{name}_by_year"], field)
                out[name] = value.transpose(*field.dims).astype(field.dtype).assign_attrs(
                    field.attrs)
        return out.assign_coords({tdim: times})

    def _onto_dynamic(self, times, chunks):
        """Dynamic baselines: the fields already live on dates; dates the baseline does not cover
        get NaN and ``baseline_undefined``. With an in-base correction, reference years use
        the residual percentile of their own year.
        """
        ds, tdim = self._dataset, self.grid.time_dim
        corrected = self.baseline.in_base_correction != "none"
        names = ["threshold", "reference", "invalid_reason"]
        fields = ds[names + (["trend", "seasonal"] if corrected else [])]
        by_year = (ds[["residual_threshold_by_year", "invalid_reason_by_year"]]
                   if corrected else xr.Dataset())
        if chunks:
            fields = fields.chunk(chunks)
            by_year = by_year.chunk(chunks) if corrected else by_year
        out = fields.reindex({tdim: times})
        domain = ds[self.grid.domain_mask]
        dims = ds["invalid_reason"].dims
        gap = out["invalid_reason"].isnull()
        out["invalid_reason"] = xr.where(
            gap, xr.where(domain, BASELINE_UNDEFINED, OUTSIDE_DOMAIN),
            out["invalid_reason"].fillna(0)).transpose(*dims).astype(np.uint8)
        if corrected:
            inside = _reference_mask(times, self.baseline)
            year = pd.DatetimeIndex(times).year.to_numpy()
            first = self.baseline.reference_years[0]
            index = {YEAR: xr.DataArray(np.where(inside, year - first, 0), dims=tdim),
                     DOY: xr.DataArray(
                         doy_positions(times, self.baseline.leap_day_policy) - 1, dims=tdim)}
            own = by_year.isel(index).drop_vars([YEAR, DOY])
            reference = out["trend"] + out["seasonal"]
            reason = xr.where(~domain, OUTSIDE_DOMAIN,
                              xr.where(reference.isnull(), BASELINE_UNDEFINED, 0)
                              | own["invalid_reason_by_year"]).astype(np.uint8)
            inside = xr.DataArray(inside, dims=tdim)
            new = {"invalid_reason": reason,
                   "reference": reference.where(reason == 0),
                   "threshold": (reference + own["residual_threshold_by_year"]).where(
                       reason == 0)}
            for name, value in new.items():
                field = out[name]
                out[name] = xr.where(inside, value, field).transpose(*field.dims).astype(
                    field.dtype).assign_attrs(field.attrs)
            out = out.drop_vars(["trend", "seasonal"])
        return out.assign_coords({tdim: times})

    def to_dataset(self) -> xr.Dataset:
        ds = self._dataset.copy(deep=False)
        ds.attrs.update({
            "Conventions": CONVENTIONS,
            "xhw_schema": "BaselineResult",
            "xhw_schema_version": SCHEMA_VERSIONS["BaselineResult"],
            "xhw_grid": self.grid.to_json(),
            "xhw_baseline": self.baseline.to_json(),
            "xhw_definition": self.definition.to_json(),
            "xhw_provenance": self.provenance.to_json(),
            "baseline_id": self.baseline_id,
            "definition_id": self.definition_id,
        })
        ds["invalid_reason"].attrs.update({"flag_masks": BASELINE_REASON_MASKS,
                                           "flag_meanings": BASELINE_REASON_MEANINGS})
        return ds

    def check_invariants(self) -> None:
        """Value invariants (computes the data)."""
        ds, problems = self._dataset, []
        reason = np.asarray(ds["invalid_reason"].values)
        valid = reason == 0
        for name in ("threshold", "reference"):
            if not np.all(np.isfinite(np.asarray(ds[name].values)) | ~valid):
                problems.append(f"valid positions without a finite {name}")
        if not time_layout(ds) and not np.all(
                (np.asarray(ds["sample_count"].values) >= self.baseline.min_samples) | ~valid):
            problems.append("valid positions with fewer than min_samples samples")
        if not np.all((reason & ~np.uint8(1 | 4 | 8)) == 0):
            problems.append("unknown invalid_reason bits")
        domain = ds[self.grid.domain_mask]
        outside = (ds["invalid_reason"] & OUTSIDE_DOMAIN) != 0
        if not bool((outside == ~domain).all()):
            problems.append("outside_domain bit != ~domain_mask")
        if time_layout(ds):
            problems += self._dynamic_problems()
        if self.baseline.in_base_correction != "none":
            year_reason = np.asarray(ds["invalid_reason_by_year"].values)
            year_valid = year_reason == 0
            first = by_year_names(ds)[0]
            if not np.all(np.isfinite(np.asarray(ds[first].values)) | ~year_valid):
                problems.append(f"valid in-base positions without a finite {first}")
            if not np.all((np.asarray(ds["sample_count_by_year"].values)
                           >= self.baseline.min_samples) | ~year_valid):
                problems.append("valid in-base positions with fewer than min_samples samples")
            outside = (ds["invalid_reason_by_year"] & OUTSIDE_DOMAIN) != 0
            if not bool((outside == ~domain).all()):
                problems.append("outside_domain bit of invalid_reason_by_year != ~domain_mask")
        if problems:
            raise InvariantError("; ".join(problems))

    def _dynamic_problems(self) -> list[str]:
        """reference == trend + seasonal and threshold == reference + Q_R(d(t))."""
        ds, problems = self._dataset, []
        valid = ds["invalid_reason"] == 0
        total = ds["trend"] + ds["seasonal"]
        if not bool((abs(ds["reference"] - total) <= 1e-9 * (1 + abs(total))).where(
                valid, True).all()):
            problems.append("reference != trend + seasonal on valid dates")
        pos = doy_positions(ds[self.grid.time_dim].values, self.baseline.leap_day_policy)
        q = ds["residual_threshold"].isel({DOY: xr.DataArray(pos - 1,
                                                             dims=self.grid.time_dim)})
        gap = ds["threshold"] - ds["reference"] - q
        if not bool((abs(gap) <= 1e-9 * (1 + abs(ds["threshold"]))).where(valid, True).all()):
            problems.append("threshold != reference + residual_threshold on valid dates")
        q_valid = ds["residual_invalid_reason"] == 0
        if not bool(((ds["sample_count"] >= self.baseline.min_samples)
                     & ds["residual_threshold"].notnull()).where(q_valid, True).all()):
            problems.append("valid residual percentiles without enough samples")
        return problems


def _check_structure(ds: xr.Dataset, grid: GridSpec, baseline: BaselineSpec,
                     definition: DefinitionSpec) -> None:
    doy_mode = baseline.leap_day_policy != "not_applicable"
    combined = definition.combination == "and_daily"
    dynamic = time_layout(ds)
    if dynamic:
        require(baseline.method != "fixed_climatology",
                "a baseline on the time axis needs a dynamic BaselineSpec.method ")
    base = {*grid.spatial_dims, *((COMPONENT,) if combined else ())}
    expected = base | ({DOY} if doy_mode else set())
    if dynamic:  #: fields on the time axis, residual percentile per day of year
        tdim = grid.time_dim
        layout = {**{n: base | {tdim} for n in ("trend", "seasonal", "reference", "threshold",
                                                "invalid_reason")},
                  **{n: expected for n in ("residual_threshold", "sample_count",
                                           "residual_invalid_reason")}}
        require(tdim in ds.coords and ds.sizes[tdim] > 0, f"time coordinate {tdim!r} missing")
        values = np.asarray(ds[tdim].values, dtype="datetime64[ns]")
        require(bool(np.all(np.diff(values) > np.timedelta64(0, "ns"))),
                f"{tdim!r} must be strictly increasing")
        require(ds["residual_invalid_reason"].dtype == np.uint8,
                "'residual_invalid_reason' must be uint8")
    else:
        layout = {n: expected for n in ("threshold", "reference", "sample_count",
                                        "invalid_reason")}
    for name, dims in layout.items():
        require(name in ds.data_vars, f"required variable {name!r} is missing")
        require(set(ds[name].dims) == dims, f"{name!r} dims {ds[name].dims} != {dims}")
    if combined:
        check_component_coords(ds, definition, lags=False)
    if doy_mode:
        require(np.array_equal(ds[DOY].values, np.arange(1, N_DOY + 1)),
                f"{DOY!r} coordinate must be 1..366")
    units = {temperature_units(n, ds[n].attrs.get("units")) for n in ("threshold", "reference")}
    require(len(units) == 1, "threshold and reference units differ")
    for name in ("threshold", "reference"):
        require(ds[name].dtype.kind == "f", f"{name!r} must be floating point")
        require(ds[name].attrs.get("units_metadata") == ON_SCALE,
                f"{name!r} needs units_metadata={ON_SCALE!r}")
    require(ds["sample_count"].dtype.kind in "iu", "'sample_count' must be integer")
    require(ds["invalid_reason"].dtype == np.uint8, "'invalid_reason' must be uint8")
    names = by_year_names(ds)
    present = [name for name in (*BY_YEAR, *BY_YEAR_DYNAMIC) if name in ds.variables]
    if baseline.in_base_correction == "none":
        require(not present, f"{present} need BaselineSpec.in_base_correction ")
    else:
        for name in names:
            require(name in ds.data_vars, f"required variable {name!r} is missing")
            require(set(ds[name].dims) == expected | {YEAR},
                    f"{name!r} dims {ds[name].dims} != {expected | {YEAR}}")
        require(np.array_equal(ds[YEAR].values, baseline.reference_years),
                f"{YEAR!r} coordinate must list the reference years {baseline.reference_years}")
        require(ds[names[0]].dtype.kind == "f", f"{names[0]!r} must be float")
        require(ds[names[1]].dtype.kind in "iu", f"{names[1]!r} must be integer")
        require(ds[names[2]].dtype == np.uint8, f"{names[2]!r} must be uint8")
        require(ds[names[0]].attrs.get("units") == ds["threshold"].attrs.get("units"),
                f"{names[0]} and threshold units differ")
    require(grid.domain_mask in ds.variables, f"domain mask {grid.domain_mask!r} is missing")
    require(set(ds[grid.domain_mask].dims) == set(grid.spatial_dims)
            and ds[grid.domain_mask].dtype == bool, "domain mask must be static and boolean")
    check_serializable_attrs("dataset", ds.attrs)
    for name, var in ds.variables.items():
        check_serializable_attrs(f"variable {name!r}", var.attrs)


# --- fitting ---------------------------------------------------------------------------------


def input_variables(variable, definition: DefinitionSpec) -> tuple[str, ...]:
    """Dataset variable names: one name, or a tuple aligned with ``definition.components``."""
    if definition.combination == "and_daily":
        n = len(definition.components)
        require(isinstance(variable, tuple) and len(variable) == n
                and all(isinstance(v, str) for v in variable) and len(set(variable)) == n,
                f"and_daily needs `variable` as a tuple of {n} distinct dataset variable names "
                "aligned with definition.components")
        return variable
    require(isinstance(variable, str), "variable must be the name of a dataset variable")
    return (variable,)


def daily_statistics(definition: DefinitionSpec) -> tuple[str, ...]:
    if definition.combination == "and_daily":
        return tuple(c.daily_statistic for c in definition.components)
    return (definition.daily_statistic,)


def check_input(ds: xr.Dataset, names: tuple[str, ...], grid: GridSpec) -> str:
    """Dims and units of the input variables; returns their common canonical unit."""
    units = set()
    for name in names:
        require(name in ds.data_vars, f"variable {name!r} not found")
        da = ds[name]
        require(set(da.dims) == {grid.time_dim, *grid.spatial_dims},
                f"{name!r} dims {da.dims} do not match the grid")
        units.add(temperature_units(name, da.attrs.get("units")))
    if len(units) > 1:
        raise UnitsError(f"input variables {names} have different units; convert explicitly")
    return units.pop()


def fit_baseline(ds: xr.Dataset, *, variable: str | tuple[str, ...], grid: GridSpec,
                 baseline: BaselineSpec, definition: DefinitionSpec,
                 provenance: Provenance | None = None) -> BaselineResult:
    """Fit a fixed climatology and percentile threshold per day of year and location.

    Uses observed values only (``filled`` days excluded) inside `baseline.reference_period` .
    Lazy for dask input with the time dimension in one chunk. Emits `XHWDataQualityWarning`
    when sampling is below recommendations. For ``combination="and_daily"``, `variable` is a
    tuple aligned with the components and the result has a ``component`` dimension. With
    ``baseline.in_base_correction`` the result also holds ``threshold_by_year`` (with
    ``sample_count_by_year`` and ``invalid_reason_by_year``) for each reference year, m times
    the size of the threshold; a warning gives the estimate above 1 GB.
    """
    return _fit(ds, variable=variable, grid=grid, baseline=baseline, definition=definition,
                provenance=provenance)


def _fit(ds, *, variable, grid, baseline, definition, provenance=None, replica=None):
    """`fit_baseline`; with `replica` = k (``mean_threshold`` only), ``threshold_by_year`` holds
    the k-th replica of each year instead of their mean (used by `in_base_rates`).
    """
    require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
    require(all(isinstance(s, t) for s, t in ((grid, GridSpec), (baseline, BaselineSpec),
                                              (definition, DefinitionSpec))),
            "grid, baseline and definition must be GridSpec, BaselineSpec and DefinitionSpec")
    if baseline.method not in ("fixed_climatology", "detrended", "shifting", "stl",
                               "state_space", "gam", "ssa"):
        raise NotSupportedError(f"baseline method {baseline.method!r} is not implemented")
    if baseline.computed_on != "temperature":
        raise NotSupportedError("temporal detection fits baselines on temperature, not anomalies")
    grid.check(ds)
    names = input_variables(variable, definition)
    units = check_input(ds, names, grid)
    steps = daily_steps(ds, names, daily_statistics(definition), definition.day_definition)
    for name in names:
        require_single_time_chunk(ds[name], grid.time_dim, "fit_baseline")
    if baseline.method != "fixed_climatology":
        from ._dynamic import fit_dynamic

        return fit_dynamic(ds, names=names, units=units, steps=steps, grid=grid,
                           baseline=baseline, definition=definition, provenance=provenance,
                           replica=replica)
    times = ds[grid.time_dim].values
    in_ref = _reference_mask(times, baseline)
    require(bool(in_ref.any()), "no data inside baseline.reference_period")
    doy_mode = baseline.leap_day_policy != "not_applicable"
    if doy_mode:
        sets = _window_indices(doy_positions(times, baseline.leap_day_policy), in_ref,
                               baseline.doy_window_days // 2)
    else:
        sets = {0: np.flatnonzero(in_ref)}
    absolute = None
    if definition.threshold_kind == "absolute":
        value, value_units = definition.absolute_threshold
        absolute = _to_units(value, temperature_units("absolute_threshold", value_units), units)
    if definition.combination == "and_daily":
        pcts = [c.percentile for c in definition.components]
    else:
        pcts = [definition.percentile if absolute is None else None]
    correction = baseline.in_base_correction
    years = baseline.reference_years
    year_sets = None
    if correction != "none":
        require(absolute is None, "in_base_correction needs a percentile threshold ")
        year_of = pd.DatetimeIndex(times).year.to_numpy()
        year_sets = _year_sets(sets, year_of, years, correction, replica)
        cells = math.prod(ds.sizes[d] for d in grid.spatial_dims) * len(names)
        size = cells * len(years) * (N_DOY if doy_mode else 1) * 13
        if size > _BY_YEAR_WARN_BYTES:
            warnings.warn(f"in_base_correction={correction!r}: the per-year thresholds of "
                          f"{len(years)} years need ~{size / 1e9:.1f} GB ", stacklevel=3)
    domain = ds[grid.domain_mask]
    out_dims = [DOY] if doy_mode else []
    fitted = []
    for name, pct in zip(names, pcts):
        observed = ds[name].where(~ds["filled"]) if "filled" in ds else ds[name]
        kwargs = dict(sets=sets, doy_mode=doy_mode,
                      feb29=baseline.leap_day_policy == "hobday_366_interp",
                      pct=50.0 if pct is None else pct,
                      method=NUMPY_PERCENTILE_METHOD[baseline.percentile_estimator],
                      smooth=baseline.smoothing_window_days if doy_mode else None,
                      min_samples=baseline.min_samples, absolute=absolute)
        thr, ref, count, reason = xr.apply_ufunc(
            _fit_core, observed, input_core_dims=[[grid.time_dim]],
            output_core_dims=[out_dims] * 4, dask="parallelized", kwargs=kwargs,
            output_dtypes=[float, float, np.int32, np.uint8],
            dask_gufunc_kwargs={"output_sizes": {DOY: N_DOY}} if doy_mode else None,
        )
        if count.chunks is None:  # data-dependent checks only when already computed
            inside = domain.broadcast_like(count).transpose(*count.dims).values
            _quality_warnings(baseline, definition, pct, in_ref, doy_mode, count.values, inside)
        else:
            _quality_warnings(baseline, definition, pct, in_ref, doy_mode, None, None)
        fits = (thr, ref, count, reason)
        if year_sets is not None:
            by_kwargs = {k: v for k, v in kwargs.items() if k != "sets"}
            fits += xr.apply_ufunc(
                _fit_years_core, observed, input_core_dims=[[grid.time_dim]],
                output_core_dims=[[YEAR, *out_dims]] * 3, dask="parallelized",
                kwargs=dict(by_kwargs, year_sets=year_sets),
                output_dtypes=[float, np.int32, np.uint8],
                dask_gufunc_kwargs={"output_sizes": {YEAR: len(years),
                                                     **({DOY: N_DOY} if doy_mode else {})}},
            )
        fitted.append(fits)
    if definition.combination == "and_daily":
        labels = [c.variable for c in definition.components]
        fits = tuple(
            xr.concat([f[k] for f in fitted], dim=COMPONENT).assign_coords({COMPONENT: labels})
            for k in range(len(fitted[0])))
    else:
        fits = fitted[0]
    thr, ref, count, reason = fits[:4]
    attrs = {"units": ds[names[0]].attrs["units"], "units_metadata": ON_SCALE}
    out = xr.Dataset({
        "threshold": thr.where(domain).assign_attrs(attrs, long_name="detection threshold"),
        "reference": ref.where(domain).assign_attrs(attrs, long_name="reference climatology"),
        "sample_count": count.assign_attrs(long_name="observed samples per day of year"),
        "invalid_reason": xr.where(domain, reason, OUTSIDE_DOMAIN).astype(np.uint8),
        grid.domain_mask: domain,
    })
    if year_sets is not None:
        thr_y, count_y, reason_y = fits[4:]
        out["threshold_by_year"] = thr_y.where(domain).assign_attrs(
            attrs, long_name=f"threshold of each reference year ({correction})")
        out["sample_count_by_year"] = count_y.assign_attrs(
            long_name="observed samples per day of year without the reference year")
        out["invalid_reason_by_year"] = xr.where(domain, reason_y, OUTSIDE_DOMAIN).astype(
            np.uint8)
        out = out.assign_coords({YEAR: np.array(years, dtype=np.int64)})
    if doy_mode:
        out = out.assign_coords({DOY: np.arange(1, N_DOY + 1)})
    lead_dims = ((YEAR,) if year_sets is not None else ()) + (
        (COMPONENT,) if definition.combination == "and_daily" else ()) + (
        (DOY,) if doy_mode else ())
    out = out.transpose(*lead_dims, *grid.spatial_dims)
    if provenance is None:
        params = (("variable", ",".join(names)), ("excluded_filled_days", "filled" in ds))
        steps.append(ProcessingStep(name="fit_baseline", params=params))
        provenance = Provenance.new(inputs=(baseline.source,), steps=tuple(steps))
    return BaselineResult.from_dataset(out, grid=grid, baseline=baseline, definition=definition,
                                       provenance=provenance)
