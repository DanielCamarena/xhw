# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""build_catalog: EventCatalog with temporal metrics from an XMask (temporal detection ).

Metrics are computed over the whole event interval (joined gap days included, as in the
reference implementation) using observed and filled days; missing days are excluded from
intensities and counted in ``n_missing_days``. Censored events have partial metrics
(provisional in monitoring); their censor flags say so.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from ._catalog import EVENT_DIM, EventCatalog, event_rows, index_var
from ._mask import (
    COMPONENT,
    DIFFERENCE,
    LAG_COORD,
    ON_SCALE,
    XMask,
    columns,
    component_series,
)
from ._schema import check_choice, require
from ._specs import ProcessingStep, Provenance

CATEGORY_MEANINGS = "none moderate strong severe extreme"


def _rates(j: np.ndarray, peak: np.ndarray, prev: np.ndarray, has_prev: np.ndarray,
           nxt: np.ndarray, has_next: np.ndarray) -> tuple:
    """Onset/decline rates of the oriented anomaly, half-day convention of Hobday/Oliver,
    for a batch of events of the same length (rows of `j`).

    `prev`/`nxt` are the oriented anomalies of the neighbouring days, used where `has_prev`/
    `has_next` hold; where they do not (the event touches the window edge or an invalid day,
    treated alike) the rate uses the event's own end. A peak on the last day at an edge uses a
    one-day denominator, symmetric to the onset rule.
    """
    last = j.shape[1] - 1
    rows = np.arange(j.shape[0])
    top = j[rows, peak]
    first_day, last_day = j[:, 0], j[:, -1]
    onset = np.where(has_prev,
                     (top - 0.5 * (first_day + prev)) / (peak + 0.5),
                     (top - first_day) / np.where(peak > 0, peak, 1))
    decline = np.where(has_next,
                       (top - 0.5 * (last_day + nxt)) / (last - peak + 0.5),
                       (top - last_day) / np.where(peak < last, last - peak, 1))
    return onset, decline


def _metrics(mask: XMask, rows, series: dict, filled) -> dict[str, np.ndarray]:
    """Metrics of one series: (time, locations) arrays of temperature, reference and threshold
    aligned on the reference day (component values of day D + lag_days for and_daily).

    Events are processed in batches of equal duration, so every statistic is one vectorised
    reduction over the days of the batch instead of a call per event.
    """
    g, ds, d = mask.grid, mask.dataset, mask.definition
    get = dict(series, valid=columns(ds["valid"], g.time_dim, g.spatial_dims))
    times = ds[g.time_dim].values
    sign = 1.0 if d.tail == "upper" else -1.0
    extreme = "temperature_max" if d.tail == "upper" else "temperature_min"
    names = ["n_filled_days", "n_missing_days", "peak_date", "intensity_max", "intensity_mean",
             "intensity_std", "intensity_cumulative", "intensity_max_rel_threshold",
             "intensity_mean_rel_threshold", "intensity_cumulative_rel_threshold", extreme,
             "temperature_mean", "rate_onset", "rate_decline"]
    if d.category_scheme is not None:
        names += ["category_max", "days_category_1", "days_category_2", "days_category_3",
                  "days_category_4"]
    n = len(rows)
    if n == 0:
        return {k: np.asarray([]) for k in names}
    out: dict[str, np.ndarray] = {}
    col = np.fromiter((r.col for r in rows), np.int64, n)
    start = np.fromiter((r.start for r in rows), np.int64, n)
    end = np.fromiter((r.end for r in rows), np.int64, n)
    length = end - start + 1
    by_length = np.argsort(length, kind="stable")
    last = times.size - 1
    pick = np.nanmax if sign > 0 else np.nanmin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for pos in np.split(by_length, np.flatnonzero(np.diff(length[by_length])) + 1):
            c, s, e = col[pos], start[pos], end[pos]
            size = int(length[pos[0]])
            rank = np.arange(pos.size)
            day = s[:, None] + np.arange(size)
            cell = c[:, None]
            temp = get["temperature"][day, cell]
            ref = get["reference"][day, cell]
            thr = get["threshold"][day, cell]
            anomaly = temp - ref
            oriented = sign * anomaly
            peak = np.nanargmax(oriented, axis=1)
            rel = temp - thr

            def neighbour(i: np.ndarray):
                ok = (i >= 0) & (i <= last)
                i = np.clip(i, 0, last)
                ok &= get["valid"][i, c]
                return ok, sign * (get["temperature"][i, c] - get["reference"][i, c])

            has_prev, prev = neighbour(s - 1)
            has_next, nxt = neighbour(e + 1)
            onset, decline = _rates(oriented, peak, prev, has_prev, nxt, has_next)
            batch = {
                "n_filled_days": filled[day, cell].sum(axis=1) if filled is not None
                else np.zeros(pos.size, np.int64),
                "n_missing_days": np.isnan(temp).sum(axis=1),
                "peak_date": times[s + peak],
                "intensity_max": anomaly[rank, peak],
                "intensity_mean": np.nanmean(anomaly, axis=1),
                "intensity_std": np.nanstd(anomaly, axis=1),
                "intensity_cumulative": np.nansum(anomaly, axis=1),
                "intensity_max_rel_threshold": rel[rank, peak],
                "intensity_mean_rel_threshold": np.nanmean(rel, axis=1),
                "intensity_cumulative_rel_threshold": np.nansum(rel, axis=1),
                extreme: pick(temp, axis=1),
                "temperature_mean": np.nanmean(temp, axis=1),
                "rate_onset": onset,
                "rate_decline": decline,
            }
            if d.category_scheme is not None:
                scale = thr - ref
                ratio = np.where(scale != 0, rel / np.where(scale != 0, scale, 1), np.nan)
                cats = np.floor(1.0 + ratio)
                finite = np.isfinite(cats)
                top = np.where(finite.any(axis=1), np.nanmax(cats, axis=1), 0)
                batch["category_max"] = np.minimum(4, np.maximum(0, top)).astype(np.int64)
                for k in (1, 2, 3):
                    batch[f"days_category_{k}"] = (cats == k).sum(axis=1)
                batch["days_category_4"] = (cats >= 4).sum(axis=1)
            for k, values in batch.items():
                if k not in out:
                    out[k] = np.empty(n, dtype=values.dtype)
                out[k][pos] = values
    return out


def _all_metrics(mask: XMask, rows) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Metrics per event; for ``and_daily`` per event and component (dims event, component).
    Counts of the joint row (``n_filled_days``) stay 1-D.
    """
    g, ds = mask.grid, mask.dataset
    filled = columns(ds["filled"], g.time_dim, g.spatial_dims) if "filled" in ds else None
    series = component_series(ds, mask.definition, g.time_dim, g.spatial_dims,
                              ("temperature", "reference", "threshold"))
    per = [_metrics(mask, rows, s, filled) for s in series]
    if mask.definition.combination != "and_daily":
        return per[0], (EVENT_DIM,)
    out = {k: np.stack([m[k] for m in per], axis=1).reshape(len(rows), len(per))
           for k in per[0] if k != "n_filled_days"}
    out["n_filled_days"] = per[0]["n_filled_days"]
    return out, (EVENT_DIM, COMPONENT)


def _attrs(units: str, cumulative: bool = False, rate: bool = False) -> dict:
    suffix = " d" if cumulative else " d-1" if rate else ""
    return {"units": units + suffix, "units_metadata": DIFFERENCE}


def build_catalog(mask: XMask, *, analysis_mode: str,
                  provenance: Provenance | None = None) -> EventCatalog:
    """Build the EventCatalog of `mask` (computes the mask data; catalogs are eager).

    ``retrospective``: qualified events only. ``monitoring``: also edge candidates. The
    baseline must be available before the window in monitoring mode.
    """
    require(isinstance(mask, XMask), "mask must be an XMask")
    check_choice("analysis_mode", analysis_mode, ("retrospective", "monitoring"))
    rows = event_rows(mask, analysis_mode)
    times = mask.dataset[mask.grid.time_dim].values
    n = len(rows)
    field = list(zip(*(r.fields for r in rows))) if rows else [[] for _ in range(7)]
    data = {
        "event_id": (EVENT_DIM, np.arange(1, n + 1, dtype=np.int64)),
        "start": (EVENT_DIM, times[[r.start for r in rows]]),
        "end": (EVENT_DIM, times[[r.end for r in rows]]),
        "duration_days": (EVENT_DIM, np.array([r.end - r.start + 1 for r in rows], np.int64)),
        "n_exceedance_days": (EVENT_DIM, np.array(field[1], np.int64)),
        "n_joined_gap_days": (EVENT_DIM, np.array(field[2], np.int64)),
        "left_censored": (EVENT_DIM, np.array(field[3], bool)),
        "right_censored": (EVENT_DIM, np.array(field[4], bool)),
        "censor_reason": (EVENT_DIM, np.array(field[5], np.uint8)),
        "qualified": (EVENT_DIM, np.array(field[0], bool)),
        "contains_forecast": (EVENT_DIM, np.array(field[6], bool)),
    }
    for k, dim in enumerate(mask.grid.spatial_dims):
        data[index_var(dim)] = (EVENT_DIM, np.array([r.loc[k] for r in rows], np.int64))
    ds = xr.Dataset(data)
    if "temperature" in mask.dataset:
        units = mask.dataset["temperature"].attrs["units"]
        metrics, dims = _all_metrics(mask, rows)
        if len(dims) == 2:
            comps = mask.definition.components
            lags = np.array([c.lag_days for c in comps], np.int64)
            ds = ds.assign_coords({COMPONENT: [c.variable for c in comps],
                                   LAG_COORD: (COMPONENT, lags)})
        for name, values in metrics.items():
            if name == "n_filled_days":
                ds[name] = (EVENT_DIM, values.astype(np.int64))
            elif name.startswith(("n_", "days_category")):
                ds[name] = (dims, values.astype(np.int64))
            elif name == "peak_date":
                ds[name] = (dims, values.astype("datetime64[ns]"))
            elif name == "category_max":
                ds[name] = (dims, values.astype(np.uint8),
                            {"flag_values": np.arange(5, dtype=np.uint8),
                             "flag_meanings": CATEGORY_MEANINGS,
                             "category_scheme": mask.definition.category_scheme})
            elif name.startswith("temperature"):
                ds[name] = (dims, values.astype(float),
                            {"units": units, "units_metadata": ON_SCALE})
            else:
                ds[name] = (dims, values.astype(float),
                            _attrs(units, cumulative="cumulative" in name,
                                   rate=name.startswith("rate")))
        for name in ("intensity_mean", "intensity_std", "intensity_cumulative"):
            ds[name].attrs["long_name"] = ("over the event interval, joined gap days included, "
                                           "missing days excluded")
    if provenance is None:
        provenance = Provenance.new(parents=(mask.provenance.run_id,), steps=(
            ProcessingStep(name="build_catalog", params=(("analysis_mode", analysis_mode),)),))
    return EventCatalog.from_dataset(ds, mask=mask, granularity="point_temporal",
                                     analysis_mode=analysis_mode, provenance=provenance)
