# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Neighbour fraction, ingestion of externally produced objects and their comparison (object
tracking).

`ingest_objects` lets xHW consume the objects of another method (a neural network, Ocetrac,
marEx, a visual analysis) with its provenance declared, as an ObjectField with
``origin="external"``; `compare_objects` compares any two ObjectFields of the same mask. xHW
does not train or run those methods (SPEC): it only reads their result and keeps the evidence
apart.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

from . import _morph, _track
from ._errors import NotSupportedError, ValidationError, XHWDataQualityWarning
from ._mask import XMask
from ._objects import (
    OBJECT_DIM,
    TRACK_DIM,
    ObjectField,
    build_geometry,
    cell_weights,
    flat,
    make_table,
    set_coordinate_vars,
    static_names,
    vertical_thickness,
)
from ._recipe import Footprint
from ._schema import check_choice, require
from ._specs import ProcessingStep, Provenance, SourceRef


def _supported(grid) -> None:
    require(grid.support in ("line", "surface", "volume"),
            "objects need a line, a surface or a volume support", NotSupportedError)


def _lift(a: np.ndarray, grid) -> np.ndarray:
    """A line is one row: (time, x) -> (time, 1, x)."""
    return a[:, None, :] if grid.support == "line" else a


def neighbor_fraction(mask: XMask, *, footprint: Footprint,
                      layer: str = "exceedance") -> xr.DataArray:
    """Fraction of the footprint (the cell included) that is set in `layer`, weighted by the cell
    measure and normalised by the valid weight of the footprint: land and invalid cells never
    lower it, and it is NaN where a cell has no valid cell around it. A continuous measure of
    spatial coherence that needs no binarisation (declared numerical rules). Computes the
    mask's data.
    """
    require(isinstance(mask, XMask), "mask must be an XMask")
    check_choice("layer", layer, ("exceedance", "event_membership"))
    g, md = mask.grid, mask.dataset
    _supported(g)
    lat, lon = _morph.axes(md, g)
    order = (g.time_dim, *g.spatial_dims)
    weights, _ = cell_weights(md, g)
    valid = _lift(np.asarray(md["valid"].transpose(*order).values), g)
    field = _lift(np.asarray(md[layer].transpose(*order).values), g) & valid
    offsets = _morph.build_offsets(footprint, lat, lon, g.periodic_x, _morph.is_volume(g),
                                   _morph.is_line(g), vertical_thickness(md, g))
    values = _morph.neighbour_fraction(field, valid, weights.reshape(-1, weights.shape[-1])
                                       if g.support == "line" else weights, offsets, g.periodic_x)
    values = values.reshape((valid.shape[0], *(md.sizes[d] for d in g.spatial_dims)))
    return xr.DataArray(values, dims=order, coords=md["valid"].transpose(*order).coords,
                        name="neighbor_fraction",
                        attrs={"long_name": f"weighted fraction of {layer} in the footprint",
                               "units": "1"})


# --- ingestion -------------------------------------------------------------------------------


def _sequential(ids: np.ndarray) -> np.ndarray:
    """Renumber positive ids 1..k by first appearance in raster order (day, y, x)."""
    flat = ids.ravel()
    positive = flat > 0
    values, first = np.unique(flat[positive], return_index=True)
    lut = np.zeros(int(values.max()) + 1 if values.size else 1, dtype=np.int32)
    lut[values[np.argsort(first, kind="stable")]] = np.arange(1, values.size + 1,
                                                              dtype=np.int32)
    return lut[ids]


def _check_connectivity(raw, g, md, lat, lon, connectivity, kind, on_violation) -> None:
    """Patches: every label of a day is one connected patch under `connectivity`. Tracks: two
    neighbouring cells of a day never belong to different identities.
    """
    volume, line = _morph.is_volume(g), _morph.is_line(g)
    offsets = _morph.build_offsets(connectivity, lat, lon, g.periodic_x, volume, line,
                                   vertical_thickness(md, g))
    days = raw[:, None, :] if line else raw
    times = md[g.time_dim].values
    found = []
    for t in np.flatnonzero((days > 0).any(axis=tuple(range(1, days.ndim)))):
        if kind == "patches":
            for label, n in _morph.split_labels(days[t], offsets, g.periodic_x).items():
                found.append((t, f"label {label}", f"forms {n} patches"))
        else:
            for a, b in _morph.touching_labels(days[t], offsets, g.periodic_x):
                found.append((t, f"labels {a} and {b}", "touch"))
    if not found:
        return
    shown = "; ".join(f"{who} on {np.datetime_as_string(times[t], unit='D')} {what}"
                      for t, who, what in found[:5])
    problem = ("labels are not a single connected patch" if kind == "patches"
               else "pairs of identities touch")
    message = (f"{len(found)} {problem} under the declared connectivity "
               f"({shown}{'; ...' if len(found) > 5 else ''})")
    if on_violation == "error":
        raise ValidationError(message)
    warnings.warn(message, XHWDataQualityWarning, stacklevel=3)


def ingest_objects(labels: xr.DataArray, *, mask: XMask, kind: str, source: SourceRef,
                   method: str, source_layer: str = "exceedance",
                   analysis_mode: str = "retrospective",
                   provenance: Provenance | None = None,
                   connectivity: Footprint | None = None,
                   on_violation: str = "error") -> ObjectField:
    """Wrap integer labels produced elsewhere as an ObjectField of `mask`.

    ``kind="patches"``: `labels` identify the patches of each day (the numbers may repeat or
    skip across days; they are renumbered 1..n_t in raster order and the original value is
    kept in ``object_external_id``). ``kind="tracks"``: `labels` are identities of objects
    that live several days (0 = background); daily patches are the cells of one identity on
    one day, the identities are renumbered 1..k by first day and first cell, the original
    value is kept in ``track_external_id``. Labels must be non-negative integers on the mask's
    grid and time axis and must not lie on invalid cells. `source` and `method` declare where
    the objects come from; `source_layer` is the layer of `mask` that ``added`` and
    ``removed`` are measured against.

    `connectivity` (a `Footprint`) declares the neighbourhood the external method used to
    define a patch (8 neighbours for Ocetrac and marEx, for example) and checks it, with
    longitude periodic as in the grid. ``kind="patches"``: every label of every day must be a
    single connected patch. ``kind="tracks"``: an identity may be several patches of one day
    (they are joined through other days), so the check is the converse: two neighbouring cells
    of one day cannot belong to different identities. A violation is reported with its day:
    ``on_violation="error"`` raises, ``"warn"`` emits an `XHWDataQualityWarning`. The check
    catches a declared neighbourhood that is larger than the one the method used (a label that
    is not connected; touching identities); it cannot tell a method that used a larger
    neighbourhood than the one declared for tracks. Without it nothing is assumed and the
    comparison with an xHW recipe of another neighbourhood differs by construction. The
    declared footprint is recorded in the provenance.
    """
    require(isinstance(mask, XMask), "mask must be an XMask")
    require(isinstance(source, SourceRef), "source must be a SourceRef")
    check_choice("kind", kind, ("patches", "tracks"))
    check_choice("source_layer", source_layer, ("exceedance", "event_membership"))
    check_choice("analysis_mode", analysis_mode, ("retrospective", "monitoring"))
    check_choice("on_violation", on_violation, ("error", "warn"))
    require(connectivity is None or isinstance(connectivity, Footprint),
            "connectivity must be a Footprint")
    require(isinstance(method, str) and method.strip() != "", "method must be a non-empty string")
    g, md = mask.grid, mask.dataset
    _supported(g)
    order = (g.time_dim, *g.spatial_dims)
    require(isinstance(labels, xr.DataArray) and set(labels.dims) == set(order),
            f"labels must be a DataArray with dims {order}")
    labels = labels.transpose(*order)
    for dim in order:
        require(labels.sizes[dim] == md.sizes[dim], f"labels differ from the mask along {dim!r}")
    require(bool((labels[g.time_dim].values == md[g.time_dim].values).all()),
            "labels and mask have different time axes")
    raw = np.asarray(labels.values)
    require(raw.dtype.kind in "iu" or (raw.dtype.kind == "f" and np.isfinite(raw).all()
                                       and (raw == np.round(raw)).all()),
            "labels must be integers without NaN")
    raw = raw.astype(np.int64)
    require(bool((raw >= 0).all()), "labels must be >= 0 (0 = background)")
    valid = np.asarray(md["valid"].transpose(*order).values)
    if bool(((raw > 0) & ~valid).any()):
        raise ValidationError("labels lie on cells that are not valid in the mask")
    lat, lon, geometry = build_geometry(md, g)
    if connectivity is not None:
        _check_connectivity(raw, g, md, lat, lon, connectivity, kind, on_violation)
    weights, unit = cell_weights(md, g)
    weights = weights.reshape(-1, weights.shape[-1])         # the flat view (time, z*y, x)
    shape = raw.shape
    raw = flat(raw)
    T = raw.shape[0]
    times = md[g.time_dim].values
    # daily patches: one per (day, id) for tracks, one per (day, label) for patches
    daily = np.zeros(raw.shape, dtype=np.int32)
    for t in np.flatnonzero((raw > 0).any(axis=(1, 2))):
        _, inverse = np.unique(raw[t], return_inverse=True)
        daily[t] = _sequential(np.where(raw[t] > 0, inverse.reshape(raw[t].shape) + 1, 0)[None])[0]
    table, clat, clon, (tt, yy, xx, gid) = make_table(
        daily, weights, geometry["lat"], geometry["lon"])
    external = np.zeros(table.n, dtype=np.int64)
    external[gid] = raw[tt, yy, xx]
    ds = xr.Dataset({"valid": md["valid"].transpose(*order)})
    for name in static_names(g):
        if name is not None and name in md.data_vars:
            ds[name] = md[name]
    obj_track = np.zeros(table.n, dtype=np.int32)
    summary = None
    processed = daily > 0
    if kind == "tracks":
        ids = _sequential(raw.astype(np.int64).clip(0, np.iinfo(np.int64).max))
        k = int(ids.max())
        original = np.zeros(k + 1, dtype=np.int64)
        original[ids[raw > 0]] = raw[raw > 0]
        obj_track[gid] = ids[tt, yy, xx]
        summary = _track.track_summary(table, obj_track, np.zeros(table.n, bool), T, 1)
        ds["track_id"] = (order, ids.reshape(shape))
        ds["track_external_id"] = (TRACK_DIM, original[1:])
    area_attrs = {"units": unit, "long_name": "summed cell measure of the patch"}
    ds["label"] = (order, daily.reshape(shape))
    ds["object_time"] = (OBJECT_DIM, times[table.day])
    ds["object_label"] = (OBJECT_DIM, table.label.astype(np.int32))
    ds["object_track"] = (OBJECT_DIM, obj_track)
    ds["object_area"] = (OBJECT_DIM, table.area, area_attrs)
    ds["object_cells"] = (OBJECT_DIM, table.cells.astype(np.int64))
    set_coordinate_vars(ds, g, geometry, table, weights, clat, clon, (yy, xx, gid))
    ds["object_external_id"] = (OBJECT_DIM, external)
    if summary is not None:
        m = summary["start"].size
        ds["track_number"] = (TRACK_DIM, np.arange(1, m + 1, dtype=np.int32))
        ds["track_start"] = (TRACK_DIM, times[summary["start"].astype(int)])
        ds["track_end"] = (TRACK_DIM, times[summary["end"].astype(int)])
        ds["track_duration_days"] = (TRACK_DIM, summary["duration"].astype(np.int64))
        ds["track_active_days"] = (TRACK_DIM, summary["active"].astype(np.int64))
        ds["track_bridged_days"] = (TRACK_DIM, summary["bridged"].astype(np.int64))
        ds["track_area_max"] = (TRACK_DIM, summary["area_max"], area_attrs)
        ds["track_area_mean"] = (TRACK_DIM, summary["area_mean"], area_attrs)
        ds["track_area_days"] = (TRACK_DIM, summary["area_days"],
                                 {"units": f"{unit} day", "long_name": "area summed over days"})
        ds["track_n_multi_overlap_days"] = (TRACK_DIM, np.zeros(m, np.int64))
        ds["track_left_censored"] = (TRACK_DIM, summary["start"] == 0)
        ds["track_right_censored"] = (TRACK_DIM, summary["end"] == T - 1)
        ds["track_qualified"] = (TRACK_DIM, np.ones(m, bool))
        ds["track_parent"] = (TRACK_DIM, np.zeros(m, np.int32))
        ds["track_merged_into"] = (TRACK_DIM, np.zeros(m, np.int32))
    source_cells = np.asarray(md[source_layer].transpose(*order).values) & valid
    processed = processed.reshape(shape)
    ds["processed_mask"] = (order, processed)
    ds["added"] = (order, processed & ~source_cells)
    ds["removed"] = (order, source_cells & ~processed)
    if provenance is None:
        provenance = Provenance.new(
            inputs=(source,), parents=(mask.provenance.run_id,),
            steps=(ProcessingStep(name="ingest_objects",
                                  params=(("kind", kind), ("method", method),
                                          ("analysis_mode", analysis_mode))
                                  + ((("connectivity", connectivity.spec_id),)
                                     if connectivity is not None else ())),))
    return ObjectField._make(
        ds, grid=g, recipe=None, analysis_mode=analysis_mode, provenance=provenance,
        definition_id=mask.definition_id, baseline_id=mask.baseline_id,
        source_mask_run_id=mask.provenance.run_id, origin="external", final_after_days=0,
        source_layer=source_layer)


# --- comparison ------------------------------------------------------------------------------


def compare_objects(a: ObjectField, b: ObjectField) -> xr.Dataset:
    """Compare two ObjectFields of the same mask (e.g. xHW against an ingested method).

    Returns a Dataset with, along time, ``jaccard`` of the processed masks (NaN where both are
    empty), ``cells_a``, ``cells_b``, ``n_objects_a`` and ``n_objects_b``; and, along
    ``object_a`` (the daily patches of `a`), ``best_iou`` (intersection over union, by area,
    with the best patch of `b` of the same day; 0 if none overlaps) and ``best_match_b`` (its
    label, 0 if none). The patch comparison needs ``label`` in both.
    """
    require(isinstance(a, ObjectField) and isinstance(b, ObjectField), "expected ObjectFields")
    require(a.source_mask_run_id == b.source_mask_run_id, "the objects come from different masks")
    require(a.grid == b.grid, "the objects have different grids")
    g = a.grid
    _supported(g)
    order = (g.time_dim, *g.spatial_dims)
    space = tuple(range(1, len(order)))              # every axis but time
    da, db = a.dataset, b.dataset
    pa = np.asarray(da["processed_mask"].transpose(*order).values)
    pb = np.asarray(db["processed_mask"].transpose(*order).values)
    both, either = (pa & pb).sum(axis=space), (pa | pb).sum(axis=space)
    with np.errstate(invalid="ignore", divide="ignore"):
        jaccard = np.where(either > 0, both / either, np.nan)
    out = xr.Dataset({"jaccard": (g.time_dim, jaccard),
                      "cells_a": (g.time_dim, pa.sum(axis=space)),
                      "cells_b": (g.time_dim, pb.sum(axis=space))},
                     coords={g.time_dim: da[g.time_dim]})
    if a.has_label and b.has_label:
        la = np.asarray(da["label"].transpose(*order).values)
        lb = np.asarray(db["label"].transpose(*order).values)
        weights = cell_weights(da, g)[0].ravel()
        ta, tb = _table(da, g.time_dim), _table(db, g.time_dim)
        iou = np.zeros(ta.n)
        match = np.zeros(ta.n, dtype=np.int32)
        for t in np.flatnonzero((la > 0).any(axis=space) & (lb > 0).any(axis=space)):
            x, y, shared = _track.overlap_pairs(la[t].ravel(), lb[t].ravel(), weights)
            gx, gy = ta.base[t] + x - 1, tb.base[t] + y - 1
            value = shared / (ta.area[gx] + tb.area[gy] - shared)
            for i in np.lexsort((y, -value)):      # best first; ties: lowest label of b
                if value[i] > iou[gx[i]]:
                    iou[gx[i]], match[gx[i]] = value[i], y[i]
        out["n_objects_a"] = (g.time_dim, np.diff(ta.base))
        out["n_objects_b"] = (g.time_dim, np.diff(tb.base))
        out["best_iou"] = ("object_a", iou)
        out["best_match_b"] = ("object_a", match)
        out["object_a_time"] = ("object_a", da["object_time"].values)
        out["object_a_label"] = ("object_a", da["object_label"].values)
    return out


def _table(ds: xr.Dataset, time_dim: str) -> _track.Table:
    """Only the fields `compare_objects` needs: areas and the first object of each day."""
    day = np.searchsorted(ds[time_dim].values, ds["object_time"].values)
    base = np.concatenate(([0], np.cumsum(np.bincount(day, minlength=ds.sizes[time_dim]))))
    zeros = np.zeros(day.size, dtype=np.int64)
    return _track.Table(day=zeros, label=zeros, area=np.asarray(ds["object_area"].values),
                        cells=zeros, first_cell=zeros, base=base)
