# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Optional partition of merged patches by their nearest parent cell.

The default genealogy behavior is not to partition: a merged patch is one node with several
parents. This operation, off unless called, returns a **new** ObjectField in which every patch
with two or more linked parents on the previous day is split into one part per parent: each
cell goes to the parent with the nearest cell on the previous day (Euclidean distance on the
index grid, periodic in x when the grid is; in a volume over (z, y, x)), as marEx's
``nn_partitioning`` does on structured grids. The first parent in raster order keeps the
patch. Days are processed in order, so the parents of a day are the parts of the previous one.
A part may be split in pieces.

It changes the objects (their number, areas and merges, and how events connect), never the
detected mask: the processed cells are the same, and `check_against` still holds. It is not
equivalent to the partition of marEx: marEx consolidates identities as it goes, so who is a
parent can differ; no equivalence is claimed without a comparison with marEx itself.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from . import _track
from ._errors import NotSupportedError
from ._lineage import LinkPolicy, field_arrays
from ._mask import XMask
from ._objects import ObjectField, _assemble, build_geometry
from ._schema import require
from ._specs import ProcessingStep, Provenance


def _relabel(day: np.ndarray) -> np.ndarray:
    """Labels 1..n of one day in order of first cell (raster order)."""
    flat_day = day.ravel()
    values, first = np.unique(flat_day, return_index=True)
    keep = values > 0
    values, first = values[keep], first[keep]
    lut = np.zeros(int(flat_day.max()) + 1 if flat_day.size else 1, np.int32)
    lut[values[np.argsort(first)]] = np.arange(1, values.size + 1, dtype=np.int32)
    return lut[day]


def _nearest(parents: np.ndarray, spatial_shape, periodic: bool) -> np.ndarray:
    """For every cell, the label of the nearest cell of `parents` (labels > 0) on the index grid."""
    grid = parents.reshape(spatial_shape)
    if periodic:
        grid = np.concatenate([grid, grid, grid], axis=-1)
    _, index = distance_transform_edt(grid == 0, return_indices=True)
    nearest = grid[tuple(index)]
    if periodic:
        nx = spatial_shape[-1]
        nearest = nearest[..., nx:2 * nx]
    return nearest.reshape(parents.shape)


def partition_merges(objects: ObjectField, mask: XMask, *,
                     policy: LinkPolicy | None = None,
                     provenance: Provenance | None = None) -> ObjectField:
    """Split every merged patch of `objects` among its parents (see the module for marEx).

    `mask` is the XMask the objects were built from; `policy` decides which overlaps are links
    (default: that of the track step, as in `build_lineage`). Only gap-free links of
    consecutive days are partitioned. Returns a new ObjectField with the same recipe and
    analysis mode, tables and tracks recomputed, and a provenance step ``partition_merges``.
    """
    require(isinstance(objects, ObjectField), "objects must be an ObjectField")
    require(isinstance(mask, XMask), "mask must be an XMask")
    require(objects.has_label, "the ObjectField has no patches (no segment step)")
    require(objects.recipe is not None, "partition_merges needs objects built by xHW "
            "(external objects can be partitioned by their own method and ingested)",
            NotSupportedError)
    require(mask.provenance.run_id == objects.source_mask_run_id,
            "mask is not the source of the objects")
    if policy is None:
        policy = (LinkPolicy.from_track(objects.recipe.steps[-1]) if objects.recipe.has_track
                  else LinkPolicy())
    g, md = mask.grid, mask.dataset
    labels, weights, unit, _ = field_arrays(objects)
    out = labels.copy()
    rows, nx = out.shape[1:]
    spatial_shape = ((md.sizes[g.spatial_dims[0]], rows // md.sizes[g.spatial_dims[0]], nx)
                     if g.support == "volume" else (rows, nx))
    n_split = 0
    for t in range(1, out.shape[0]):
        a, b = out[t - 1], out[t]
        if not a.any() or not b.any():
            continue
        la, lb, shared = _track.overlap_pairs(a, b, weights)
        area_a = np.bincount(a.ravel(), weights=weights.ravel(), minlength=int(a.max()) + 1)
        area_b = np.bincount(b.ravel(), weights=weights.ravel(), minlength=int(b.max()) + 1)
        value = _track.measure(policy.measure, shared, area_a[la], area_b[lb])
        linked = (shared > 0) & (value >= policy.threshold)
        la, lb = la[linked], lb[linked]
        day = b.copy()
        next_label = int(b.max()) + 1
        for child in np.unique(lb):
            parents = np.sort(la[lb == child])
            if parents.size < 2:
                continue
            n_split += 1
            nearest = _nearest(np.where(np.isin(a, parents), a, 0), spatial_shape,
                               g.periodic_x)
            cells = b == child
            for i, parent in enumerate(parents):
                if i == 0:
                    continue                          # the first parent keeps the patch
                day[cells & (nearest == parent)] = next_label
                next_label += 1
        out[t] = _relabel(day)
    order = (g.time_dim, *g.spatial_dims)
    lift = (lambda x: x[:, None, :]) if g.support == "line" else (lambda x: x)
    valid = lift(np.asarray(md["valid"].transpose(*order).values))
    source = lift(np.asarray(md[objects.source_layer].transpose(*order).values)) & valid
    processed = lift(np.asarray(objects.dataset["processed_mask"].transpose(*order).values))
    if provenance is None:
        provenance = Provenance.new(parents=(objects.provenance.run_id, mask.provenance.run_id),
                                    steps=(ProcessingStep(
                                        name="partition_merges",
                                        params=(("policy_id", policy.spec_id),
                                                ("method", "nearest_parent_cell"),
                                                ("n_partitioned", n_split))),))
    # the cells do not change: the processed mask of the objects is passed as it is (with a
    # track step `_assemble` derives it again from the recomputed tracks)
    return _assemble(mask, objects.recipe, objects.analysis_mode, provenance, source, processed,
                     out, weights, unit, build_geometry(md, g)[2], md[g.time_dim].values, order)
