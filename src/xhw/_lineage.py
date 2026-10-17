# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""LineageGraph and `build_lineage`: genealogy of the daily patches of an ObjectField.

The node of the graph is the daily patch of object tracking. The edges are every pair of
patches of consecutive days that share cells, with four overlap measures, plus the links
across empty days that the gap rule of object tracking chooses; ``linked`` says which edges
count under a `LinkPolicy`. Edges below the threshold stay in the table. Everything else is
derived from the linked edges:

- node type by life cycle (``isolated``, ``birth``, ``death``, ``continuation``, ``complex``)
  and the flags ``is_split`` (two or more linked children) and ``is_merge`` (two or more
  linked parents);
- branches: maximal chains of 1 -> 1 links; a split ends the parent branch and starts one
  branch per child, a merge ends the incoming branches and starts one ("terminate and
  create");
- the dominance annotation of every linked edge, with the rule of object tracking monitoring
  (largest ``dominance_measure``, ties by ``tie_rule``), and ``dominant_tied`` when a tie
  decided;
- spacetime events: in ``retrospective`` the connected components of the linked edges,
  numbered like the ``track_id`` of object tracking (by first day, then first cell); in
  ``monitoring`` events are assigned forward and never rewritten: when events join, by default
  both are frozen with ``merged_into`` pointing to a new event (``merge_rule="new_event"``),
  or the dominant one survives as in object tracking (``merge_rule="survivor"``). The metrics
  of the event they join are computed over the union of their nodes .

Works on lines, surfaces and volumes of structured objects: the edges are computed on the flat
view of object tracking, so they do not depend on the support; the measures are in the cell
measure of the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr

from . import _lineage_metrics as lm
from . import _track
from ._errors import InvariantError, SchemaError
from ._mask import check_entity, check_stored_id, load_attr_json, read_only, strip_xhw_attrs
from ._objects import OBJECT_DIM, ObjectField, build_geometry, cell_weights, flat, make_table
from ._recipe import DOMINANCE_MEASURES, LINK_MEASURES, TIE_RULES
from ._schema import (
    CONVENTIONS,
    SCHEMA_VERSIONS,
    SpecBase,
    check_choice,
    check_serializable_attrs,
    is_int,
    is_number,
    require,
)
from ._specs import GridSpec, ProcessingStep, Provenance, _spec

TRANSITION_DIM, BRANCH_DIM, BRANCH_LINK_DIM, EVENT_DIM = (
    "transition", "branch", "branch_link", "spacetime_event")
NODE_TYPES = ("isolated", "birth", "death", "continuation", "complex")
MERGE_RULES = ("new_event", "survivor")
MEASURES = ("overlap_smaller", "iou", "source_fraction", "target_fraction")
_TRANSITION_VARS = ("source_object", "target_object", "source_time", "target_time", "gap_days",
                    "shared", *MEASURES, "linked", "is_dominant_for_source",
                    "is_dominant_for_target", "dominant_tied")
_NODE_VARS = ("object_time", "object_label", "in_degree", "out_degree", "node_type", "is_split",
              "is_merge", "object_branch", "object_spacetime_event", "provisional", "backend_id")
_GROUP_METRICS = ("area_max", "peak_area_day", "area_days", "footprint", "net_displacement",
                  "path_length", "left_censored", "right_censored", "provisional")
_BRANCH_VARS = ("branch_number", "branch_start", "branch_end", "branch_duration_days",
                "branch_active_days", "branch_bridged_days", "branch_spacetime_event",
                *(f"branch_{name}" for name in _GROUP_METRICS))
_LINK_VARS = ("branch_link_parent", "branch_link_child")
_EVENT_VARS = ("event_number", "event_start", "event_end", "event_duration_days",
               "event_active_days", "event_bridged_days", "event_n_objects", "event_n_branches",
               "event_n_splits", "event_n_merges", "event_merged_into",
               *(f"event_{name}" for name in _GROUP_METRICS))
_ATTRS = ("Conventions", "xhw_schema", "xhw_schema_version", "xhw_grid", "xhw_policy",
          "policy_id", "xhw_provenance", "definition_id", "baseline_id", "source_objects_run_id",
          "source_mask_run_id", "analysis_mode", "merge_rule", "final_after_days",
          "window_start", "window_end")
_TIE = 1e-12          # relative tolerance under which two dominance scores are equal


# --- policy ----------------------------------------------------------------------------------


@_spec
class LinkPolicy(SpecBase):
    """Which edges of the transition table count as links (genealogy, ).

    ``measure`` is thresholded by ``threshold`` with ``>=`` (the comparator of object
    tracking): ``overlap_smaller`` (shared / area of the smaller patch, the rule of object
    tracking, marEx and Ocetrac), ``iou`` (shared / union), ``source_fraction`` (shared / area
    of the earlier patch) or ``target_fraction`` (shared / area of the later patch). The
    default, ``overlap_smaller >= 0`` with a positive shared area, links any shared cell.
    ``max_gap_days``: empty days a patch may be away and still be linked (the gap rule of
    object tracking: one candidate per reappearing patch). ``dominance_measure`` and
    ``tie_rule``: the dominance annotation and the bridged candidate (as in the object
    tracking track step). `from_track` builds the policy of an object tracking track step.
    """

    measure: Literal["overlap_smaller", "iou", "source_fraction",
                     "target_fraction"] = "overlap_smaller"
    threshold: float = 0.0
    max_gap_days: int = 0
    dominance_measure: Literal["overlap_smaller", "iou", "source_fraction", "target_fraction",
                               "shared"] = "overlap_smaller"
    tie_rule: Literal["lowest_id", "largest_area"] = "lowest_id"

    def __post_init__(self) -> None:
        check_choice("measure", self.measure, LINK_MEASURES)
        check_choice("dominance_measure", self.dominance_measure, DOMINANCE_MEASURES)
        check_choice("tie_rule", self.tie_rule, TIE_RULES)
        require(is_number(self.threshold) and 0.0 <= self.threshold <= 1.0,
                "threshold must be in [0, 1] (the measures are fractions)")
        require(is_int(self.max_gap_days) and self.max_gap_days >= 0,
                "max_gap_days must be an integer >= 0")

    @classmethod
    def from_track(cls, step) -> LinkPolicy:
        """The policy of an object tracking track step (its threshold, gap and rules)."""
        require(step.op == "track", "from_track needs a track step")
        return cls(measure=step.rule("link_measure"), threshold=float(step.min_overlap),
                   max_gap_days=int(step.max_gap_days),
                   dominance_measure=step.rule("dominance_measure"),
                   tie_rule=step.rule("tie_rule"))


# --- inputs ----------------------------------------------------------------------------------


def field_arrays(objects: ObjectField):
    """``(labels, weights, unit, table)`` of an ObjectField: the flat view (time, z*y, x) of
    object tracking (a line is one row) and its object table.
    """
    g, ds = objects.grid, objects.dataset
    labels = np.asarray(ds["label"].transpose(g.time_dim, *g.spatial_dims).values)
    if g.support == "line":
        labels = labels[:, None, :]
    labels = flat(labels)
    weights, unit = cell_weights(ds, g)
    weights = weights.reshape(-1, weights.shape[-1])
    table = make_table(labels, weights, None, None)[0]
    require(table.n == ds.sizes[OBJECT_DIM], "label field and object table disagree")
    return labels, weights, unit, table


def _components(table, src, dst, cells_per_day: int) -> np.ndarray:
    """Connected components of the edges, numbered 1.. by first day then first cell (the
    numbering of the retrospective tracks of object tracking).
    """
    z = np.zeros(src.size)
    return _track.retrospective_tracks(table, _track.Links(src, dst, z, z.astype(int), z),
                                       cells_per_day)


def _edges(labels, weights, table, policy: LinkPolicy):
    """All consecutive-day pairs that share cells, plus the bridges of the gap rule; sorted by
    (source, target).
    """
    src, dst, shared = [], [], []
    counts = np.diff(table.base)
    for t in range(labels.shape[0] - 1):
        if counts[t] and counts[t + 1]:
            a, b, sh = _track.overlap_pairs(labels[t], labels[t + 1], weights)
            keep = sh > 0
            src.append(table.base[t] + a[keep] - 1)
            dst.append(table.base[t + 1] + b[keep] - 1)
            shared.append(sh[keep])
    links = _track.build_links(labels, weights, table, min_overlap=policy.threshold,
                               max_gap_days=policy.max_gap_days, link_measure=policy.measure,
                               dominance_measure=policy.dominance_measure,
                               tie_rule=policy.tie_rule)
    bridge = links.gap > 0
    n_cons = sum(x.size for x in src)
    src = np.concatenate(src + [links.src[bridge]]).astype(np.int64)
    dst = np.concatenate(dst + [links.dst[bridge]]).astype(np.int64)
    shared = np.concatenate(shared + [links.shared[bridge]]).astype(float)
    gap = np.concatenate([np.zeros(n_cons, np.int64), links.gap[bridge].astype(np.int64)])
    a_src, a_dst = table.area[src], table.area[dst]
    values = {name: _track.measure(name, shared, a_src, a_dst) for name in MEASURES}
    linked = np.concatenate([values[policy.measure][:n_cons] >= policy.threshold,
                             np.ones(int(bridge.sum()), bool)])
    order = np.lexsort((dst, src))
    out = {"src": src, "dst": dst, "shared": shared, "gap": gap, "linked": linked, **values}
    return {k: v[order] for k, v in out.items()}, links


def _dominance(group, other, score, area_other, n, by_area: bool):
    """For each node of `group`, its dominant edge among the given (linked) edges: the largest
    `score`, ties by the tie rule. Returns (is dominant, won by a tie) per edge.
    """
    m = group.size
    dominant, tied = np.zeros(m, bool), np.zeros(m, bool)
    if m == 0:
        return dominant, tied
    keys = (other, -area_other, -score, group) if by_area else (other, -score, group)
    order = np.lexsort(keys)
    g_sorted = group[order]
    first = np.ones(m, bool)
    first[1:] = g_sorted[1:] != g_sorted[:-1]
    winners = order[first]
    dominant[winners] = True
    best = np.full(n, -np.inf)
    np.maximum.at(best, group, score)
    near = np.isclose(score, best[group], rtol=_TIE, atol=0.0)
    n_best = np.bincount(group, weights=near, minlength=n)
    tied[winners] = n_best[group[winners]] > 1
    return dominant, tied


def _node_types(in_deg, out_deg) -> np.ndarray:
    kind = np.full(in_deg.size, NODE_TYPES.index("continuation"), np.int8)
    kind[(in_deg == 0) & (out_deg > 0)] = NODE_TYPES.index("birth")
    kind[(in_deg > 0) & (out_deg == 0)] = NODE_TYPES.index("death")
    kind[(in_deg == 0) & (out_deg == 0)] = NODE_TYPES.index("isolated")
    kind[(in_deg >= 2) & (out_deg >= 2)] = NODE_TYPES.index("complex")
    return kind


def _monitoring_events(n, src, dst, score, area, merge_rule: str, by_area: bool):
    """Forward event ids that are never rewritten. Returns (event per node, merged_into per
    event, 1-based with a leading 0).
    """
    event = np.zeros(n, np.int32)
    merged = [0]
    order = np.lexsort((src, dst))
    d_sorted, s_sorted, sc_sorted = dst[order], src[order], score[order]
    start = np.searchsorted(d_sorted, np.arange(n + 1))

    def live(e: int) -> int:
        while merged[e]:
            e = merged[e]
        return e

    def new() -> int:
        merged.append(0)
        return len(merged) - 1

    for obj in range(n):
        lo, hi = start[obj], start[obj + 1]
        preds = list(zip(sc_sorted[lo:hi].tolist(), s_sorted[lo:hi].tolist()))
        if not preds:
            event[obj] = new()
            continue
        joined = sorted({live(int(event[p])) for _, p in preds})
        if len(joined) == 1:
            event[obj] = joined[0]
        elif merge_rule == "new_event":
            c = new()
            for e in joined:
                merged[e] = c
            event[obj] = c
        # the dominant parent's event survives (the rule of object tracking)
        else:
            key = ((lambda ro: (-ro[0], -area[ro[1]], live(int(event[ro[1]])), ro[1])) if by_area
                   else (lambda ro: (-ro[0], live(int(event[ro[1]])), ro[1])))
            keep = live(int(event[min(preds, key=key)[1]]))
            for e in joined:
                if e != keep:
                    merged[e] = keep
            event[obj] = keep
    return event, np.array(merged, np.int32)


def pd_isnull(a) -> np.ndarray:
    """NaN or NaT, elementwise (False for integers and booleans)."""
    a = np.asarray(a)
    if a.dtype.kind in "fc":
        return np.isnan(a)
    if a.dtype.kind in "mM":
        return np.isnat(a)
    return np.zeros(a.shape, bool)


def _reaches(start, goal, merged) -> np.ndarray:
    """Whether following ``merged_into`` from each event of `start` reaches `goal`."""
    ok = np.zeros(np.size(start), bool)
    for i, (e, target) in enumerate(zip(np.asarray(start).tolist(), np.asarray(goal).tolist())):
        while e and e != target:
            e = int(merged[e])
        ok[i] = e == target
    return ok


# --- the container ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LineageGraph:
    """Genealogy of the daily patches of one ObjectField under one `LinkPolicy` (genealogy).

    Dimensions: ``transition`` (edges), ``object`` (nodes, the rows of the ObjectField's
    object table), ``branch``, ``branch_link`` (parent -> child branch relations) and
    ``spacetime_event``. The transition table is the view `transitions`. Build it with
    `build_lineage`; read a serialized one with ``from_dataset(ds)``.
    """

    _dataset: xr.Dataset
    grid: GridSpec
    policy: LinkPolicy
    analysis_mode: str
    merge_rule: str
    provenance: Provenance
    definition_id: str
    baseline_id: str
    source_objects_run_id: str
    source_mask_run_id: str
    final_after_days: int
    window_start: str
    window_end: str

    @classmethod
    def _make(cls, ds: xr.Dataset, **fields) -> LineageGraph:
        ds = strip_xhw_attrs(ds)
        ds.attrs = {k: v for k, v in ds.attrs.items() if k not in _ATTRS}
        graph = cls(read_only(ds), **fields)
        graph._validate()
        return graph

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, *, copy: bool = False) -> LineageGraph:
        """Read a dataset produced by `to_dataset` (schema policy and ids are enforced)."""
        require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
        check_entity(ds, "LineageGraph")
        for name in _ATTRS[3:]:
            require(name in ds.attrs, f"serialized attribute {name!r} is missing", SchemaError)
        policy = LinkPolicy.from_dict(load_attr_json(ds, "xhw_policy"))
        check_stored_id(ds, "policy_id", policy)
        return cls._make(
            ds.copy(deep=True) if copy else ds,
            grid=GridSpec.from_dict(load_attr_json(ds, "xhw_grid")), policy=policy,
            analysis_mode=ds.attrs["analysis_mode"], merge_rule=ds.attrs["merge_rule"],
            provenance=Provenance.from_dict(load_attr_json(ds, "xhw_provenance")),
            definition_id=ds.attrs["definition_id"], baseline_id=ds.attrs["baseline_id"],
            source_objects_run_id=ds.attrs["source_objects_run_id"],
            source_mask_run_id=ds.attrs["source_mask_run_id"],
            final_after_days=int(ds.attrs["final_after_days"]),
            window_start=ds.attrs["window_start"], window_end=ds.attrs["window_end"])

    # -- access -------------------------------------------------------------------------

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset.copy(deep=False)

    def transitions(self) -> xr.Dataset:
        """The transition table: the edges, one row per pair of patches."""
        return self._dataset[[name for name in _TRANSITION_VARS]].copy(deep=False)

    def members(self, *, spacetime_event: int | None = None,
                branch: int | None = None) -> np.ndarray:
        """Nodes (object rows) of one branch or one spacetime event. The nodes of an event
        include those of the events it absorbed (monitoring).
        """
        require((spacetime_event is None) != (branch is None),
                "give exactly one of spacetime_event or branch")
        ds = self._dataset
        if branch is not None:
            require(1 <= branch <= ds.sizes[BRANCH_DIM], f"no branch {branch}")
            return np.flatnonzero(np.asarray(ds["object_branch"].values) == branch)
        require(1 <= spacetime_event <= ds.sizes[EVENT_DIM],
                f"no spacetime event {spacetime_event}")
        event = np.asarray(ds["object_spacetime_event"].values)
        merged = np.concatenate(([0], np.asarray(ds["event_merged_into"].values)))
        return np.flatnonzero(_reaches(event, np.full(event.size, spacetime_event), merged))

    def series(self, *, spacetime_event: int | None = None,
               branch: int | None = None) -> xr.Dataset:
        """Level-2 series of one branch or event: per day with a patch, the size, the number of
        patches, the size-weighted centroid and, when the graph has intensity, the daily
        maximum and mean. A view of the node table; nothing is stored.
        """
        ds = self._dataset
        nodes = self.members(spacetime_event=spacetime_event, branch=branch)
        times = np.asarray(ds["object_time"].values)[nodes]
        days, inv = np.unique(times, return_inverse=True)
        area = np.asarray(ds["object_area"].values, float)[nodes]
        total = np.bincount(inv, weights=area, minlength=days.size)
        safe = np.where(total > 0, total, 1.0)
        out = xr.Dataset(coords={"time": days})
        out["area"] = ("time", total, dict(ds["object_area"].attrs))
        out["n_objects"] = ("time", np.bincount(inv, minlength=days.size))
        if "object_lat" in ds:
            phi = np.deg2rad(np.asarray(ds["object_lat"].values)[nodes])
            lam = np.deg2rad(np.asarray(ds["object_lon"].values)[nodes])
            vec = [np.bincount(inv, weights=area * f, minlength=days.size) for f in (
                np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))]
            out["centroid_lat"] = ("time", np.rad2deg(np.arctan2(vec[2], np.hypot(vec[0], vec[1]))),
                                   {"units": "degrees_north"})
            out["centroid_lon"] = ("time", np.rad2deg(np.arctan2(vec[1], vec[0])),
                                   {"units": "degrees_east"})
        for name in ("object_depth", "object_along"):
            if name in ds:
                values = np.asarray(ds[name].values, float)[nodes]
                out["centroid_" + name.split("_", 1)[1]] = (
                    "time", np.bincount(inv, weights=area * values, minlength=days.size) / safe,
                    dict(ds[name].attrs))
        if "object_intensity_max" in ds:
            peak = np.full(days.size, np.nan)
            np.fmax.at(peak, inv, np.asarray(ds["object_intensity_max"].values)[nodes])
            out["intensity_max"] = ("time", peak, dict(ds["object_intensity_max"].attrs))
            weighted = "unweighted" not in ds["object_intensity_mean"].attrs.get("cell_methods", "")
            name = "object_intensity_valid_measure" if weighted else "object_intensity_valid_cells"
            w = np.asarray(ds[name].values, float)[nodes]
            mean = np.asarray(ds["object_intensity_mean"].values)[nodes]
            den = np.bincount(inv, weights=w, minlength=days.size)
            num = np.bincount(inv, weights=w * np.where(np.isfinite(mean), mean, 0.0),
                              minlength=days.size)
            out["intensity_mean"] = ("time", np.where(den > 0, num / np.where(den > 0, den, 1.0),
                                                     np.nan),
                                     dict(ds["object_intensity_mean"].attrs))
        return out

    def maps(self, objects: ObjectField, *, spacetime_event: int | None = None,
             branch: int | None = None, field: xr.DataArray | None = None) -> xr.Dataset:
        """Level-3 maps of one branch or event, computed on demand from the label field of
        `objects` (the ObjectField the graph was built from): ``frequency`` (days each cell
        was occupied; the footprint is ``frequency > 0``), ``first_day``, ``last_day`` and
        ``peak_area`` (the cells on the day of the largest size). With `field` (off by
        default): ``intensity_mean`` and ``intensity_max`` over the occupied days of each
        cell, ``intensity_cumulative`` (sum over those days, value x day),
        ``day_of_max_intensity`` and ``peak_intensity`` (the field on the day of the largest
        intensity, NaN elsewhere).
        """
        require(objects.provenance.run_id == self.source_objects_run_id,
                "the maps need the ObjectField the graph was built from")
        nodes = self.members(spacetime_event=spacetime_event, branch=branch)
        g, ods = objects.grid, objects.dataset
        labels, _, _, table = field_arrays(objects)
        T, rows, nx = labels.shape
        times = ods[g.time_dim].values
        member = np.zeros(table.n, bool)
        member[nodes] = True
        freq = np.zeros((rows, nx), np.int64)
        first = np.full((rows, nx), -1, np.int64)
        last = np.full((rows, nx), -1, np.int64)
        size = np.zeros(T)
        values = _field_values(field, objects) if field is not None else None
        vsum, vmax = np.zeros((rows, nx)), np.full((rows, nx), np.nan)
        valid_days = np.zeros((rows, nx), np.int64)
        vday = np.full((rows, nx), -1, np.int64)
        daily_max = np.full(T, np.nan)
        flat_w = field_arrays(objects)[1]
        for t in np.unique(table.day[nodes]):
            lab = labels[t]
            on = (lab > 0) & member[np.where(lab > 0, table.base[t] + lab - 1, 0)]
            freq += on
            first[on & (first < 0)] = t
            last[on] = t
            size[t] = float(flat_w[on].sum())
            if values is not None:
                v = np.where(on, values[t], np.nan)
                finite = np.isfinite(v)
                valid_days += finite
                vsum += np.where(finite, v, 0.0)
                better = on & ~(v <= vmax) & np.isfinite(v)
                vmax = np.where(better, v, vmax)
                vday[better] = t
                daily_max[t] = np.nanmax(v) if np.isfinite(v).any() else np.nan
        peak_t = int(np.argmax(size))
        shape = tuple(ods.sizes[d] for d in g.spatial_dims)

        def back(a):
            return a.reshape(shape)

        def stamp(index):
            out = np.full(index.shape, np.datetime64("NaT", "ns"))
            out[index >= 0] = times[index[index >= 0]]
            return back(out)

        lab_peak = labels[peak_t]
        on_peak = (lab_peak > 0) & member[np.where(lab_peak > 0,
                                                   table.base[peak_t] + lab_peak - 1, 0)]
        out = xr.Dataset({"frequency": (g.spatial_dims, back(freq)),
                          "first_day": (g.spatial_dims, stamp(first)),
                          "last_day": (g.spatial_dims, stamp(last)),
                          "peak_area": (g.spatial_dims, back(on_peak))},
                         coords={k: v for k, v in ods["valid"].isel({g.time_dim: 0}).coords.items()
                                 if k != g.time_dim})
        out["frequency"].attrs["units"] = "day"
        out.attrs["peak_area_day"] = str(np.datetime_as_string(times[peak_t], unit="D"))
        if values is not None:
            out["intensity_mean"] = (
                g.spatial_dims,
                back(np.where(valid_days > 0, vsum / np.maximum(valid_days, 1), np.nan)),
                {"units": field.attrs.get("units", "")},
            )
            out["intensity_max"] = (g.spatial_dims, back(vmax), {"units": field.attrs.get(
                "units", "")})
            out["intensity_cumulative"] = (
                g.spatial_dims,
                back(np.where(valid_days > 0, vsum, np.nan)),
                {"units": f"{field.attrs.get('units', '')} day"},
            )
            out["day_of_max_intensity"] = (g.spatial_dims, stamp(vday))
            ok = np.isfinite(daily_max)
            if ok.any():
                tp = int(np.nanargmax(np.where(ok, daily_max, -np.inf)))
                lab = labels[tp]
                on = (lab > 0) & member[np.where(lab > 0, table.base[tp] + lab - 1, 0)]
                out["peak_intensity"] = (g.spatial_dims, back(np.where(on, values[tp], np.nan)),
                                         {"units": field.attrs.get("units", "")})
                out.attrs["peak_intensity_day"] = str(np.datetime_as_string(times[tp], unit="D"))
        return out

    def to_event_catalog(self, mask, *, provenance: Provenance | None = None):
        """The spacetime events as an `EventCatalog` with ``granularity="object_tracked"``: one
        row per event, no location dims. `mask` is the XMask of the objects (it gives the
        definition and the baseline). ``n_exceedance_days`` and ``n_joined_gap_days`` are the
        days with a patch and the bridged days; the metrics of the graph follow without the
        ``event_`` prefix. In monitoring the rows include the events frozen by a join
        (``merged_into`` > 0).
        """
        from ._catalog import WINDOW_END, WINDOW_START, EventCatalog
        from ._mask import XMask

        require(isinstance(mask, XMask), "mask must be an XMask")
        require(mask.provenance.run_id == self.source_mask_run_id,
                "the mask is not the source of the objects of this graph")
        require(mask.definition_id == self.definition_id and mask.baseline_id == self.baseline_id,
                "definition or baseline differs from the mask")
        ds = self._dataset
        k = ds.sizes[EVENT_DIM]
        start, end = ds["event_start"].values, ds["event_end"].values
        w0, w1 = np.datetime64(self.window_start, "ns"), np.datetime64(self.window_end, "ns")
        reason = ((start == w0) * WINDOW_START + (end == w1) * WINDOW_END).astype(np.uint8)
        rows = xr.Dataset({
            "event_id": ("event", ds["event_number"].values.astype(np.int64)),
            "start": ("event", start), "end": ("event", end),
            "duration_days": ("event", ds["event_duration_days"].values.astype(np.int64)),
            "n_exceedance_days": ("event", ds["event_active_days"].values.astype(np.int64)),
            "n_joined_gap_days": ("event", ds["event_bridged_days"].values.astype(np.int64)),
            "left_censored": ("event", start == w0), "right_censored": ("event", end == w1),
            "qualified": ("event", np.ones(k, bool)),
            "contains_forecast": ("event", np.zeros(k, bool)),
            "censor_reason": ("event", reason),
        })
        skip = {"event_number", "event_start", "event_end", "event_duration_days",
                "event_active_days", "event_bridged_days", "event_left_censored",
                "event_right_censored"}
        for name, var in ds.data_vars.items():
            if var.dims == (EVENT_DIM,) and name not in skip:
                rows[name.split("_", 1)[1]] = ("event", var.values, dict(var.attrs))
        if provenance is None:
            provenance = Provenance.new(
                parents=(mask.provenance.run_id, self.provenance.run_id),
                steps=(ProcessingStep(name="to_event_catalog",
                                      params=(("granularity", "object_tracked"),)),))
        return EventCatalog.from_dataset(rows, mask=mask, granularity="object_tracked",
                                         analysis_mode=self.analysis_mode,
                                         provenance=provenance)

    def check_catalog(self, catalog) -> None:
        """Check an ``object_tracked`` EventCatalog against this graph (ids, rows and metrics)."""
        require(catalog.granularity == "object_tracked", "not an object_tracked catalog")
        require(catalog.source_mask_run_id == self.source_mask_run_id,
                "catalog and graph come from different masks")
        require(self.provenance.run_id in catalog.provenance.parents,
                "the catalog does not list this graph as a parent")
        rows, ds = catalog.dataset, self._dataset
        pairs = [("event_id", "event_number"), ("start", "event_start"), ("end", "event_end"),
                 ("n_exceedance_days", "event_active_days"),
                 ("n_joined_gap_days", "event_bridged_days")]
        pairs += [(n.split("_", 1)[1], n) for n, v in ds.data_vars.items()
                  if v.dims == (EVENT_DIM,) and n.split("_", 1)[1] in rows]
        for row_name, graph_name in pairs:
            a, b = np.asarray(rows[row_name].values), np.asarray(ds[graph_name].values)
            same = (a.shape == b.shape) and bool(np.all((a == b) | (pd_isnull(a) & pd_isnull(b))))
            if not same:
                raise InvariantError(f"catalog {row_name!r} differs from the graph")

    def node_type_names(self) -> np.ndarray:
        """Node types as strings (the dataset stores int8 codes)."""
        return np.asarray(NODE_TYPES, dtype=object)[np.asarray(self._dataset["node_type"].values)]

    # -- serialization ------------------------------------------------------------------

    def to_dataset(self) -> xr.Dataset:
        ds = self._dataset.copy(deep=False)
        ds.attrs.update({
            "Conventions": CONVENTIONS,
            "xhw_schema": "LineageGraph",
            "xhw_schema_version": SCHEMA_VERSIONS["LineageGraph"],
            "xhw_grid": self.grid.to_json(),
            "xhw_policy": self.policy.to_json(),
            "policy_id": self.policy.spec_id,
            "xhw_provenance": self.provenance.to_json(),
            "definition_id": self.definition_id,
            "baseline_id": self.baseline_id,
            "source_objects_run_id": self.source_objects_run_id,
            "source_mask_run_id": self.source_mask_run_id,
            "analysis_mode": self.analysis_mode,
            "merge_rule": self.merge_rule,
            "final_after_days": int(self.final_after_days),
            "window_start": self.window_start,
            "window_end": self.window_end,
        })
        for name in ("linked", "is_dominant_for_source", "is_dominant_for_target",
                     "dominant_tied", "is_split", "is_merge", "provisional",
                     "event_left_censored", "event_right_censored"):
            ds[name].attrs.update({"flag_values": np.array([0, 1], dtype=np.int8),
                                   "flag_meanings": "false true"})
        ds["node_type"].attrs.update({"flag_values": np.arange(len(NODE_TYPES), dtype=np.int8),
                                      "flag_meanings": " ".join(NODE_TYPES)})
        return ds

    # -- validation ---------------------------------------------------------------------

    def _validate(self) -> None:
        ds = self._dataset
        check_choice("analysis_mode", self.analysis_mode, ("retrospective", "monitoring"))
        check_choice("merge_rule", self.merge_rule, MERGE_RULES)
        require(isinstance(self.policy, LinkPolicy), "policy must be a LinkPolicy")
        for names, dim in ((_TRANSITION_VARS, TRANSITION_DIM), (_NODE_VARS, OBJECT_DIM),
                           (_BRANCH_VARS, BRANCH_DIM), (_LINK_VARS, BRANCH_LINK_DIM),
                           (_EVENT_VARS, EVENT_DIM)):
            for name in names:
                require(name in ds.data_vars and ds[name].dims == (dim,),
                        f"variable {name!r} is missing or not along {dim!r}")
        if "object_intensity_mean" in ds:
            for name in ("object_intensity_valid_cells", "object_intensity_valid_measure"):
                require(name in ds and ds[name].dims == (OBJECT_DIM,),
                        f"missing required intensity support variable {name!r} "
                        "(LineageGraph schema 0.1)", SchemaError)
        check_serializable_attrs("dataset", ds.attrs)
        for name, var in ds.variables.items():
            check_serializable_attrs(f"variable {name!r}", var.attrs)

    def check_invariants(self) -> None:
        """Rules of the graph: degrees, types, dominance, branches and events agree with the
        linked edges (1-D tables, cheap).
        """
        ds, problems = self._dataset, []

        def check(ok, message):
            if not bool(np.all(ok)):
                problems.append(message)

        v = {name: np.asarray(ds[name].values) for name in ds.data_vars}
        n = v["in_degree"].size
        src, dst, linked = v["source_object"], v["target_object"], v["linked"]
        check((src >= 0) & (src < n) & (dst >= 0) & (dst < n), "edge ends outside the nodes")
        check(v["source_time"] < v["target_time"], "edges must go forward in time")
        order = np.lexsort((dst, src))
        check(order == np.arange(src.size), "edges not sorted by (source, target)")
        check(v["shared"] > 0, "edges without shared cells")
        days = (v["target_time"] - v["source_time"]) // np.timedelta64(1, "D")
        check(v["gap_days"] == days - 1, "gap_days != target day - source day - 1")
        check((v["gap_days"] == 0) | linked, "a bridge across empty days must be linked")
        cons = v["gap_days"] == 0
        check(linked[cons] == (v[self.policy.measure][cons] >= self.policy.threshold),
              "linked != the policy on consecutive days")
        check(v["gap_days"] <= self.policy.max_gap_days, "bridge longer than max_gap_days")
        in_deg = np.bincount(dst[linked], minlength=n)
        out_deg = np.bincount(src[linked], minlength=n)
        check(v["in_degree"] == in_deg, "in_degree != linked edges into the node")
        check(v["out_degree"] == out_deg, "out_degree != linked edges out of the node")
        check(v["node_type"] == _node_types(in_deg, out_deg), "node_type != degrees")
        check(v["is_split"] == (out_deg >= 2), "is_split != two or more children")
        check(v["is_merge"] == (in_deg >= 2), "is_merge != two or more parents")
        for flag, group, deg in (("is_dominant_for_source", src, out_deg),
                                 ("is_dominant_for_target", dst, in_deg)):
            check(~v[flag] | linked, f"{flag} on an edge that is not linked")
            per = np.bincount(group[v[flag]], minlength=n)
            check(per == (deg > 0), f"{flag}: one dominant edge per node with linked edges")
        check(~v["dominant_tied"] | v["is_dominant_for_source"] | v["is_dominant_for_target"],
              "dominant_tied on an edge that is not dominant")
        branch = v["object_branch"]
        cont = linked & (out_deg[src] == 1) & (in_deg[dst] == 1)
        check(branch[src[cont]] == branch[dst[cont]], "a 1 -> 1 link changes branch")
        check(branch[src[linked & ~cont]] != branch[dst[linked & ~cont]],
              "a split or merge link stays in one branch")
        pairs = set(zip(v["branch_link_parent"].tolist(), v["branch_link_child"].tolist()))
        cut = linked & ~cont
        expect = set(zip(branch[src[cut]].tolist(), branch[dst[cut]].tolist()))
        check(pairs == expect, "branch links != the split and merge links")
        k = v["event_number"].size
        event = v["object_spacetime_event"]
        check((event >= 1) & (event <= k), "object_spacetime_event outside the event table")
        check(v["event_number"] == np.arange(1, k + 1), "event numbers must be 1..k")
        merged = np.concatenate(([0], v["event_merged_into"]))
        if self.analysis_mode == "retrospective":
            check(merged == 0, "retrospective events are never merged into others")
            check(event[src[linked]] == event[dst[linked]], "a link joins two events")
            check(v["branch_spacetime_event"][branch - 1] == event,
                  "a branch spans two spacetime events")
        else:
            check(_reaches(event[src[linked]], event[dst[linked]], merged),
                  "a link leads to an event that did not absorb the source event")
            check(_reaches(event, v["branch_spacetime_event"][branch - 1], merged),
                  "the event of a branch did not absorb the events of its nodes")
        if "object_intensity_mean" in v:
            count, weight = v["object_intensity_valid_cells"], v["object_intensity_valid_measure"]
            check(np.isfinite(count) & (count >= 0) & (count <= v["object_cells"])
                  & (count == np.floor(count)), "invalid intensity cell count")
            check(np.isfinite(weight) & (weight >= 0)
                  & (weight <= v["object_area"] * (1 + 1e-12)),
                  "invalid intensity valid measure")
            check((count == 0) == (weight == 0), "intensity denominators disagree")
            for name in ("object_intensity_mean", "object_intensity_max",
                         "object_intensity_cumulative"):
                check(np.isfinite(v[name]) == (count > 0), f"{name}: missing support mismatch")
        if problems:
            raise InvariantError("; ".join(dict.fromkeys(problems)))

    def check_against(self, objects: ObjectField) -> None:
        """Compare with the ObjectField: run ids, nodes and the edges recomputed from its label
        field (computes the field).
        """
        require(objects.provenance.run_id == self.source_objects_run_id,
                "source objects run id differs")
        require(objects.grid == self.grid, "grid differs from the objects")
        expected = build_lineage(objects, policy=self.policy, analysis_mode=self.analysis_mode,
                                 merge_rule=self.merge_rule, provenance=self.provenance)
        a, b = self._dataset, expected._dataset
        for name in (*_TRANSITION_VARS, *_NODE_VARS):
            if not np.array_equal(np.asarray(a[name].values), np.asarray(b[name].values)):
                raise InvariantError(f"{name!r} differs from the graph rebuilt from the objects")


# --- build -----------------------------------------------------------------------------------


def _field_values(field, objects: ObjectField) -> np.ndarray:
    """Validate cell identity before materializing the intensity field's flat view."""
    g, ods = objects.grid, objects.dataset
    require(isinstance(field, xr.DataArray), "field must be an xarray.DataArray")
    dims = (g.time_dim, *g.spatial_dims)
    require(set(field.dims) == set(dims), f"field must have the dims {dims}")
    for dim in dims:
        require(field.sizes[dim] == ods.sizes[dim],
                f"field dimension {dim!r} must have the size of the objects")
    names = {g.time_dim, *(d for d in g.spatial_dims if d in ods.coords)}
    names.update(n for n in (g.coords.lat, g.coords.lon, g.coords.depth, g.coords.along)
                 if n is not None)
    for name in sorted(names):
        require(name in field.coords, f"field coordinate {name!r} is missing; supply the "
                "objects' coordinates after explicitly aligning the field")
        a, b = field[name], ods[name]
        require(set(a.dims) == set(b.dims) and all(a.sizes[d] == b.sizes[d] for d in a.dims),
                f"field coordinate {name!r} has incompatible dimensions")
        require(a.transpose(*b.dims).variable.equals(b.variable),
                f"field coordinate {name!r} differs from the objects; explicitly align "
                "the field before calling build_lineage or maps (no automatic reordering)")
    values = np.asarray(field.transpose(*dims).values, dtype=float)
    if g.support == "line":
        values = values[:, None, :]
    return flat(values)


def _volume_extras(objects: ObjectField) -> dict:
    """Column of every cell of the flat view, the horizontal area of each column (object
    tracking, or 1 per column without an area measure) and the depth of every cell.
    """
    g, ods = objects.grid, objects.dataset
    nz, ny, nx = (ods.sizes[d] for d in g.spatial_dims)
    rows = np.arange(nz * ny)
    columns = ((rows % ny)[:, None] * nx + np.arange(nx)[None, :]).ravel()
    if g.cell_measure.kind == "area":
        column_weight = np.asarray(ods[g.cell_measure.var].transpose(*g.spatial_dims[1:]).values,
                                   float).ravel()
        column_unit = "m2"
    else:
        column_weight, column_unit = np.ones(ny * nx), "1"
    depth = build_geometry(ods, g)[2]["depth"].ravel()
    return {"columns": columns, "column_weight": column_weight, "column_unit": column_unit,
            "depth": depth, "depth_units": ods[g.vertical.coordinate].attrs.get("units", "m")}


def _group_vars(ds, prefix, dim, own, k, m, parent, labels, weights, table, times, unit,
                support, along_units, extra, with_intensity, weighted, field_units):
    """Level-1 variables of the branches or the events  from the metrics of their daily tables
    (`m`); the footprint is propagated through the merge forest (`parent`).
    """
    T = labels.shape[0]
    stamp = (lambda i: times[i]) if k else (lambda i: times[:0])
    ds[f"{prefix}_number"] = (dim, np.arange(1, k + 1, dtype=np.int32))
    ds[f"{prefix}_start"] = (dim, stamp(m["start"]))
    ds[f"{prefix}_end"] = (dim, stamp(m["end"]))
    duration = (m["end"] - m["start"] + 1).astype(np.int64)
    ds[f"{prefix}_duration_days"] = (dim, duration)
    ds[f"{prefix}_active_days"] = (dim, m["active"].astype(np.int64))
    ds[f"{prefix}_bridged_days"] = (dim, duration - m["active"].astype(np.int64))
    if prefix == "event":
        ds["event_n_objects"] = (dim, m["n"].astype(np.int64))
        ds["event_n_splits"] = (dim, m["split"].astype(np.int64))
        ds["event_n_merges"] = (dim, m["merge"].astype(np.int64))
    ds[f"{prefix}_left_censored"] = (dim, m["start"] == 0)
    ds[f"{prefix}_right_censored"] = (dim, m["end"] == T - 1)
    foot = lm.footprint_metrics(labels, weights, table, np.arange(table.n), own, k,
                                parent=parent, columns=extra.get("columns"),
                                column_weight=extra.get("column_weight"),
                                depth=extra.get("depth"))
    ds[f"{prefix}_area_max"] = (dim, m["size_max"], {
        "units": unit, "long_name": "largest daily size (length, area or volume)"})
    ds[f"{prefix}_peak_area_day"] = (dim, stamp(m["peak_day"]))
    ds[f"{prefix}_area_days"] = (dim, m["size_days"], {
        "units": f"{unit} day", "long_name": "daily sizes summed over days (not the footprint)"})
    ds[f"{prefix}_footprint"] = (dim, foot["footprint"], {
        "units": unit, "long_name": "union of the cells occupied at some time"})
    ds[f"{prefix}_net_displacement"] = (dim, m["net_displacement"], {
        "units": along_units, "long_name": "distance between the first and the last daily "
                                           "centroid (horizontal; along the line on a line)"})
    ds[f"{prefix}_path_length"] = (dim, m["path_length"], {
        "units": along_units, "long_name": "summed distance between consecutive daily "
                                           "centroids (horizontal; along the line on a line)"})
    ds[f"{prefix}_provisional"] = (dim, m["provisional"] > 0)
    if support == "volume":
        ds[f"{prefix}_net_vertical_displacement"] = (dim, m["net_vertical_displacement"], {
            "units": extra["depth_units"], "long_name": "depth of the last daily centroid minus "
                                                        "that of the first (positive = deeper)"})
        ds[f"{prefix}_vertical_path_length"] = (dim, m["vertical_path_length"], {
            "units": extra["depth_units"]})
        ds[f"{prefix}_footprint_horizontal"] = (dim, foot["footprint_horizontal"], {
            "units": extra["column_unit"],
            "long_name": "union of the columns occupied at some level"})
        ds[f"{prefix}_depth_min"] = (dim, foot["depth_min"], {"units": extra["depth_units"]})
        ds[f"{prefix}_depth_max"] = (dim, foot["depth_max"], {"units": extra["depth_units"]})
    if with_intensity:
        method = ("cell measure x day: mean (weighted)" if weighted
                  else "cell-days: mean (unweighted, an approximation)")
        ds[f"{prefix}_intensity_max"] = (dim, m["intensity_max"], {"units": field_units})
        ds[f"{prefix}_intensity_mean"] = (dim, m["intensity_mean"],
                                          {"units": field_units, "cell_methods": method})
        ds[f"{prefix}_intensity_cumulative"] = (dim, m["intensity_cumulative"],
                                                {"units": f"{field_units} {unit} day"})
        peak = m["peak_intensity_day"]
        out = np.full(k, np.datetime64("NaT", "ns"))
        out[peak >= 0] = times[peak[peak >= 0]]
        ds[f"{prefix}_peak_intensity_day"] = (dim, out)


def build_lineage(objects: ObjectField, *, policy: LinkPolicy | None = None,
                  analysis_mode: str | None = None, merge_rule: str = "new_event",
                  field: xr.DataArray | None = None, weighted: bool = True,
                  provenance: Provenance | None = None) -> LineageGraph:
    """Genealogy of the daily patches of `objects` (genealogy).

    `policy` defaults to the policy of the recipe's track step (`LinkPolicy.from_track`), so
    the retrospective spacetime events are the ``track_id`` of object tracking (when its
    minimum duration is 1), or to `LinkPolicy` when the objects were not tracked.
    `analysis_mode` defaults to that of the objects. `merge_rule` (monitoring):
    ``"new_event"`` (default) freezes the events that join and opens a new one; ``"survivor"``
    keeps the dominant one, as object tracking does with tracks. `field` (off by default) adds
    intensity metrics: a DataArray on the grid and days of the objects, for example
    ``mask.intensity``; the daily spatial mean is weighted by the cell measure unless
    ``weighted=False``. Computes the label field (eager).
    """
    require(isinstance(objects, ObjectField), "objects must be an ObjectField")
    require(objects.has_label, "the ObjectField has no patches (no segment step)")
    if policy is None:
        recipe = objects.recipe
        policy = (LinkPolicy.from_track(recipe.steps[-1]) if recipe is not None
                  and recipe.has_track else LinkPolicy())
    require(isinstance(policy, LinkPolicy), "policy must be a LinkPolicy")
    mode = objects.analysis_mode if analysis_mode is None else analysis_mode
    check_choice("analysis_mode", mode, ("retrospective", "monitoring"))
    check_choice("merge_rule", merge_rule, MERGE_RULES)
    g, ods = objects.grid, objects.dataset
    labels, weights, unit, table = field_arrays(objects)
    n, T = table.n, labels.shape[0]
    times = ods[g.time_dim].values
    cells_per_day = labels.shape[1] * labels.shape[2]
    e, _ = _edges(labels, weights, table, policy)
    src, dst, linked = e["src"], e["dst"], e["linked"]
    by_area = policy.tie_rule == "largest_area"

    in_deg = np.bincount(dst[linked], minlength=n)
    out_deg = np.bincount(src[linked], minlength=n)
    node_type = _node_types(in_deg, out_deg)
    score = _track.measure(policy.dominance_measure, e["shared"], table.area[src],
                           table.area[dst])
    idx = np.flatnonzero(linked)
    dom_src, tie_src = _dominance(src[idx], dst[idx], score[idx], table.area[dst[idx]], n, by_area)
    dom_dst, tie_dst = _dominance(dst[idx], src[idx], score[idx], table.area[src[idx]], n, by_area)
    flags = {name: np.zeros(src.size, bool) for name in
             ("is_dominant_for_source", "is_dominant_for_target", "dominant_tied")}
    flags["is_dominant_for_source"][idx] = dom_src
    flags["is_dominant_for_target"][idx] = dom_dst
    flags["dominant_tied"][idx] = (dom_src & tie_src) | (dom_dst & tie_dst)

    cont = linked & (out_deg[src] == 1) & (in_deg[dst] == 1)
    branch = (_components(table, src[cont], dst[cont], cells_per_day) if n
              else np.zeros(0, np.int32))
    split_merge = linked & ~cont
    link_pairs = np.unique(np.stack([branch[src[split_merge]], branch[dst[split_merge]]], 1)
                           .reshape(-1, 2), axis=0) if n else np.zeros((0, 2), np.int32)

    if mode == "retrospective":
        event = (_components(table, src[linked], dst[linked], cells_per_day) if n
                 else np.zeros(0, np.int32))
        merged = np.zeros(int(event.max()) + 1 if n else 1, np.int32)
    else:
        event, merged = _monitoring_events(n, src[linked], dst[linked], score[linked],
                                           table.area, merge_rule, by_area)
    k = merged.size - 1
    nb = int(branch.max()) if n else 0
    day = table.day

    # provisional (monitoring): the result of the last days can still change
    delay = max(int(objects.final_after_days), int(policy.max_gap_days))
    provisional = (day >= T - 1 - delay) if mode == "monitoring" else np.zeros(n, bool)

    ds = xr.Dataset()
    ds["source_object"] = (TRANSITION_DIM, src)
    ds["target_object"] = (TRANSITION_DIM, dst)
    ds["source_time"] = (TRANSITION_DIM, times[day[src]])
    ds["target_time"] = (TRANSITION_DIM, times[day[dst]])
    ds["gap_days"] = (TRANSITION_DIM, e["gap"])
    ds["shared"] = (TRANSITION_DIM, e["shared"], {"units": unit,
                                                   "long_name": "shared cell measure"})
    for name in MEASURES:
        ds[name] = (TRANSITION_DIM, e[name], {"units": "1"})
    ds["linked"] = (TRANSITION_DIM, linked)
    for name, values in flags.items():
        ds[name] = (TRANSITION_DIM, values)

    ds["object_time"] = (OBJECT_DIM, times[day])
    ds["object_label"] = (OBJECT_DIM, table.label.astype(np.int32))
    ds["in_degree"] = (OBJECT_DIM, in_deg.astype(np.int64))
    ds["out_degree"] = (OBJECT_DIM, out_deg.astype(np.int64))
    ds["node_type"] = (OBJECT_DIM, node_type)
    ds["is_split"] = (OBJECT_DIM, out_deg >= 2)
    ds["is_merge"] = (OBJECT_DIM, in_deg >= 2)
    ds["object_branch"] = (OBJECT_DIM, branch.astype(np.int32))
    ds["object_spacetime_event"] = (OBJECT_DIM, event.astype(np.int32))
    ds["provisional"] = (OBJECT_DIM, provisional)
    backend = (np.asarray(ods["object_external_id"].values, np.int64)
               if "object_external_id" in ods else np.full(n, -1, np.int64))
    ds["backend_id"] = (OBJECT_DIM, backend, {"long_name": "identifier of the external method "
                                              "(object_external_id), -1 when built by xHW"})

    # node attributes the series and the metrics use (copied from the ObjectField)
    ds["object_area"] = ods["object_area"].variable
    ds["object_cells"] = ods["object_cells"].variable
    names = {"line": ("object_along",), "surface": ("object_lat", "object_lon"),
             "volume": ("object_lat", "object_lon", "object_depth")}[g.support]
    for name in names:
        ds[name] = ods[name].variable
    coords = {name.split("_", 1)[1]: np.asarray(ods[name].values, float) for name in names}
    values = _field_values(field, objects) if field is not None else None
    node_int = (lm.node_intensity(labels, weights, table, values, weighted)
                if values is not None else None)
    field_units = field.attrs.get("units", "") if field is not None else ""
    if node_int is not None:
        method = ("cell measure: mean (weighted)" if weighted
                  else "cells: mean (unweighted, an approximation)")
        ds["object_intensity_valid_cells"] = (
            OBJECT_DIM,
            node_int["count"].astype(np.int64),
            {"units": "1", "long_name": "cells with finite field"},
        )
        ds["object_intensity_valid_measure"] = (
            OBJECT_DIM,
            node_int["weight"],
            {"units": unit, "long_name": "finite field support"},
        )
        ds["object_intensity_max"] = (OBJECT_DIM, node_int["max"], {"units": field_units})
        ds["object_intensity_mean"] = (OBJECT_DIM, node_int["mean"],
                                       {"units": field_units, "cell_methods": method})
        ds["object_intensity_cumulative"] = (OBJECT_DIM, np.where(
            node_int["count"] > 0, node_int["cumulative"], np.nan),
                                             {"units": f"{field_units} {unit} day"})

    # branches and events: every node in its own group; in monitoring an event also covers the
    # events merged into it, through daily tables propagated up the merge forest
    columns = lm.node_columns(table, g.support, coords, node_int, weighted, flags={
        "split": out_deg >= 2, "merge": in_deg >= 2, "provisional": provisional})
    extra = _volume_extras(objects) if g.support == "volume" else {}
    along_units = ods["object_along"].attrs.get("units", "") if g.support == "line" else "m"
    b0 = (branch - 1).astype(np.int64)
    e0 = (event - 1).astype(np.int64)
    parent = merged[1:].astype(np.int64) - 1
    for prefix, dim, own, kk, par in (("branch", BRANCH_DIM, b0, nb, None),
                                      ("event", EVENT_DIM, e0, k, parent)):
        tables = lm.group_tables(own, day, columns, kk, T, par)
        metrics = lm.table_metrics(*tables, kk, g.support)
        _group_vars(ds, prefix, dim, own, kk, metrics, par, labels, weights, table, times,
                    unit, g.support, along_units, extra, node_int is not None, weighted,
                    field_units)
    bevent = np.zeros(nb, np.int32)
    bevent[b0] = event              # nodes are in time order: the event of the last node wins
    ds["branch_spacetime_event"] = (BRANCH_DIM, bevent)
    ds["branch_link_parent"] = (BRANCH_LINK_DIM, link_pairs[:, 0].astype(np.int32))
    ds["branch_link_child"] = (BRANCH_LINK_DIM, link_pairs[:, 1].astype(np.int32))
    ds["event_n_branches"] = (EVENT_DIM, lm.union_counts(e0, branch, k, parent))
    ds["event_merged_into"] = (EVENT_DIM, merged[1:].astype(np.int32))

    if provenance is None:
        provenance = Provenance.new(parents=(objects.provenance.run_id,), steps=(ProcessingStep(
            name="build_lineage", params=(("analysis_mode", mode), ("merge_rule", merge_rule),
                                          ("policy_id", policy.spec_id))),))
    return LineageGraph._make(
        ds, grid=g, policy=policy, analysis_mode=mode, merge_rule=merge_rule,
        provenance=provenance, definition_id=objects.definition_id,
        baseline_id=objects.baseline_id, source_objects_run_id=objects.provenance.run_id,
        source_mask_run_id=objects.source_mask_run_id,
        final_after_days=int(objects.final_after_days) if mode == "monitoring" else 0,
        window_start=str(np.datetime_as_string(times[0], unit="s")),
        window_end=str(np.datetime_as_string(times[-1], unit="s")))
