# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""ObjectField and `build_objects`: segmentation and overlap tracking of an XMask (object
tracking).

A recipe (`SpatialRecipe`) runs over a boolean working field that starts as ``exceedance`` or
``event_membership`` restricted to valid cells. The XMask is never modified : everything the
recipe adds or removes is recorded in ``added`` and ``removed`` of the resulting ObjectField,
and cells that are not valid never enter a patch.

Lines (x), surfaces (y, x) and volumes (z, y, x) on structured grids are supported; the
neighbours are those of the index grid, so curvilinear (2-D latitude and longitude) and static
sigma grids (3-D thickness and depth) need nothing more than their weights and coordinates. In
a volume the cell measure is the cell volume (genealogy, from ``area x thickness`` or a volume
measure) and the tables add the depth of the centroid; on a line it is a length and the tables
hold the along-line coordinate of the centroid. Tracking works on the flattened view (time,
z*y, x): two cells overlap when they are the same cell on two days.

Result variables (dimensions ``(time, *spatial_dims)`` unless stated): ``processed_mask``,
``added``, ``removed``, ``valid``; ``label`` (daily patches, 0 = background, present if the
recipe segments); ``track_id`` (present if it tracks); tables along ``object`` (one row per
daily patch) and ``track`` (one row per tracked object). Everything is eager.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import xarray as xr

from . import _morph, _track
from ._detect import persist_array
from ._errors import InvariantError, NotSupportedError, SchemaError, XHWDataQualityWarning
from ._mask import (
    MISSING_DATA,
    XMask,
    check_entity,
    check_stored_id,
    load_attr_json,
    read_only,
    strip_xhw_attrs,
)
from ._recipe import MinArea, SpatialRecipe
from ._schema import CONVENTIONS, SCHEMA_VERSIONS, check_choice, check_serializable_attrs, require
from ._specs import GridSpec, ProcessingStep, Provenance

OBJECT_DIM, TRACK_DIM = "object", "track"
_DISCARD_WARNING = 0.9      # fraction of patches a minimum area may discard before it warns
_OBJECT_VARS = ("object_time", "object_label", "object_track", "object_area", "object_cells")
_SUPPORT_OBJECT_VARS = {"line": ("object_along",), "surface": ("object_lat", "object_lon"),
                        "volume": ("object_lat", "object_lon", "object_depth")}
_TRACK_VARS = ("track_number", "track_start", "track_end", "track_duration_days",
               "track_active_days", "track_bridged_days", "track_area_max", "track_area_mean",
               "track_area_days", "track_n_multi_overlap_days", "track_left_censored",
               "track_right_censored", "track_qualified", "track_parent", "track_merged_into")
_FIELD_ATTRS = ("Conventions", "xhw_schema", "xhw_schema_version", "xhw_grid", "xhw_recipe",
                "xhw_provenance", "definition_id", "baseline_id", "source_mask_run_id",
                "analysis_mode", "origin", "final_after_days", "window_start", "window_end",
                "recipe_id", "source_layer")


# --- weights, tables -------------------------------------------------------------------------


def cell_weights(ds: xr.Dataset, grid: GridSpec) -> tuple[np.ndarray, str]:
    """Cell measure over the spatial dims and its unit: m (length of a line), object tracking
    (surface area), genealogy (volume of a volume grid: area x thickness or a volume measure)
    or '1' (cell count).
    """
    measure = grid.cell_measure
    if grid.support == "line":
        if measure.kind == "length":
            return np.asarray(ds[measure.var].transpose(*grid.spatial_dims).values,
                              dtype=float), "m"
        require(measure.kind == "unit", "a line needs a length or a unit measure",
                NotSupportedError)
        return np.ones(tuple(ds.sizes[d] for d in grid.spatial_dims)), "1"
    if grid.support == "volume":
        shape = tuple(ds.sizes[d] for d in grid.spatial_dims)
        if measure.kind == "unit":
            return np.ones(shape), "1"
        if measure.kind == "volume":
            w = ds[measure.var]
        else:
            require(measure.kind == "area", "a volume needs an area, a volume or a unit measure",
                    NotSupportedError)
            w = ds[measure.var] * ds[grid.vertical.thickness]
        w = (w * xr.ones_like(ds[grid.domain_mask], dtype=float)).transpose(*grid.spatial_dims)
        return np.asarray(w.values, dtype=float), "m3"
    if measure.kind == "area":
        w = np.asarray(ds[measure.var].transpose(*grid.spatial_dims).values, dtype=float)
        return w, "m2"
    require(measure.kind == "unit", "objects need an area or a unit cell measure",
            NotSupportedError)
    return np.ones(tuple(ds.sizes[d] for d in grid.spatial_dims)), "1"


def flat(a: np.ndarray) -> np.ndarray:
    """(time, [z,] y, x) -> (time, z*y, x); a surface is unchanged. Volume cells keep their
    raster order (z, y, x), so the first cell of an object is the same in both views.
    """
    return a.reshape(a.shape[0], -1, a.shape[-1])


def make_table(labels: np.ndarray, weights: np.ndarray, lat: np.ndarray, lon: np.ndarray,
               counts: np.ndarray | None = None):
    """Object table of a label field plus per-object centroid (area-weighted spherical mean).
    `labels` and `weights` are the flat view. `lat` and `lon` are the coordinates of every
    cell of that view, shaped like the weights, or 1-D (one per row and one per column);
    ``None`` skips the centroid (lines).
    """
    days, ny, nx = labels.shape
    if counts is None:
        counts = labels.reshape(days, -1).max(axis=1).astype(np.int64)
    base = np.concatenate(([0], np.cumsum(counts)))
    n = int(base[-1])
    tt, yy, xx = np.nonzero(labels)
    gid = base[tt] + labels[tt, yy, xx] - 1
    w = weights[yy, xx]
    cells = np.bincount(gid, minlength=n)
    area = np.bincount(gid, weights=w, minlength=n)
    if lat is None:
        clat = clon = None
    else:
        phi = np.deg2rad(lat[yy] if lat.ndim == 1 else lat[yy, xx])
        lam = np.deg2rad(lon[xx] if lon.ndim == 1 else lon[yy, xx])
        vec = [np.bincount(gid, weights=w * f, minlength=n) for f in (
            np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi))]
        clat = np.rad2deg(np.arctan2(vec[2], np.hypot(vec[0], vec[1])))
        clon = np.rad2deg(np.arctan2(vec[1], vec[0]))
        clon = clon % 360.0 if float(lon.min()) >= 0.0 else clon
    first = np.zeros(n, dtype=np.int64)
    if gid.size:
        unique, index = np.unique(gid, return_index=True)
        first[unique] = (yy * nx + xx)[index]
    day = np.repeat(np.arange(days, dtype=np.int64), counts)
    label = np.arange(n, dtype=np.int64) - base[day] + 1
    table = _track.Table(day=day, label=label, area=area, cells=cells, first_cell=first,
                         base=base)
    return table, clat, clon, (tt, yy, xx, gid)


def _calibration_areas(calibration, unit: str) -> np.ndarray:
    """Patch areas of a frozen calibration sample: an ObjectField or an array of areas."""
    if isinstance(calibration, ObjectField):
        require("object_area" in calibration.dataset,
                "the calibration ObjectField has no patches (no segment step)")
        found = calibration.dataset["object_area"].attrs.get("units")
        require(found == unit, f"calibration areas are in {found!r}, the field uses {unit!r}")
        return np.asarray(calibration.dataset["object_area"].values, dtype=float)
    return np.asarray(calibration, dtype=float).ravel()


def _cut(spec: MinArea, table: _track.Table, domain_weight: float, unit: str,
         sample: np.ndarray | None) -> float:
    """Minimum area (in cell-measure units) of a patch; the patches below it are discarded.
    `sample` is the frozen calibration sample of a percentile (None: the patches of this run).
    """
    if spec.kind == "none":
        return 0.0
    if spec.kind == "domain_fraction":
        return spec.value / 100.0 * domain_weight
    want = {"absolute_km": "m", "absolute_km3": "m3"}.get(spec.kind, "m2")
    require(unit == want or spec.kind == "percentile" and spec.floor_km2 is None,
            f"min_area {spec.kind!r} needs a cell measure in {want} (the grid has {unit!r})")
    if spec.kind == "absolute_km":
        return spec.value * 1e3
    if spec.kind == "absolute_km2":
        return spec.value * 1e6
    if spec.kind == "absolute_km3":
        return spec.value * 1e9
    sample = table.area if sample is None else sample
    if sample.size == 0:
        warnings.warn("min_area percentile: no patches in the reference sample; nothing is "
                      "discarded", XHWDataQualityWarning, stacklevel=3)
        return 0.0
    cut = float(np.percentile(sample, spec.value))
    return max(cut, spec.floor_km2 * 1e6) if spec.floor_km2 is not None else cut


def _apply_cut(labels, table, keep):
    """Drop the patches where `keep` is False and renumber each day 1..n_t."""
    cum = np.cumsum(keep)
    before = np.concatenate(([0], cum))[table.base[:-1]]
    new = (cum - before[table.day]).astype(np.int32)
    out = np.zeros_like(labels)
    tt, yy, xx = np.nonzero(labels)
    gid = table.base[tt] + labels[tt, yy, xx] - 1
    ok = keep[gid]
    out[tt[ok], yy[ok], xx[ok]] = new[gid[ok]]
    return out


# --- the container ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ObjectField:
    """Segmentation (and tracking) result of one XMask under one recipe.

    Build it with `build_objects`; read a serialized one with ``from_dataset(ds)``.
    ``origin="external"`` marks objects produced elsewhere and ingested with `ingest_objects`
    (their ``recipe`` is None). Cells and areas are in the cell measure of the grid.
    """

    _dataset: xr.Dataset
    grid: GridSpec
    recipe: SpatialRecipe | None
    analysis_mode: str
    provenance: Provenance
    definition_id: str
    baseline_id: str
    source_mask_run_id: str
    origin: str
    final_after_days: int
    source_layer: str

    @classmethod
    def _make(cls, ds: xr.Dataset, **fields) -> ObjectField:
        ds = strip_xhw_attrs(ds)
        ds.attrs = {k: v for k, v in ds.attrs.items() if k not in _FIELD_ATTRS}
        field = cls(read_only(ds), **fields)
        field._validate()
        return field

    @classmethod
    def from_dataset(cls, ds: xr.Dataset, *, copy: bool = False) -> ObjectField:
        """Read a dataset produced by `to_dataset` (schema policy and ids are enforced)."""
        require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
        check_entity(ds, "ObjectField")
        for name in _FIELD_ATTRS[3:]:
            if name in ("xhw_recipe", "recipe_id"):      # absent for externally produced objects
                continue
            require(name in ds.attrs, f"serialized attribute {name!r} is missing", SchemaError)
        grid = GridSpec.from_dict(load_attr_json(ds, "xhw_grid"))
        recipe = (SpatialRecipe.from_dict(load_attr_json(ds, "xhw_recipe"))
                  if ds.attrs.get("xhw_recipe") else None)
        provenance = Provenance.from_dict(load_attr_json(ds, "xhw_provenance"))
        if recipe is not None:
            check_stored_id(ds, "recipe_id", recipe)
        return cls._make(
            ds.copy(deep=True) if copy else ds, grid=grid, recipe=recipe,
            analysis_mode=ds.attrs["analysis_mode"], provenance=provenance,
            definition_id=ds.attrs["definition_id"], baseline_id=ds.attrs["baseline_id"],
            source_mask_run_id=ds.attrs["source_mask_run_id"], origin=ds.attrs["origin"],
            final_after_days=int(ds.attrs["final_after_days"]),
            source_layer=ds.attrs["source_layer"])

    # -- access -------------------------------------------------------------------------

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset.copy(deep=False)

    @property
    def has_label(self) -> bool:
        return "label" in self._dataset

    @property
    def has_track(self) -> bool:
        return "track_id" in self._dataset

    # -- serialization ------------------------------------------------------------------

    def to_dataset(self) -> xr.Dataset:
        ds = self._dataset.copy(deep=False)
        times = ds[self.grid.time_dim].values
        ds.attrs.update({
            "Conventions": CONVENTIONS,
            "xhw_schema": "ObjectField",
            "xhw_schema_version": SCHEMA_VERSIONS["ObjectField"],
            "xhw_grid": self.grid.to_json(),
            "xhw_recipe": self.recipe.to_json() if self.recipe is not None else "",
            "xhw_provenance": self.provenance.to_json(),
            "definition_id": self.definition_id,
            "baseline_id": self.baseline_id,
            "source_mask_run_id": self.source_mask_run_id,
            "analysis_mode": self.analysis_mode,
            "origin": self.origin,
            "source_layer": self.source_layer,
            "final_after_days": int(self.final_after_days),
            "window_start": str(np.datetime_as_string(times[0], unit="s")),
            "window_end": str(np.datetime_as_string(times[-1], unit="s")),
        })
        if self.recipe is not None:
            ds.attrs["recipe_id"] = self.recipe.spec_id
        for name in ("processed_mask", "added", "removed", "valid", "track_qualified",
                     "track_left_censored", "track_right_censored"):
            if name in ds:
                ds[name].attrs.update({"flag_values": np.array([0, 1], dtype=np.int8),
                                       "flag_meanings": "false true"})
        ds[self.grid.time_dim].encoding["calendar"] = self.grid.calendar
        return ds

    # -- validation ---------------------------------------------------------------------

    def _validate(self) -> None:
        ds, g = self._dataset, self.grid
        check_choice("analysis_mode", self.analysis_mode, ("retrospective", "monitoring"))
        check_choice("origin", self.origin, ("xhw", "external"))
        check_choice("source_layer", self.source_layer, ("exceedance", "event_membership"))
        require((self.recipe is None) == (self.origin == "external"),
                "a recipe is present exactly when the objects were built by xHW")
        g.check(ds)
        dims = (g.time_dim, *g.spatial_dims)
        for name in ("processed_mask", "added", "removed", "valid"):
            require(name in ds.data_vars, f"required variable {name!r} is missing")
            require(ds[name].dims == dims and ds[name].dtype == bool,
                    f"{name!r} must be boolean with dims {dims}")
        for name in ("label", "track_id"):
            if name in ds.data_vars:
                require(ds[name].dims == dims and ds[name].dtype == np.int32,
                        f"{name!r} must be int32 with dims {dims}")
        require(not self.has_track or self.has_label, "track_id needs label")
        if self.has_label:
            for name in _OBJECT_VARS + _SUPPORT_OBJECT_VARS[g.support]:
                require(name in ds.data_vars and ds[name].dims == (OBJECT_DIM,),
                        f"object variable {name!r} is missing or not along {OBJECT_DIM!r}")
        if self.has_track:
            for name in _TRACK_VARS:
                require(name in ds.data_vars and ds[name].dims == (TRACK_DIM,),
                        f"track variable {name!r} is missing or not along {TRACK_DIM!r}")
        check_serializable_attrs("dataset", ds.attrs)
        for name, var in ds.variables.items():
            check_serializable_attrs(f"variable {name!r}", var.attrs)

    def check_invariants(self) -> None:
        """Value rules of the field, the tables and their consistency (one pass over the data)."""
        ds, g = self._dataset, self.grid
        T = ds.sizes[g.time_dim]
        problems: list[str] = []

        def check(ok, message):
            if not bool(np.all(ok)):
                problems.append(message)

        v = {n: np.asarray(ds[n].transpose(g.time_dim, *g.spatial_dims).values)
             for n in ("processed_mask", "added", "removed", "valid")}
        check(~v["processed_mask"] | v["valid"], "processed cells outside the valid cells")
        check(~(v["added"] & v["removed"]), "a cell is both added and removed")
        check(~v["added"] | v["processed_mask"], "added cells not in the processed mask")
        check(~v["removed"] | ~v["processed_mask"], "removed cells in the processed mask")
        if self.has_label:
            self._check_objects(ds, g, T, v, check)
        if self.has_track:
            self._check_tracks(ds, g, T, v, check)
        if problems:
            raise InvariantError("; ".join(dict.fromkeys(problems)))

    def _check_objects(self, ds, g, T, v, check):
        label = np.asarray(ds["label"].transpose(g.time_dim, *g.spatial_dims).values)
        check(label >= 0, "negative labels")
        check((label == 0) | v["valid"], "patches on invalid cells")
        day = np.searchsorted(ds[g.time_dim].values, ds["object_time"].values)
        number = np.asarray(ds["object_label"].values)
        counts = np.bincount(day, minlength=T)
        check(np.bincount(day, weights=(number > 0), minlength=T) == counts,
              "object labels must be >= 1")
        check(label.reshape(T, -1).max(axis=1) == counts,
              "labels of a day are not 1..n_t (object table disagrees)")
        order = np.lexsort((number, day))
        check(order == np.arange(order.size), "object table not ordered by (day, label)")
        cells = np.bincount(
            (np.concatenate(([0], np.cumsum(counts)))[np.nonzero(label)[0]]
             + label[label > 0] - 1), minlength=number.size)
        check(cells == ds["object_cells"].values, "object_cells != cells of the label field")
        check(np.asarray(ds["object_area"].values) > 0, "non-positive object areas")
        if self.recipe is not None and self.recipe.steps[-1].op in ("segment", "track"):
            if not self.has_track:
                check((label > 0) == v["processed_mask"], "processed_mask != patches")

    def _check_tracks(self, ds, g, T, v, check):
        track = np.asarray(ds["track_id"].transpose(g.time_dim, *g.spatial_dims).values)
        label = np.asarray(ds["label"].transpose(g.time_dim, *g.spatial_dims).values)
        check((track == 0) | (label > 0), "tracked cells without a patch")
        number = np.asarray(ds["track_number"].values)
        k = number.size
        check(number == np.arange(1, k + 1), "track numbers must be 1..k")
        obj_track = np.asarray(ds["object_track"].values)
        check((obj_track >= 0) & (obj_track <= k), "object_track outside the track table")
        day = np.searchsorted(ds[g.time_dim].values, ds["object_time"].values)
        start, end = np.asarray(ds["track_start"].values), np.asarray(ds["track_end"].values)
        times = ds[g.time_dim].values
        s, e = np.searchsorted(times, start), np.searchsorted(times, end)
        check(s <= e, "track start after end")
        duration = np.asarray(ds["track_duration_days"].values)
        check(duration == e - s + 1, "track_duration_days != end - start + 1")
        active = np.asarray(ds["track_active_days"].values)
        check((active >= 1) & (active <= duration), "track_active_days outside [1, duration]")
        check(np.asarray(ds["track_bridged_days"].values) == duration - active,
              "track_bridged_days != duration - active days")
        member = obj_track > 0
        if member.any():
            t0 = np.full(k, T)
            t1 = np.full(k, -1)
            np.minimum.at(t0, obj_track[member] - 1, day[member])
            np.maximum.at(t1, obj_track[member] - 1, day[member])
            check((t0 == s) & (t1 == e), "track span != span of its objects")
            keys = np.unique(((obj_track[member] - 1).astype(np.int64) * T + day[member]))
            check(np.bincount(keys // T, minlength=k) == active,
                  "track_active_days != days with an object")
        check(np.asarray(ds["track_left_censored"].values) == (s == 0),
              "left censoring != track starting on the first day")
        check(np.asarray(ds["track_right_censored"].values) == (e == T - 1),
              "right censoring != track ending on the last day")
        qualified = np.asarray(ds["track_qualified"].values)
        if self.recipe is not None:
            minimum = self.recipe.steps[-1].min_duration_days
            check(qualified == (duration >= minimum), "track_qualified != duration >= minimum")
        if self.analysis_mode == "retrospective":
            check(qualified, "retrospective objects hold qualified tracks only")
        lut = np.concatenate(([False], qualified))
        expect = lut[track] if self.recipe is None or self.recipe.steps[-1].op == "track" else None
        if expect is not None:
            check(v["processed_mask"] == expect, "processed_mask != cells of qualified tracks")
        check(np.asarray(ds["track_parent"].values) >= 0, "negative track_parent")

    def check_against(self, mask: XMask) -> None:
        """Compare with the source mask: ids, grid and ``processed == (source + added) -
        removed`` (computes the mask's data). A full rebuild comparison is `build_objects` +
        equality.
        """
        require(mask.definition_id == self.definition_id, "definition_id differs from the mask")
        require(mask.baseline_id == self.baseline_id, "baseline_id differs from the mask")
        require(mask.provenance.run_id == self.source_mask_run_id, "source mask run id differs")
        require(mask.grid == self.grid, "grid differs from the mask")
        g, ds = self.grid, self._dataset
        order = (g.time_dim, *g.spatial_dims)
        md = mask.dataset
        valid = np.asarray(md["valid"].transpose(*order).values)
        require(bool((np.asarray(ds["valid"].transpose(*order).values) == valid).all()),
                "valid differs from the mask")
        source = np.asarray(md[self.source_layer].transpose(*order).values) & valid
        added = np.asarray(ds["added"].transpose(*order).values)
        removed = np.asarray(ds["removed"].transpose(*order).values)
        processed = np.asarray(ds["processed_mask"].transpose(*order).values)
        if not bool((processed == ((source | added) & ~removed)).all()):
            raise InvariantError("processed_mask != (source | added) & ~removed")
        if not bool(((added & source) | (removed & ~source)).sum() == 0):
            raise InvariantError("added cells in the source or removed cells outside it")


# --- build -----------------------------------------------------------------------------------


def _delay(recipe: SpatialRecipe, mask: XMask) -> int:
    """Days after which the result of a day cannot change any more (monitoring)."""
    d = mask.definition
    late = d.min_duration_days + d.max_gap_days - 1
    total = late if recipe.input_layer == "event_membership" else 0
    for step in recipe.steps:
        if step.op == "persist":
            m = step.min_duration_days or d.min_duration_days
            g = step.max_gap_days if step.max_gap_days is not None else d.max_gap_days
            total += m + g - 1
        elif step.op == "track":
            total += step.max_gap_days
    return int(total)


def build_objects(mask: XMask, *, recipe: SpatialRecipe | None = None,
                  analysis_mode: str = "retrospective", calibration=None,
                  provenance: Provenance | None = None) -> ObjectField:
    """Run `recipe` (default: ``SpatialRecipe.default``) on `mask` and return the ObjectField.

    ``retrospective``: tracks are the connected components of the overlap graph and a
    percentile minimum area uses every patch of the period. ``monitoring``: identities are
    assigned forward and never rewritten, tracks shorter than the minimum duration are kept as
    candidates (``track_qualified`` False) and ``final_after_days`` says after how many days a
    day's result is definitive. A percentile minimum area must not use days that come later,
    and the mask of a monitoring run starts after the baseline reference period, so it needs a
    frozen `calibration` sample: an ObjectField (typically a retrospective run over the
    reference period without area filter) or an array of patch areas in the cell-measure
    units; it is also accepted in retrospective mode, where it replaces the patches of the
    run. Computes the mask's data (objects are eager).
    """
    require(isinstance(mask, XMask), "mask must be an XMask")
    recipe = recipe if recipe is not None else SpatialRecipe.default()
    require(isinstance(recipe, SpatialRecipe), "recipe must be a SpatialRecipe")
    check_choice("analysis_mode", analysis_mode, ("retrospective", "monitoring"))
    g, md = mask.grid, mask.dataset
    require(g.time_axes == "single", "objects of forecast masks are not supported yet",
            NotSupportedError)
    lat, lon = _morph.axes(md, g)
    periodic, volume, line = g.periodic_x, _morph.is_volume(g), _morph.is_line(g)
    order = (g.time_dim, *g.spatial_dims)
    times = md[g.time_dim].values
    nz = md.sizes[g.spatial_dims[0]] if volume else 1
    cells_lat, cells_lon = (None, None) if line else _morph.cell_coords(lat, lon, nz)
    geometry = _geometry(md, g, cells_lat, cells_lon)
    if analysis_mode == "monitoring":
        require(mask.baseline.available_before(times[0]),
                "monitoring requires a baseline available before the window ")
    weights, unit = cell_weights(md, g)
    flat_weights = weights.reshape(-1, weights.shape[-1])
    thickness = vertical_thickness(md, g)
    domain = np.asarray(md[g.domain_mask].transpose(*g.spatial_dims).values, dtype=bool)
    domain_weight = float(weights[domain].sum())
    lift = (lambda a: a[:, None, :]) if line else (lambda a: a)    # a line is one row
    valid = lift(np.asarray(md["valid"].transpose(*order).values))
    field = lift(np.asarray(md[recipe.input_layer].transpose(*order).values)) & valid
    source = field.copy()
    definition = mask.definition
    eligible = valid | ((lift(np.asarray(md["invalid_reason"].transpose(*order).values))
                         == MISSING_DATA) & (definition.missing_policy == "as_non_exceedance"))
    labels = None
    for step in recipe.steps:
        if step.op == "persist":
            membership, _ = persist_array(
                field, eligible,
                min_days=step.min_duration_days or definition.min_duration_days,
                max_gap=(step.max_gap_days if step.max_gap_days is not None
                         else definition.max_gap_days),
                rule=step.join_rule or definition.join_rule)
            field, labels = membership & valid, None
        elif step.op in ("open", "close"):
            offsets = _morph.build_offsets(step.footprint, lat, lon, periodic, volume, line,
                                           thickness)
            field, labels = _morph.morphology(step.op, field, valid, offsets, periodic), None
        elif step.op == "segment":
            offsets = _morph.build_offsets(step.footprint, lat, lon, periodic, volume, line,
                                           thickness)
            labels, counts = _morph.label_field(field, step.footprint, offsets, periodic)
            labels = flat(labels)
            table = make_table(labels, flat_weights, None, None, counts)[0]
            sample = None
            if step.min_area.kind == "percentile":
                if calibration is not None:
                    sample = _calibration_areas(calibration, unit)
                else:
                    require(analysis_mode == "retrospective",
                            "a percentile minimum area in monitoring mode needs a frozen "
                            "`calibration` sample (it cannot use later days)")
            cut = _cut(step.min_area, table, domain_weight, unit, sample)
            if cut > 0.0:
                keep = table.area >= cut
                if table.n and not keep.any():
                    warnings.warn(
                        f"min_area: the cut ({cut:.6g} {unit}) is larger than every patch "
                        f"(largest {table.area.max():.6g}); no patch survives",
                        XHWDataQualityWarning, stacklevel=2)
                elif table.n and (table.n - int(keep.sum())) > _DISCARD_WARNING * table.n:
                    warnings.warn(
                        f"min_area ({step.min_area.kind} {step.min_area.value:g}) discards "
                        f"{table.n - int(keep.sum())} of {table.n} patches "
                        f"({100.0 * (1 - keep.mean()):.0f} %): the cut ({cut:.6g} {unit}) may not "
                        "suit the size of this domain; check `MinArea` (a percentile or a "
                        "smaller fraction scales better to large domains)",
                        XHWDataQualityWarning, stacklevel=2)
                labels = _apply_cut(labels, table, keep)
            field = (labels > 0).reshape(field.shape)
            last_segment = step
    if labels is None and recipe.has_segment:
        # a step after the last segment changed the field: its patches are the components of
        # the final field under that segment's neighbourhood (the area cut was already applied)
        offsets = _morph.build_offsets(last_segment.footprint, lat, lon, periodic, volume, line,
                                       thickness)
        labels = flat(_morph.label_field(field, last_segment.footprint, offsets, periodic)[0])
    return _assemble(mask, recipe, analysis_mode, provenance, source, field, labels,
                     flat_weights, unit, geometry, times, order)


def set_coordinate_vars(ds, g, geometry, table, weights, clat, clon, cells) -> None:
    """Centroid variables of the object table of a line, a surface or a volume."""
    yy, xx, gid = cells
    tiny = np.finfo(float).tiny
    if g.support == "line":
        ds["object_along"] = (OBJECT_DIM, np.bincount(
            gid, weights=weights[yy, xx] * geometry["along"][xx], minlength=table.n)
            / np.maximum(table.area, tiny),
            {"units": geometry["along_units"],
             "long_name": "length-weighted mean along-line coordinate of the object"})
    else:
        ds["object_lat"] = (OBJECT_DIM, clat, {"units": "degrees_north"})
        ds["object_lon"] = (OBJECT_DIM, clon, {"units": "degrees_east"})
    if g.support == "volume":
        ds["object_depth"] = (OBJECT_DIM, np.bincount(
            gid, weights=weights[yy, xx] * geometry["depth"][yy, xx], minlength=table.n)
            / np.maximum(table.area, tiny),
            {"units": geometry["depth_units"],
             "long_name": "volume-weighted mean depth of the object"})


def vertical_thickness(md, g) -> np.ndarray | None:
    """Thickness (m) of every cell of a volume over (z, y, x), from `VerticalSpec.thickness` (1-D
    or static 3-D); None for lines and surfaces. Used by the physical vertical connectivity
    (``Footprint.vertical_gap_m``, physical vertical connectivity).
    """
    if g.support != "volume":
        return None
    thick = (xr.zeros_like(md[g.domain_mask], dtype=float) + md[g.vertical.thickness])
    return np.asarray(thick.transpose(*g.spatial_dims).values, dtype=float)


def static_names(g) -> list:
    """Static variables of the mask that the object dataset keeps (domain, measure, thickness)."""
    names = [g.domain_mask, g.cell_measure.var]
    if g.vertical is not None:
        names.append(g.vertical.thickness)
    return names


def build_geometry(md, g):
    """``(lat, lon, geometry)``: the axes of the grid and the per-cell coordinates of the flat
    view used by the object tables.
    """
    lat, lon = _morph.axes(md, g)
    nz = md.sizes[g.spatial_dims[0]] if _morph.is_volume(g) else 1
    cells = (None, None) if _morph.is_line(g) else _morph.cell_coords(lat, lon, nz)
    return lat, lon, _geometry(md, g, *cells)


def _geometry(md, g, cells_lat, cells_lon) -> dict:
    """Coordinates of every cell of the flat view for the object tables."""
    out = {"lat": cells_lat, "lon": cells_lon}
    shape = tuple(md.sizes[d] for d in g.spatial_dims)
    if g.support == "volume":
        coordinate = md[g.vertical.coordinate]
        broadcast = (xr.zeros_like(md[g.domain_mask], dtype=float) + coordinate).transpose(
            *g.spatial_dims)
        out["depth"] = np.asarray(broadcast.values, dtype=float).reshape(-1, shape[-1])
        out["depth_units"] = coordinate.attrs.get("units", "m")
    if g.support == "line":
        out["along"] = np.asarray(md[g.coords.along].values, dtype=float)
        out["along_units"] = md[g.coords.along].attrs.get("units", "")
    return out


def _assemble(mask, recipe, mode, provenance, source, field, labels, weights, unit, geometry,
              times, order) -> ObjectField:
    """`labels` and `weights` are the flat view (time, z*y, x), see `flat`."""
    g, md = mask.grid, mask.dataset
    T, (ny, nx) = times.size, weights.shape
    shape = (T, *(md.sizes[d] for d in g.spatial_dims))
    source, processed = source.reshape(shape), field.reshape(shape)
    ds = xr.Dataset({"valid": md["valid"].transpose(*order)})
    roles = (g.coords.lat, g.coords.lon, g.coords.depth, g.coords.along)
    ds = ds.assign_coords({name: md[name] for name in roles if name is not None})
    for name in static_names(g):
        if name is not None and name in md.data_vars:
            ds[name] = md[name]
    if labels is not None:
        table, clat, clon, (tt, yy, xx, gid) = make_table(
            labels, weights, geometry["lat"], geometry["lon"])
        obj_track = np.zeros(table.n, dtype=np.int32)
        summary = parent = merged = qualified = None
        if recipe.has_track:
            step = recipe.steps[-1]
            links = _track.build_links(
                labels, weights, table, min_overlap=step.min_overlap,
                max_gap_days=step.max_gap_days, link_measure=step.rule("link_measure"),
                dominance_measure=step.rule("dominance_measure"), tie_rule=step.rule("tie_rule"))
            if mode == "monitoring":
                obj_track, parent_all, merged_all = _track.monitoring_tracks(
                    table, links, dominance_measure=step.rule("dominance_measure"),
                    tie_rule=step.rule("tie_rule"), branch_rule=step.rule("branch_rule"))
            else:
                obj_track = _track.retrospective_tracks(table, links, ny * nx)
            flagged = _track.multi_overlap(table, links)
            summary = _track.track_summary(table, obj_track, flagged, T, step.min_duration_days)
            k = summary["start"].size
            qualified = summary["duration"] >= step.min_duration_days
            if mode == "monitoring":
                parent, merged = parent_all[1:], merged_all[1:]
            else:    # unqualified tracks are dropped and the rest renumbered 1..k'
                lut = np.zeros(k + 1, dtype=np.int32)
                lut[1:][qualified] = np.arange(1, int(qualified.sum()) + 1, dtype=np.int32)
                obj_track = lut[obj_track]
                summary = {name: values[qualified] for name, values in summary.items()}
                parent = merged = np.zeros(int(qualified.sum()), dtype=np.int32)
                qualified = np.ones(int(qualified.sum()), dtype=bool)
            track_id = np.zeros(labels.shape, dtype=np.int32)
            track_id[tt, yy, xx] = obj_track[gid]
            processed = track_id > 0
            if mode == "monitoring":
                processed = np.concatenate(([False], qualified))[track_id]
            processed = processed.reshape(shape)
            ds["track_id"] = (order, track_id.reshape(shape))
        ds["label"] = (order, labels.reshape(shape))
        area_attrs = {"units": unit, "long_name": "summed cell measure of the patch"}
        ds["object_time"] = (OBJECT_DIM, times[table.day])
        ds["object_label"] = (OBJECT_DIM, table.label.astype(np.int32))
        ds["object_track"] = (OBJECT_DIM, obj_track)
        ds["object_area"] = (OBJECT_DIM, table.area, area_attrs)
        ds["object_cells"] = (OBJECT_DIM, table.cells.astype(np.int64))
        set_coordinate_vars(ds, g, geometry, table, weights, clat, clon, (yy, xx, gid))
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
            ds["track_n_multi_overlap_days"] = (TRACK_DIM,
                                                summary["n_multi_overlap_days"].astype(np.int64))
            ds["track_left_censored"] = (TRACK_DIM, summary["start"] == 0)
            ds["track_right_censored"] = (TRACK_DIM, summary["end"] == T - 1)
            ds["track_qualified"] = (TRACK_DIM, np.asarray(qualified, bool))
            ds["track_parent"] = (TRACK_DIM, np.asarray(parent, np.int32))
            ds["track_merged_into"] = (TRACK_DIM, np.asarray(merged, np.int32))
    ds["processed_mask"] = (order, processed)
    ds["added"] = (order, processed & ~source)
    ds["removed"] = (order, source & ~processed)
    if provenance is None:
        provenance = Provenance.new(parents=(mask.provenance.run_id,), steps=(ProcessingStep(
            name="build_objects", params=(("analysis_mode", mode),
                                          ("recipe_id", recipe.spec_id))),))
    return ObjectField._make(
        ds, grid=g, recipe=recipe, analysis_mode=mode, provenance=provenance,
        definition_id=mask.definition_id, baseline_id=mask.baseline_id,
        source_mask_run_id=mask.provenance.run_id, origin="xhw",
        final_after_days=_delay(recipe, mask) if mode == "monitoring" else 0,
        source_layer=recipe.input_layer)
