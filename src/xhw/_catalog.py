# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""EventCatalog: tabular events (and monitoring candidates) derived from an XMask.

Two granularities. ``point_temporal``: events per location (cell), from the mask.
``object_tracked``: one row per spacetime event of a LineageGraph
(`LineageGraph.to_event_catalog`); it has no location dims, its rows may overlap in time, and
``n_exceedance_days`` / ``n_joined_gap_days`` are the days with a patch and the bridged days.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import xarray as xr

from ._errors import InvariantError, NotSupportedError, SchemaError, ValidationError
from ._mask import (
    BOOL_VARS,
    COMPONENT,
    XMask,
    check_component_coords,
    check_entity,
    check_stored_id,
    columns,
    load_attr_json,
    read_only,
    runs,
    strip_xhw_attrs,
)
from ._schema import (
    CONVENTIONS,
    SCHEMA_VERSIONS,
    check_choice,
    check_serializable_attrs,
    require,
)
from ._specs import BaselineSpec, DefinitionSpec, Provenance

EVENT_DIM = "event"
#: `censor_reason` bits.
WINDOW_START, WINDOW_END, MISSING_DATA, FORECAST_HORIZON = 1, 2, 4, 8
CENSOR_MASKS = np.array([1, 2, 4, 8], dtype=np.uint8)
CENSOR_MEANINGS = "window_start window_end missing_data forecast_horizon"

_INT_VARS = ("event_id", "duration_days", "n_exceedance_days", "n_joined_gap_days")
_BOOL_VARS = ("left_censored", "right_censored", "qualified", "contains_forecast")
_TIME_VARS = ("start", "end")
_CATALOG_ATTRS = ("granularity", "analysis_mode", "location_dims", "source_mask_run_id",
                  "window_start", "window_end", "phenomenon")
_ONE_DAY = np.timedelta64(1, "D")


def index_var(dim: str) -> str:
    return f"index_{dim}"


class EventRow(NamedTuple):
    """One expected catalog row derived from a mask (shared by build_catalog/check_against)."""

    col: int
    loc: tuple
    start: int
    end: int
    fields: tuple  # (qualified, n_exc, n_gap, left, right, reason, contains_forecast)


def event_rows(mask: XMask, analysis_mode: str) -> list[EventRow]:
    """Events (membership runs) and, in monitoring, edge candidates, with censoring.

    A neighbouring invalid day censors the event on that side (``missing_data``); the first
    and last days of the window censor as ``window_start``/``window_end``
    (``forecast_horizon`` for forecast masks). Computes the mask data.
    """
    g, ds = mask.grid, mask.dataset
    times = ds[g.time_dim].values
    exc, mem, gap, valid = (columns(ds[n], g.time_dim, g.spatial_dims) for n in BOOL_VARS)
    forecast = g.time_axes == "forecast"
    seg = np.asarray(ds["segment"].values) if forecast else np.zeros(times.size, np.int8)
    shape = tuple(ds.sizes[d] for d in g.spatial_dims)
    last = times.size - 1
    min_days = mask.definition.min_duration_days

    def fields(col: int, s: int, e: int, qualified: bool) -> tuple:
        reason = 0
        if s == 0:
            reason |= WINDOW_START
        elif not valid[s - 1, col]:
            reason |= MISSING_DATA
        left = reason != 0
        if e == last:
            reason |= FORECAST_HORIZON if forecast else WINDOW_END
        elif not valid[e + 1, col]:
            reason |= MISSING_DATA
        right = e == last or not valid[e + 1, col]
        return (qualified, int(exc[s:e + 1, col].sum()), int(gap[s:e + 1, col].sum()),
                bool(left), bool(right), reason, bool(np.any(seg[s:e + 1] == 1)))

    rows: list[EventRow] = []
    for col in range(exc.shape[1]):
        loc = tuple(int(i) for i in np.unravel_index(col, shape)) if shape else ()
        found = [(s, e, True) for s, e in runs(mem[:, col])]
        if analysis_mode == "monitoring":
            for s, e in runs(exc[:, col] & ~mem[:, col]):
                touches = s == 0 or e == last or not valid[s - 1, col] or not valid[e + 1, col]
                if e - s + 1 < min_days and touches:
                    found.append((s, e, False))
        for s, e, qualified in sorted(found):
            rows.append(EventRow(col, loc, s, e, fields(col, s, e, qualified)))
    return rows


@dataclass(frozen=True, eq=False)
class EventCatalog:
    """Events of one phenomenon, one row per event (dimension ``event``).

    ``analysis_mode="retrospective"`` holds qualified events only. ``"monitoring"`` also
    records candidates: exceedance runs shorter than the minimum duration that touch a window
    edge or missing data (``qualified=False``). Events touching an edge are censored in both
    modes. Global key: ``(phenomenon, provenance.run_id, event_id)``.
    """

    _dataset: xr.Dataset
    definition: DefinitionSpec
    baseline: BaselineSpec
    provenance: Provenance
    granularity: str
    analysis_mode: str
    location_dims: tuple[str, ...]
    source_mask_run_id: str
    window_start: str
    window_end: str

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        *,
        mask: XMask | None = None,
        granularity: str | None = None,
        analysis_mode: str | None = None,
        provenance: Provenance | None = None,
        copy: bool = False,
    ) -> EventCatalog:
        """Build a catalog for `mask`, or read one produced by `to_dataset` (no keywords).

        Catalog rows are 1-D and small: their value rules are validated at construction.
        """
        require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
        if mask is None and granularity is None and analysis_mode is None and provenance is None:
            check_entity(ds, "EventCatalog")
            definition = DefinitionSpec.from_dict(load_attr_json(ds, "xhw_definition"))
            baseline = BaselineSpec.from_dict(load_attr_json(ds, "xhw_baseline"))
            provenance = Provenance.from_dict(load_attr_json(ds, "xhw_provenance"))
            check_stored_id(ds, "definition_id", definition)
            check_stored_id(ds, "baseline_id", baseline)
            for name in _CATALOG_ATTRS:
                require(name in ds.attrs, f"serialized attribute {name!r} is missing", SchemaError)
            require(ds.attrs["phenomenon"] == definition.phenomenon,
                    "phenomenon attribute differs from the definition", SchemaError)
            fields: dict[str, Any] = {
                "granularity": ds.attrs["granularity"],
                "analysis_mode": ds.attrs["analysis_mode"],
                "location_dims": tuple(json.loads(ds.attrs["location_dims"])),
                "source_mask_run_id": ds.attrs["source_mask_run_id"],
                "window_start": ds.attrs["window_start"],
                "window_end": ds.attrs["window_end"],
            }
        elif mask is not None and None not in (granularity, analysis_mode, provenance):
            require(isinstance(mask, XMask), "mask must be an XMask")
            require(isinstance(provenance, Provenance), "provenance must be Provenance")
            require(mask.provenance.run_id in provenance.parents,
                    "catalog provenance must list the source mask run id as a parent")
            definition, baseline = mask.definition, mask.baseline
            if granularity == "object_tracked":
                require(mask.grid.support in ("line", "surface", "volume"),
                        "object_tracked catalogs need a line, surface or volume mask",
                        NotSupportedError)
            times = mask.dataset[mask.grid.time_dim].values
            fields = {
                "granularity": granularity,
                "analysis_mode": analysis_mode,
                "location_dims": (mask.grid.spatial_dims if granularity == "point_temporal"
                                  else ()),
                "source_mask_run_id": mask.provenance.run_id,
                "window_start": str(np.datetime_as_string(times[0], unit="s")),
                "window_end": str(np.datetime_as_string(times[-1], unit="s")),
            }
        else:
            raise ValidationError(
                "pass mask, granularity, analysis_mode and provenance, or none of them"
            )
        ds = strip_xhw_attrs(ds.copy(deep=True) if copy else ds)
        ds.attrs = {k: v for k, v in ds.attrs.items() if k not in _CATALOG_ATTRS}
        catalog = cls(read_only(ds), definition, baseline, provenance, **fields)
        catalog._validate()
        return catalog

    # -- access -------------------------------------------------------------------------

    @property
    def dataset(self) -> xr.Dataset:
        """Shallow copy of the rows; replacing variables in it does not affect the catalog."""
        return self._dataset.copy(deep=False)

    @property
    def phenomenon(self) -> str:
        return self.definition.phenomenon

    @property
    def definition_id(self) -> str:
        return self.definition.spec_id

    @property
    def baseline_id(self) -> str:
        return self.baseline.spec_id

    # -- serialization ------------------------------------------------------------------

    def to_dataset(self) -> xr.Dataset:
        ds = self._dataset.copy(deep=False)
        ds.attrs.update({
            "Conventions": CONVENTIONS,
            "xhw_schema": "EventCatalog",
            "xhw_schema_version": SCHEMA_VERSIONS["EventCatalog"],
            "xhw_definition": self.definition.to_json(),
            "xhw_baseline": self.baseline.to_json(),
            "xhw_provenance": self.provenance.to_json(),
            "definition_id": self.definition_id,
            "baseline_id": self.baseline_id,
            "phenomenon": self.phenomenon,
            "granularity": self.granularity,
            "analysis_mode": self.analysis_mode,
            "location_dims": json.dumps(list(self.location_dims)),
            "source_mask_run_id": self.source_mask_run_id,
            "window_start": self.window_start,
            "window_end": self.window_end,
        })
        ds["censor_reason"].attrs.update({"flag_masks": CENSOR_MASKS,
                                          "flag_meanings": CENSOR_MEANINGS})
        return ds

    # -- validation ---------------------------------------------------------------------

    def _validate(self) -> None:
        check_choice("granularity", self.granularity, ("point_temporal", "object_tracked"))
        if self.granularity == "object_tracked":
            require(self.location_dims == (), "object_tracked catalogs have no location dims")
        check_choice("analysis_mode", self.analysis_mode, ("retrospective", "monitoring"))
        start, end = np.datetime64(self.window_start, "ns"), np.datetime64(self.window_end, "ns")
        require(start <= end, "window_start must not be after window_end")
        if self.analysis_mode == "monitoring":
            require(self.baseline.available_before(start),
                    "monitoring requires a baseline available before the window ")
        ds = self._dataset
        names = (*_INT_VARS, *_BOOL_VARS, *_TIME_VARS, "censor_reason",
                 *(index_var(d) for d in self.location_dims))
        for name in names:
            require(name in ds.data_vars, f"required catalog variable {name!r} is missing")
            require(ds[name].dims == (EVENT_DIM,), f"{name!r} must be 1-D along {EVENT_DIM!r}")
        allowed = [(EVENT_DIM,)]
        if self.definition.combination == "and_daily":   # per-component metrics
            allowed.append((EVENT_DIM, COMPONENT))
            if any(var.dims == allowed[1] for var in ds.data_vars.values()):
                check_component_coords(ds, self.definition)
        for name, var in ds.data_vars.items():
            require(var.dims in allowed, f"{name!r} dims {var.dims} not in {allowed}")
        for name in (*_INT_VARS, *(index_var(d) for d in self.location_dims)):
            require(ds[name].dtype == np.int64, f"{name!r} must be int64")
        for name in _BOOL_VARS:
            require(ds[name].dtype == bool, f"{name!r} must be boolean")
        for name in _TIME_VARS:
            require(ds[name].dtype.kind == "M", f"{name!r} must be datetime64")
        require(ds["censor_reason"].dtype == np.uint8, "'censor_reason' must be uint8")
        check_serializable_attrs("catalog", ds.attrs)
        for name, var in ds.variables.items():
            check_serializable_attrs(f"variable {name!r}", var.attrs)
        self.check_invariants()

    def check_invariants(self) -> None:
        """Row-level rules (1-D, cheap; also run at construction)."""
        ds, problems = self._dataset, []
        v = {name: np.asarray(ds[name].values) for name in ds.data_vars}
        start, end = v["start"], v["end"]
        w0, w1 = np.datetime64(self.window_start, "ns"), np.datetime64(self.window_end, "ns")
        reason = v["censor_reason"]
        left, right, qualified = v["left_censored"], v["right_censored"], v["qualified"]

        def check(ok: Any, message: str) -> None:
            if not bool(np.all(ok)):
                problems.append(message)

        check(np.unique(v["event_id"]).size == v["event_id"].size, "event_id values not unique")
        check(start <= end, "start after end")
        check((start >= w0) & (end <= w1), "events outside the catalog window")
        days = (end - start) // _ONE_DAY + 1
        check(v["duration_days"] == days, "duration_days != end - start + 1")
        check(v["duration_days"] == v["n_exceedance_days"] + v["n_joined_gap_days"],
              "duration_days != n_exceedance_days + n_joined_gap_days")
        check((v["n_exceedance_days"] >= 1) & (v["n_joined_gap_days"] >= 0), "negative counts")
        min_days = self.definition.min_duration_days
        check(~qualified | (v["duration_days"] >= min_days),
              "qualified events shorter than min_duration_days")
        check(qualified | ((v["duration_days"] < min_days) & (v["n_joined_gap_days"] == 0)
                           & (left | right)),
              "candidates must be short, gap-free and censored")
        if self.analysis_mode == "retrospective":
            check(qualified, "retrospective catalogs cannot hold candidates ")
        check((reason & ~np.uint8(15)) == 0, "unknown censor_reason bits")
        check((left | right) == (reason != 0), "censored flags != (censor_reason != 0)")
        check(((reason & WINDOW_START) == 0) | (left & (start == w0)),
              "window_start censoring needs left_censored and start == window_start")
        check((start != w0) | ((reason & WINDOW_START) != 0), "event at window start not censored")
        at_end = (reason & (WINDOW_END | FORECAST_HORIZON)) != 0
        check(~at_end | (right & (end == w1)), "window/horizon censoring needs end == window_end")
        check((end != w1) | at_end, "event at window end not censored")
        check(((reason & WINDOW_END) == 0) | ((reason & FORECAST_HORIZON) == 0),
              "window_end and forecast_horizon are exclusive")
        check(((reason & FORECAST_HORIZON) == 0) | v["contains_forecast"],
              "forecast_horizon censoring requires contains_forecast")
        if self.granularity != "object_tracked":     # spacetime events may overlap in time
            keys = [v[index_var(d)] for d in self.location_dims]
            # sorted by (location, start): events of one location are then adjacent, so an
            # overlap is always between neighbours (one pass instead of one per location)
            order = np.lexsort((start, *reversed(keys)))
            first, last = order[:-1], order[1:]
            overlap = start[last] <= end[first]
            for key in keys:
                overlap &= key[first] == key[last]
            for i in np.flatnonzero(overlap):
                loc = tuple(key[first[i]] for key in keys)
                problems.append(f"overlapping events at location {loc}")
        if problems:
            raise InvariantError("; ".join(dict.fromkeys(problems)))

    def check_against(self, mask: XMask) -> None:
        """Verify rows, counts and censoring against a mask (computes the mask's data). For an
        ``object_tracked`` catalog only the ids and the window are checked here; its rows are
        checked against the graph with `LineageGraph.check_catalog`.
        """
        require(mask.definition_id == self.definition_id, "definition_id differs from the mask")
        require(mask.baseline_id == self.baseline_id, "baseline_id differs from the mask")
        require(mask.provenance.run_id == self.source_mask_run_id, "source mask run id differs")
        if self.granularity == "object_tracked":
            times = mask.dataset[mask.grid.time_dim].values
            require(np.datetime64(self.window_start, "ns") == times[0]
                    and np.datetime64(self.window_end, "ns") == times[-1],
                    "catalog window differs from the mask time range")
            return
        require(mask.grid.spatial_dims == self.location_dims, "location dims differ from the mask")
        times = mask.dataset[mask.grid.time_dim].values
        require(np.datetime64(self.window_start, "ns") == times[0]
                and np.datetime64(self.window_end, "ns") == times[-1],
                "catalog window differs from the mask time range")
        expected = {(r.loc, r.start, r.end): r.fields for r in event_rows(mask, self.analysis_mode)}
        position = {t: i for i, t in enumerate(times)}
        v = {name: np.asarray(self._dataset[name].values) for name in self._dataset.data_vars}
        actual: dict[tuple, tuple] = {}
        for i in range(v["start"].size):
            loc = tuple(int(v[index_var(d)][i]) for d in self.location_dims)
            s, e = position.get(v["start"][i]), position.get(v["end"][i])
            require(s is not None and e is not None, f"event {v['event_id'][i]} dates not in mask")
            actual[(loc, s, e)] = (bool(v["qualified"][i]), int(v["n_exceedance_days"][i]),
                                   int(v["n_joined_gap_days"][i]), bool(v["left_censored"][i]),
                                   bool(v["right_censored"][i]), int(v["censor_reason"][i]),
                                   bool(v["contains_forecast"][i]))
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            wrong = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])
            raise InvariantError(
                f"catalog does not match mask: missing rows {missing}, unexpected rows {extra}, "
                f"rows with wrong counts/censoring {wrong} "
                "(fields: qualified, n_exc, n_gap, left, right, reason, forecast)"
            )
