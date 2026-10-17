# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""XMask: detection evidence of one phenomenon on its native grid."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from ._errors import InvariantError, SchemaError, ValidationError
from ._schema import (
    CONVENTIONS,
    SCHEMA_VERSIONS,
    check_schema_version,
    check_serializable_attrs,
    require,
    temperature_units,
)
from ._specs import BaselineSpec, DefinitionSpec, GridSpec, Provenance

BOOL_VARS = ("exceedance", "event_membership", "joined_gap", "valid")
FLOAT_VARS = ("threshold", "reference")
#: dimension of per-component variables of ``combination="and_daily"``
COMPONENT = "component"
LAG_COORD = "lag_days"
ON_SCALE = "temperature: on_scale"
DIFFERENCE = "temperature: difference"

#: `invalid_reason` bits; new bits may only be added (compatible change).
OUTSIDE_DOMAIN, MISSING_DATA, BASELINE_UNDEFINED, INSUFFICIENT_SAMPLES = 1, 2, 4, 8
LOW_DATA_QUALITY = 16  # series with too many filled days
KNOWN_REASON_BITS = 1 | 2 | 4 | 8 | 16
INVALID_REASON_MASKS = np.array([1, 2, 4, 8, 16], dtype=np.uint8)
INVALID_REASON_MEANINGS = ("outside_domain missing_data baseline_undefined insufficient_samples "
                           "low_data_quality")

_MARINE = ("ocean",)
_ATMOSPHERIC = ("land", "air_over_ocean", "air_over_land")
_XHW_ATTRS = ("Conventions", "xhw_schema", "xhw_schema_version", "xhw_grid", "xhw_definition",
              "xhw_baseline", "xhw_provenance", "definition_id", "baseline_id")


def read_only(ds: xr.Dataset) -> xr.Dataset:
    """Shallow copy whose numpy-backed non-index variables are read-only views (no data copy)."""
    ds = ds.copy(deep=False)
    for var in ds.variables.values():
        if isinstance(var, xr.IndexVariable):
            continue
        data = var.data
        if isinstance(data, np.ndarray) and data.flags.writeable:
            view = data.view()
            view.flags.writeable = False
            var.data = view
    return ds


def strip_xhw_attrs(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=False)
    ds.attrs = {k: v for k, v in ds.attrs.items() if k not in _XHW_ATTRS}
    return ds


def load_attr_json(ds: xr.Dataset, name: str) -> dict:
    require(name in ds.attrs, f"serialized attribute {name!r} is missing", SchemaError)
    try:
        return json.loads(ds.attrs[name])
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"attribute {name!r} is not valid JSON") from exc


def check_entity(ds: xr.Dataset, entity: str) -> None:
    require(ds.attrs.get("xhw_schema") == entity, f"dataset is not a serialized {entity}",
            SchemaError)
    check_schema_version(entity, ds.attrs.get("xhw_schema_version", ""))


def check_stored_id(ds: xr.Dataset, name: str, spec) -> None:
    require(ds.attrs.get(name) == spec.spec_id,
            f"{name} does not match its serialized spec (tampered or inconsistent metadata)",
            SchemaError)


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of True runs in a 1-D boolean array."""
    padded = np.concatenate(([False], np.asarray(flags, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(s), int(e) - 1) for s, e in zip(edges[::2], edges[1::2])]


def columns(da: xr.DataArray, time_dim: str, spatial_dims: tuple[str, ...]) -> np.ndarray:
    """Values as a (time, locations) array; computes lazy data (explicit checks only)."""
    values = np.asarray(da.transpose(time_dim, *spatial_dims).values)
    return values.reshape(values.shape[0], -1)


def lead(values: np.ndarray, lag: int, fill) -> np.ndarray:
    """Value of day D+lag on day D along axis 0; the last `lag` days get `fill`."""
    if lag == 0:
        return values
    out = np.full_like(values, fill)
    out[:-lag] = values[lag:]
    return out


def component_series(ds: xr.Dataset, definition: DefinitionSpec, time_dim: str,
                     spatial_dims: tuple[str, ...], names: tuple[str, ...]) -> list[dict]:
    """Per component: (time, locations) arrays of `names` aligned on the reference day.

    For ``and_daily`` the value of component c on reference day D is its value on D+lag_c; a
    single definition yields one entry without shift. Computes lazy data.
    """
    if definition.combination != "and_daily":
        return [{n: columns(ds[n], time_dim, spatial_dims) for n in names}]
    out = []
    for i, comp in enumerate(definition.components):
        out.append({n: lead(columns(ds[n].isel({COMPONENT: i}), time_dim, spatial_dims).astype(
            float), comp.lag_days, np.nan) for n in names})
    return out


@dataclass(frozen=True, eq=False)
class XMask:
    """Detection evidence: exceedance, event membership, joined gaps and validity.

    Build it with `from_dataset`. `dataset` returns a shallow copy whose numpy data are
    read-only; any change requires a new instance. Spatial preprocessing never rewrites this
    evidence: `processed_mask` is a separate entity.
    """

    _dataset: xr.Dataset
    grid: GridSpec
    definition: DefinitionSpec
    baseline: BaselineSpec
    provenance: Provenance

    # -- construction -------------------------------------------------------------------

    @classmethod
    def from_dataset(
        cls,
        ds: xr.Dataset,
        *,
        grid: GridSpec | None = None,
        definition: DefinitionSpec | None = None,
        baseline: BaselineSpec | None = None,
        provenance: Provenance | None = None,
        copy: bool = False,
    ) -> XMask:
        """Validate structure (without computing data) and build an immutable mask.

        Pass the four specs to build a new mask, or none of them to read a dataset produced by
        `to_dataset` (schema policy and identifier integrity are then enforced). `copy=True`
        also copies numpy data to isolate the mask from the caller's arrays.
        """
        require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
        given = [grid, definition, baseline, provenance]
        if all(spec is None for spec in given):
            check_entity(ds, "XMask")
            grid = GridSpec.from_dict(load_attr_json(ds, "xhw_grid"))
            definition = DefinitionSpec.from_dict(load_attr_json(ds, "xhw_definition"))
            baseline = BaselineSpec.from_dict(load_attr_json(ds, "xhw_baseline"))
            provenance = Provenance.from_dict(load_attr_json(ds, "xhw_provenance"))
            check_stored_id(ds, "definition_id", definition)
            check_stored_id(ds, "baseline_id", baseline)
        elif any(spec is None for spec in given):
            raise ValidationError("pass all of grid, definition, baseline and provenance, or none")
        require(isinstance(grid, GridSpec), "grid must be GridSpec")
        require(isinstance(definition, DefinitionSpec), "definition must be DefinitionSpec")
        require(isinstance(baseline, BaselineSpec), "baseline must be BaselineSpec")
        require(isinstance(provenance, Provenance), "provenance must be Provenance")
        ds = strip_xhw_attrs(ds.copy(deep=True) if copy else ds)
        _check_structure(ds, grid, definition)
        return cls(read_only(ds), grid, definition, baseline, provenance)

    # -- access -------------------------------------------------------------------------

    @property
    def dataset(self) -> xr.Dataset:
        """Shallow copy of the data; replacing variables in it does not affect the mask."""
        return self._dataset.copy(deep=False)

    @property
    def definition_id(self) -> str:
        return self.definition.spec_id

    @property
    def baseline_id(self) -> str:
        return self.baseline.spec_id

    def intensity(self) -> xr.DataArray:
        """Signed intensity temperature - reference (lazy); a temperature difference."""
        ds = self._dataset
        require("temperature" in ds, "intensity needs the optional 'temperature' variable")
        out = ds["temperature"] - ds["reference"]
        # Physical roles may be stored as Dataset variables rather than coordinates.
        # Preserve cell identity on the derived DataArray used by lineage intensity.
        roles = (self.grid.coords.lat, self.grid.coords.lon, self.grid.coords.depth,
                 self.grid.coords.along)
        out = out.assign_coords({name: ds[name] for name in roles
                                 if name is not None and name in ds
                                 and set(ds[name].dims) <= set(out.dims)})
        out.attrs = {"units": ds["temperature"].attrs["units"], "units_metadata": DIFFERENCE,
                     "long_name": "signed intensity relative to the reference climatology"}
        return out.rename("intensity")

    # -- serialization ------------------------------------------------------------------

    def to_dataset(self) -> xr.Dataset:
        """Dataset with JSON specs in attributes, CF flag attributes and the declared calendar."""
        ds = self._dataset.copy(deep=False)
        ds.attrs.update({
            "Conventions": CONVENTIONS,
            "xhw_schema": "XMask",
            "xhw_schema_version": SCHEMA_VERSIONS["XMask"],
            "xhw_grid": self.grid.to_json(),
            "xhw_definition": self.definition.to_json(),
            "xhw_baseline": self.baseline.to_json(),
            "xhw_provenance": self.provenance.to_json(),
            "definition_id": self.definition_id,
            "baseline_id": self.baseline_id,
        })
        for name in BOOL_VARS:
            ds[name].attrs.update({"flag_values": np.array([0, 1], dtype=np.int8),
                                   "flag_meanings": "false true"})
        ds["invalid_reason"].attrs.update({"flag_masks": INVALID_REASON_MASKS,
                                           "flag_meanings": INVALID_REASON_MEANINGS})
        if "filled" in ds:
            ds["filled"].attrs.update({"flag_values": np.array([0, 1], dtype=np.int8),
                                       "flag_meanings": "observed interpolated"})
        if "segment" in ds:
            ds["segment"].attrs.update({"flag_values": np.array([0, 1], dtype=np.int8),
                                        "flag_meanings": "observed forecast"})
        ds[self.grid.time_dim].encoding["calendar"] = self.grid.calendar
        return ds

    # -- value invariants (explicit, computes data) -------------------------------------

    def check_invariants(self) -> None:
        """Evaluate all value invariants; reads every variable (cost: one pass over the data)."""
        ds, g, d = self._dataset, self.grid, self.definition
        get = {name: columns(ds[name], g.time_dim, g.spatial_dims)
               for name in (*BOOL_VARS, "invalid_reason")}
        exc, mem, gap, valid = (get[n] for n in BOOL_VARS)
        names = ("threshold", "temperature") if "temperature" in ds else ("threshold",)
        series = component_series(ds, d, g.time_dim, g.spatial_dims, names)
        reason = get["invalid_reason"]
        problems: list[str] = []

        def check(ok: Any, message: str) -> None:
            if not bool(np.all(ok)):
                problems.append(message)

        check(~exc | valid, "exceedance on invalid days")
        #: an invalid day may belong to an event only as a joined gap, only because
        # of missing data, and only under missing_policy="as_non_exceedance".
        bridged = gap & (reason == MISSING_DATA) & (d.missing_policy == "as_non_exceedance")
        check(~mem | valid | bridged, "event_membership on invalid days")
        check(gap == (mem & ~exc), "joined_gap != event_membership & ~exceedance")
        check(valid == (reason == 0), "valid != (invalid_reason == 0)")
        check((reason & ~np.uint8(KNOWN_REASON_BITS)) == 0, "unknown invalid_reason bits")
        for s in series:
            check(np.isfinite(s["threshold"]) | ~valid, "valid days without a finite threshold")
        domain = np.asarray(ds[g.domain_mask].transpose(*g.spatial_dims).values).reshape(-1)
        outside = (reason & OUTSIDE_DOMAIN) != 0
        check(outside == ~domain[np.newaxis, :], "outside_domain bit != ~domain_mask")
        if "temperature" in ds:
            expected = np.ones(exc.shape, dtype=bool)
            for s in series:   # and_daily: AND over components on their own dates
                check(np.isfinite(s["temperature"]) | ~valid,
                      "valid days without a finite temperature")
                with np.errstate(invalid="ignore"):
                    expected &= _compare(d.comparator, s["temperature"], s["threshold"])
            check((exc == expected) | ~valid, f"exceedance != temperature {d.comparator} threshold")
        for col in range(mem.shape[1]):
            for start, end in runs(mem[:, col]):
                if end - start + 1 < d.min_duration_days:
                    problems.append(f"membership run shorter than min_duration at location {col}")
                if not (exc[start, col] and exc[end, col]):
                    problems.append(f"event does not start/end on exceedance at location {col}")
            for start, end in runs(gap[:, col]):
                if end - start + 1 > d.max_gap_days:
                    problems.append(f"joined gap longer than max_gap_days at location {col}")
        if "segment" in ds:
            seg = np.asarray(ds["segment"].values)
            check(np.isin(seg, (0, 1)), "segment values must be 0 or 1")
            check(np.diff(seg.astype(int)) >= 0, "forecast segment must follow observations")
        if problems:
            raise InvariantError("; ".join(dict.fromkeys(problems)))


def _compare(comparator: str, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return {">": np.greater, ">=": np.greater_equal, "<": np.less,
            "<=": np.less_equal}[comparator](left, right)


def check_component_coords(ds: xr.Dataset, definition: DefinitionSpec,
                           lags: bool = True) -> None:
    """``component`` (and ``lag_days``) coordinates must match the definition's components."""
    names = [c.variable for c in definition.components]
    require(COMPONENT in ds.coords and [str(v) for v in ds[COMPONENT].values] == names,
            f"{COMPONENT!r} coordinate must list the component variables {names}")
    if lags:
        expected = [c.lag_days for c in definition.components]
        require(LAG_COORD in ds.coords and ds[LAG_COORD].dims == (COMPONENT,)
                and [int(v) for v in ds[LAG_COORD].values] == expected,
                f"{LAG_COORD!r} coordinate along {COMPONENT!r} must be {expected}")


def _check_structure(ds: xr.Dataset, grid: GridSpec, definition: DefinitionSpec) -> None:
    """Structural validation only: names, dims, dtypes, units and attributes (no data read)."""
    grid.check(ds)
    expected_dims = {grid.time_dim, *grid.spatial_dims}
    combined = definition.combination == "and_daily"
    component_dims = expected_dims | {COMPONENT} if combined else expected_dims
    for name in (*BOOL_VARS, *FLOAT_VARS, "invalid_reason"):
        require(name in ds.data_vars, f"required variable {name!r} is missing")
    for name in (*BOOL_VARS, "invalid_reason"):
        require(set(ds[name].dims) == expected_dims,
                f"{name!r} dims {ds[name].dims} != {sorted(expected_dims)}")
    for name in (*FLOAT_VARS, "temperature"):
        if name in ds.data_vars:
            require(set(ds[name].dims) == component_dims,
                    f"{name!r} dims {ds[name].dims} != {sorted(component_dims)}")
    if combined:
        check_component_coords(ds, definition)
    for name in BOOL_VARS:
        require(ds[name].dtype == bool, f"{name!r} must be boolean")
    if "filled" in ds.data_vars:
        require(set(ds["filled"].dims) == expected_dims and ds["filled"].dtype == bool,
                "'filled' must be boolean with the mask dims ")
    require(ds["invalid_reason"].dtype == np.uint8, "'invalid_reason' must be uint8")
    units = set()
    for name in (*FLOAT_VARS, "temperature"):
        if name not in ds.data_vars:
            continue
        require(ds[name].dtype.kind == "f", f"{name!r} must be floating point")
        units.add(temperature_units(name, ds[name].attrs.get("units")))
        require(ds[name].attrs.get("units_metadata") == ON_SCALE,
                f"{name!r} needs units_metadata={ON_SCALE!r}")
    require(len(units) == 1, f"threshold/reference/temperature units differ: {sorted(units)}")
    if "temperature" in ds.data_vars and "standard_name" in ds["temperature"].attrs:
        require(ds["temperature"].attrs["standard_name"] == definition.variable,
                "temperature standard_name differs from definition.variable")
    if definition.phenomenon in ("mhw", "mcs"):
        require(grid.domain in _MARINE, f"{definition.phenomenon} requires an ocean domain")
    else:
        require(grid.domain in _ATMOSPHERIC, "ahw requires a land or air domain")
    if grid.time_axes == "forecast":
        require("segment" in ds.data_vars, "forecast masks need a 'segment' variable")
        require(ds["segment"].dims == (grid.time_dim,) and ds["segment"].dtype == np.int8,
                "'segment' must be int8 along time")
    else:
        require("segment" not in ds.data_vars, "'segment' is only allowed for forecast masks")
    check_serializable_attrs("dataset", ds.attrs)
    for name, var in ds.variables.items():
        check_serializable_attrs(f"variable {name!r}", var.attrs)
