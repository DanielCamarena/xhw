# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Immutable metadata contracts of data contracts: specs, source references and provenance.

All classes are frozen, keyword-only dataclasses without scientific defaults. Validation in
``__post_init__`` only inspects the metadata itself; `GridSpec.check` compares a spec with a
dataset reading 1-D coordinates and static variables only.
"""

from __future__ import annotations

import platform
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from typing import Any, ClassVar, Literal

import numpy as np
import xarray as xr

from ._errors import CalendarNotSupportedError, NotSupportedError, ValidationError
from ._schema import (
    SpecBase,
    check_choice,
    check_nonempty_str,
    check_optional_str,
    check_pairs,
    is_int,
    is_number,
    measure_units,
    parse_iso,
    parse_iso_utc,
    require,
    temperature_units,
)

SUPPORTED_CALENDARS = ("standard", "gregorian", "proleptic_gregorian")
_GREGORIAN_START = np.datetime64("1582-10-15")
_ONE_DAY = np.timedelta64(1, "D")


def _spec(cls):
    return dataclass(frozen=True, kw_only=True, slots=True)(cls)


def _set(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


# --- provenance -------------------------------------------------------------------------


@_spec
class SourceRef(SpecBase):
    """Reproducible reference to an input product (dataset id, version, query, status)."""

    dataset_id: str
    variable: str
    accessed: str
    product_status: Literal["final", "preliminary", "unknown"]
    version: str | None = None
    query: str | None = None
    checksum: str | None = None

    def __post_init__(self) -> None:
        check_nonempty_str("dataset_id", self.dataset_id)
        check_nonempty_str("variable", self.variable)
        parse_iso("accessed", self.accessed)
        check_choice("product_status", self.product_status, ("final", "preliminary", "unknown"))
        for name in ("version", "query", "checksum"):
            check_optional_str(name, getattr(self, name))


def _step_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool)) or is_number(value)


@_spec
class ProcessingStep(SpecBase):
    """One serializable processing step."""

    name: str
    params: tuple[tuple[str, str | int | float | bool | None], ...] = ()

    def __post_init__(self) -> None:
        check_nonempty_str("name", self.name)
        check_pairs("params", self.params, _step_value)


@_spec
class Provenance(SpecBase):
    """Run identity, software versions, inputs, parents and processing steps."""

    _nested: ClassVar[dict[str, type]] = {"inputs": SourceRef, "steps": ProcessingStep}

    run_id: str
    created_utc: str
    software: tuple[tuple[str, str], ...]
    inputs: tuple[SourceRef, ...] = ()
    parents: tuple[str, ...] = ()
    steps: tuple[ProcessingStep, ...] = ()

    def __post_init__(self) -> None:
        check_nonempty_str("run_id", self.run_id)
        parse_iso("created_utc", self.created_utc)
        check_pairs("software", self.software, lambda v: isinstance(v, str))
        require(all(isinstance(s, SourceRef) for s in self.inputs), "inputs must be SourceRef")
        require(isinstance(self.parents, tuple), "parents must be a tuple of run ids")
        for parent in self.parents:
            check_nonempty_str("parents entry", parent)
        require(all(isinstance(s, ProcessingStep) for s in self.steps),
                "steps must be ProcessingStep")

    @classmethod
    def new(
        cls,
        *,
        inputs: tuple[SourceRef, ...] = (),
        parents: tuple[str, ...] = (),
        steps: tuple[ProcessingStep, ...] = (),
    ) -> Provenance:
        """Create provenance for a new run: random run id, UTC timestamp, software versions."""
        software = (
            ("numpy", np.__version__),
            ("python", platform.python_version()),
            ("xarray", xr.__version__),
            ("xhw", version("xhw")),
        )
        return cls(
            run_id=str(uuid.uuid4()),
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            software=software,
            inputs=tuple(inputs),
            parents=tuple(parents),
            steps=tuple(steps),
        )


# --- grid -------------------------------------------------------------------------------


@_spec
class CoordRoles(SpecBase):
    """Names of coordinate variables by role; lat/lon may be 2-D."""

    lat: str | None = None
    lon: str | None = None
    depth: str | None = None
    along: str | None = None

    def __post_init__(self) -> None:
        for name in ("lat", "lon", "depth", "along"):
            check_optional_str(name, getattr(self, name))


@_spec
class CellMeasure(SpecBase):
    """Physical weight of each cell; 'unit' (cell counts) must be declared explicitly."""

    kind: Literal["none", "unit", "length", "area", "volume"]
    var: str | None = None
    units: str | None = None

    def __post_init__(self) -> None:
        check_choice("cell_measure.kind", self.kind, ("none", "unit", "length", "area", "volume"))
        if self.kind in ("none", "unit"):
            require(self.var is None and self.units is None,
                    f"cell_measure kind {self.kind!r} takes no variable or units")
        else:
            check_nonempty_str("cell_measure.var", self.var)
            measure_units(self.kind, self.units)


@_spec
class VerticalSpec(SpecBase):
    """Vertical axis of a volume: coordinate and layer thickness variable."""

    coordinate: str
    thickness: str
    thickness_units: str
    time_dependent: bool

    def __post_init__(self) -> None:
        check_nonempty_str("vertical.coordinate", self.coordinate)
        check_nonempty_str("vertical.thickness", self.thickness)
        measure_units("length", self.thickness_units)
        require(isinstance(self.time_dependent, bool), "vertical.time_dependent must be bool")
        if self.time_dependent:
            raise NotSupportedError(
                "time-dependent vertical geometry (e.g. CROCO sigma) is rejected in data contracts "
            )


@_spec
class NativeGrid(SpecBase):
    """Identity of the analysed grid; regridded data must reference its origin."""

    _nested: ClassVar[dict[str, type]] = {"regridded_from": SourceRef}

    product: str
    resolution: str
    is_native: bool
    regridded_from: SourceRef | None = None

    def __post_init__(self) -> None:
        check_nonempty_str("native.product", self.product)
        check_nonempty_str("native.resolution", self.resolution)
        require(isinstance(self.is_native, bool), "native.is_native must be bool")
        if self.is_native:
            require(self.regridded_from is None, "native grid cannot declare regridded_from")
        else:
            require(isinstance(self.regridded_from, SourceRef),
                    "non-native grid requires regridded_from ")


_SUPPORT_TOPOLOGY = {
    "point": ("none",),
    "line": ("transect",),
    "surface": ("structured", "unstructured"),
    "volume": ("structured",),
}
_SUPPORT_MEASURES = {
    "point": ("none",),
    "line": ("length", "unit"),
    "surface": ("area", "unit"),
    "volume": ("volume", "area", "unit"),
}
_DOMAINS = ("ocean", "land", "air_over_ocean", "air_over_land")


@_spec
class GridSpec(SpecBase):
    """Spatial support, topology, calendar, weights and domain of a daily field.

    ``support`` names the geometry of the data: point T(t), line T(x,t), surface T(x,y,t),
    volume T(x,y,z,t). It is not the detection algorithm.
    """

    _nested: ClassVar[dict[str, type]] = {
        "coords": CoordRoles, "cell_measure": CellMeasure, "vertical": VerticalSpec,
        "native": NativeGrid,
    }

    support: Literal["point", "line", "surface", "volume"]
    topology: Literal["none", "transect", "structured", "unstructured"]
    time_dim: str
    spatial_dims: tuple[str, ...]
    coords: CoordRoles
    calendar: str
    frequency: Literal["1D"]
    time_axes: Literal["single", "forecast"]
    periodic_x: bool
    cell_measure: CellMeasure
    vertical: VerticalSpec | None
    domain: Literal["ocean", "land", "air_over_ocean", "air_over_land"]
    domain_mask: str
    domain_fraction: str | None
    native: NativeGrid

    def __post_init__(self) -> None:
        check_choice("support", self.support, tuple(_SUPPORT_TOPOLOGY))
        check_choice("topology", self.topology, _SUPPORT_TOPOLOGY[self.support])
        if self.topology == "unstructured":
            raise NotSupportedError("unstructured topology is rejected in data contracts ")
        check_nonempty_str("time_dim", self.time_dim)
        require(isinstance(self.spatial_dims, tuple), "spatial_dims must be a tuple")
        for dim in self.spatial_dims:
            check_nonempty_str("spatial_dims entry", dim)
        dims = (self.time_dim, *self.spatial_dims)
        require(len(set(dims)) == len(dims), "time_dim and spatial_dims must be distinct")
        expected = {"point": (0, 1), "line": (1,), "surface": (2,), "volume": (3,)}[self.support]
        require(len(self.spatial_dims) in expected,
                f"support {self.support!r} requires {expected} spatial dims")
        require(isinstance(self.coords, CoordRoles), "coords must be CoordRoles")
        if self.support == "line":
            require(self.coords.along is not None, "line support requires coords.along")
        if self.support in ("surface", "volume"):
            require(self.coords.lat is not None and self.coords.lon is not None,
                    f"{self.support} support requires coords.lat and coords.lon")
        if self.calendar not in SUPPORTED_CALENDARS:
            raise CalendarNotSupportedError(
                f"calendar {self.calendar!r} is not supported in data contracts; supported: "
                f"{SUPPORTED_CALENDARS}. Other calendars need cftime and an "
            )
        check_choice("frequency", self.frequency, ("1D",))
        check_choice("time_axes", self.time_axes, ("single", "forecast"))
        require(isinstance(self.periodic_x, bool), "periodic_x must be declared as bool")
        if self.periodic_x:
            require(self.support in ("surface", "volume") and self.coords.lon is not None,
                    "periodic_x requires a surface/volume support with coords.lon")
        require(isinstance(self.cell_measure, CellMeasure), "cell_measure must be CellMeasure")
        check_choice(f"cell_measure.kind for {self.support}", self.cell_measure.kind,
                     _SUPPORT_MEASURES[self.support])
        if self.support == "volume":
            require(isinstance(self.vertical, VerticalSpec), "volume support requires vertical")
            require(self.coords.depth == self.vertical.coordinate,
                    "coords.depth must name the vertical coordinate")
        else:
            require(self.vertical is None, "vertical is only allowed for volume support")
        check_choice("domain", self.domain, _DOMAINS)
        check_nonempty_str("domain_mask", self.domain_mask)
        check_optional_str("domain_fraction", self.domain_fraction)
        require(isinstance(self.native, NativeGrid), "native must be NativeGrid")

    # -- dataset checks: 1-D coordinates and static variables only ------------------------

    def check(self, ds: xr.Dataset) -> None:
        """Validate `ds` against this grid; never reads (time x space) variables."""
        require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
        for dim in (self.time_dim, *self.spatial_dims):
            require(dim in ds.dims, f"dimension {dim!r} not found in dataset")
        self._check_time(ds)
        for role in ("lat", "lon", "depth", "along"):
            name = getattr(self.coords, role)
            if name is not None:
                require(name in ds.variables, f"coordinate {role}={name!r} not found")
                self._check_static(ds, name, role)
        if self.coords.lat is not None:
            lat = np.asarray(ds[self.coords.lat].values, dtype=float)
            require(bool(np.all(np.abs(lat[np.isfinite(lat)]) <= 90.0)),
                    "latitude outside [-90, 90]")
        if self.periodic_x:
            self._check_periodic(ds)
        if self.coords.along is not None:
            along = np.asarray(ds[self.coords.along].values, dtype=float)
            steps = np.diff(along)
            require(bool(np.all(steps > 0) or np.all(steps < 0)),
                    "along-line coordinate must be strictly monotonic")
        self._check_domain(ds)
        self._check_measure(ds)
        if self.vertical is not None:
            self._check_static(ds, self.vertical.coordinate, "vertical coordinate")
            require(self.spatial_dims[0] in ds[self.vertical.coordinate].dims,
                    "vertical coordinate must vary along the first spatial dim (1-D, or static "
                    "over the spatial dims for sigma grids)")
            self._check_positive(ds, self.vertical.thickness, "vertical thickness")
            require(self.spatial_dims[0] in ds[self.vertical.thickness].dims,
                    "vertical thickness must vary along the vertical dim")
        if self.time_axes == "forecast":
            for name in ("forecast_reference_time", "member"):
                require(name in ds.coords and ds.coords[name].ndim == 0,
                        f"forecast fields need scalar coordinate {name!r} ")

    def _check_time(self, ds: xr.Dataset) -> None:
        require(self.time_dim in ds.coords, f"time dimension {self.time_dim!r} has no coordinate")
        time = ds[self.time_dim]
        values = time.values
        if values.dtype.kind != "M":
            raise CalendarNotSupportedError(
                "time is not datetime64 "
            )
        found = time.encoding.get("calendar", time.attrs.get("calendar"))
        if found is not None and found not in SUPPORTED_CALENDARS:
            raise CalendarNotSupportedError(
                f"dataset calendar {found!r} is not supported "
            )
        if values.size:
            require(bool(values.min().astype("datetime64[D]") >= _GREGORIAN_START),
                    "dates before 1582-10-15 are not supported ")
        steps = np.diff(values)
        if np.any(steps == np.timedelta64(0, "ns")):
            raise ValidationError("duplicate time steps; deduplicate explicitly")
        if np.any(steps < np.timedelta64(0, "ns")):
            raise ValidationError("time is not strictly increasing")
        if np.any(steps != _ONE_DAY):
            raise ValidationError(
                "time is not a regular daily axis (missing or irregular days); reindex explicitly "
                "and mark added days as missing"
            )

    def _check_static(self, ds: xr.Dataset, name: str, what: str) -> None:
        require(self.time_dim not in ds[name].dims, f"{what} {name!r} must be static (no time)")
        require(set(ds[name].dims) <= set(self.spatial_dims),
                f"{what} {name!r} dims {ds[name].dims} not within {self.spatial_dims}")

    def _check_periodic(self, ds: xr.Dataset) -> None:
        lon = ds[self.coords.lon]
        x_dim = self.spatial_dims[-1]
        other = [d for d in lon.dims if d != x_dim]
        values = np.asarray(lon.isel({d: 0 for d in other}).values, dtype=float)
        require(values.ndim == 1 and values.size > 1, "periodic_x needs a longitude along x")
        step = float(np.median(np.abs(np.diff(values))))
        span = float(values.max() - values.min()) + step
        require(abs(span - 360.0) <= 0.5 * step,
                "periodic_x=True but longitude does not cover 360 degrees")

    def _check_domain(self, ds: xr.Dataset) -> None:
        require(self.domain_mask in ds.variables, f"domain mask {self.domain_mask!r} not found")
        self._check_static(ds, self.domain_mask, "domain mask")
        require(set(ds[self.domain_mask].dims) == set(self.spatial_dims),
                "domain mask must span all spatial dims")
        require(ds[self.domain_mask].dtype == bool, "domain mask must be boolean")
        if self.domain_fraction is not None:
            require(self.domain_fraction in ds.variables,
                    f"domain fraction {self.domain_fraction!r} not found")
            self._check_static(ds, self.domain_fraction, "domain fraction")
            frac = np.asarray(ds[self.domain_fraction].values, dtype=float)
            require(bool(np.all((frac >= 0.0) & (frac <= 1.0))),
                    "domain fraction must lie in [0, 1] without NaN")

    def _check_measure(self, ds: xr.Dataset) -> None:
        if self.cell_measure.var is None:
            return
        self._check_positive(ds, self.cell_measure.var, "cell measure")

    def _check_positive(self, ds: xr.Dataset, name: str, what: str) -> None:
        require(name in ds.variables, f"{what} {name!r} not found")
        self._check_static(ds, name, what)
        values = np.asarray(ds[name].values, dtype=float)
        require(bool(np.all(np.isfinite(values) & (values > 0))),
                f"{what} {name!r} must be finite and > 0")


# --- detection definition -----------------------------------------------------------------

_DAY_OFFSET = re.compile(r"offset:[+-](0\d|1[0-4]):[0-5]\d")


def _check_day_definition(value: Any) -> None:
    check_nonempty_str("day_definition", value)
    if value in ("native", "UTC") or _DAY_OFFSET.fullmatch(value):
        return
    if value.startswith("iana:") or value == "solar":
        raise NotSupportedError(f"day_definition {value!r} is reserved, not enabled ")
    raise ValidationError(
        f"day_definition {value!r} must be 'native', 'UTC' or 'offset:+HH:MM' "
    )


def clock_offset(name: str, value: Any) -> np.timedelta64:
    """Offset from UTC of a clock written as 'UTC' or 'offset:+HH:MM' ('native' rejected)."""
    _check_day_definition(value)
    require(value != "native", f"{name} must be 'UTC' or 'offset:+HH:MM', not 'native'")
    if value == "UTC":
        return np.timedelta64(0, "m")
    sign = -1 if value[7] == "-" else 1
    return np.timedelta64(sign * (int(value[8:10]) * 60 + int(value[11:13])), "m")


_DAILY_STATISTICS = ("native", "mean", "max", "min")


@_spec
class DailyComponent(SpecBase):
    """One daily variable of a combined definition.

    ``lag_days`` pairs the component value of day D+lag with the reference day D: each
    component is compared with its own threshold on its own date and the exceedance is
    shifted.
    """

    variable: str
    daily_statistic: Literal["native", "mean", "max", "min"]
    percentile: float
    lag_days: int

    def __post_init__(self) -> None:
        check_nonempty_str("component variable", self.variable)
        check_choice("component daily_statistic", self.daily_statistic, _DAILY_STATISTICS)
        require(is_number(self.percentile) and 0.0 < float(self.percentile) < 100.0,
                "component percentile must be a number in (0, 100)")
        _set(self, "percentile", float(self.percentile))
        require(is_int(self.lag_days) and self.lag_days in (0, 1),
                "lag_days must be 0 or 1 ")


@_spec
class DefinitionSpec(SpecBase):
    """Threshold-run definition: exceedance rule plus persistence and gap joining.

    ``combination="and_daily"`` combines the daily exceedances of several `components` (AND)
    before persistence and gaps are applied; the top-level ``variable`` is then a label of the
    set, ``daily_statistic="native"`` and ``percentile=None``.
    """

    _nested: ClassVar[dict[str, type]] = {"components": DailyComponent}
    _minor_added: ClassVar[dict[str, Any]] = {"components": None}

    profile: str
    phenomenon: Literal["mhw", "mcs", "ahw"]
    variable: str
    combination: Literal["single", "and_daily"]
    daily_statistic: Literal["native", "mean", "max", "min"]
    day_definition: str
    tail: Literal["upper", "lower"]
    comparator: Literal[">", ">=", "<", "<="]
    threshold_kind: Literal["percentile", "absolute"]
    percentile: float | None
    absolute_threshold: tuple[float, str] | None
    min_duration_days: int
    max_gap_days: int
    join_rule: Literal["qualify_then_join", "join_then_qualify"]
    missing_policy: Literal["break", "as_non_exceedance"]
    season_months: tuple[int, ...] | None
    category_scheme: Literal["hobday2018", "schlegel2021"] | None
    components: tuple[DailyComponent, ...] | None = None

    def __post_init__(self) -> None:
        check_nonempty_str("profile", self.profile)
        check_choice("phenomenon", self.phenomenon, ("mhw", "mcs", "ahw"))
        check_nonempty_str("variable", self.variable)
        check_choice("combination", self.combination, ("single", "and_daily"))
        check_choice("daily_statistic", self.daily_statistic, _DAILY_STATISTICS)
        _check_day_definition(self.day_definition)
        if self.phenomenon == "ahw":
            require(self.day_definition != "native", "ahw requires an explicit day_definition")
        check_choice("tail", self.tail, ("upper", "lower"))
        expected_tail = "lower" if self.phenomenon == "mcs" else "upper"
        require(self.tail == expected_tail, f"{self.phenomenon} requires tail={expected_tail!r}")
        allowed = (">", ">=") if self.tail == "upper" else ("<", "<=")
        check_choice(f"comparator for tail {self.tail!r}", self.comparator, allowed)
        check_choice("threshold_kind", self.threshold_kind, ("percentile", "absolute"))
        if self.combination == "and_daily":
            self._check_components()
        elif self.threshold_kind == "percentile":
            require(is_number(self.percentile) and 0.0 < float(self.percentile) < 100.0,
                    "percentile must be a number in (0, 100)")
            require(self.absolute_threshold is None, "percentile excludes absolute_threshold")
            _set(self, "percentile", float(self.percentile))
        else:
            require(self.percentile is None, "absolute threshold excludes percentile")
            value = self.absolute_threshold
            require(isinstance(value, tuple) and len(value) == 2 and is_number(value[0]),
                    "absolute_threshold must be (value, units)")
            temperature_units("absolute_threshold", value[1])
            _set(self, "absolute_threshold", (float(value[0]), value[1]))
        if self.combination == "single":
            require(self.components is None,
                    "components are only allowed with combination='and_daily'")
        require(is_int(self.min_duration_days) and self.min_duration_days >= 1,
                "min_duration_days must be an int >= 1")
        require(is_int(self.max_gap_days) and self.max_gap_days >= 0,
                "max_gap_days must be an int >= 0")
        check_choice("join_rule", self.join_rule, ("qualify_then_join", "join_then_qualify"))
        check_choice("missing_policy", self.missing_policy, ("break", "as_non_exceedance"))
        if self.season_months is not None:
            months = self.season_months
            require(isinstance(months, tuple) and months and all(is_int(m) and 1 <= m <= 12
                                                                  for m in months),
                    "season_months must be a non-empty tuple of months 1..12")
            require(list(months) == sorted(set(months)), "season_months must be sorted, unique")
        if self.category_scheme is not None:
            check_choice("category_scheme", self.category_scheme, ("hobday2018", "schlegel2021"))
            expected = "upper" if self.category_scheme == "hobday2018" else "lower"
            require(self.tail == expected,
                    f"{self.category_scheme} categories require tail={expected!r} "
                    "")

    def _check_components(self) -> None:
        comps = self.components
        require(isinstance(comps, tuple) and len(comps) >= 2
                and all(isinstance(c, DailyComponent) for c in comps),
                "and_daily requires components: a tuple of >= 2 DailyComponent ")
        require(comps[0].lag_days == 0, "the first (reference) component must have lag_days=0")
        require(len({c.variable for c in comps}) == len(comps),
                "component variables must be unique")
        require(self.daily_statistic == "native",
                "and_daily sets daily statistics per component; use daily_statistic='native'")
        require(self.threshold_kind == "percentile" and self.percentile is None
                and self.absolute_threshold is None,
                "and_daily uses per-component percentiles: threshold_kind='percentile', "
                "percentile=None, absolute_threshold=None")
        require(self.category_scheme is None, "and_daily has no category scheme")


# --- baseline -----------------------------------------------------------------------------

PERCENTILE_ESTIMATORS = tuple(f"hyndman_fan_{k}" for k in range(1, 10))
#: Hyndman & Fan (1996) type -> NumPy >= 1.22 `method` name.
NUMPY_PERCENTILE_METHOD = dict(zip(PERCENTILE_ESTIMATORS, (
    "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf",
    "hazen", "weibull", "linear", "median_unbiased", "normal_unbiased",
)))
LEAP_DAY_POLICIES = ("hobday_366_interp", "calendar_doy", "not_applicable")
#: Treatment of the years inside the reference period.
IN_BASE_CORRECTIONS = ("none", "mean_threshold", "leave_one_year_out")
BASELINE_METHODS = ("fixed_climatology", "shifting", "detrended", "stl", "gam", "ssa",
                    "state_space")
#: Dynamic baselines: reference = trend + seasonal, threshold = reference + residual percentile
#:. Each implemented method has a closed parameter registry: name -> (default, check).
DYNAMIC_METHODS = BASELINE_METHODS[1:]


def _int_in(low: int, high: int | None = None):
    return lambda v: is_int(v) and v >= low and (high is None or v <= high)


def _choice_of(*values: str):
    return lambda v: isinstance(v, str) and v in values


def _non_negative(v) -> bool:
    return is_number(v) and v >= 0


def _positive(v) -> bool:
    return is_number(v) and v > 0


def _seasonal_window(v) -> bool:
    return v == "periodic" or (is_int(v) and v >= 7 and v % 2 == 1)


_METHOD_PARAMS: dict[str, dict[str, tuple[Any, Any]]] = {
    # polynomial trend + annual harmonics (or a day-of-year climatology of the detrended series)
    "detrended": {"trend_degree": (1, _int_in(0, 3)),
                  "seasonal": ("harmonic", _choice_of("harmonic", "climatology")),
                  "n_harmonics": (3, _int_in(1, 6)),
                  "min_fit_years": (3, _int_in(1))},
    # day-of-year climatology of a moving window of years
    "shifting": {"window_years": (30, _int_in(2)), "min_years": (10, _int_in(1))},
    # STL (Cleveland et al. 1990) on the 366-position grid; 20-year trend window: keeps >= 90 %
    # of a 1-year event; "periodic" = fixed seasonal cycle, as R's s.window="periodic"
    "stl": {"trend_window_days": (7305, _int_in(3)),
            "seasonal_window": ("periodic", _seasonal_window),
            "robust": (0, _int_in(0, 1)),
            "min_fit_years": (3, _int_in(2)),
            "engine": ("vectorized", _choice_of("vectorized", "statsmodels"))},
    # local linear trend + trigonometric seasonal (Kalman filter/smoother); variances relative to
    # the irregular one (the smoothed and filtered states depend only on these ratios). The slope
    # variance is given as a time scale T in days (ratio = T**-4, the scale of the equivalent
    # cubic smoothing spline): 5600 days (~15 years) keeps >= 90 % of a 1-year event
    "state_space": {"n_harmonics": (3, _int_in(1, 6)),
                    "level_ratio": (0.0, _non_negative),
                    "trend_timescale_days": (5600, _int_in(30)),
                    "seasonal_ratio": (0.0, _non_negative),
                    "min_fit_years": (3, _int_in(1))},
    # penalised B-splines: cubic in time (1.5 per decade keeps >= 90 % of a 1-year event,
    # ) + cyclic over the year; lam as in pygam (default 0.6)
    "gam": {"trend_splines_per_decade": (1.5, _positive),
            "seasonal_splines": (20, _int_in(4)),
            "trend_lam": (0.6, _non_negative),
            "seasonal_lam": (0.6, _non_negative),
            "min_fit_years": (3, _int_in(1)),
            "engine": ("vectorized", _choice_of("vectorized", "pygam"))},
    # SSA with a 15-year window; trend = eigentriples whose dominant period exceeds 30 years,
    # i.e. only the zero frequency of that window: a threshold equal to a frequency of the window
    # would split ties by rounding; keeps >= 90 % of a 1-year event
    "ssa": {"window_days": (5479, _int_in(366)),
            "n_components": (12, _int_in(2, 60)),
            "trend_period_days": (10958, _int_in(366)),
            "n_harmonics": (3, _int_in(1, 6)),
            "min_fit_years": (3, _int_in(2))},
}
_CAUSAL_PARAMS: dict[str, tuple[Any, Any]] = {
    "causal_scheme": ("cutoff", _choice_of("cutoff", "expanding")),
    "lag_days": (0, _int_in(0)),
    "max_extrapolation_days": (30, _int_in(0)),
    "refit_every_days": (30, _int_in(1)),
}


def _param_value(value: Any) -> bool:
    return isinstance(value, str) or is_number(value)


def dynamic_params(spec) -> dict[str, Any]:
    """Effective parameters of an implemented dynamic method: validated against its closed
    registry, with defaults filled in. Checked when fitting, not when building the Spec, so
    that every Spec valid before dynamic baselines stays valid.
    """
    registry = _METHOD_PARAMS.get(spec.method)
    require(registry is not None, f"method {spec.method!r} has no parameter registry yet")
    allowed = dict(registry)
    if spec.fit_mode == "causal":
        allowed.update(_CAUSAL_PARAMS)
    given = dict(spec.params)
    unknown = set(given) - set(allowed)
    require(not unknown, f"unknown params for {spec.method!r} ({spec.fit_mode}): "
            f"{sorted(unknown)}; allowed: {sorted(allowed)}")
    out = {}
    for name, (default, check) in allowed.items():
        value = given.get(name, default)
        require(check(value), f"params[{name!r}]={value!r} is not valid for {spec.method!r}")
        out[name] = value
    if spec.method == "shifting":
        require(out["min_years"] <= out["window_years"], "min_years cannot exceed window_years")
    if spec.method == "stl":
        require(not (out["robust"] and out["engine"] == "vectorized"),
                "robust STL needs engine='statsmodels' (the vectorised engine shares one "
                "LOESS operator among cells, which robustness weights break)")
    if spec.fit_mode == "causal" and out["causal_scheme"] == "expanding":
        require(out["lag_days"] + out["refit_every_days"] <= out["max_extrapolation_days"],
                "expanding needs lag_days + refit_every_days <= max_extrapolation_days")
    return out


@_spec
class BaselineSpec(SpecBase):
    """Reference climatology/threshold method, training period, estimator and availability.

    ``in_base_correction`` gives each year of the reference period its own threshold so that
    it is not compared with a threshold estimated from itself: ``leave_one_year_out`` (the
    other years) or ``mean_threshold`` (mean of the thresholds with the year replaced by each
    other year, Zhang et al. 2005). Years outside the reference period are unchanged.
    """

    _nested: ClassVar[dict[str, type]] = {"source": SourceRef}
    _minor_added: ClassVar[dict[str, Any]] = {"in_base_correction": "none"}

    method: Literal["fixed_climatology", "shifting", "detrended", "stl", "gam", "ssa",
                    "state_space"]
    computed_on: Literal["temperature", "anomaly"]
    reference_period: tuple[str, str]
    source: SourceRef
    doy_window_days: int | None
    smoothing_window_days: int | None
    percentile_estimator: str
    leap_day_policy: Literal["hobday_366_interp", "calendar_doy", "not_applicable"]
    min_samples: int
    fit_mode: Literal["retrospective", "causal"]
    available_until: str | None
    params: tuple[tuple[str, str | int | float], ...] = ()
    in_base_correction: Literal["none", "mean_threshold", "leave_one_year_out"] = "none"

    def __post_init__(self) -> None:
        check_choice("method", self.method, BASELINE_METHODS)
        check_choice("computed_on", self.computed_on, ("temperature", "anomaly"))
        period = self.reference_period
        require(isinstance(period, tuple) and len(period) == 2,
                "reference_period must be (start, end)")
        start = parse_iso_utc("reference_period start", period[0])
        end = parse_iso_utc("reference_period end", period[1])
        require(start <= end, "reference_period start must not be after end")
        require(isinstance(self.source, SourceRef), "source must be SourceRef")
        if self.doy_window_days is not None:
            require(is_int(self.doy_window_days) and self.doy_window_days >= 1
                    and self.doy_window_days % 2 == 1, "doy_window_days must be an odd int >= 1")
        if self.smoothing_window_days is not None:
            require(is_int(self.smoothing_window_days) and self.smoothing_window_days >= 1
                    and self.smoothing_window_days % 2 == 1,
                    "smoothing_window_days must be an odd int >= 1")
        check_choice("percentile_estimator", self.percentile_estimator, PERCENTILE_ESTIMATORS)
        check_choice("leap_day_policy", self.leap_day_policy, LEAP_DAY_POLICIES)
        require((self.leap_day_policy == "not_applicable") == (self.doy_window_days is None),
                "leap_day_policy 'not_applicable' iff doy_window_days is None ")
        require(is_int(self.min_samples) and self.min_samples >= 1, "min_samples must be >= 1")
        check_choice("fit_mode", self.fit_mode, ("retrospective", "causal"))
        if self.fit_mode == "causal":
            require(self.available_until is not None, "causal baselines need available_until")
        if self.available_until is not None:
            until = parse_iso_utc("available_until", self.available_until)
            require(until >= end, "available_until cannot precede the end of reference_period")
        check_pairs("params", self.params, _param_value)
        check_choice("in_base_correction", self.in_base_correction, IN_BASE_CORRECTIONS)
        if self.in_base_correction != "none":
            require(self.method in ("fixed_climatology", "detrended", "shifting", "stl",
                                    "state_space", "gam", "ssa"),
                    "in_base_correction applies to fixed_climatology and implemented dynamic "
                    "baselines ")
            years = range(start.year, end.year + 1)
            require(len(years) >= 2, "in_base_correction needs a reference period of >= 2 years")

    @property
    def reference_years(self) -> tuple[int, ...]:
        """Calendar years that intersect the reference period (labels of the in-base thresholds)."""
        start = parse_iso_utc("reference_period start", self.reference_period[0])
        end = parse_iso_utc("reference_period end", self.reference_period[1])
        return tuple(range(start.year, end.year + 1))

    def available_before(self, instant: np.datetime64) -> bool:
        """True if the baseline uses no information at or after `instant`."""
        limit = self.available_until if self.fit_mode == "causal" else self.reference_period[1]
        cutoff = np.datetime64(parse_iso_utc("limit", limit))
        return bool(cutoff < instant)


# --- coupling ----------------------------------------------------

_COUPLING_PARAMS: dict[str, dict[str, Any]] = {
    "same_support": {},
    "coastal_pairs": {"pairs_table": str, "max_distance_km": float},
    "coupling_grid": {"grid": str, "remap": str, "binarization_threshold": float},
    "column_projection": {"depth_min_m": float, "depth_max_m": float},
}


@_spec
class CouplingSpec(SpecBase):
    """Declarative AHW-MHW/MCS association rule; validates metadata without computing an
    association.
    """

    strategy: Literal["event_first", "mask_first"]
    phenomena: tuple[str, str]
    spatial_rule: Literal["same_support", "coupling_grid", "coastal_pairs", "column_projection"]
    spatial_params: tuple[tuple[str, str | float], ...]
    lag_window_days: tuple[int, int]
    membership_basis: Literal["exceedance", "event_membership"]
    min_phase_days: int
    matching: Literal["many_to_many"]

    def __post_init__(self) -> None:
        check_choice("strategy", self.strategy, ("event_first", "mask_first"))
        require(isinstance(self.phenomena, tuple) and len(self.phenomena) == 2,
                "phenomena must be (atmospheric, marine)")
        check_choice("phenomena[0]", self.phenomena[0], ("ahw",))
        check_choice("phenomena[1]", self.phenomena[1], ("mhw", "mcs"))
        check_choice("spatial_rule", self.spatial_rule, tuple(_COUPLING_PARAMS))
        check_pairs("spatial_params", self.spatial_params, _param_value)
        params = dict(self.spatial_params)
        required = _COUPLING_PARAMS[self.spatial_rule]
        require(set(params) == set(required),
                f"{self.spatial_rule} requires exactly the parameters {sorted(required)}")
        coerced = []
        for key, value in self.spatial_params:
            if required[key] is float:
                require(is_number(value), f"{key} must be a number")
                value = float(value)
            else:
                check_nonempty_str(key, value)
            coerced.append((key, value))
        _set(self, "spatial_params", tuple(coerced))
        params = dict(coerced)
        if self.spatial_rule == "coastal_pairs":
            require(params["max_distance_km"] > 0, "max_distance_km must be > 0")
        if self.spatial_rule == "coupling_grid":
            check_choice("remap", params["remap"], ("fractional_coverage",))
            require(0.0 < params["binarization_threshold"] <= 1.0,
                    "binarization_threshold must be in (0, 1]")
        if self.spatial_rule == "column_projection":
            require(0.0 <= params["depth_min_m"] < params["depth_max_m"],
                    "column_projection needs 0 <= depth_min_m < depth_max_m")
        lag = self.lag_window_days
        require(isinstance(lag, tuple) and len(lag) == 2 and all(is_int(v) for v in lag)
                and lag[0] <= lag[1], "lag_window_days must be (min, max) ints with min <= max")
        check_choice("membership_basis", self.membership_basis, ("exceedance", "event_membership"))
        require(is_int(self.min_phase_days) and self.min_phase_days >= 1,
                "min_phase_days must be an explicit int >= 1")
        check_choice("matching", self.matching, ("many_to_many",))

    def check_catalogs(self, atmospheric, marine) -> None:
        """Check that two EventCatalogs match the declared phenomena (no association computed)."""
        require(atmospheric.phenomenon == self.phenomena[0],
                f"atmospheric catalog is {atmospheric.phenomenon!r}, "
                f"expected {self.phenomena[0]!r}")
        require(marine.phenomenon == self.phenomena[1],
                f"marine catalog is {marine.phenomenon!r}, expected {self.phenomena[1]!r}")
        require(atmospheric.provenance.run_id != marine.provenance.run_id,
                "catalogs of different phenomena must come from different runs")
