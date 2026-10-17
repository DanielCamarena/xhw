# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Canonical serialization, schema versions, identifiers and shared validators."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import warnings
from datetime import date, datetime, timezone
from typing import Any, ClassVar

import numpy as np

from ._errors import SchemaError, UnitsError, ValidationError, XHWSchemaWarning

#: Schema version of every serialized entity, "MAJOR.MINOR".
SCHEMA_VERSIONS: dict[str, str] = {
    "SourceRef": "0.1",
    "ProcessingStep": "0.1",
    "Provenance": "0.1",
    "CoordRoles": "0.1",
    "CellMeasure": "0.1",
    "VerticalSpec": "0.1",
    "NativeGrid": "0.1",
    "GridSpec": "0.1",
    "DailyComponent": "0.1",
    "DefinitionSpec": "0.1",
    "BaselineSpec": "0.1",
    "CouplingSpec": "0.1",
    "BaselineResult": "0.1",
    "XMask": "0.1",
    "EventCatalog": "0.1",
    "Footprint": "0.1",
    "MinArea": "0.1",
    "RecipeStep": "0.1",
    "SpatialRecipe": "0.1",
    "ObjectField": "0.1",
    "LinkPolicy": "0.1",
    "LineageGraph": "0.1",
}

#: Global `Conventions` attribute written on serialized datasets.
CONVENTIONS = "CF-1.11 xHW-0.1"


def parse_version(version: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", str(version))
    if match is None:
        raise SchemaError(f"schema version {version!r} is not 'MAJOR.MINOR'")
    return int(match.group(1)), int(match.group(2))


def check_schema_version(entity: str, found: str) -> bool:
    """Apply the reading policy; return True when the found MINOR is newer than known."""
    known_major, known_minor = parse_version(SCHEMA_VERSIONS[entity])
    major, minor = parse_version(found)
    if major != known_major:
        raise SchemaError(
            f"{entity} schema {found} is incompatible with supported {SCHEMA_VERSIONS[entity]}; "
            "use a dataset with the supported schema major version"
        )
    if minor > known_minor:
        warnings.warn(
            f"{entity} schema {found} is newer than supported {SCHEMA_VERSIONS[entity]}; "
            "unknown fields are ignored",
            XHWSchemaWarning,
            stacklevel=3,
        )
        return True
    return False


def canonical_json(obj: Any) -> str:
    """Canonical JSON used for storage and hashing (sorted keys, ASCII, no NaN)."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                          allow_nan=False)
    except ValueError as exc:
        raise ValidationError(f"value is not serializable to canonical JSON: {exc}") from exc


def _to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return value._fields_dict()
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _id_jsonable(value: Any) -> Any:
    if isinstance(value, SpecBase):
        return value._id_fields()
    if isinstance(value, tuple):
        return [_id_jsonable(v) for v in value]
    return _to_jsonable(value)


def _to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuples(v) for v in value)
    return value


class SpecBase:
    """Mixin for frozen metadata dataclasses: dict/JSON round-trip and content identifiers."""

    __slots__ = ()
    #: field name -> nested SpecBase type (element type for tuple fields)
    _nested: ClassVar[dict[str, type]] = {}
    #: Designated optional fields whose defaults are omitted from the canonical payload.
    _minor_added: ClassVar[dict[str, Any]] = {}

    @classmethod
    def _entity(cls) -> str:
        return cls.__name__

    def _fields_dict(self) -> dict[str, Any]:
        return {f.name: _to_jsonable(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict with entity type and schema version."""
        return {
            "type": self._entity(),
            "schema_version": SCHEMA_VERSIONS[self._entity()],
            "fields": self._fields_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def _from_fields(cls, data: dict[str, Any], *, lenient: bool = False):
        if not isinstance(data, dict):
            raise SchemaError(f"{cls.__name__} fields must be a mapping")
        fields = dataclasses.fields(cls)
        required = {f.name for f in fields if f.init and f.default is dataclasses.MISSING
                    and f.default_factory is dataclasses.MISSING}
        missing = required - set(data)
        if missing:
            raise SchemaError(f"missing {cls.__name__} fields: {sorted(missing)}")
        known = {f.name for f in fields}
        unknown = set(data) - known
        if unknown and not lenient:
            raise SchemaError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
        kwargs: dict[str, Any] = {}
        for name in known & set(data):
            value = data[name]
            nested = cls._nested.get(name)
            if nested is not None and value is not None:
                if isinstance(value, list):
                    try:
                        value = tuple(nested._from_fields(v, lenient=lenient) for v in value)
                    except SchemaError as exc:
                        raise SchemaError(f"{cls.__name__}.{name}: {exc}") from exc
                else:
                    try:
                        value = nested._from_fields(value, lenient=lenient)
                    except SchemaError as exc:
                        raise SchemaError(f"{cls.__name__}.{name}: {exc}") from exc
            else:
                value = _to_tuples(value)
            kwargs[name] = value
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        """Rebuild from `to_dict` output, applying the schema reading policy."""
        if not isinstance(data, dict) or data.get("type") != cls._entity():
            raise SchemaError(f"expected a serialized {cls._entity()}")
        newer = check_schema_version(cls._entity(), data.get("schema_version", ""))
        return cls._from_fields(data.get("fields"), lenient=newer)

    @classmethod
    def from_json(cls, text: str):
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"invalid {cls._entity()} JSON: {exc}") from exc
        return cls.from_dict(data)

    def _id_fields(self) -> dict[str, Any]:
        """Fields hashed by `spec_id`: designated optional fields are left out while they
        hold their default, here and in every nested Spec. Non-default values enter the
        identifier payload.
        """
        fields = {f.name: _id_jsonable(getattr(self, f.name)) for f in dataclasses.fields(self)}
        for name, default in self._minor_added.items():
            if fields.get(name) == _to_jsonable(default):
                fields.pop(name)
        return fields

    @property
    def spec_id(self) -> str:
        """Content identifier "sha256:<64 hex>" (entity type + schema MAJOR, no MINOR)."""
        fields = self._id_fields()
        major, _ = parse_version(SCHEMA_VERSIONS[self._entity()])
        payload = {"type": self._entity(), "schema_major": major, "fields": fields}
        return "sha256:" + hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()

    @property
    def short_id(self) -> str:
        """First 12 hex characters of `spec_id`, for display only (never as a key)."""
        return self.spec_id.split(":", 1)[1][:12]


# --- shared validators -------------------------------------------------------------------


def require(condition: bool, message: str, error: type[ValidationError] = ValidationError) -> None:
    if not condition:
        raise error(message)


def check_choice(name: str, value: Any, allowed: tuple) -> None:
    if value not in allowed:
        raise ValidationError(f"{name}={value!r} is not one of {allowed}")


def is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def is_number(value: Any) -> bool:
    return is_int(value) or isinstance(value, (float, np.floating))


def check_nonempty_str(name: str, value: Any) -> None:
    require(isinstance(value, str) and value.strip() != "", f"{name} must be a non-empty string")


def check_optional_str(name: str, value: Any) -> None:
    if value is not None:
        check_nonempty_str(name, value)


def parse_iso(name: str, value: Any) -> datetime:
    """Parse an ISO-8601 date or datetime string (a trailing 'Z' means UTC)."""
    check_nonempty_str(name, value)
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            parsed = date.fromisoformat(value)
            return datetime(parsed.year, parsed.month, parsed.day)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{name}={value!r} is not an ISO-8601 date/datetime") from exc


def parse_iso_utc(name: str, value: Any) -> datetime:
    """Like `parse_iso`, returning a naive datetime in UTC (naive input is taken as UTC)."""
    parsed = parse_iso(name, value)
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def check_pairs(name: str, value: Any, value_check) -> None:
    require(isinstance(value, tuple), f"{name} must be a tuple of (key, value) pairs")
    keys = []
    for pair in value:
        require(isinstance(pair, tuple) and len(pair) == 2, f"{name} entries must be (key, value)")
        check_nonempty_str(f"{name} key", pair[0])
        require(value_check(pair[1]), f"{name}[{pair[0]!r}] has an unsupported value {pair[1]!r}")
        keys.append(pair[0])
    require(len(keys) == len(set(keys)), f"{name} keys must be unique")


# --- units (CF strings, no unit library; ) ---------------------------------------

_TEMPERATURE_ALIASES = {
    "degC": "degC", "degree_Celsius": "degC", "degrees_Celsius": "degC", "celsius": "degC",
    "Celsius": "degC", "°C": "degC", "degree_C": "degC", "degrees_C": "degC",
    "K": "K", "kelvin": "K", "Kelvin": "K",
}
_MEASURE_UNITS = {
    "length": {"m": "m"},
    "area": {"m2": "m2", "m^2": "m2", "m**2": "m2"},
    "volume": {"m3": "m3", "m^3": "m3", "m**3": "m3"},
}


def temperature_units(name: str, units: Any) -> str:
    """Return the canonical temperature unit ('degC' or 'K') or raise UnitsError."""
    if not isinstance(units, str) or units not in _TEMPERATURE_ALIASES:
        raise UnitsError(f"{name} has unknown or missing temperature units {units!r}")
    return _TEMPERATURE_ALIASES[units]


def measure_units(kind: str, units: Any) -> str:
    table = _MEASURE_UNITS[kind]
    if not isinstance(units, str) or units not in table:
        raise UnitsError(f"{kind} measure units {units!r} not in {sorted(table)}")
    return table[units]


# --- attribute serializability ------------------------------------------------


def check_serializable_attrs(owner: str, attrs: dict) -> None:
    """Reject attribute values that NetCDF cannot store (None, bool, dict, objects)."""
    for key, value in attrs.items():
        ok = isinstance(value, str) or is_number(value)
        if isinstance(value, (list, tuple)):
            ok = all(is_number(v) for v in value) or all(isinstance(v, str) for v in value)
        if isinstance(value, np.ndarray):
            ok = value.ndim <= 1 and value.dtype.kind in "iuf"
        if not ok:
            raise ValidationError(
                f"attribute {key!r} of {owner} has non-serializable value {value!r}"
            )
