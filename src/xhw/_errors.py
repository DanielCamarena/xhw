# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Exception and warning types raised by xHW contracts."""


class XHWError(Exception):
    """Base class for all xHW errors."""


class ValidationError(XHWError, ValueError):
    """A contract, specification or dataset violates an xHW rule."""


class CalendarNotSupportedError(ValidationError):
    """The calendar is valid CF but not supported yet."""


class UnitsError(ValidationError):
    """Units are missing, unknown or incompatible."""


class NotSupportedError(ValidationError):
    """The value is representable in the contract but rejected in this version."""


class SchemaError(ValidationError):
    """Serialized metadata is missing, inconsistent or from an unsupported schema version."""


class InvariantError(ValidationError):
    """Value-level invariants of a container do not hold."""


class XHWSchemaWarning(UserWarning):
    """Serialized metadata comes from a newer compatible schema version."""


class XHWDataQualityWarning(UserWarning):
    """Configuration or data fall below recommended sampling for a reliable estimate."""
