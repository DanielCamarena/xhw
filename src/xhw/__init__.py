# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Thermal extremes, baselines, spatial objects and their genealogy on native grids.

The public API is exactly xhw.__all__; internal modules are private.
"""

from importlib.metadata import version

from ._aggregate import aggregate_daily
from ._baseline import BaselineResult, fit_baseline
from ._catalog import EventCatalog
from ._detect import detect, interpolate_gaps
from ._errors import (
    CalendarNotSupportedError,
    InvariantError,
    NotSupportedError,
    SchemaError,
    UnitsError,
    ValidationError,
    XHWDataQualityWarning,
    XHWError,
    XHWSchemaWarning,
)
from ._events import build_catalog
from ._ingest import compare_objects, ingest_objects, neighbor_fraction
from ._lineage import LineageGraph, LinkPolicy, build_lineage
from ._mask import XMask
from ._objects import ObjectField, build_objects
from ._partition import partition_merges
from ._rates import in_base_rates
from ._recipe import Footprint, MinArea, RecipeStep, SpatialRecipe
from ._specs import (
    BaselineSpec,
    CellMeasure,
    CoordRoles,
    CouplingSpec,
    DailyComponent,
    DefinitionSpec,
    GridSpec,
    NativeGrid,
    ProcessingStep,
    Provenance,
    SourceRef,
    VerticalSpec,
)

__version__ = version("xhw")

__all__ = [
    "BaselineResult",
    "BaselineSpec",
    "CalendarNotSupportedError",
    "CellMeasure",
    "CoordRoles",
    "CouplingSpec",
    "DailyComponent",
    "DefinitionSpec",
    "EventCatalog",
    "Footprint",
    "GridSpec",
    "InvariantError",
    "LineageGraph",
    "LinkPolicy",
    "MinArea",
    "NativeGrid",
    "NotSupportedError",
    "ObjectField",
    "ProcessingStep",
    "Provenance",
    "RecipeStep",
    "SchemaError",
    "SourceRef",
    "SpatialRecipe",
    "UnitsError",
    "ValidationError",
    "VerticalSpec",
    "XHWDataQualityWarning",
    "XHWError",
    "XHWSchemaWarning",
    "XMask",
    "__version__",
    "aggregate_daily",
    "build_catalog",
    "build_lineage",
    "build_objects",
    "compare_objects",
    "detect",
    "fit_baseline",
    "in_base_rates",
    "ingest_objects",
    "interpolate_gaps",
    "neighbor_fraction",
    "partition_merges",
]
