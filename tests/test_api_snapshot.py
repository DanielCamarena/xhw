# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Frozen public API: any change here needs a conscious update and possibly an.

Incompatible ( + 0.MINOR bump): removing/renaming a class, field or method; changing a field's
type or making it required; tightening validation; changing how spec_id is computed.
"""

import dataclasses

import xhw

EXPECTED_ALL = [
    "BaselineResult", "BaselineSpec", "CalendarNotSupportedError", "CellMeasure", "CoordRoles",
    "CouplingSpec", "DailyComponent", "DefinitionSpec", "EventCatalog",
    "Footprint", "GridSpec",
    "InvariantError", "LineageGraph", "LinkPolicy", "MinArea", "NativeGrid",
    "NotSupportedError", "ObjectField",
    "ProcessingStep", "Provenance", "RecipeStep",
    "SchemaError", "SourceRef", "SpatialRecipe", "UnitsError", "ValidationError", "VerticalSpec",
    "XHWDataQualityWarning", "XHWError", "XHWSchemaWarning", "XMask", "__version__",
    "aggregate_daily", "build_catalog", "build_lineage", "build_objects", "compare_objects",
    "detect",
    "fit_baseline", "in_base_rates", "ingest_objects", "interpolate_gaps", "neighbor_fraction",
    "partition_merges",
]  # temporal detection additions: BaselineResult, fit_baseline, XHWDataQualityWarning,
# interpolate_gaps, detect, build_catalog. atmospheric detection additions:
# DailyComponent, aggregate_daily. atmospheric detection: in_base_rates.
# object tracking additions: Footprint, MinArea, RecipeStep, SpatialRecipe,
# ObjectField, build_objects, ingest_objects, compare_objects, neighbor_fraction.
# genealogy additions: LinkPolicy, LineageGraph, build_lineage,
# partition_merges

# (field name, annotation, required)
EXPECTED_FIELDS = {
    "SourceRef": [
        ("dataset_id", "str", True), ("variable", "str", True), ("accessed", "str", True),
        ("product_status", "Literal['final', 'preliminary', 'unknown']", True),
        ("version", "str | None", False), ("query", "str | None", False),
        ("checksum", "str | None", False),
    ],
    "ProcessingStep": [
        ("name", "str", True),
        ("params", "tuple[tuple[str, str | int | float | bool | None], ...]", False),
    ],
    "Provenance": [
        ("run_id", "str", True), ("created_utc", "str", True),
        ("software", "tuple[tuple[str, str], ...]", True),
        ("inputs", "tuple[SourceRef, ...]", False), ("parents", "tuple[str, ...]", False),
        ("steps", "tuple[ProcessingStep, ...]", False),
    ],
    "CoordRoles": [
        ("lat", "str | None", False), ("lon", "str | None", False),
        ("depth", "str | None", False), ("along", "str | None", False),
    ],
    "CellMeasure": [
        ("kind", "Literal['none', 'unit', 'length', 'area', 'volume']", True),
        ("var", "str | None", False), ("units", "str | None", False),
    ],
    "VerticalSpec": [
        ("coordinate", "str", True), ("thickness", "str", True),
        ("thickness_units", "str", True), ("time_dependent", "bool", True),
    ],
    "NativeGrid": [
        ("product", "str", True), ("resolution", "str", True), ("is_native", "bool", True),
        ("regridded_from", "SourceRef | None", False),
    ],
    "GridSpec": [
        ("support", "Literal['point', 'line', 'surface', 'volume']", True),
        ("topology", "Literal['none', 'transect', 'structured', 'unstructured']", True),
        ("time_dim", "str", True), ("spatial_dims", "tuple[str, ...]", True),
        ("coords", "CoordRoles", True), ("calendar", "str", True),
        ("frequency", "Literal['1D']", True), ("time_axes", "Literal['single', 'forecast']", True),
        ("periodic_x", "bool", True), ("cell_measure", "CellMeasure", True),
        ("vertical", "VerticalSpec | None", True),
        ("domain", "Literal['ocean', 'land', 'air_over_ocean', 'air_over_land']", True),
        ("domain_mask", "str", True), ("domain_fraction", "str | None", True),
        ("native", "NativeGrid", True),
    ],
    "DefinitionSpec": [
        ("profile", "str", True), ("phenomenon", "Literal['mhw', 'mcs', 'ahw']", True),
        ("variable", "str", True),
        ("combination", "Literal['single', 'and_daily']", True),  #
        ("daily_statistic", "Literal['native', 'mean', 'max', 'min']", True),
        ("day_definition", "str", True), ("tail", "Literal['upper', 'lower']", True),
        ("comparator", "Literal['>', '>=', '<', '<=']", True),
        ("threshold_kind", "Literal['percentile', 'absolute']", True),
        ("percentile", "float | None", True),
        ("absolute_threshold", "tuple[float, str] | None", True),
        ("min_duration_days", "int", True), ("max_gap_days", "int", True),
        ("join_rule", "Literal['qualify_then_join', 'join_then_qualify']", True),
        ("missing_policy", "Literal['break', 'as_non_exceedance']", True),  #
        ("season_months", "tuple[int, ...] | None", True),
        ("category_scheme", "Literal['hobday2018', 'schlegel2021'] | None", True),  #
        ("components", "tuple[DailyComponent, ...] | None", False),  #
    ],
    "DailyComponent": [
        ("variable", "str", True),
        ("daily_statistic", "Literal['native', 'mean', 'max', 'min']", True),
        ("percentile", "float", True), ("lag_days", "int", True),
    ],
    "BaselineSpec": [
        ("method", "Literal['fixed_climatology', 'shifting', 'detrended', 'stl', 'gam', 'ssa', "
                   "'state_space']", True),  #: values added (compatible)
        ("computed_on", "Literal['temperature', 'anomaly']", True),
        ("reference_period", "tuple[str, str]", True), ("source", "SourceRef", True),
        ("doy_window_days", "int | None", True), ("smoothing_window_days", "int | None", True),
        ("percentile_estimator", "str", True),
        ("leap_day_policy", "Literal['hobday_366_interp', 'calendar_doy', 'not_applicable']",
         True),
        ("min_samples", "int", True),
        ("fit_mode", "Literal['retrospective', 'causal']", True),
        ("available_until", "str | None", True),
        ("params", "tuple[tuple[str, str | int | float], ...]", False),
        ("in_base_correction", "Literal['none', 'mean_threshold', 'leave_one_year_out']",
         False),  #
    ],
    "CouplingSpec": [
        ("strategy", "Literal['event_first', 'mask_first']", True),
        ("phenomena", "tuple[str, str]", True),
        ("spatial_rule",
         "Literal['same_support', 'coupling_grid', 'coastal_pairs', 'column_projection']", True),
        ("spatial_params", "tuple[tuple[str, str | float], ...]", True),
        ("lag_window_days", "tuple[int, int]", True),
        ("membership_basis", "Literal['exceedance', 'event_membership']", True),
        ("min_phase_days", "int", True), ("matching", "Literal['many_to_many']", True),
    ],
    "XMask": [
        ("_dataset", "xr.Dataset", True), ("grid", "GridSpec", True),
        ("definition", "DefinitionSpec", True), ("baseline", "BaselineSpec", True),
        ("provenance", "Provenance", True),
    ],
    "BaselineResult": [
        ("_dataset", "xr.Dataset", True), ("grid", "GridSpec", True),
        ("baseline", "BaselineSpec", True), ("definition", "DefinitionSpec", True),
        ("provenance", "Provenance", True),
    ],
    "EventCatalog": [
        ("_dataset", "xr.Dataset", True), ("definition", "DefinitionSpec", True),
        ("baseline", "BaselineSpec", True), ("provenance", "Provenance", True),
        ("granularity", "str", True), ("analysis_mode", "str", True),
        ("location_dims", "tuple[str, ...]", True), ("source_mask_run_id", "str", True),
        ("window_start", "str", True), ("window_end", "str", True),
    ],
    "Footprint": [
        ("kind", "Literal['connectivity', 'radius', 'offsets']", True),
        ("connectivity", "int | None", False),
        ("radius_km", "float | None", False),
        ("offsets", "tuple[tuple[int, ...], ...] | None", False),
        ("vertical_gap_m", "float | None", False),
    ],
    "MinArea": [
        ("kind", "Literal['none', 'percentile', 'domain_fraction', 'absolute_km', "
                 "'absolute_km2', 'absolute_km3']", True),
        ("value", "float | None", False),
        ("floor_km2", "float | None", False),
    ],
    "RecipeStep": [
        ("op", "Literal['persist', 'open', 'close', 'segment', 'track']", True),
        ("footprint", "Footprint | None", False),
        ("min_area", "MinArea | None", False),
        ("min_duration_days", "int | None", False),
        ("max_gap_days", "int | None", False),
        ("join_rule", "Literal['qualify_then_join', 'join_then_qualify'] | None", False),
        ("min_overlap", "float | None", False),
        # genealogy: track rules, None = object tracking behaviour
        ("link_measure",
         "Literal['overlap_smaller', 'iou', 'source_fraction', 'target_fraction'] | None", False),
        ("dominance_measure", "Literal['overlap_smaller', 'iou', 'source_fraction', "
                              "'target_fraction', 'shared'] | None", False),
        ("tie_rule", "Literal['lowest_id', 'largest_area'] | None", False),
        ("branch_rule", "Literal['inherit', 'terminate_and_create'] | None", False),
    ],
    "SpatialRecipe": [
        ("input_layer", "Literal['exceedance', 'event_membership']", True),
        ("steps", "tuple[RecipeStep, ...]", True),
    ],
    "LinkPolicy": [
        ("measure", "Literal['overlap_smaller', 'iou', 'source_fraction', 'target_fraction']",
         False),
        ("threshold", "float", False), ("max_gap_days", "int", False),
        ("dominance_measure", "Literal['overlap_smaller', 'iou', 'source_fraction', "
                              "'target_fraction', 'shared']", False),
        ("tie_rule", "Literal['lowest_id', 'largest_area']", False),
    ],
    "LineageGraph": [
        ("_dataset", "xr.Dataset", True), ("grid", "GridSpec", True),
        ("policy", "LinkPolicy", True), ("analysis_mode", "str", True),
        ("merge_rule", "str", True), ("provenance", "Provenance", True),
        ("definition_id", "str", True), ("baseline_id", "str", True),
        ("source_objects_run_id", "str", True), ("source_mask_run_id", "str", True),
        ("final_after_days", "int", True), ("window_start", "str", True),
        ("window_end", "str", True),
    ],
    "ObjectField": [
        ("_dataset", "xr.Dataset", True), ("grid", "GridSpec", True),
        ("recipe", "SpatialRecipe | None", True), ("analysis_mode", "str", True),
        ("provenance", "Provenance", True), ("definition_id", "str", True),
        ("baseline_id", "str", True), ("source_mask_run_id", "str", True),
        ("origin", "str", True), ("final_after_days", "int", True),
        ("source_layer", "str", True),
    ],
}

EXPECTED_METHODS = {
    "BaselineResult": {"from_dataset", "to_dataset", "check_invariants", "expand", "dataset",
                       "baseline_id", "definition_id", "doy_mode"},
    "XMask": {"from_dataset", "to_dataset", "check_invariants", "intensity", "dataset",
                    "definition_id", "baseline_id"},
    "EventCatalog": {"from_dataset", "to_dataset", "check_invariants", "check_against",
                     "dataset", "phenomenon", "definition_id", "baseline_id"},
    "ObjectField": {"from_dataset", "to_dataset", "check_invariants", "check_against", "dataset",
                    "has_label", "has_track"},
    "LineageGraph": {"from_dataset", "to_dataset", "check_invariants", "check_against",
                     "dataset", "transitions", "members", "series", "maps", "node_type_names",
                     "to_event_catalog", "check_catalog"},
}


def normalize(annotation: str) -> str:
    return annotation.replace('"', "'")


def test_public_names():
    assert sorted(xhw.__all__) == sorted(EXPECTED_ALL)


def test_dataclass_fields():
    for name, expected in EXPECTED_FIELDS.items():
        cls = getattr(xhw, name)
        actual = [
            (f.name, normalize(f.type),
             f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING)
            for f in dataclasses.fields(cls)
        ]
        assert actual == expected, name


def test_container_methods():
    for name, expected in EXPECTED_METHODS.items():
        public = {m for m in vars(getattr(xhw, name)) if not m.startswith("_")}
        assert expected <= public, name
