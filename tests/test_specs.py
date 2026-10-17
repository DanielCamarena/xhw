# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Spec contracts: dict/JSON round-trip, identifiers, schema policy and validation rules."""

import dataclasses
import re

import pytest
from _fixtures import SOURCE, baseline, f7_coastal_pair, grid, hobday, provenance, surface_grid

import xhw

SPECS = [
    hobday(), baseline(), grid(), surface_grid(), f7_coastal_pair().coupling, provenance("run"),
    SOURCE,
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: type(s).__name__)
def test_dict_and_json_round_trip_preserve_spec_and_id(spec):
    cls = type(spec)
    again = cls.from_dict(spec.to_dict())
    assert again == spec
    assert again.spec_id == spec.spec_id
    assert cls.from_json(spec.to_json()) == spec


def test_spec_id_format_and_short_id():
    spec = hobday()
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", spec.spec_id)
    assert spec.short_id == spec.spec_id[7:19]


def test_spec_id_depends_on_content_not_on_int_float_spelling():
    assert hobday(percentile=90).spec_id == hobday(percentile=90.0).spec_id
    assert hobday(percentile=90.0).spec_id != hobday(percentile=95.0).spec_id


def test_spec_id_includes_entity_type():
    a = xhw.CoordRoles()
    b = xhw.CellMeasure(kind="none")
    assert a.spec_id != b.spec_id


def test_specs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        hobday().percentile = 95.0


def test_schema_policy_major_rejected_minor_warns_unknown_field_rejected():
    data = hobday().to_dict()
    with pytest.raises(xhw.SchemaError):
        xhw.DefinitionSpec.from_dict({**data, "schema_version": "1.0"})
    with pytest.raises(xhw.SchemaError):
        xhw.DefinitionSpec.from_dict({**data, "fields": {**data["fields"], "new_field": 1}})
    newer = {**data, "schema_version": "0.9", "fields": {**data["fields"], "new_field": 1}}
    with pytest.warns(xhw.XHWSchemaWarning):
        assert xhw.DefinitionSpec.from_dict(newer) == hobday()
    with pytest.raises(xhw.SchemaError):
        xhw.BaselineSpec.from_dict(data)


def test_provenance_new_records_software_and_unique_run():
    a, b = xhw.Provenance.new(inputs=(SOURCE,)), xhw.Provenance.new()
    assert a.run_id != b.run_id
    assert {"numpy", "python", "xarray", "xhw"} == {name for name, _ in a.software}


# --- one negative test per validation rule ------------------------------------------------

DEFINITION_ERRORS = [
    ({"comparator": "<"}, xhw.ValidationError),                      # tail/comparator
    ({"phenomenon": "mcs", "category_scheme": None}, xhw.ValidationError),  # mcs needs lower
    ({"phenomenon": "mcs", "tail": "lower", "comparator": "<"}, xhw.ValidationError),  # cold cat.
    ({"percentile": 100.0}, xhw.ValidationError),
    ({"absolute_threshold": (25.0, "degC")}, xhw.ValidationError),   # percentile + absolute
    ({"threshold_kind": "absolute", "percentile": None, "absolute_threshold": (25.0, "degF")},
     xhw.UnitsError),
    ({"min_duration_days": 0}, xhw.ValidationError),
    ({"min_duration_days": True}, xhw.ValidationError),
    ({"max_gap_days": -1}, xhw.ValidationError),
    ({"day_definition": "local:America/Lima"}, xhw.ValidationError),
    ({"day_definition": "iana:America/Lima"}, xhw.NotSupportedError),
    ({"day_definition": "solar"}, xhw.NotSupportedError),
    ({"phenomenon": "ahw", "variable": "air_temperature"}, xhw.ValidationError),  # ahw native
    ({"combination": "and_daily"}, xhw.ValidationError),        # needs components
    ({"season_months": (12, 1, 2)}, xhw.ValidationError),
    ({"missing_policy": "bridge"}, xhw.ValidationError),
    ({"join_rule": "whatever"}, xhw.ValidationError),
]


@pytest.mark.parametrize("overrides,error", DEFINITION_ERRORS)
def test_definition_rules(overrides, error):
    with pytest.raises(error):
        hobday(**overrides)


def test_absolute_threshold_is_accepted_with_temperature_units():
    spec = hobday(threshold_kind="absolute", percentile=None, absolute_threshold=(25, "K"))
    assert spec.absolute_threshold == (25.0, "K")


BASELINE_ERRORS = [
    {"doy_window_days": 10},
    {"doy_window_days": None},                                   # needs not_applicable
    {"leap_day_policy": "not_applicable"},                       # needs doy_window None
    {"leap_day_policy": "drop_feb29"},                           # removed
    {"percentile_estimator": "linear"},                          # names are hyndman_fan_k
    {"fit_mode": "causal"},                                      # needs available_until
    {"available_until": "1990-01-01"},                           # before end of period
    {"reference_period": ("2000-01-01", "1990-01-01")},
    {"smoothing_window_days": 30},
    {"params": (("a", 1), ("a", 2))},
]


@pytest.mark.parametrize("overrides", BASELINE_ERRORS)
def test_baseline_rules(overrides):
    with pytest.raises(xhw.ValidationError):
        baseline(**overrides)


def test_constant_threshold_baseline_uses_not_applicable():
    spec = baseline(doy_window_days=None, leap_day_policy="not_applicable")
    assert spec.leap_day_policy == "not_applicable"


GRID_ERRORS = [
    ({"calendar": "noleap"}, xhw.CalendarNotSupportedError),
    ({"calendar": "360_day"}, xhw.CalendarNotSupportedError),
    ({"support": "surface", "topology": "unstructured", "spatial_dims": ("cell",)},
     xhw.NotSupportedError),
    ({"periodic_x": None}, xhw.ValidationError),                 # periodicity must be declared
    ({"periodic_x": True}, xhw.ValidationError),                 # point cannot be periodic
    ({"support": "volume", "topology": "structured", "spatial_dims": ("z", "y", "x"),
      "cell_measure": xhw.CellMeasure(kind="unit")}, xhw.ValidationError),  # no vertical
    ({"support": "line", "topology": "transect", "spatial_dims": ("along",),
      "coords": xhw.CoordRoles(along="along"),
      "cell_measure": xhw.CellMeasure(kind="area", var="a", units="m2")}, xhw.ValidationError),
    ({"native": None}, xhw.ValidationError),
    ({"domain": "sea"}, xhw.ValidationError),
    ({"frequency": "1H"}, xhw.ValidationError),
    ({"spatial_dims": ("time",)}, xhw.ValidationError),
]


@pytest.mark.parametrize("overrides,error", GRID_ERRORS)
def test_grid_rules(overrides, error):
    with pytest.raises(error):
        grid(**overrides)


def test_nested_grid_rules():
    with pytest.raises(xhw.ValidationError):
        xhw.CellMeasure(kind="unit", var="cell_area")            # unit takes no variable
    with pytest.raises(xhw.UnitsError):
        xhw.CellMeasure(kind="area", var="cell_area", units="km2")
    with pytest.raises(xhw.NotSupportedError):
        xhw.VerticalSpec(coordinate="s_rho", thickness="dz", thickness_units="m",
                         time_dependent=True)
    with pytest.raises(xhw.ValidationError):
        xhw.NativeGrid(product="ERA5 on OSTIA grid", resolution="0.05", is_native=False)
    with pytest.raises(xhw.ValidationError):
        xhw.SourceRef(dataset_id="x", variable="sst", accessed="yesterday",
                      product_status="final")
    with pytest.raises(xhw.ValidationError):
        xhw.SourceRef(dataset_id="x", variable="sst", accessed="2026-01-01",
                      product_status="draft")


def coupling(**overrides):
    params = dict(strategy="event_first", phenomena=("ahw", "mhw"), spatial_rule="coastal_pairs",
                  spatial_params=(("pairs_table", "t"), ("max_distance_km", 50.0)),
                  lag_window_days=(-5, 5), membership_basis="event_membership",
                  min_phase_days=1, matching="many_to_many")
    params.update(overrides)
    return xhw.CouplingSpec(**params)


COUPLING_ERRORS = [
    {"spatial_params": (("pairs_table", "t"),)},                             # missing distance
    {"spatial_params": (("pairs_table", "t"), ("max_distance_km", 5.0), ("x", 1.0))},
    {"spatial_params": (("pairs_table", "t"), ("max_distance_km", 0.0))},
    {"spatial_rule": "coupling_grid",
     "spatial_params": (("grid", "g"), ("remap", "fractional_coverage"),
                        ("binarization_threshold", 0.0))},
    {"spatial_rule": "column_projection",
     "spatial_params": (("depth_min_m", 50.0), ("depth_max_m", 10.0))},
    {"lag_window_days": (5, -5)},
    {"min_phase_days": 0},
    {"phenomena": ("mhw", "ahw")},
    {"matching": "one_to_one"},
    {"membership_basis": "both"},
]


@pytest.mark.parametrize("overrides", COUPLING_ERRORS)
def test_coupling_rules(overrides):
    with pytest.raises(xhw.ValidationError):
        coupling(**overrides)


def test_coupling_normalizes_numeric_parameters():
    spec = coupling(spatial_params=(("pairs_table", "t"), ("max_distance_km", 50)))
    assert dict(spec.spatial_params)["max_distance_km"] == 50.0
    assert spec.spec_id == coupling().spec_id


def test_initial_entity_schema_versions():
    from xhw._schema import SCHEMA_VERSIONS

    assert set(SCHEMA_VERSIONS.values()) == {"0.1"}
    for spec in SPECS:
        assert spec.to_dict()["schema_version"] == "0.1"
