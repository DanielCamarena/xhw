# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""NetCDF4 round-trip in tmp_path and serialized-metadata integrity."""

import json

import numpy as np
import pytest
import xarray as xr
from _fixtures import ALL, f1_point_mhw, f4_line, f7_coastal_pair

import xhw

ENGINE = "h5netcdf"


def write_read(ds, path):
    ds.to_netcdf(path, engine=ENGINE)
    with xr.open_dataset(path, engine=ENGINE) as opened:
        return opened.load()


def fixtures():
    items = {name: build() for name, build in ALL.items()}
    items["F4-monitoring"] = f4_line("monitoring")
    pair = f7_coastal_pair()
    items["F7-ahw"], items["F7-mhw"] = pair.ahw, pair.mhw
    return items


@pytest.mark.parametrize("name", [*ALL, "F4-monitoring", "F7-ahw", "F7-mhw"])
def test_netcdf_round_trip_preserves_mask_and_catalog(name, tmp_path):
    fixture = fixtures()[name]
    mask, catalog = fixture.mask, fixture.catalog

    stored = write_read(mask.to_dataset(), tmp_path / "mask.nc")
    again = xhw.XMask.from_dataset(stored)
    assert (again.grid, again.definition, again.baseline, again.provenance) == (
        mask.grid, mask.definition, mask.baseline, mask.provenance)
    assert again.definition_id == mask.definition_id
    assert stored.attrs["Conventions"] == "CF-1.11 xHW-0.1"
    assert stored[mask.grid.time_dim].encoding["calendar"] == mask.grid.calendar
    for var in ("exceedance", "event_membership", "joined_gap", "valid"):
        assert again.dataset[var].dtype == bool
    assert again.dataset["invalid_reason"].dtype == np.uint8
    for var in ("threshold", "reference", "temperature"):
        assert again.dataset[var].attrs["units"] == mask.dataset[var].attrs["units"]
        assert again.dataset[var].attrs["units_metadata"] == "temperature: on_scale"
    for var in mask.dataset.data_vars:
        np.testing.assert_array_equal(again.dataset[var].values, mask.dataset[var].values)
    again.check_invariants()

    rows = write_read(catalog.to_dataset(), tmp_path / "catalog.nc")
    cat = xhw.EventCatalog.from_dataset(rows)
    assert (cat.analysis_mode, cat.window_start, cat.window_end, cat.location_dims) == (
        catalog.analysis_mode, catalog.window_start, catalog.window_end, catalog.location_dims)
    for var in catalog.dataset.data_vars:
        np.testing.assert_array_equal(cat.dataset[var].values, catalog.dataset[var].values)
        assert cat.dataset[var].dtype == catalog.dataset[var].dtype
    cat.check_against(again)


def test_to_dataset_does_not_mutate_the_container():
    fixture = f1_point_mhw()
    fixture.mask.to_dataset()
    fixture.catalog.to_dataset()
    assert fixture.mask.dataset.attrs == {}
    assert "flag_values" not in fixture.mask.dataset["exceedance"].attrs
    assert "calendar" not in fixture.mask.dataset["time"].encoding
    assert "flag_masks" not in fixture.catalog.dataset["censor_reason"].attrs


def tampered(ds, attr, key, value):
    data = json.loads(ds.attrs[attr])
    data["fields"][key] = value
    ds = ds.copy()
    ds.attrs[attr] = json.dumps(data)
    return ds


def test_tampered_spec_is_detected_by_stored_id(tmp_path):
    stored = write_read(f1_point_mhw().mask.to_dataset(), tmp_path / "m.nc")
    with pytest.raises(xhw.SchemaError, match="definition_id"):
        xhw.XMask.from_dataset(tampered(stored, "xhw_definition", "percentile", 95.0))


def test_schema_version_policy_on_datasets():
    ds = f1_point_mhw().mask.to_dataset()
    with pytest.raises(xhw.SchemaError):
        xhw.XMask.from_dataset(ds.assign_attrs(xhw_schema_version="1.0"))
    with pytest.warns(xhw.XHWSchemaWarning):
        xhw.XMask.from_dataset(ds.assign_attrs(xhw_schema_version="0.7"))
    with pytest.raises(xhw.SchemaError):
        xhw.EventCatalog.from_dataset(ds)                       # not a catalog
    with pytest.raises(xhw.SchemaError):
        xhw.XMask.from_dataset(ds.drop_attrs())
