# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""XMask: fixtures F1-F9, structural rules, value invariants and immutability."""

import dataclasses

import numpy as np
import pytest
import xarray as xr
from _fixtures import (
    ALL,
    baseline,
    f1_point_mhw,
    f2_point_mcs,
    f5_surface,
    f7_coastal_pair,
    f8_forecast,
    grid,
    hobday,
    provenance,
)

import xhw


def all_masks():
    masks = {name: build().mask for name, build in ALL.items()}
    pair = f7_coastal_pair()
    masks["F7-ahw"], masks["F7-mhw"] = pair.ahw.mask, pair.mhw.mask
    return masks


@pytest.mark.parametrize("name", [*ALL, "F7-ahw", "F7-mhw"])
def test_fixture_masks_satisfy_invariants(name):
    all_masks()[name].check_invariants()


def test_supports_cover_point_line_surface_volume():
    supports = {m.grid.support for m in all_masks().values()}
    assert supports == {"point", "line", "surface", "volume"}


def test_intensity_is_signed_difference():
    heat = f1_point_mhw().mask.intensity()
    cold = f2_point_mcs().mask.intensity()
    assert heat.values[1] == 3.0 and cold.values[1] == -3.0  # VALIDATION: +3 and -3
    assert heat.attrs["units_metadata"] == "temperature: difference"


# --- rebuilding helpers -------------------------------------------------------------------

def rebuild(mask, **changes):
    ds = mask.dataset.copy(deep=True)
    for name, value in changes.items():
        ds[name] = value
    return xhw.XMask.from_dataset(ds, grid=mask.grid, definition=mask.definition,
                                        baseline=mask.baseline, provenance=mask.provenance)


def edited(mask, name, index, value):
    data = np.array(mask.dataset[name].values)
    data[index] = value
    return rebuild(mask, **{name: (mask.dataset[name].dims, data, mask.dataset[name].attrs)})


# --- structural rules (construction, no data read) ---------------------------------------

def test_missing_required_variable():
    ds = f1_point_mhw().mask.dataset.drop_vars("joined_gap")
    mask = f1_point_mhw().mask
    with pytest.raises(xhw.ValidationError, match="joined_gap"):
        xhw.XMask.from_dataset(ds, grid=mask.grid, definition=mask.definition,
                                     baseline=mask.baseline, provenance=mask.provenance)


def test_partial_specs_rejected():
    mask = f1_point_mhw().mask
    with pytest.raises(xhw.ValidationError):
        xhw.XMask.from_dataset(mask.dataset, grid=mask.grid)


@pytest.mark.parametrize("change,error", [
    ({"exceedance": ("time", np.zeros(14))}, xhw.ValidationError),              # not bool
    ({"invalid_reason": ("time", np.zeros(14, np.int32))}, xhw.ValidationError),
    ({"threshold": ("time", np.full(14, 285.0), {"units": "K",
                                                 "units_metadata": "temperature: on_scale"})},
     xhw.ValidationError),                                                       # units differ
    ({"threshold": ("time", np.full(14, 12.0))}, xhw.UnitsError),                # no units
    ({"reference": ("time", np.full(14, 10.0), {"units": "degC"})}, xhw.ValidationError),
    ({"segment": ("time", np.zeros(14, np.int8))}, xhw.ValidationError),         # not forecast
    ({"valid": (("time",), np.ones(14, bool), {"note": None})}, xhw.ValidationError),
    ({"valid": (("time",), np.ones(14, bool), {"flag": True})}, xhw.ValidationError),
])
def test_structural_rules(change, error):
    with pytest.raises(error):
        rebuild(f1_point_mhw().mask, **change)


def test_phenomenon_must_match_domain():
    mask = f1_point_mhw().mask
    with pytest.raises(xhw.ValidationError, match="ocean"):
        xhw.XMask.from_dataset(mask.dataset, grid=grid(domain="land"),
                                     definition=hobday(), baseline=baseline(),
                                     provenance=provenance("x"))


def test_forecast_mask_requires_segment_and_scalar_coords():
    mask = f8_forecast().mask
    with pytest.raises(xhw.ValidationError, match="segment"):
        rebuild_without = mask.dataset.drop_vars("segment")
        xhw.XMask.from_dataset(rebuild_without, grid=mask.grid,
                                     definition=mask.definition, baseline=mask.baseline,
                                     provenance=mask.provenance)
    with pytest.raises(xhw.ValidationError, match="member"):
        xhw.XMask.from_dataset(mask.dataset.drop_vars("member"), grid=mask.grid,
                                     definition=mask.definition, baseline=mask.baseline,
                                     provenance=mask.provenance)


# --- value invariants (explicit check_invariants) ----------------------------------------

@pytest.mark.parametrize("name,index,value,message", [
    ("exceedance", 0, True, "exceedance != temperature"),
    ("valid", 3, False, "valid != "),
    ("invalid_reason", 3, 2, "valid != "),
    ("joined_gap", 6, False, "joined_gap"),
    ("event_membership", 0, True, "joined_gap"),
])
def test_broken_invariants_are_detected_f1(name, index, value, message):
    mask = edited(f1_point_mhw().mask, name, index, value)
    with pytest.raises(xhw.InvariantError, match=message):
        mask.check_invariants()


def test_short_membership_run_is_detected():
    mask = f2_point_mcs().mask
    mem = np.zeros(7, bool)
    mem[1:4] = True
    exc = mem.copy()
    temp = np.where(exc, 7.0, 10.0)
    broken = rebuild(mask, event_membership=("time", mem), exceedance=("time", exc),
                     temperature=("time", temp, mask.dataset["temperature"].attrs))
    with pytest.raises(xhw.InvariantError, match="shorter than min_duration"):
        broken.check_invariants()


def test_gap_longer_than_max_gap_is_detected():
    mask = f1_point_mhw().mask
    exc = np.array(mask.dataset["exceedance"].values)
    exc[5] = False                                  # joined gap becomes 3 days (5, 6, 7)
    gap = np.array(mask.dataset["joined_gap"].values)
    gap[5] = True
    temp = np.array(mask.dataset["temperature"].values)
    temp[5] = 10.0
    broken = rebuild(mask, exceedance=("time", exc), joined_gap=("time", gap),
                     temperature=("time", temp, mask.dataset["temperature"].attrs))
    with pytest.raises(xhw.InvariantError, match="max_gap_days"):
        broken.check_invariants()


def test_land_cell_with_valid_data_is_detected():
    mask = f5_surface().mask
    with pytest.raises(xhw.InvariantError, match="outside_domain"):
        edited(mask, "invalid_reason", (0, 0, 0), 0).check_invariants()


# --- immutability --------------------------------------------------------------

def test_mask_attributes_cannot_be_reassigned():
    mask = f1_point_mhw().mask
    with pytest.raises(dataclasses.FrozenInstanceError):
        mask.grid = grid()


def test_mask_values_are_read_only_through_dataset():
    mask = f1_point_mhw().mask
    with pytest.raises(ValueError):
        mask.dataset["exceedance"][0] = True
    with pytest.raises(ValueError):
        mask.dataset["exceedance"].values[0] = True


def test_replacing_variables_in_dataset_copy_does_not_affect_mask():
    mask = f1_point_mhw().mask
    copy = mask.dataset
    copy["exceedance"] = ("time", np.ones(14, bool))
    assert not mask.dataset["exceedance"].values[0]


def test_copy_true_isolates_caller_arrays_copy_false_shares_them():
    source = f1_point_mhw().mask
    ds = source.dataset.copy(deep=True)
    shared = xhw.XMask.from_dataset(ds, grid=source.grid, definition=source.definition,
                                          baseline=source.baseline, provenance=source.provenance)
    isolated = xhw.XMask.from_dataset(ds, grid=source.grid, definition=source.definition,
                                            baseline=source.baseline,
                                            provenance=source.provenance, copy=True)
    ds["temperature"].values[0] = 99.0              # caller mutates its own array afterwards
    assert shared.dataset["temperature"].values[0] == 99.0   # documented copy=False behaviour
    assert isolated.dataset["temperature"].values[0] == 10.0


def test_masks_compare_by_identity():
    a = f1_point_mhw().mask
    b = xhw.XMask.from_dataset(a.dataset, grid=a.grid, definition=a.definition,
                                     baseline=a.baseline, provenance=a.provenance)
    assert a != b and a == a
    assert a.definition_id == b.definition_id


def test_non_dataset_input_rejected():
    with pytest.raises(xhw.ValidationError):
        xhw.XMask.from_dataset(xr.DataArray([1]), grid=grid(), definition=hobday(),
                                     baseline=baseline(), provenance=provenance("x"))
