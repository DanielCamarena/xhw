# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""object tracking segmentation: `build_objects`, minimum area, recipe order, bookkeeping and the
container.
"""

import numpy as np
import pytest
import xarray as xr
from _fixtures import baseline, hobday
from _object_fixtures import make_mask, recipe, seg

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
SHAPE = (6, 8, 8)


def build(mask, *steps, layer="exceedance", **kw):
    obj = xhw.build_objects(mask, recipe=recipe(*steps, layer=layer), **kw)
    obj.check_invariants()
    obj.check_against(mask)
    return obj


def values(obj, name):
    return np.asarray(obj.dataset[name].transpose("time", "y", "x").values)


# --- segmentation -------------------------------------------------------------------------------


def test_patches_are_numbered_per_day_in_raster_order_and_tabulated():
    exc = np.zeros(SHAPE, bool)
    exc[0, 0:2, 5:7] = True        # 4 cells, first in raster order
    exc[0, 5, 0] = True            # 1 cell
    exc[1, 3, 3] = True
    obj = build(make_mask(exc, unit_measure=True), seg())
    lab = values(obj, "label")
    assert lab[0, 0, 5] == 1 and lab[0, 5, 0] == 2 and lab[1, 3, 3] == 1 and lab.max() == 2
    ds = obj.dataset
    assert ds["object_cells"].values.tolist() == [4, 1, 1]
    assert ds["object_label"].values.tolist() == [1, 2, 1]
    day = np.datetime64("2000-01-01", "ns")
    assert (ds["object_time"].values == [day, day, day + np.timedelta64(1, "D")]).all()
    assert ds["object_area"].attrs["units"] == "1"          # a unit measure: cell counts
    assert ds["object_area"].values.tolist() == [4.0, 1.0, 1.0]


def test_default_recipe_discards_patches_smaller_than_five_percent_of_the_domain():
    exc = np.zeros(SHAPE, bool)
    exc[0, 0, 0:3] = True          # 3 cells = 4.7 % of 64: dropped
    exc[0, 4, 0:4] = True          # 4 cells = 6.25 %: kept
    mask = make_mask(exc, unit_measure=True)
    obj = xhw.build_objects(mask)
    obj.check_invariants()
    lab = values(obj, "label")
    assert (lab[0, 4, 0:4] == 1).all() and not lab[0, 0].any() and lab.max() == 1
    assert values(obj, "removed")[0, 0, 0:3].all() and not values(obj, "added").any()
    assert obj.recipe == xhw.SpatialRecipe.default()


def test_the_domain_fraction_uses_the_area_of_the_valid_domain_not_the_cell_count():
    exc = np.zeros((1, 6, 6), bool)
    exc[0, 0, 0] = True            # 1 cell
    exc[0, 3, 0:2] = True          # 2 cells in a row: 2 a_row > 5 % of the domain area
    mask = make_mask(exc)
    area = mask.dataset["cell_area"].values
    assert 2 * area[3, 0] > 0.05 * area.sum() > area[0, 0]
    lab = values(xhw.build_objects(mask), "label")
    assert lab[0, 3, 0] == 1 and not lab[0, 0].any()
    land = np.ones((6, 6), bool)
    land[:, 3:] = False            # half of the domain is land: 5 % of 18 cells = 0.9 cells
    half = xhw.build_objects(make_mask(exc, domain=land, unit_measure=True))
    assert values(half, "label")[0, 0, 0] == 1             # the single cell now survives


def test_default_recipe_applies_the_persistence_of_the_definition_after_the_area_cut():
    exc = np.zeros((8, 8, 8), bool)
    exc[0:3, 4:8, 0:4] = True      # 16 cells (25 %) for 3 days: survives
    exc[0:2, 0:3, 0:3] = True      # 9 cells (14 %) for 2 days: too short for min 3 days
    exc[5, 4:8, 4:8] = True        # 16 cells but only 1 day
    mask = make_mask(exc, unit_measure=True, definition=hobday(min_duration_days=3,
                                                              max_gap_days=0))
    obj = xhw.build_objects(mask)
    obj.check_invariants()
    obj.check_against(mask)
    kept = values(obj, "processed_mask")
    assert kept[0:3, 4:8, 0:4].all() and kept.sum() == 48
    assert obj.has_label and values(obj, "label").max() == 1       # patches of the final field


def test_a_minimum_area_that_discards_most_patches_warns():
    exc = np.zeros((1, 20, 20), bool)
    exc[0, 0:10:2, 0:10:4] = True                              # 15 single-cell patches
    exc[0, 14:20, 14:20] = True                                # one of 36 cells (9 %)
    mask = make_mask(exc, unit_measure=True)
    with pytest.warns(xhw.XHWDataQualityWarning, match=r"discards \d+ of \d+ patches"):
        obj = xhw.build_objects(mask, recipe=recipe(seg(area=xhw.MinArea(
            kind="domain_fraction", value=5.0))))
    assert values(obj, "label").max() == 1


@pytest.mark.parametrize("area, kept", [
    (xhw.MinArea(kind="none"), {1, 2, 3, 4}),
    (xhw.MinArea(kind="domain_fraction", value=2.0), {2, 3, 4}),   # 2 % of 100 cells = 2 cells
    (xhw.MinArea(kind="domain_fraction", value=3.0), {3, 4}),      # `>=`: 3 cells survive
    (xhw.MinArea(kind="domain_fraction", value=3.1), {4}),
    (xhw.MinArea(kind="percentile", value=50.0), {3, 4}),          # median of 1..4 = 2.5
    (xhw.MinArea(kind="percentile", value=0.0), {1, 2, 3, 4}),
])
def test_minimum_area_kinds_on_patches_of_one_to_four_cells(area, kept):
    exc = np.zeros((1, 10, 10), bool)
    for i, size in enumerate((1, 2, 3, 4)):
        exc[0, 2 * i, 0:size] = True
    obj = build(make_mask(exc, unit_measure=True), seg(area=area))
    assert set(obj.dataset["object_cells"].values.tolist()) == kept


def test_absolute_area_is_in_square_kilometres_and_needs_an_area_measure():
    exc = np.zeros((1, 6, 6), bool)
    exc[0, 0, 0:2] = True
    exc[0, 3, 0:4] = True
    mask = make_mask(exc)
    one = float(mask.dataset["cell_area"].values[3, 0]) / 1e6        # km2 of a cell
    keep_big = build(mask, seg(area=xhw.MinArea(kind="absolute_km2", value=3.5 * one)))
    assert keep_big.dataset["object_cells"].values.tolist() == [4]
    keep_all = build(mask, seg(area=xhw.MinArea(kind="absolute_km2", value=0.5 * one)))
    assert keep_all.dataset["object_cells"].values.tolist() == [2, 4]
    with pytest.raises(xhw.ValidationError, match="cell measure in m2"):
        xhw.build_objects(make_mask(exc, unit_measure=True),
                          recipe=recipe(seg(area=xhw.MinArea(kind="absolute_km2", value=1.0))))


def test_a_percentile_floor_is_an_absolute_lower_bound():
    exc = np.zeros((1, 6, 6), bool)
    exc[0, 0, 0] = exc[0, 3, 0:2] = True
    mask = make_mask(exc)
    cell = float(mask.dataset["cell_area"].values[3, 0]) / 1e6
    plain = build(mask, seg(area=xhw.MinArea(kind="percentile", value=0.0)))
    floored = build(mask, seg(area=xhw.MinArea(kind="percentile", value=0.0,
                                               floor_km2=1.5 * cell)))
    assert plain.dataset.sizes["object"] == 2
    assert floored.dataset["object_cells"].values.tolist() == [2]


def test_a_cut_larger_than_every_patch_warns_and_leaves_no_patch():
    exc = np.zeros((1, 6, 6), bool)
    exc[0, 0, 0] = True
    everything = xhw.MinArea(kind="domain_fraction", value=100.0)      # the whole domain
    with pytest.warns(xhw.XHWDataQualityWarning, match="no patch survives"):
        obj = xhw.build_objects(make_mask(exc, unit_measure=True),
                                recipe=recipe(seg(area=everything)))
    assert obj.dataset.sizes["object"] == 0 and not values(obj, "processed_mask").any()


def test_a_percentile_in_monitoring_needs_a_frozen_calibration_sample():
    exc = np.zeros((2, 8, 8), bool)
    exc[:, 0, 0:3] = True
    mask = make_mask(exc, unit_measure=True)                  # baseline 1970-1999, window 2000
    percentile = recipe(seg(area=xhw.MinArea(kind="percentile", value=50.0)))
    with pytest.raises(xhw.ValidationError, match="calibration"):
        xhw.build_objects(mask, recipe=percentile, analysis_mode="monitoring")
    # the cut comes from the calibration areas only: 50 % of [2, 4, 6, 8] = 5 cells
    obj = xhw.build_objects(mask, recipe=percentile, analysis_mode="monitoring",
                            calibration=np.array([2.0, 4.0, 6.0, 8.0]))
    assert obj.dataset.sizes["object"] == 0
    obj = xhw.build_objects(mask, recipe=percentile, analysis_mode="monitoring",
                            calibration=np.array([1.0, 2.0, 3.0]))   # median 2 cells: keeps 3
    assert obj.dataset["object_cells"].values.tolist() == [3, 3]


def test_a_calibration_object_field_gives_its_patch_areas_and_units_must_agree():
    exc = np.zeros((2, 8, 8), bool)
    exc[:, 0, 0:3] = True
    mask = make_mask(exc, unit_measure=True)
    reference = xhw.build_objects(mask, recipe=recipe(seg()))
    percentile = recipe(seg(area=xhw.MinArea(kind="percentile", value=90.0)))
    obj = xhw.build_objects(mask, recipe=percentile, analysis_mode="monitoring",
                            calibration=reference)
    assert obj.dataset.sizes["object"] == 2        # cut = 3 cells (all calibration patches are 3)
    other = make_mask(exc)                         # area measure in object tracking
    with pytest.raises(xhw.ValidationError, match="calibration areas are in"):
        xhw.build_objects(other, recipe=percentile, analysis_mode="monitoring",
                          calibration=reference)


# --- bookkeeping, evidence kept apart -----------------------------------------------------------


def test_the_mask_is_never_modified_and_the_bookkeeping_is_exact():
    exc = np.zeros(SHAPE, bool)
    exc[0, 1:4, 1:4] = True
    exc[0, 2, 2] = False                            # a hole
    exc[1, 6, 6] = True                             # an isolated cell
    mask = make_mask(exc)
    before = mask.dataset.copy(deep=True)
    obj = build(mask, xhw.RecipeStep.morph("close", xhw.Footprint.connected(2)),
                seg(area=xhw.MinArea.default()))
    assert mask.dataset.identical(before)           #,
    added, removed, processed = (values(obj, n) for n in ("added", "removed", "processed_mask"))
    assert added[0, 2, 2] and added.sum() == 1      # the closing fills the hole, recorded
    assert removed[1, 6, 6] and removed.sum() == 1  # the isolated cell is under the minimum area
    source = exc & mask.dataset["valid"].transpose("time", "y", "x").values
    assert np.array_equal(processed, (source | added) & ~removed)


def test_cells_that_are_not_valid_never_enter_a_patch_or_a_closing():
    exc = np.zeros(SHAPE, bool)
    exc[0, 3, 0:3] = exc[0, 3, 4:7] = True          # a gap at x = 3...
    domain = np.ones((8, 8), bool)
    domain[3, 3] = False                            #... that is land
    obj = build(make_mask(exc, domain=domain, unit_measure=True),
                xhw.RecipeStep.morph("close", xhw.Footprint.connected(2)),
                seg(area=xhw.MinArea(kind="domain_fraction", value=1.0)))
    lab = values(obj, "label")
    assert lab[0, 3, 3] == 0 and lab[0, 3, 0] != lab[0, 3, 4] and lab.max() == 2
    assert not values(obj, "processed_mask")[0, 3, 3] and not values(obj, "added").any()


def test_input_layer_event_membership_segments_the_event_cells():
    exc = np.zeros(SHAPE, bool)
    exc[0:3, 2, 2] = exc[0:3, 2, 3] = True
    mem = exc.copy()
    mem[3, 2, 2] = mem[3, 2, 3] = True              # joined days: membership without exceedance
    mask = make_mask(exc, mem=mem, unit_measure=True)
    from_exc = build(mask, seg(), layer="exceedance")
    from_mem = build(mask, seg(), layer="event_membership")
    assert not values(from_exc, "label")[3].any() and values(from_mem, "label")[3].any()


def test_recipes_do_not_commute_a_filter_before_the_persistence_is_not_the_same_as_after():
    # cell A exceeds days 0-4 and cell B (its neighbour) days 2-6: A+B form a 2-cell patch on 2-4
    exc = np.zeros((8, 6, 6), bool)
    exc[0:5, 2, 2] = True
    exc[2:7, 2, 3] = True
    mask = make_mask(exc, unit_measure=True)
    two_cells = xhw.MinArea(kind="domain_fraction", value=5.0)     # 5 % of 36 = 1.8 cells
    persist = xhw.RecipeStep.persist(min_duration_days=5, max_gap_days=0,
                                     join_rule="qualify_then_join")
    s_then_t = build(mask, seg(area=two_cells), persist)
    t_then_s = build(mask, persist, seg(area=two_cells))
    assert not values(s_then_t, "processed_mask").any()      # 3 days of coherence only
    kept = values(t_then_s, "processed_mask")
    assert kept[2:5, 2, 2:4].all() and kept.sum() == 6       # A+B on days 2-4, none elsewhere


def test_persist_without_values_inherits_the_definition_of_the_mask():
    exc = np.zeros(SHAPE, bool)
    exc[0:3, 1, 1] = True
    definition = hobday(min_duration_days=3, max_gap_days=0)
    mask = make_mask(exc, definition=definition, unit_measure=True)
    out = build(mask, xhw.RecipeStep.persist(), seg(area=xhw.MinArea.none()))
    assert values(out, "processed_mask")[0:3, 1, 1].all()
    short = make_mask(np.where(np.arange(6)[:, None, None] < 2, exc, False),
                      definition=definition, unit_measure=True)
    out = build(short, xhw.RecipeStep.persist(), seg(area=xhw.MinArea.none()))
    assert not values(out, "processed_mask").any()


# --- geometry -------------------------------------------------------------------------------------


def test_the_centroid_is_an_area_weighted_spherical_mean_without_a_jump_at_the_seam():
    exc = np.zeros((1, 6, 12), bool)
    exc[0, 2, 11] = exc[0, 2, 0] = True              # two cells either side of longitude 0
    mask = make_mask(exc, periodic=True, step=30.0, lat0=-75.0)
    obj = build(mask, seg(rank=1))
    assert obj.dataset["object_cells"].values.tolist() == [2]
    # two cells 30 degrees apart at latitude -15: their spherical mean is the midpoint of the great
    # circle, a little poleward of the parallel (tan(lat) = tan(-15 deg) / cos(15 deg))
    midpoint = np.rad2deg(np.arctan(np.tan(np.deg2rad(-15.0)) / np.cos(np.deg2rad(15.0))))
    assert obj.dataset["object_lon"].values[0] == pytest.approx(345.0, abs=1e-9)  # 330 and 360
    assert obj.dataset["object_lat"].values[0] == pytest.approx(midpoint, abs=1e-9)
    # a patch that is not at the seam: the mean longitude is the plain mean
    exc2 = np.zeros((1, 6, 12), bool)
    exc2[0, 2, 3:5] = True
    obj2 = build(make_mask(exc2, periodic=True, step=30.0, lat0=-75.0), seg())
    assert obj2.dataset["object_lon"].values[0] == pytest.approx(105.0, abs=1e-6)


def test_a_patch_wraps_in_longitude_only_if_the_grid_is_periodic():
    exc = np.zeros((1, 4, 8), bool)
    exc[0, 1, 0] = exc[0, 1, 7] = True
    one = build(make_mask(exc, periodic=True), seg())
    two = build(make_mask(exc, periodic=False), seg())
    assert one.dataset.sizes["object"] == 1 and two.dataset.sizes["object"] == 2


def test_a_patch_never_wraps_in_latitude():
    exc = np.zeros((1, 6, 6), bool)
    exc[0, 0, 2] = exc[0, 5, 2] = True
    assert build(make_mask(exc, periodic=True), seg()).dataset.sizes["object"] == 2


# --- the container --------------------------------------------------------------------------------


def test_round_trip_through_netcdf_keeps_everything(tmp_path):
    exc = np.zeros(SHAPE, bool)
    exc[0:4, 1:3, 1:4] = True
    mask = make_mask(exc, unit_measure=True)
    obj = build(mask, seg(), xhw.RecipeStep.track(min_overlap=0.0, max_gap_days=0,
                                                  min_duration_days=2))
    path = tmp_path / "objects.nc"
    obj.to_dataset().to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf") as opened:
        back = xhw.ObjectField.from_dataset(opened.load())
    back.check_invariants()
    assert back.recipe == obj.recipe and back.grid == obj.grid
    assert back.analysis_mode == obj.analysis_mode and back.origin == "xhw"
    assert back.source_mask_run_id == mask.provenance.run_id
    for name in obj.dataset.data_vars:
        np.testing.assert_array_equal(back.dataset[name].values, obj.dataset[name].values, name)


def test_the_dataset_is_read_only_and_a_tampered_id_is_refused():
    exc = np.zeros(SHAPE, bool)
    exc[0, 1, 1:4] = True
    obj = build(make_mask(exc, unit_measure=True), seg())
    with pytest.raises(ValueError):
        obj._dataset["processed_mask"].values[0, 0, 0] = True
    ds = obj.to_dataset()
    ds.attrs["recipe_id"] = "sha256:" + "0" * 64
    with pytest.raises(xhw.SchemaError, match="recipe_id"):
        xhw.ObjectField.from_dataset(ds)


@pytest.mark.parametrize("tamper, message", [
    (lambda ds: (ds["processed_mask"].values.__setitem__((0, 0, 0), True),
                 ds["valid"].values.__setitem__((0, 0, 0), False)), "outside the valid"),
    (lambda ds: (ds["added"].values.__setitem__((0, 0, 0), True),
                 ds["removed"].values.__setitem__((0, 0, 0), True)), "both added and removed"),
    (lambda ds: ds["object_cells"].values.__setitem__(0, 99), "object_cells"),
    (lambda ds: ds["label"].values.__setitem__((0, 5, 5), 7), "labels of a day"),
])
def test_check_invariants_catches_tampering(tamper, message):
    exc = np.zeros(SHAPE, bool)
    exc[0, 1, 1:4] = True
    ds = build(make_mask(exc, unit_measure=True), seg()).to_dataset().copy(deep=True)
    tamper(ds)
    with pytest.raises(xhw.InvariantError, match=message):
        xhw.ObjectField.from_dataset(ds).check_invariants()


def test_check_against_refuses_a_different_mask():
    exc = np.zeros(SHAPE, bool)
    exc[0, 1, 1:4] = True
    obj = build(make_mask(exc, unit_measure=True), seg())
    other = make_mask(exc, unit_measure=True)       # another run id
    with pytest.raises(xhw.ValidationError, match="run id"):
        obj.check_against(other)


def test_objects_need_a_line_a_surface_or_a_volume():
    from _fixtures import f1_point_mhw

    with pytest.raises(xhw.NotSupportedError, match="line, a surface or a volume"):
        xhw.build_objects(f1_point_mhw().mask)


def test_a_chunked_mask_gives_the_same_objects():
    pytest.importorskip("dask")
    exc = np.zeros(SHAPE, bool)
    exc[0:3, 2:4, 2:5] = True
    mask = make_mask(exc, unit_measure=True)
    chunked = xhw.XMask.from_dataset(mask.dataset.chunk({"time": 2}), grid=mask.grid,
                                           definition=mask.definition, baseline=mask.baseline,
                                           provenance=mask.provenance)
    a, b = xhw.build_objects(mask), xhw.build_objects(chunked)
    for name in a.dataset.data_vars:
        np.testing.assert_array_equal(a.dataset[name].values, b.dataset[name].values, name)


def test_forecast_masks_are_not_supported_yet():
    from _fixtures import f8_forecast

    with pytest.raises(xhw.NotSupportedError):
        xhw.build_objects(f8_forecast().mask)


def test_monitoring_needs_a_baseline_available_before_the_window():
    exc = np.zeros(SHAPE, bool)
    exc[0:3, 1, 1:4] = True
    mask = make_mask(exc, unit_measure=True)
    assert mask.baseline.available_before(np.datetime64("2000-01-01"))
    inside = make_mask(exc, unit_measure=True, base=baseline(reference_period=(
        "1995-01-01", "2005-12-31")))
    with pytest.raises(xhw.ValidationError, match="available before"):
        xhw.build_objects(inside, analysis_mode="monitoring")
