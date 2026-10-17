# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""structured objects on every structured grid: lines (x), curvilinear surfaces and static sigma
volumes.

The neighbours are those of the index grid, so these grids only change the weights and the
coordinates; the tests check exactly that (equality with the regular grid, brute-force radius,
volumes and depths of known cells).
"""

import numpy as np
import pytest
from _object_fixtures import (
    make_line_mask,
    make_mask,
    make_volume_mask,
    recipe,
    rotated_coordinates,
    seg,
)

import xhw
from xhw import _morph

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")


def build(mask, *steps, layer="exceedance", **kw):
    obj = xhw.build_objects(mask, recipe=recipe(*steps, layer=layer), **kw)
    obj.check_invariants()
    obj.check_against(mask)
    return obj


# --- lines --------------------------------------------------------------------------------------


def cells(shape, *items):
    out = np.zeros(shape, bool)
    for item in items:
        out[item] = True
    return out


def test_line_patches_follow_the_neighbours_of_the_index_with_length_and_along_centroid():
    exc = cells((3, 9), (0, slice(1, 3)), (0, 5), (1, slice(2, 4)))
    length = np.array([100., 200., 300., 400., 500., 600., 700., 800., 900.])
    along = np.array([0., 100., 300., 600., 1000., 1500., 2100., 2800., 3600.])
    mask = make_line_mask(exc, length=length, along=along)
    obj = build(mask, seg())
    ds = obj.dataset
    assert ds["object_cells"].values.tolist() == [2, 1, 2]
    assert ds["object_area"].attrs["units"] == "m"
    np.testing.assert_allclose(ds["object_area"].values, [500., 600., 700.])
    np.testing.assert_allclose(ds["object_along"].values,
                               [(200 * 100 + 300 * 300) / 500, 1500.,
                                (300 * 300 + 400 * 600) / 700])
    assert ds["label"].dims == ("time", "along") and "object_lat" not in ds


def test_line_connectivity_offsets_and_unsupported_footprints():
    exc = cells((1, 9), (0, 1), (0, 3), (0, 4))
    mask = make_line_mask(exc)
    for rank in (1, 2, 3):
        assert build(mask, seg(rank)).dataset["object_cells"].values.tolist() == [1, 2]
    jump = xhw.Footprint.explicit([(-2,), (-1,), (1,), (2,)])
    obj = build(mask, xhw.RecipeStep.segment(footprint=jump, min_area=xhw.MinArea.none()))
    assert obj.dataset["object_cells"].values.tolist() == [3]       # 1 and 3 joined across the gap
    with pytest.raises(xhw.NotSupportedError, match="radius"):
        xhw.build_objects(mask, recipe=recipe(xhw.RecipeStep.segment(
            footprint=xhw.Footprint.disc(5.0), min_area=xhw.MinArea.none())))
    with pytest.raises(xhw.ValidationError, match=r"\(dx,\)"):
        xhw.build_objects(mask, recipe=recipe(xhw.RecipeStep.segment(
            footprint=xhw.Footprint.explicit([(0, 1), (0, -1)]), min_area=xhw.MinArea.none())))


def test_line_morphology_does_not_invent_cells_against_the_end():
    exc = cells((1, 9), (0, 1), (0, 2), (0, 5), (0, 6))
    closed = build(make_line_mask(exc), xhw.RecipeStep.morph("close", xhw.Footprint.connected(1)),
                   seg())
    # the gap 3-4 lies between two parts and is filled; cells 0 and 7 touch the end of the line
    # or an unset neighbour that is not between two parts, so they stay empty
    assert closed.dataset["processed_mask"].values[0].tolist() == [
        False, True, True, True, True, True, True, False, False]
    exc = cells((1, 9), (0, 0), (0, 1), (0, 4), (0, 5), (0, 6), (0, 8))
    opened = build(make_line_mask(exc), xhw.RecipeStep.morph("open", xhw.Footprint.connected(1)),
                   seg())
    # runs shorter than the three-cell element disappear, except a run that starts at the end
    # of the line (the missing neighbour does not erode it)
    assert opened.dataset["processed_mask"].values[0].tolist() == [
        True, True, False, False, True, True, True, False, False]


def test_line_tracking_default_recipe_and_minimum_length():
    exc = np.zeros((6, 12), bool)
    for t in range(6):
        exc[t, t:t + 4] = True              # four cells drifting one cell a day
    mask = make_line_mask(exc)
    obj = build(mask, seg(), xhw.RecipeStep.track(min_overlap=0.5, min_duration_days=3))
    ds = obj.dataset
    assert ds["track_number"].size == 1 and ds["track_duration_days"].values.tolist() == [6]
    np.testing.assert_allclose(ds["track_area_max"].values, [4000.0])
    kept = build(mask, seg(area=xhw.MinArea(kind="absolute_km", value=4.0)))
    assert kept.dataset["object_cells"].values.tolist() == [4] * 6
    dropped = build(mask, seg(area=xhw.MinArea(kind="absolute_km", value=4.1)))
    assert dropped.dataset["object_cells"].size == 0
    default = xhw.build_objects(mask)
    default.check_invariants()
    assert default.dataset["processed_mask"].values.sum() == 24    # 4 x 6; the domain is 12 cells
    with pytest.raises(xhw.ValidationError, match="m2"):
        xhw.build_objects(mask, recipe=recipe(seg(area=xhw.MinArea(kind="absolute_km2",
                                                                      value=1.0))))


SOURCE = xhw.SourceRef(dataset_id="external:demo", variable="mask", accessed="2026-10-06",
                       product_status="final", version="0.1")


def test_line_objects_round_trip():
    mask = make_line_mask(cells((2, 6), (0, slice(1, 3)), (1, slice(1, 3))))
    obj = build(mask, seg(), xhw.RecipeStep.track(min_duration_days=1))
    again = xhw.ObjectField.from_dataset(obj.to_dataset(), copy=True)
    again.check_invariants()


def test_neighbor_fraction_on_a_line_weights_by_length():
    length = np.array([100., 200., 300., 400., 500.])
    mask = make_line_mask(cells((2, 5), (0, 2)), length=length)
    frac = xhw.neighbor_fraction(mask, footprint=xhw.Footprint.connected(1))
    assert frac.dims == ("time", "along")
    assert frac.values[0, 2] == pytest.approx(300 / (200 + 300 + 400))       # itself and both sides
    assert frac.values[0, 0] == 0.0 and frac.values[0, 1] == pytest.approx(300 / (100 + 200 + 300))
    assert frac.values[0, 4] == 0.0 and not frac.values[1].any()


@pytest.mark.parametrize("kind", ["patches", "tracks"])
def test_ingesting_the_own_labels_of_a_line_reproduces_its_object_tables(kind):
    length = np.array([100., 200., 300., 400., 500., 600., 700.])
    along = np.array([0., 100., 300., 600., 1000., 1500., 2100.])
    exc = cells((3, 7), (0, 1), (0, 2), (0, 5), (1, 2), (1, 3), (2, 3))
    mask = make_line_mask(exc, length=length, along=along)
    own = build(mask, seg(), xhw.RecipeStep.track(min_duration_days=1))
    labels = own.dataset["label" if kind == "patches" else "track_id"]
    got = xhw.ingest_objects(labels, mask=mask, kind=kind, source=SOURCE, method="own")
    got.check_invariants()
    got.check_against(mask)
    for name in ("object_cells", "object_area", "object_along"):
        np.testing.assert_allclose(got.dataset[name].values, own.dataset[name].values,
                                   err_msg=name)
    assert "object_lat" not in got.dataset
    cmp = xhw.compare_objects(own, got)
    assert (cmp["best_iou"] == 1.0).all()


# --- curvilinear grids -------------------------------------------------------------------


def values(obj, name):
    return np.asarray(obj.dataset[name].values)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("steps", [
    (seg(1),),
    (seg(2),),
    (xhw.RecipeStep.segment(footprint=xhw.Footprint.disc(150.0), min_area=xhw.MinArea.none()),),
    (xhw.RecipeStep.morph("close", xhw.Footprint.disc(120.0)), seg(1)),
    (xhw.RecipeStep.morph("open", xhw.Footprint.connected(2)), seg(1),
     xhw.RecipeStep.track(min_overlap=0.3, max_gap_days=1, min_duration_days=2)),
])
def test_a_curvilinear_copy_of_a_regular_grid_gives_identical_objects(steps, periodic):
    rng = np.random.default_rng(11 + periodic)
    exc = rng.random((6, 9, 12)) < 0.35
    domain = rng.random((9, 12)) < 0.85
    regular = make_mask(exc, periodic=periodic, domain=domain)
    curved = make_mask(exc, periodic=periodic, domain=domain, curvilinear=True)
    a, b = build(regular, *steps), build(curved, *steps)
    assert set(a.dataset.data_vars) == set(b.dataset.data_vars)
    for name in a.dataset.data_vars:
        x, y = values(a, name), values(b, name)
        if x.dtype.kind == "f":
            np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-9, err_msg=name)
        else:
            assert (x == y).all(), name


def brute_force_neighbours(lat, lon, y, x, radius_km, periodic):
    ny, nx = lat.shape
    out = set()
    for j in range(ny):
        for i in range(nx):
            if (j, i) == (y, x):
                continue
            if _morph.haversine_km(lat[y, x], lon[y, x], lat[j, i], lon[j, i]) <= radius_km:
                out.add((j, i))
    return out


@pytest.mark.parametrize("radius", [60.0, 130.0, 250.0])
def test_the_radius_footprint_of_a_rotated_grid_is_the_set_of_cells_within_the_distance(radius):
    lat, lon = rotated_coordinates(14, 16, angle_deg=25.0, step=0.75)
    offsets = _morph.build_offsets(xhw.Footprint.disc(radius), lat, lon, False)
    ny, nx = lat.shape
    for y, x in [(0, 0), (6, 7), (13, 15), (3, 12), (10, 2)]:
        found = set()
        for dy, dx, rows in offsets:
            j, i = y + dy, x + dx
            if 0 <= j < ny and 0 <= i < nx and (rows is None or rows[y, x]):
                found.add((j, i))
        assert found == brute_force_neighbours(lat, lon, y, x, radius, False), (y, x)


def test_the_curvilinear_footprint_is_symmetric_so_the_morphology_is_consistent():
    lat, lon = rotated_coordinates(10, 12, angle_deg=40.0, step=0.9)
    rng = np.random.default_rng(5)
    field = rng.random((3, 10, 12)) < 0.4
    valid = np.ones_like(field)
    offsets = _morph.build_offsets(xhw.Footprint.disc(110.0), lat, lon, False)
    opened = _morph.morphology("open", field, valid, offsets, False)
    closed = _morph.morphology("close", field, valid, offsets, False)
    assert (opened <= field).all() and (field <= closed).all()
    assert (_morph.morphology("open", opened, valid, offsets, False) == opened).all()


def test_curvilinear_objects_use_the_two_dimensional_coordinates_and_areas():
    lat, lon = rotated_coordinates(8, 10, angle_deg=30.0, step=1.0)
    exc = np.zeros((2, 8, 10), bool)
    exc[0, 3, 4] = True
    exc[1, 5:7, 2] = True
    mask = make_mask(exc, coords2d=(lat, lon))
    obj = build(mask, seg())
    ds = obj.dataset
    np.testing.assert_allclose(ds["object_lat"].values[0], lat[3, 4], atol=1e-9)
    np.testing.assert_allclose(ds["object_lon"].values[0], lon[3, 4], atol=1e-9)
    assert ds["object_cells"].values.tolist() == [1, 2]
    again = xhw.ObjectField.from_dataset(obj.to_dataset(), copy=True)
    again.check_invariants()


def test_curvilinear_mixed_one_and_two_dimensional_coordinates_are_refused():
    mask = make_mask(np.ones((1, 4, 4), bool), curvilinear=True)
    ds = mask.dataset
    ds = ds.assign_coords(lat=("y", np.arange(4.0)))        # 1-D latitude, 2-D longitude
    mixed = xhw.XMask.from_dataset(ds, grid=mask.grid, definition=mask.definition,
                                         baseline=mask.baseline, provenance=mask.provenance)
    with pytest.raises(xhw.NotSupportedError, match="2-D latitude and longitude"):
        xhw.build_objects(mixed)


# --- static sigma volumes -----------------------------------------------------------------------


def sigma_arrays(nz=3, ny=6, nx=6):
    base = np.array([10.0, 20.0, 30.0])[:nz, None, None]
    thickness = base * (1.0 + 0.2 * np.arange(ny))[None, :, None] * np.ones((nz, ny, nx))
    top = np.cumsum(thickness, axis=0) - thickness / 2          # mid-depth of each cell
    return thickness, top


def test_sigma_volume_uses_the_three_dimensional_thickness_and_depth():
    thickness, depth = sigma_arrays()
    exc = np.zeros((2, 3, 6, 6), bool)
    exc[0, 0:2, 2, 2] = True
    exc[0, 1, 4, 4] = True
    mask = make_volume_mask(exc, sigma=(thickness, depth))
    area = mask.dataset["cell_area"].values
    obj = build(mask, seg())
    ds = obj.dataset
    w0, w1 = area[2, 2] * thickness[0, 2, 2], area[2, 2] * thickness[1, 2, 2]
    np.testing.assert_allclose(ds["object_area"].values[0], w0 + w1)
    np.testing.assert_allclose(ds["object_depth"].values[0],
                               (w0 * depth[0, 2, 2] + w1 * depth[1, 2, 2]) / (w0 + w1))
    assert ds["object_area"].attrs["units"] == "m3"
    default = xhw.build_objects(mask)               # domain fraction over the sigma volume
    default.check_invariants()
    again = xhw.ObjectField.from_dataset(obj.to_dataset(), copy=True)
    again.check_invariants()


def test_sigma_volume_equals_the_z_volume_when_the_thickness_does_not_vary():
    rng = np.random.default_rng(2)
    exc = rng.random((3, 3, 6, 6)) < 0.3
    flat_dz, flat_depth = (10.0, 20.0, 30.0), (5.0, 20.0, 45.0)
    thickness = np.broadcast_to(np.array(flat_dz)[:, None, None], (3, 6, 6)).copy()
    depth = np.broadcast_to(np.array(flat_depth)[:, None, None], (3, 6, 6)).copy()
    a = build(make_volume_mask(exc), seg(2))
    b = build(make_volume_mask(exc, sigma=(thickness, depth)), seg(2))
    for name in ("label", "processed_mask", "object_area", "object_depth", "object_lat"):
        np.testing.assert_allclose(values(a, name), values(b, name), err_msg=name)


def test_a_vertical_coordinate_that_does_not_vary_with_z_is_refused():
    thickness, depth = sigma_arrays()
    mask = make_volume_mask(np.ones((1, 3, 6, 6), bool), sigma=(thickness, depth))
    ds = mask.dataset.copy()
    ds["depth"] = (("y", "x"), np.ones((6, 6)))           # a surface field, not a vertical axis
    with pytest.raises(xhw.ValidationError, match="first spatial dim"):
        mask.grid.check(ds)
