# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""object tracking on volumes (z, y, x): 3-D connectivity and morphology, cell volumes, depth,
tracking.
"""

import numpy as np
import pytest
from _fixtures import hobday
from _object_fixtures import make_mask, make_volume_mask, recipe, seg

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
SHAPE = (4, 3, 6, 6)       # time, z, y, x
DZ = (10.0, 20.0, 30.0)
DEPTH = (5.0, 20.0, 45.0)


def build(mask, *steps, layer="exceedance", **kw):
    obj = xhw.build_objects(mask, recipe=recipe(*steps, layer=layer), **kw)
    obj.check_invariants()
    obj.check_against(mask)
    return obj


def values(obj, name):
    return np.asarray(obj.dataset[name].transpose("time", "z", "y", "x").values)


def cells(*items, shape=SHAPE):
    out = np.zeros(shape, bool)
    for item in items:
        out[item] = True
    return out


@pytest.mark.parametrize("second, expected", [
    ((0, 1, 2, 2), {1: 1, 2: 1, 3: 1}),        # a face (dz = 1): joined at every rank
    ((0, 1, 3, 2), {1: 2, 2: 1, 3: 1}),        # an edge (dz = 1, dy = 1): 18 or 26 neighbours
    ((0, 1, 3, 3), {1: 2, 2: 2, 3: 1}),        # a corner (1, 1, 1): only the 26 neighbours
])
def test_vertical_neighbours_follow_the_6_18_26_connectivity(second, expected):
    exc = cells((0, 0, 2, 2), second)
    mask = make_volume_mask(exc, dz=DZ, depth=DEPTH)
    for rank, n_patches in expected.items():
        obj = build(mask, seg(rank))
        assert obj.dataset["object_cells"].size == n_patches, rank


def test_explicit_offsets_label_like_the_rank_connectivity_also_with_periodic_longitude():
    rng = np.random.default_rng(3)
    exc = rng.random(SHAPE) < 0.3
    for periodic in (False, True):
        mask = make_volume_mask(exc, periodic=periodic)
        full = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dz, dy, dx) != (0, 0, 0)]
        by_rank = values(build(mask, seg(3)), "label")
        by_offsets = values(build(mask, xhw.RecipeStep.segment(
            footprint=xhw.Footprint.explicit(full), min_area=xhw.MinArea.none())), "label")
        assert (by_rank == by_offsets).all()
        faces = [(dz, dy, dx) for dz, dy, dx in full if abs(dz) + abs(dy) + abs(dx) == 1]
        assert (values(build(mask, seg(1)), "label")
                == values(build(mask, xhw.RecipeStep.segment(
                    footprint=xhw.Footprint.explicit(faces),
                    min_area=xhw.MinArea.none())), "label")).all()


def test_longitude_wraps_in_a_periodic_volume():
    exc = cells((0, 1, 2, 0), (0, 1, 2, 5))
    plain = build(make_volume_mask(exc), seg(1))
    wrapped = build(make_volume_mask(exc, periodic=True), seg(1))
    assert plain.dataset["object_cells"].size == 2
    assert wrapped.dataset["object_cells"].size == 1


def test_the_object_measure_is_the_volume_and_the_depth_is_volume_weighted():
    exc = cells((0, 0, 2, 2), (0, 1, 2, 2))
    mask = make_volume_mask(exc, dz=DZ, depth=DEPTH)
    area = float(mask.dataset["cell_area"].values[2, 2])
    obj = build(mask, seg())
    ds = obj.dataset
    assert ds["object_area"].attrs["units"] == "m3"
    np.testing.assert_allclose(ds["object_area"].values, [area * (10.0 + 20.0)])
    np.testing.assert_allclose(ds["object_depth"].values, [(5.0 * 10 + 20.0 * 20) / 30.0])
    np.testing.assert_allclose(ds["object_lat"].values, [-8.0], atol=1e-9)
    unit = build(make_volume_mask(exc, unit_measure=True), seg())
    assert unit.dataset["object_area"].attrs["units"] == "1"
    assert unit.dataset["object_area"].values.tolist() == [2.0]


def test_the_default_recipe_uses_five_percent_of_the_volume_of_the_domain_and_persistence():
    exc = np.zeros(SHAPE, bool)
    exc[0:3, 0:3, 0:3, 0:3] = True          # 27 cells for 3 days
    exc[0:2, 0, 5, 5] = True                # one cell, two days: too small and too short
    mask = make_volume_mask(exc, unit_measure=True,
                            definition=hobday(min_duration_days=3, max_gap_days=0))
    obj = xhw.build_objects(mask)
    obj.check_invariants()
    kept = values(obj, "processed_mask")
    assert kept[0:3, 0:3, 0:3, 0:3].all() and kept.sum() == 81
    assert obj.has_label and values(obj, "label").max() == 1


def test_domain_fraction_counts_only_the_volume_of_the_domain_mask():
    exc = cells((0, 0, 0, 0), (0, 0, 0, 1))                  # 2 cells
    domain = np.zeros((3, 6, 6), bool)
    domain[0, 0:2, 0:5] = True                                # 10 cells: 2 cells = 20 %
    mask = make_volume_mask(exc, unit_measure=True, domain=domain)
    kept = build(mask, seg(area=xhw.MinArea(kind="domain_fraction", value=20.0)))
    assert kept.dataset["object_cells"].values.tolist() == [2]
    dropped = build(mask, seg(area=xhw.MinArea(kind="domain_fraction", value=21.0)))
    assert "object_cells" in dropped.dataset and dropped.dataset["object_cells"].size == 0


def test_absolute_minimum_volume_is_in_cubic_kilometres():
    exc = cells((0, 0, 2, 2), (0, 1, 2, 2), (0, 0, 5, 5))
    mask = make_volume_mask(exc, dz=DZ, depth=DEPTH)
    area = float(mask.dataset["cell_area"].values[2, 2])
    big = area * (10.0 + 20.0) / 1e9
    kept = build(mask, seg(area=xhw.MinArea(kind="absolute_km3", value=big * 0.99)))
    assert kept.dataset["object_cells"].values.tolist() == [2]
    with pytest.raises(xhw.ValidationError, match="m2"):
        xhw.build_objects(mask, recipe=recipe(seg(area=xhw.MinArea(kind="absolute_km2",
                                                                      value=1.0))))
    surface = make_mask(np.ones((1, 4, 4), bool), unit_measure=True)
    with pytest.raises(xhw.ValidationError, match="m3"):
        xhw.build_objects(surface, recipe=recipe(seg(area=xhw.MinArea(kind="absolute_km3",
                                                                         value=1.0))))


def test_a_block_that_drifts_through_the_volume_is_one_track():
    exc = np.zeros(SHAPE, bool)
    for t in range(4):
        exc[t, 0:2, 2:4, t:t + 2] = True
    mask = make_volume_mask(exc, dz=DZ, depth=DEPTH)
    obj = build(mask, seg(1), xhw.RecipeStep.track(min_overlap=0.5, min_duration_days=2))
    ds = obj.dataset
    assert ds["track_number"].size == 1 and ds["track_duration_days"].values.tolist() == [4]
    assert ds["track_id"].dims == ("time", "z", "y", "x")
    rows = mask.dataset["cell_area"].values[2:4, 0].sum()           # the block spans rows 2 and 3
    np.testing.assert_allclose(ds["track_area_max"].values, [rows * 2 * (10.0 + 20.0)], rtol=1e-6)
    # the same block one layer lower and not overlapping vertically is another object
    exc2 = np.zeros(SHAPE, bool)
    exc2[0, 0, 2:4, 2:4] = True
    exc2[1, 2, 2:4, 2:4] = True
    apart = build(make_volume_mask(exc2), seg(1), xhw.RecipeStep.track(min_duration_days=1))
    assert apart.dataset["track_number"].size == 2


def test_morphology_in_three_dimensions():
    cross = [(1, 3, 3), (0, 3, 3), (2, 3, 3), (1, 2, 3), (1, 4, 3), (1, 3, 2), (1, 3, 4)]
    exc = cells(*[(0, *c) for c in cross], (0, 1, 0, 0))
    mask = make_volume_mask(exc)
    opened = build(mask, xhw.RecipeStep.morph("open", xhw.Footprint.connected(1)), seg())
    kept = values(opened, "processed_mask")[0]
    assert kept.sum() == 7 and not kept[1, 0, 0]          # the isolated cell does not survive
    gap = cells((0, 0, 3, 3), (0, 2, 3, 3))                # two layers with an empty one between
    for rank, filled in ((1, False), (3, True)):
        closed = build(make_volume_mask(gap),
                       xhw.RecipeStep.morph("close", xhw.Footprint.connected(rank)), seg())
        assert bool(values(closed, "processed_mask")[0, 1, 3, 3]) is filled


def test_a_volume_object_field_round_trips_and_checks_its_invariants():
    exc = cells((0, 0, 2, 2), (0, 1, 2, 2), (1, 0, 2, 2), (1, 1, 2, 3))
    mask = make_volume_mask(exc)
    obj = build(mask, seg(), xhw.RecipeStep.track(min_duration_days=1))
    again = xhw.ObjectField.from_dataset(obj.to_dataset(), copy=True)
    again.check_invariants()
    assert again.dataset["object_depth"].identical(obj.dataset["object_depth"])


def test_footprints_that_do_not_fit_the_support_are_refused():
    mask = make_volume_mask(cells((0, 0, 2, 2)))
    with pytest.raises(xhw.NotSupportedError, match="radius"):
        xhw.build_objects(mask, recipe=recipe(xhw.RecipeStep.segment(
            footprint=xhw.Footprint.disc(100.0), min_area=xhw.MinArea.none())))
    with pytest.raises(xhw.ValidationError, match="dz"):
        xhw.build_objects(mask, recipe=recipe(xhw.RecipeStep.segment(
            footprint=xhw.Footprint.explicit([(0, 1), (0, -1)]), min_area=xhw.MinArea.none())))
    surface = make_mask(np.ones((1, 4, 4), bool), unit_measure=True)
    with pytest.raises(xhw.ValidationError, match="dy, dx"):
        xhw.build_objects(surface, recipe=recipe(xhw.RecipeStep.segment(
            footprint=xhw.Footprint.explicit([(0, 0, 1), (0, 0, -1)]),
            min_area=xhw.MinArea.none())))


def test_the_offsets_of_a_footprint_must_be_symmetric_and_of_one_length():
    with pytest.raises(xhw.ValidationError, match="negation"):
        xhw.Footprint.explicit([(0, 0, 1)])
    with pytest.raises(xhw.ValidationError, match="same length"):
        xhw.Footprint.explicit([(0, 1), (0, -1), (0, 0, 1), (0, 0, -1)])
    line = xhw.Footprint.explicit([(0, 0, 1), (0, 0, -1)])
    assert xhw.Footprint.from_json(line.to_json()).offsets == ((0, 0, 1), (0, 0, -1))


SOURCE = xhw.SourceRef(dataset_id="external:demo", variable="mask", accessed="2026-10-06",
                       product_status="final", version="0.1")


def test_neighbor_fraction_in_a_volume_weights_the_cells_by_their_volume():
    mask = make_volume_mask(cells((0, 1, 2, 2)), dz=DZ, depth=DEPTH)       # the middle layer
    area = float(mask.dataset["cell_area"].values[2, 2])
    frac = xhw.neighbor_fraction(mask, footprint=xhw.Footprint.connected(1))
    assert frac.dims == ("time", "z", "y", "x")
    centre = area * 20.0
    # the centre cell: itself plus 6 neighbours (above 10 m, below 30 m, four of 20 m)
    expected = centre / (centre + area * (10.0 + 30.0) + 4 * centre)
    assert frac.values[0, 1, 2, 2] == pytest.approx(expected, rel=1e-3)
    above = frac.values[0, 0, 2, 2]               # the cell above sees the centre as its neighbour
    assert above == pytest.approx(centre / (area * 10.0 + centre + 4 * area * 10.0), rel=1e-3)
    assert frac.values[0, 0, 0, 0] == 0.0 and not frac.values[1].any()


@pytest.mark.parametrize("kind", ["patches", "tracks"])
def test_ingesting_the_own_labels_of_a_volume_reproduces_its_object_tables(kind):
    exc = cells((0, 0, 2, 2), (0, 1, 2, 2), (1, 0, 2, 3), (1, 1, 2, 3), (2, 2, 4, 4))
    mask = make_volume_mask(exc, dz=DZ, depth=DEPTH)
    own = build(mask, seg(), xhw.RecipeStep.track(min_duration_days=1))
    labels = own.dataset["label" if kind == "patches" else "track_id"]
    got = xhw.ingest_objects(labels, mask=mask, kind=kind, source=SOURCE, method="own")
    got.check_invariants()
    got.check_against(mask)
    for name in ("object_cells", "object_area", "object_depth", "object_lat", "object_lon"):
        np.testing.assert_allclose(got.dataset[name].values, own.dataset[name].values,
                                   err_msg=name)
    assert got.dataset["label"].dims == ("time", "z", "y", "x")
    cmp = xhw.compare_objects(own, got)
    assert (cmp["jaccard"].dropna("time") == 1.0).all() and (cmp["best_iou"] == 1.0).all()
