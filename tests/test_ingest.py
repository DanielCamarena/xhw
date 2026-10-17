# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""object tracking: neighbour fraction, ingestion of external objects and comparison."""

import numpy as np
import pytest
import xarray as xr
from _object_fixtures import make_mask, recipe, seg

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
SOURCE = xhw.SourceRef(dataset_id="external:cnn-demo", variable="mask", accessed="2026-10-05",
                       product_status="final", version="0.1")


def as_labels(mask, array):
    return xr.DataArray(np.asarray(array), dims=("time", "y", "x"),
                        coords={"time": mask.dataset["time"].values})


def square(days, *cells, ny=6, nx=6):
    out = np.zeros((days, ny, nx), int)
    for t, y, x, value in cells:
        out[t, y, x] = value
    return out


# --- neighbour fraction -----------------------------------------------------------------------


def test_neighbor_fraction_of_the_exceedance_of_a_mask():
    exc = np.zeros((2, 5, 5), bool)
    exc[0, 2, 2] = True
    mask = make_mask(exc, unit_measure=True)
    frac = xhw.neighbor_fraction(mask, footprint=xhw.Footprint.connected(2))
    assert frac.dims == ("time", "y", "x") and frac.name == "neighbor_fraction"
    assert frac.values[0, 2, 2] == pytest.approx(1 / 9)
    assert frac.values[0, 0, 0] == 0.0 and not frac.values[1].any()
    assert frac.values[0, 1, 1] == pytest.approx(1 / 9)


def test_neighbor_fraction_ignores_land_and_is_nan_without_valid_neighbours():
    exc = np.zeros((1, 3, 3), bool)
    exc[0, 1, 1] = True
    domain = np.ones((3, 3), bool)
    domain[0, 1] = False
    mask = make_mask(exc, domain=domain, unit_measure=True)
    frac = xhw.neighbor_fraction(mask, footprint=xhw.Footprint.connected(2)).values[0]
    assert frac[0, 0] == pytest.approx(1 / 3)           # 3 valid cells in its footprint, not 4
    assert frac[0, 1] == pytest.approx(1 / 5)           # the land cell: 5 valid cells around it
    nothing = make_mask(exc, domain=np.zeros((3, 3), bool), unit_measure=True)
    assert np.isnan(xhw.neighbor_fraction(nothing, footprint=xhw.Footprint.connected(2))).all()


# --- ingestion --------------------------------------------------------------------------------


def test_patches_are_renumbered_per_day_and_the_original_ids_are_kept():
    mask = make_mask(np.ones((3, 6, 6), bool), unit_measure=True)
    raw = square(3, (0, 0, 0, 7), (0, 0, 1, 7), (0, 3, 3, 4), (1, 2, 2, 100), (2, 5, 5, 1))
    obj = xhw.ingest_objects(as_labels(mask, raw), mask=mask, kind="patches", source=SOURCE,
                             method="cnn-demo")
    obj.check_invariants()
    obj.check_against(mask)
    ds = obj.dataset
    assert obj.origin == "external" and obj.recipe is None and not obj.has_track
    assert ds["object_external_id"].values.tolist() == [7, 4, 100, 1]
    assert ds["object_label"].values.tolist() == [1, 2, 1, 1]
    assert ds["object_cells"].values.tolist() == [2, 1, 1, 1]
    assert obj.provenance.inputs == (SOURCE,)
    assert obj.provenance.steps[0].name == "ingest_objects"
    assert obj.provenance.parents == (mask.provenance.run_id,)
    back = xhw.ObjectField.from_dataset(obj.to_dataset())
    assert back.origin == "external" and back.recipe is None


def test_tracks_are_renumbered_by_first_day_and_first_cell_with_the_original_ids_kept():
    mask = make_mask(np.ones((4, 6, 6), bool), unit_measure=True)
    raw = square(4, (0, 4, 4, 90), (0, 1, 1, 30), (1, 4, 4, 90), (1, 1, 1, 30), (2, 3, 3, 55),
                 (3, 3, 3, 55))
    obj = xhw.ingest_objects(as_labels(mask, raw), mask=mask, kind="tracks", source=SOURCE,
                             method="cnn-demo")
    obj.check_invariants()
    ds = obj.dataset
    assert ds["track_external_id"].values.tolist() == [30, 90, 55]     # day 0 raster order, then 55
    assert ds["object_track"].values.tolist() == [1, 2, 1, 2, 3, 3]
    assert ds["track_duration_days"].values.tolist() == [2, 2, 2]
    assert ds["track_left_censored"].values.tolist() == [True, True, False]
    assert ds["track_right_censored"].values.tolist() == [False, False, True]
    assert ds["track_area_days"].values.tolist() == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("make, message", [
    (lambda m: as_labels(m, np.full((3, 6, 6), -1)), ">= 0"),
    (lambda m: as_labels(m, np.full((3, 6, 6), 0.5)), "integers"),
    (lambda m: as_labels(m, np.full((3, 6, 6), np.nan)), "integers"),
    (lambda m: xr.DataArray(np.zeros((2, 6, 6), int), dims=("time", "y", "x")),
     "differ from the mask"),
    (lambda m: xr.DataArray(np.zeros((3, 6, 6), int), dims=("time", "y", "x"),
                            coords={"time": m.dataset["time"].values + np.timedelta64(1, "D")}),
     "different time axes"),
    (lambda m: xr.DataArray(np.zeros((6, 6), int), dims=("y", "x")), "dims"),
])
def test_invalid_labels_are_refused(make, message):
    mask = make_mask(np.ones((3, 6, 6), bool), unit_measure=True)
    with pytest.raises(xhw.ValidationError, match=message):
        xhw.ingest_objects(make(mask), mask=mask, kind="patches", source=SOURCE, method="x")


def test_labels_on_invalid_cells_and_unknown_kinds_are_refused():
    domain = np.ones((6, 6), bool)
    domain[0, 0] = False
    mask = make_mask(np.ones((3, 6, 6), bool), domain=domain, unit_measure=True)
    raw = square(3, (0, 0, 0, 1))
    with pytest.raises(xhw.ValidationError, match="not valid"):
        xhw.ingest_objects(as_labels(mask, raw), mask=mask, kind="patches", source=SOURCE,
                           method="x")
    with pytest.raises(xhw.ValidationError, match="kind"):
        xhw.ingest_objects(as_labels(mask, square(3)), mask=mask, kind="blobs", source=SOURCE,
                           method="x")
    with pytest.raises(xhw.ValidationError, match="method"):
        xhw.ingest_objects(as_labels(mask, square(3)), mask=mask, kind="patches", source=SOURCE,
                           method=" ")


# --- comparison -------------------------------------------------------------------------------


def test_identical_objects_have_jaccard_and_iou_one():
    exc = np.zeros((3, 6, 6), bool)
    exc[:, 1:3, 1:3] = True
    exc[1, 5, 5] = True
    mask = make_mask(exc, unit_measure=True)
    own = xhw.build_objects(mask, recipe=recipe(seg()))
    raw = np.asarray(own.dataset["label"].transpose("time", "y", "x").values)
    other = xhw.ingest_objects(as_labels(mask, raw), mask=mask, kind="patches", source=SOURCE,
                               method="same")
    cmp = xhw.compare_objects(own, other)
    assert cmp["jaccard"].values.tolist() == [1.0, 1.0, 1.0]
    assert (cmp["best_iou"].values == 1.0).all() and (cmp["best_match_b"].values > 0).all()
    assert cmp["n_objects_a"].values.tolist() == [1, 2, 1]


def test_iou_and_jaccard_with_a_partly_overlapping_patch():
    exc = np.zeros((2, 6, 6), bool)
    exc[0, 2, 0:4] = True                               # 4 cells
    mask = make_mask(exc, unit_measure=True)
    own = xhw.build_objects(mask, recipe=recipe(seg()))
    raw = square(2, (0, 2, 2, 1), (0, 2, 3, 1), (0, 2, 4, 1), (0, 2, 5, 1))   # cols 2-5
    other = xhw.ingest_objects(as_labels(mask, raw), mask=mask, kind="patches", source=SOURCE,
                               method="shifted")
    # b has cells where the mask is not exceeding: cells 4, 5 are valid but not in the source
    cmp = xhw.compare_objects(own, other)
    assert cmp["jaccard"].values[0] == pytest.approx(2 / 6)
    assert np.isnan(cmp["jaccard"].values[1])           # both empty on day 1
    assert cmp["best_iou"].values.tolist() == [pytest.approx(2 / 6)]
    assert cmp["best_match_b"].values.tolist() == [1]
    assert cmp["cells_a"].values.tolist() == [4, 0] and cmp["cells_b"].values.tolist() == [4, 0]


def test_comparison_needs_the_same_mask_and_grid():
    exc = np.zeros((2, 6, 6), bool)
    exc[:, 1, 1:3] = True
    a = xhw.build_objects(make_mask(exc, unit_measure=True), recipe=recipe(seg()))
    b = xhw.build_objects(make_mask(exc, unit_measure=True), recipe=recipe(seg()))
    with pytest.raises(xhw.ValidationError, match="different masks"):
        xhw.compare_objects(a, b)


# --- declared connectivity -----------------------------------------------------------------------


def test_the_declared_connectivity_is_checked_label_by_label():
    mask = make_mask(np.ones((2, 6, 6), bool), unit_measure=True)
    raw = square(2, (0, 1, 1, 3), (0, 2, 2, 3), (1, 4, 4, 5), (1, 4, 5, 5))   # diagonal, domino
    kw = dict(mask=mask, kind="patches", source=SOURCE, method="cnn-demo")
    ok = xhw.ingest_objects(as_labels(mask, raw), connectivity=xhw.Footprint.connected(2), **kw)
    assert ok.provenance.steps[0].params[-1] == ("connectivity",
                                                  xhw.Footprint.connected(2).spec_id)
    with pytest.raises(xhw.ValidationError, match=r"label 3 on 2000-01-01 forms 2 patches"):
        xhw.ingest_objects(as_labels(mask, raw), connectivity=xhw.Footprint.connected(1), **kw)
    with pytest.warns(xhw.XHWDataQualityWarning, match="1 labels are not a single"):
        xhw.ingest_objects(as_labels(mask, raw), connectivity=xhw.Footprint.connected(1),
                           on_violation="warn", **kw)
    xhw.ingest_objects(as_labels(mask, raw), **kw)          # nothing is assumed without it


def test_a_label_in_two_separate_zones_is_a_violation_under_any_connectivity():
    mask = make_mask(np.ones((1, 6, 6), bool), unit_measure=True)
    raw = square(1, (0, 0, 0, 9), (0, 5, 5, 9))
    for rank in (1, 2, 3):
        with pytest.raises(xhw.ValidationError, match="forms 2 patches"):
            xhw.ingest_objects(as_labels(mask, raw), mask=mask, kind="patches", source=SOURCE,
                               method="cnn-demo", connectivity=xhw.Footprint.connected(rank))


def test_the_connectivity_check_wraps_longitude_only_on_a_periodic_grid():
    raw = square(1, (0, 2, 0, 4), (0, 2, 5, 4))          # one label on both sides of the seam
    for periodic, fine in ((False, False), (True, True)):
        mask = make_mask(np.ones((1, 6, 6), bool), unit_measure=True, periodic=periodic)
        call = lambda: xhw.ingest_objects(  # noqa: E731
            as_labels(mask, raw), mask=mask, kind="patches", source=SOURCE, method="cnn-demo",
            connectivity=xhw.Footprint.connected(1))
        if fine:
            call()
        else:
            with pytest.raises(xhw.ValidationError, match="forms 2 patches"):
                call()


def test_for_tracks_the_check_is_that_neighbouring_cells_share_their_identity():
    mask = make_mask(np.ones((1, 6, 6), bool), unit_measure=True)
    kw = dict(mask=mask, kind="tracks", source=SOURCE, method="cnn-demo")
    # one identity in two separate zones is fine for a track (joined through other days)
    apart = square(1, (0, 0, 0, 9), (0, 5, 5, 9))
    xhw.ingest_objects(as_labels(mask, apart), connectivity=xhw.Footprint.connected(2), **kw)
    # two identities side by side are not: the declared neighbourhood would have merged them
    touching = square(1, (0, 2, 2, 1), (0, 2, 3, 2), (0, 4, 0, 3), (0, 5, 1, 4))
    for rank in (1, 2):
        with pytest.raises(xhw.ValidationError, match="pairs of identities touch"):
            xhw.ingest_objects(as_labels(mask, touching),
                               connectivity=xhw.Footprint.connected(rank), **kw)
    diagonal = square(1, (0, 1, 1, 1), (0, 2, 2, 2))
    xhw.ingest_objects(as_labels(mask, diagonal), connectivity=xhw.Footprint.connected(1), **kw)
    with pytest.raises(xhw.ValidationError, match="pairs of identities touch"):
        xhw.ingest_objects(as_labels(mask, diagonal), connectivity=xhw.Footprint.connected(2), **kw)
