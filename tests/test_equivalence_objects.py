# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""object tracking: xHW objects against the labelling of Ocetrac 0.1.7 and marEx 4.1.2.

Reads the versioned outputs of `objects_case.py reference` (neither tool is needed here):

- tracking by overlap: xHW retrospective tracks with 8-neighbour patches and ``min_overlap =
  t`` give the same partition of the cells as Ocetrac's
  ``label_temporal_neighbor(contain_thresh=t)``, including the inclusive ``>=`` at exactly
  0.5;
- daily 8-neighbour patches with periodic longitude equal marEx's
  ``dask_image.ndmeasure.label(..., wrap_axes=(2,))`` of the structured, time-independent
  path;
- with min_overlap 0 and 4-neighbour patches, xHW tracks equal a face-connected (x, y, t)
  labelling (``scipy.ndimage.label``, 6 neighbours), a property independent of both tools;
- deviation, documented: the 26-neighbour (x, y, t) structure of Ocetrac ``label_3d`` and of
  marEx with ``time_connectivity=True`` merges objects that only touch diagonally between
  days, so it never gives more components than xHW and gives fewer on some masks.
- ingestion (`xhw.ingest_objects`): the labels that Ocetrac and marEx actually produced are
  read back as external objects with their declared neighbourhood checked, and compared with
  xHW's own recipe (same patches, IoU 1) on surfaces, and with the same labelling primitives
  on a line and a volume (the tools themselves have no line or volume API);
- deviation, documented: the 26-neighbour tracks are not made of connected daily patches,
  which the declared-connectivity check of the ingestion reports.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from _object_fixtures import (  # noqa: E402
    make_line_mask,
    make_mask,
    make_volume_mask,
    recipe,
    seg,
)

import xhw
from xhw._morph import build_offsets, label_field

REF = Path(__file__).parent / "reference" / "objects"


@pytest.fixture(scope="module")
def reference():
    path = REF / "labels.npz"
    if not (REF / "manifest.json").exists() or not path.exists():
        pytest.fail("missing object reference files: reinstall the complete source distribution "
                    "")
    manifest = json.loads((REF / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"]["labels.npz"]
    return np.load(path), manifest


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """The two label arrays group the cells identically (0 is the background in both)."""
    if not np.array_equal(a > 0, b > 0):
        return False
    pairs = np.unique(np.stack([a[a > 0], b[b > 0]], axis=1), axis=0)
    return (np.unique(pairs[:, 0]).size == pairs.shape[0]
            and np.unique(pairs[:, 1]).size == pairs.shape[0])


def tracks(mask, *, rank, overlap, periodic=False):
    obj = xhw.build_objects(
        make_mask(mask, unit_measure=True, periodic=periodic),
        recipe=recipe(seg(rank=rank), xhw.RecipeStep.track(min_overlap=overlap, max_gap_days=0,
                                                           min_duration_days=1)))
    obj.check_invariants()
    return np.asarray(obj.dataset["track_id"].transpose("time", "y", "x").values)


def names(reference):
    """The (time, y, x) surface masks (the line and volume masks have their own tests)."""
    return sorted(k[3:] for k in reference[0].files
                  if k.startswith("in_") and k[3:] not in ("line", "volume"))


def test_tracking_by_overlap_equals_ocetrac_label_temporal_neighbor(reference):
    data, manifest = reference
    checked = 0
    for name in names(reference):
        if name == "seam":
            continue                    # Ocetrac does not wrap longitude in this function
        thresholds = manifest["boundary_thresholds"] if name == "boundary" else manifest[
            "thresholds"]
        for thr in thresholds:
            ours = tracks(data[f"in_{name}"], rank=2, overlap=thr)
            theirs = data[f"ocetrac_{name}_{thr}"]
            assert same_partition(ours, theirs), (name, thr)
            checked += 1
    assert checked >= 20


def test_the_inclusive_boundary_is_the_same_in_both_tools(reference):
    data, _ = reference
    mask = data["in_boundary"]
    for thr, linked in ((0.5, True), (0.51, False)):
        theirs = data[f"ocetrac_boundary_{thr}"]
        assert (np.unique(theirs[theirs > 0]).size == 1) is linked
        assert (np.unique(tracks(mask, rank=2, overlap=thr)[mask]).size == 1) is linked


def test_daily_patches_equal_the_marex_labelling_call_with_periodic_longitude(reference):
    data, _ = reference
    fp = xhw.Footprint.connected(2)
    for name in names(reference):
        mask = data[f"in_{name}"]
        ny, nx = mask.shape[1:]
        lat, lon = np.arange(ny, dtype=float), np.arange(nx, dtype=float)
        ours, counts = label_field(mask, fp, build_offsets(fp, lat, lon, True), True)
        offset = np.concatenate(([0], np.cumsum(counts)[:-1]))
        global_ids = np.where(ours > 0, ours + offset[:, None, None], 0)
        assert same_partition(global_ids, data[f"marex2d_{name}"]), name


def test_the_seam_object_is_one_patch_with_periodic_longitude(reference):
    data, _ = reference
    mask = data["in_seam"]
    fp = xhw.Footprint.connected(2)
    ny, nx = mask.shape[1:]
    lat, lon = np.arange(ny, dtype=float), np.arange(nx, dtype=float)
    wrapped, _ = label_field(mask, fp, build_offsets(fp, lat, lon, True), True)
    plain, _ = label_field(mask, fp, build_offsets(fp, lat, lon, False), False)
    assert wrapped.max() == 2 and plain.max() == 3
    assert (data["marex2d_seam"][0] > 0).sum() == mask[0].sum()


def test_four_neighbour_tracks_equal_a_face_connected_space_time_labelling(reference):
    data, _ = reference
    for name in names(reference):
        ours = tracks(data[f"in_{name}"], rank=1, overlap=0.0)
        assert same_partition(ours, data[f"scipy_faces_{name}"]), name


def test_the_26_neighbour_structures_of_both_tools_merge_more_than_xhw(reference):
    data, _ = reference
    fewer = 0
    for name in names(reference):
        if name == "seam":
            continue
        mask = data[f"in_{name}"]
        ours = np.unique(tracks(mask, rank=2, overlap=0.0)).size - 1
        for key in (f"ocetrac3d_{name}", f"marex3d_{name}"):
            theirs = np.unique(data[key][data[key] > 0]).size
            assert theirs <= ours, (key, theirs, ours)
            fewer += theirs < ours
    assert fewer > 0          # the deviation is real on these masks, not hypothetical


def test_a_dataset_round_trip_does_not_change_the_reference_masks(reference):
    data, _ = reference
    mask = data["in_blobs1"]
    field = xr.DataArray(mask, dims=("time", "y", "x"))
    assert field.dtype == bool and mask.shape == (14, 16, 20)


# --- ingestion of what the tools produced -----------------------------------------------------

LEVELS = dict(dz=(10.0, 20.0, 30.0, 40.0, 50.0), depth=(5.0, 20.0, 45.0, 80.0, 125.0))
SOURCE = xhw.SourceRef(dataset_id="external:ocetrac-marex", variable="labels",
                       accessed="2026-10-06", product_status="final", version="0.1.7/4.1.2")


def labels_of(mask, array):
    dims = ("time", *mask.grid.spatial_dims)
    return xr.DataArray(np.asarray(array), dims=dims,
                        coords={"time": mask.dataset["time"].values})


def ingest(mask, array, *, kind, rank, **kw):
    return xhw.ingest_objects(labels_of(mask, array), mask=mask, kind=kind, source=SOURCE,
                              method="reference", connectivity=xhw.Footprint.connected(rank),
                              **kw)


def assert_same_patches(own, ingested):
    cmp = xhw.compare_objects(own, ingested)
    assert (cmp["jaccard"].dropna("time") == 1.0).all()
    assert (cmp["best_iou"] == 1.0).all()
    assert (cmp["n_objects_a"] == cmp["n_objects_b"]).all()


def test_marex_daily_labels_are_ingested_as_the_same_patches_as_xhw(reference):
    data, _ = reference
    for name in names(reference):
        mask = make_mask(data[f"in_{name}"], unit_measure=True, periodic=True)
        ingested = ingest(mask, data[f"marex2d_{name}"], kind="patches", rank=2)
        ingested.check_invariants()
        ingested.check_against(mask)
        own = xhw.build_objects(mask, recipe=recipe(seg(2)))
        assert_same_patches(own, ingested)


def test_ocetrac_tracks_are_ingested_as_the_same_tracks_as_xhw(reference):
    data, manifest = reference
    for name in names(reference):
        if name == "seam":
            continue
        thresholds = manifest["boundary_thresholds"] if name == "boundary" else manifest[
            "thresholds"]
        mask = make_mask(data[f"in_{name}"], unit_measure=True)
        for thr in thresholds:
            ingested = ingest(mask, data[f"ocetrac_{name}_{thr}"], kind="tracks", rank=2)
            ingested.check_invariants()
            own = xhw.build_objects(mask, recipe=recipe(seg(2), xhw.RecipeStep.track(
                min_overlap=thr, min_duration_days=1)))
            theirs = np.asarray(ingested.dataset["track_id"].values)
            assert same_partition(np.asarray(own.dataset["track_id"].values), theirs), (name, thr)
            assert ingested.dataset["track_number"].size == own.dataset["track_number"].size
            # a track is one object per day even when it has several patches that day, so the
            # tables differ by design; the cells they cover are the same
            assert (xhw.compare_objects(own, ingested)["jaccard"].dropna("time") == 1.0).all()


def test_the_26_neighbour_tracks_are_ingested_and_are_fewer_than_xhw_tracks(reference):
    data, _ = reference
    fewer = 0
    for name in names(reference):
        if name == "seam":
            continue
        mask = make_mask(data[f"in_{name}"], unit_measure=True)
        own = xhw.build_objects(mask, recipe=recipe(seg(2), xhw.RecipeStep.track(
            min_overlap=0.0, min_duration_days=1)))
        for key in (f"ocetrac3d_{name}", f"marex3d_{name}"):
            ingested = ingest(mask, data[key], kind="tracks", rank=2)
            ingested.check_invariants()
            assert ingested.dataset["track_number"].size <= own.dataset["track_number"].size
            fewer += ingested.dataset["track_number"].size < own.dataset["track_number"].size
    assert fewer > 0       # the deviation is visible in the ingested tracks, not only in the labels


def test_the_ingestion_checks_the_neighbourhood_the_tool_used(reference):
    data, _ = reference
    mask = make_mask(data["in_blobs1"], unit_measure=True, periodic=True)
    ingest(mask, data["marex2d_blobs1"], kind="patches", rank=2)         # 8 neighbours: as marEx
    ingest(mask, data["marex2d_blobs1"], kind="tracks", rank=2)          # no identities touch
    with pytest.raises(xhw.ValidationError, match="connected patch"):
        ingest(mask, data["marex2d_blobs1"], kind="patches", rank=1)     # 4 neighbours: it is not
    plain = make_mask(data["in_seam"], unit_measure=True, periodic=False)
    with pytest.raises(xhw.ValidationError, match="connected patch"):
        ingest(plain, data["marex2d_seam"], kind="patches", rank=2)      # marEx wraps longitude


def test_labelling_primitives_of_both_tools_on_a_line_ingest_as_the_xhw_patches(reference):
    data, _ = reference
    mask = make_line_mask(data["in_line"])
    own = xhw.build_objects(mask, recipe=recipe(seg(1)))
    for key in ("marex_line", "skimage_line"):
        ingested = ingest(mask, data[key], kind="patches", rank=1)
        ingested.check_invariants()
        ingested.check_against(mask)
        assert "object_along" in ingested.dataset
        assert_same_patches(own, ingested)
        np.testing.assert_allclose(ingested.dataset["object_along"].values,
                                   own.dataset["object_along"].values)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_labelling_primitives_of_both_tools_in_a_volume_ingest_as_the_xhw_patches(reference, rank):
    data, _ = reference
    for key, periodic in ((f"marex_volume_{rank}", True), (f"skimage_volume_{rank}", False)):
        mask = make_volume_mask(data["in_volume"], unit_measure=True, periodic=periodic, **LEVELS)
        ingested = ingest(mask, data[key], kind="patches", rank=rank)
        ingested.check_invariants()
        ingested.check_against(mask)
        own = xhw.build_objects(mask, recipe=recipe(seg(rank)))
        assert_same_patches(own, ingested)
        np.testing.assert_allclose(ingested.dataset["object_depth"].values,
                                   own.dataset["object_depth"].values)


def test_a_volume_label_from_a_larger_neighbourhood_is_flagged(reference):
    data, _ = reference
    mask = make_volume_mask(data["in_volume"], unit_measure=True, periodic=False, **LEVELS)
    with pytest.raises(xhw.ValidationError, match="connected patch"):
        ingest(mask, data["skimage_volume_3"], kind="patches", rank=1)   # 26-neighbour labels
