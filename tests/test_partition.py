# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""genealogy: the optional partition of merged patches by their nearest parent cell.

Off unless called; it returns a new ObjectField whose merged patches are split among their
parents by the nearest parent cell, with the same processed cells (the detection is not
touched).
"""

import numpy as np
import pytest
import xarray as xr
from _lineage_fixtures import JOIN, lay
from _object_fixtures import make_mask, make_volume_mask, recipe, seg

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")


def built(mask, mode="retrospective", track=True):
    steps = [seg()] + ([xhw.RecipeStep.track()] if track else [])
    obj = xhw.build_objects(mask, recipe=recipe(*steps), analysis_mode=mode)
    obj.check_invariants()
    return obj


def split(mask, mode="retrospective", track=True):
    obj = built(mask, mode, track)
    parts = xhw.partition_merges(obj, mask)
    parts.check_invariants()
    parts.check_against(mask)
    return obj, parts


def areas(obj):
    return obj.dataset["object_area"].values.tolist()


# A (cols 0-1) and B (cols 4-5) merge into C (cols 0-5), which lasts one more day
MERGE = lay(3, {0: [(1, 0, 2), (1, 4, 6)], 1: [(1, 0, 6)], 2: [(1, 0, 6)]}, nx=8)


def test_a_merged_patch_is_split_by_the_nearest_parent_cell():
    obj, parts = split(make_mask(MERGE, unit_measure=True))
    assert areas(obj) == [2.0, 2.0, 6.0, 6.0]
    # col 2 is nearer A (col 1), col 3 nearer B (col 4): parts 0-2 and 3-5, on days 1 and 2
    assert areas(parts) == [2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
    label = parts.dataset["label"].transpose("time", "y", "x").values
    assert label[1, 1, :6].tolist() == [1, 1, 1, 2, 2, 2]
    graph = xhw.build_lineage(parts)
    graph.check_invariants()
    assert not graph.dataset["is_merge"].values.any()
    assert graph.dataset.sizes["spacetime_event"] == 2           # A and B stay apart
    assert xhw.build_lineage(obj).dataset["is_merge"].values.tolist() == [
        False, False, True, False]                                  # the default keeps the merge


def test_the_processed_cells_and_the_provenance():
    mask = make_mask(MERGE, unit_measure=True)
    obj, parts = split(mask)
    np.testing.assert_array_equal(parts.dataset["processed_mask"].values,
                                  obj.dataset["processed_mask"].values)
    steps = [s.name for s in parts.provenance.steps]
    assert steps == ["partition_merges"]
    assert obj.provenance.run_id in parts.provenance.parents
    assert parts.recipe == obj.recipe


@pytest.mark.parametrize("mode", ["retrospective", "monitoring"])
def test_partition_in_both_modes_and_with_the_documented_join(mode):
    mask = make_mask(JOIN, unit_measure=True)
    obj, parts = split(mask, mode=mode)
    graph = xhw.build_lineage(parts)
    assert not graph.dataset["is_merge"].values.any()
    assert sum(areas(parts)) == sum(areas(obj)) == 540.0


def test_nothing_changes_without_merges():
    exc = lay(3, {0: [(1, 0, 3)], 1: [(1, 1, 4)], 2: [(1, 2, 5)]}, nx=8)
    obj, parts = split(make_mask(exc, unit_measure=True))
    for name in ("label", "object_area", "track_id", "object_track"):
        np.testing.assert_array_equal(parts.dataset[name].values, obj.dataset[name].values)


def test_the_nearest_parent_is_found_across_the_periodic_edge():
    exc = np.zeros((2, 3, 12), bool)
    exc[0, 1, [11, 0]] = True                  # A straddles longitude 0
    exc[0, 1, [5, 6]] = True                   # B
    exc[1, 1, [10, 11, 0, 1, 2, 3, 4, 5, 6]] = True
    _, parts = split(make_mask(exc, periodic=True, unit_measure=True))
    label = parts.dataset["label"].transpose("time", "y", "x").values[1, 1]
    # cell 10 is one cell from A across the edge (11), far from B
    assert label[10] == label[11] == label[0] != label[6]


def test_a_volume_is_split_over_z_y_x():
    exc = np.zeros((2, 4, 1, 3), bool)
    exc[0, 0, 0, :] = exc[0, 3, 0, :] = True   # two layers apart
    exc[1, :, 0, :] = True                     # one patch through the column
    mask = make_volume_mask(exc, dz=(10.0,) * 4, depth=(5.0, 15.0, 25.0, 35.0))
    _, parts = split(mask)
    label = parts.dataset["label"].transpose("time", "z", "y", "x").values[1, :, 0, 0]
    assert label[0] == label[1] != label[2] == label[3]


def test_objects_from_another_method_are_not_partitioned():
    mask = make_mask(MERGE, unit_measure=True)
    labels = xr.DataArray(MERGE.astype(int), dims=("time", "y", "x"),
                          coords={"time": mask.dataset["time"].values})
    source = xhw.SourceRef(dataset_id="external:demo", variable="mask", accessed="2026-10-06",
                           product_status="final", version="0.1")
    ext = xhw.ingest_objects(labels, mask=mask, kind="patches", source=source, method="demo")
    with pytest.raises(xhw.NotSupportedError, match="own method"):
        xhw.partition_merges(ext, mask)
    with pytest.raises(xhw.ValidationError, match="source"):
        xhw.partition_merges(built(mask), make_mask(MERGE, unit_measure=True))
