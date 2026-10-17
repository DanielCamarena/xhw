# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""object tracking tracking: overlap links, object gaps, identities in both analysis modes."""

import numpy as np
import pytest
from _fixtures import hobday
from _object_fixtures import make_mask, recipe, seg

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")


def patches(days, *spec, ny=4, nx=14):
    """Boolean field; each spec is (day, row, first_col, last_col) with both columns inclusive."""
    exc = np.zeros((days, ny, nx), bool)
    for t, row, a, b in spec:
        exc[t, row, a:b + 1] = True
    return exc


def track(overlap=0.0, gap=0, minimum=1):
    return xhw.RecipeStep.track(min_overlap=overlap, max_gap_days=gap, min_duration_days=minimum)


def run(exc, *, mode="retrospective", overlap=0.0, gap=0, minimum=1, layer="exceedance",
        mask=None, **kw):
    mask = mask or make_mask(exc, unit_measure=True)
    obj = xhw.build_objects(mask, recipe=recipe(seg(), track(overlap, gap, minimum), layer=layer),
                            analysis_mode=mode, **kw)
    obj.check_invariants()
    obj.check_against(mask)
    return obj


def tracks_of(obj):
    return obj.dataset["object_track"].values.tolist()


def ids(obj):
    return np.asarray(obj.dataset["track_id"].transpose("time", "y", "x").values)


# --- links ------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["retrospective", "monitoring"])
def test_a_patch_that_moves_one_cell_a_day_is_one_track(mode):
    obj = run(patches(5, (0, 1, 0, 1), (1, 1, 1, 2), (2, 1, 2, 3), (3, 1, 3, 4)), mode=mode)
    assert tracks_of(obj) == [1, 1, 1, 1]
    ds = obj.dataset
    assert ds["track_duration_days"].values.tolist() == [4]
    assert ds["track_active_days"].values.tolist() == [4]
    assert ds["track_bridged_days"].values.tolist() == [0]
    assert ds["track_area_max"].values.tolist() == [2.0]
    assert ds["track_area_days"].values.tolist() == [8.0]
    assert ds["track_area_mean"].values.tolist() == [2.0]


def test_objects_that_touch_diagonally_but_share_no_cell_are_not_linked():
    obj = run(patches(3, (0, 1, 3, 3), (1, 2, 4, 4)))
    assert tracks_of(obj) == [1, 2]            # (ocean trackers using 26-connectivity would link)


@pytest.mark.parametrize("overlap, linked", [(0.0, True), (0.5, True), (0.51, False),
                                             (1.0, False)])
def test_min_overlap_is_a_fraction_of_the_smaller_object_and_the_comparison_is_inclusive(
        overlap, linked):
    # 4 cells (cols 0-3) then 4 cells (cols 2-5): 2 shared cells = exactly 0.5 of either
    obj = run(patches(2, (0, 1, 0, 3), (1, 1, 2, 5)), overlap=overlap)
    assert (len(set(tracks_of(obj))) == 1) is linked


def test_the_overlap_is_relative_to_the_smaller_object_not_to_their_union():
    # 6 cells (cols 0-5) then 2 cells (cols 2-3) fully inside: ratio 2 / min(6, 2) = 1
    obj = run(patches(2, (0, 1, 0, 5), (1, 1, 2, 3)), overlap=1.0)
    assert tracks_of(obj) == [1, 1]


# --- merges and splits, in both modes ---------------------------------------------------------


MERGE = patches(3, (0, 1, 0, 1), (0, 1, 4, 5), (1, 1, 0, 5), (2, 1, 0, 5))
SPLIT = patches(3, (0, 1, 0, 5), (1, 1, 0, 1), (1, 1, 4, 5), (2, 1, 0, 1), (2, 1, 4, 5))


def test_a_merge_gives_one_identity_retrospectively_and_keeps_the_larger_overlap_when_monitoring():
    retro = run(MERGE)
    assert tracks_of(retro) == [1, 1, 1, 1]                    # the past is rewritten
    assert retro.dataset["track_n_multi_overlap_days"].values.tolist() == [1]   # day 1 only
    live = run(MERGE, mode="monitoring")
    assert tracks_of(live) == [1, 2, 1, 1]                     # A and B stay different on day 0
    ds = live.dataset
    assert ds["track_merged_into"].values.tolist() == [0, 1]   # B ended in the merge, recorded
    assert ds["track_parent"].values.tolist() == [0, 0]
    assert ds["track_duration_days"].values.tolist() == [3, 1]
    assert ds["track_qualified"].values.tolist() == [True, True]


def test_a_split_keeps_one_identity_retrospectively_and_gives_the_other_child_a_parent_live():
    retro = run(SPLIT)
    assert set(tracks_of(retro)) == {1}
    live = run(SPLIT, mode="monitoring")
    # the child that comes first in raster order inherits (equal overlap: lowest id)
    assert tracks_of(live) == [1, 1, 2, 1, 2]
    ds = live.dataset
    assert ds["track_parent"].values.tolist() == [0, 1]
    assert ds["track_merged_into"].values.tolist() == [0, 0]


def test_monitoring_never_rewrites_the_past_and_retrospective_does():
    full_live, full_retro = run(MERGE, mode="monitoring"), run(MERGE)
    cut = patches(1, (0, 1, 0, 1), (0, 1, 4, 5))               # only day 0
    first_live, first_retro = run(cut, mode="monitoring"), run(cut)
    day0 = (slice(0, 1),)
    assert np.array_equal(ids(first_live)[day0], ids(full_live)[day0])      # same ids on day 0
    assert not np.array_equal(ids(first_retro)[day0], ids(full_retro)[day0])


@pytest.mark.parametrize("seed", range(6))
def test_monitoring_ids_of_the_first_days_do_not_depend_on_later_days(seed):
    rng = np.random.default_rng(seed)
    exc = rng.random((10, 6, 8)) < 0.35
    mask = make_mask(exc, unit_measure=True)
    cut_mask = make_mask(exc[:6], unit_measure=True)
    kw = dict(mode="monitoring", gap=2)
    assert np.array_equal(ids(run(exc, mask=mask, **kw))[:6], ids(run(exc[:6], mask=cut_mask,
                                                                      **kw)))


# --- gaps of an object ------------------------------------------------------------------------


GAP = patches(5, (0, 1, 0, 2), (2, 1, 1, 3))           # present on day 0, away on day 1, back on 2


@pytest.mark.parametrize("mode", ["retrospective", "monitoring"])
def test_an_object_that_disappears_one_day_is_two_tracks_without_a_gap_and_one_with_it(mode):
    split = run(GAP, mode=mode, gap=0)
    assert tracks_of(split) == [1, 2]
    assert split.dataset["track_duration_days"].values.tolist() == [1, 1]
    joined = run(GAP, mode=mode, gap=1)
    ds = joined.dataset
    assert tracks_of(joined) == [1, 1]
    assert ds["track_duration_days"].values.tolist() == [3]
    assert ds["track_active_days"].values.tolist() == [2]
    assert ds["track_bridged_days"].values.tolist() == [1]


def test_a_bridge_needs_overlap_with_the_last_footprint_and_at_most_g_empty_days():
    away = patches(6, (0, 1, 0, 2), (2, 1, 8, 10))          # reappears elsewhere: no overlap
    assert tracks_of(run(away, gap=1)) == [1, 2]
    late = patches(6, (0, 1, 0, 2), (3, 1, 1, 3))           # two empty days
    assert tracks_of(run(late, gap=1)) == [1, 2]
    assert tracks_of(run(late, gap=2)) == [1, 1]


def test_with_a_larger_gap_there_are_never_more_tracks():
    rng = np.random.default_rng(11)
    exc = rng.random((14, 5, 9)) < 0.3
    counts = [run(exc, gap=g).dataset.sizes["track"] for g in range(0, 5)]
    assert counts == sorted(counts, reverse=True)


def test_each_reappearing_object_is_bridged_to_one_candidate_only_the_largest_overlap():
    # two objects vanish on day 1 and one big object reappears on day 2 overlapping both
    exc = patches(3, (0, 1, 0, 1), (0, 1, 4, 7), (2, 1, 0, 7))
    obj = run(exc, gap=1)
    # B (4 cells) overlaps the new 8-cell object by 4/4 = 1.0, A by 2/2 = 1.0: tie, lowest id (A)
    assert tracks_of(obj) == [1, 2, 1]
    assert obj.dataset["track_n_multi_overlap_days"].values.tolist() == [0, 0]   # one link only


# --- identities, duration, censoring ----------------------------------------------------------


def test_tracks_are_numbered_by_first_day_then_first_cell_in_raster_order():
    exc = patches(4, (0, 3, 9, 10), (0, 1, 2, 3), (1, 3, 9, 10), (1, 1, 2, 3), (2, 0, 0, 1),
                  (3, 0, 0, 1))
    obj = run(exc)
    # day 0: the patch at row 1 comes first in raster order, then the one at row 3; day 2: new one
    assert tracks_of(obj) == [1, 2, 1, 2, 3, 3]
    assert tracks_of(run(exc, mode="monitoring")) == [1, 2, 1, 2, 3, 3]


def test_a_minimum_duration_drops_short_tracks_retrospectively_and_flags_them_live():
    exc = patches(6, (0, 0, 0, 1), (1, 0, 0, 1), (0, 3, 6, 7), (1, 3, 6, 7), (2, 3, 6, 7),
                  (3, 3, 6, 7))
    retro = run(exc, minimum=3)
    ds = retro.dataset
    assert ds["track_number"].values.tolist() == [1]          # renumbered
    assert ds["track_duration_days"].values.tolist() == [4]
    # objects of a day are in raster order: the short patch (row 0) comes before the long one
    assert tracks_of(retro) == [0, 1, 0, 1, 1, 1]             # the short patches stay in `label`
    assert not ids(retro)[0:2, 0].any()
    processed = np.asarray(ds["processed_mask"].transpose("time", "y", "x").values)
    assert not processed[0:2, 0].any() and processed[0:4, 3, 6:8].all()
    live = run(exc, mode="monitoring", minimum=3)
    dl = live.dataset
    assert dl["track_qualified"].values.tolist() == [False, True]     # candidates are kept
    assert (ids(live)[0:2, 0, 0:2] == 1).all()
    live_processed = np.asarray(dl["processed_mask"].transpose("time", "y", "x").values)
    assert not live_processed[0:2, 0].any()                   # but never in the processed mask


def test_tracks_touching_the_window_edges_are_censored():
    exc = patches(5, (0, 1, 0, 1), (1, 1, 0, 1), (3, 2, 5, 6), (4, 2, 5, 6), (2, 0, 9, 10))
    ds = run(exc).dataset
    assert ds["track_left_censored"].values.tolist() == [True, False, False]
    assert ds["track_right_censored"].values.tolist() == [False, False, True]


def test_two_runs_give_identical_results():
    rng = np.random.default_rng(2)
    exc = rng.random((9, 6, 8)) < 0.4
    a, b = run(exc, gap=1), run(exc, gap=1)
    for name in a.dataset.data_vars:
        np.testing.assert_array_equal(a.dataset[name].values, b.dataset[name].values, name)


def test_final_after_days_counts_the_persistence_and_the_gap_in_monitoring_only():
    exc = patches(8, (0, 1, 0, 3), (1, 1, 0, 3))
    persistent = hobday(min_duration_days=5, max_gap_days=2)         # delay 5 + 2 - 1 = 6
    mask = make_mask(exc, mem=exc, definition=persistent, unit_measure=True)
    live = run(exc, mask=mask, mode="monitoring", gap=1, layer="event_membership")
    assert live.final_after_days == 7 and live.analysis_mode == "monitoring"
    assert run(exc, mask=mask, layer="event_membership").final_after_days == 0
    on_exceedance = run(exc, mask=mask, mode="monitoring", gap=1)
    assert on_exceedance.final_after_days == 1


def test_a_track_step_is_only_valid_after_a_segment_step_and_on_surface_masks():
    mask = make_mask(patches(2, (0, 1, 0, 1)), unit_measure=True)
    obj = xhw.build_objects(mask, recipe=recipe(seg(), track()))
    assert obj.has_track and obj.has_label
    plain = xhw.build_objects(mask, recipe=recipe(seg()))
    assert not plain.has_track and "track_id" not in plain.dataset


def test_tampering_with_the_track_table_is_caught():
    ds = run(patches(4, (0, 1, 0, 1), (1, 1, 0, 1), (2, 1, 0, 1))).to_dataset().copy(deep=True)
    ds["track_duration_days"].values[0] = 9
    with pytest.raises(xhw.InvariantError, match="track_duration_days"):
        xhw.ObjectField.from_dataset(ds).check_invariants()


@pytest.mark.parametrize("mode", ["retrospective", "monitoring"])
def test_a_run_without_any_patch_gives_empty_tables_not_an_error(mode):
    mask = make_mask(np.zeros((4, 5, 5), bool), unit_measure=True)
    obj = xhw.build_objects(mask, recipe=recipe(seg(), xhw.RecipeStep.track(min_duration_days=2)),
                            analysis_mode=mode)
    obj.check_invariants()
    assert obj.dataset["object_area"].size == 0 and obj.dataset["track_number"].size == 0
    assert not obj.dataset["processed_mask"].any()
