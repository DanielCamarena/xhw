# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""genealogy: LinkPolicy, the transition table and the LineageGraph.

The hand-written cases ( with and without a tie, ) are exact fixtures; the canonical
transitions (1 -> 1, 1 -> 2, 2 -> 1, 2 -> 2, split then merge, gaps, isolated, a split in
three) have hand-written expectations; and the graph is checked against object tracking (same
links, and the retrospective spacetime events are the track_id of object tracking).
"""

import numpy as np
import pytest
from _lineage_fixtures import JOIN, NO_TIE, TIE, column, edges, flip, lay, lineage, objects
from _object_fixtures import make_mask

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")


def types(graph):
    return graph.node_type_names().tolist()


# --- LinkPolicy ------------------------------------------------------------------------------


def test_the_default_policy_links_any_shared_cell():
    p = xhw.LinkPolicy()
    assert (p.measure, p.threshold, p.max_gap_days) == ("overlap_smaller", 0.0, 0)
    assert xhw.LinkPolicy.from_json(p.to_json()) == p


def test_the_policy_is_inherited_from_the_track_step():
    step = xhw.RecipeStep.track(min_overlap=0.3, max_gap_days=2, link_measure="iou",
                                dominance_measure="shared", tie_rule="largest_area")
    p = xhw.LinkPolicy.from_track(step)
    assert (p.measure, p.threshold, p.max_gap_days, p.dominance_measure, p.tie_rule) == (
        "iou", 0.3, 2, "shared", "largest_area")


@pytest.mark.parametrize("bad", [dict(threshold=1.5), dict(threshold=-0.1), dict(max_gap_days=-1),
                                 dict(measure="shared"), dict(tie_rule="random")])
def test_invalid_policies_are_rejected(bad):
    with pytest.raises(xhw.ValidationError):
        xhw.LinkPolicy(**bad)


# --- the transition table --------------------------------------------------------------------


def test_the_four_measures_of_an_edge_are_exact():
    # 6 cells then 2 cells inside them: shared 2
    graph = lineage(lay(2, {0: [(1, 0, 6)], 1: [(1, 2, 4)]}))
    t = graph.transitions()
    assert t["shared"].values.tolist() == [2.0]
    assert t["overlap_smaller"].values.tolist() == [1.0]
    assert t["iou"].values.tolist() == pytest.approx([2 / 6])
    assert t["source_fraction"].values.tolist() == pytest.approx([2 / 6])
    assert t["target_fraction"].values.tolist() == [1.0]


def test_edges_below_the_threshold_stay_in_the_table_unlinked():
    low = lineage(NO_TIE)
    high = lineage(NO_TIE, policy=xhw.LinkPolicy(threshold=0.5))
    assert edges(low, linked_only=False) == edges(high, linked_only=False)    # same six edges
    assert len(edges(low)) == 6
    assert edges(high) == [(0, 1), (1, 2), (2, 4), (4, 5)]          # P->R and R->S unlinked
    np.testing.assert_array_equal(low.transitions()["iou"].values,
                                  high.transitions()["iou"].values)


def test_the_linked_edges_are_the_links_of_m2():
    rng = np.random.default_rng(3)
    exc = rng.random((12, 6, 9)) < 0.35
    for gap, overlap in ((0, 0.0), (2, 0.0), (1, 0.4)):
        obj = objects(exc, gap=gap, overlap=overlap)
        graph = xhw.build_lineage(obj)
        graph.check_invariants()
        # object tracking links are what its retrospective tracks are made of
        event = np.asarray(column(graph, "object_spacetime_event"))
        np.testing.assert_array_equal(event, obj.dataset["object_track"].values)


def test_bridges_across_empty_days_are_linked_and_carry_their_gap():
    exc = lay(4, {0: [(1, 0, 3)], 2: [(1, 1, 4)]})
    graph = lineage(exc, gap=1)
    t = graph.transitions()
    assert edges(graph) == [(0, 1)]
    assert t["gap_days"].values.tolist() == [1]
    assert column(graph, "branch_bridged_days") == [1]
    assert column(graph, "event_bridged_days") == [1]
    assert types(graph) == ["birth", "death"]


# --- node types, branches, events: the documented cases --------------------------------------


def test_the_case_without_a_tie():
    graph = lineage(NO_TIE)                     # nodes O, P, Q, R, S, T
    assert types(graph) == ["birth", "continuation", "continuation", "continuation",
                            "continuation", "death"]
    assert column(graph, "is_split") == [False, True, False, False, False, False]
    assert column(graph, "is_merge") == [False, False, False, False, True, False]
    assert column(graph, "object_branch") == [1, 1, 2, 3, 4, 4]      # [O,P] [Q] [R] [S,T]
    assert list(zip(column(graph, "branch_link_parent"), column(graph, "branch_link_child"))) == [
        (1, 2), (1, 3), (2, 4), (3, 4)]
    assert column(graph, "object_spacetime_event") == [1] * 6
    t = graph.transitions()
    # dominant: O->P, P->Q (0.83 > 0.33), Q->S (0.92 > 0.17), S->T; no tie
    assert t["is_dominant_for_source"].values.tolist() == [True, True, False, True, True, True]
    assert t["is_dominant_for_target"].values.tolist() == [True, True, True, True, False, True]
    assert not t["dominant_tied"].values.any()


def test_the_case_without_a_tie_at_threshold_one_half():
    graph = lineage(NO_TIE, policy=xhw.LinkPolicy(threshold=0.5))
    assert types(graph) == ["birth", "continuation", "continuation", "isolated",
                            "continuation", "death"]
    assert column(graph, "object_branch") == [1, 1, 1, 2, 1, 1]      # [O,P,Q,S,T] and [R]
    assert column(graph, "object_spacetime_event") == [1, 1, 1, 2, 1, 1]
    assert column(graph, "branch_link_parent") == []


@pytest.mark.parametrize("flipped", [False, True])
def test_the_case_with_a_tie_and_its_reversed_numbering(flipped):
    graph = lineage(flip(TIE) if flipped else TIE)
    # the branches do not depend on the numbering
    assert column(graph, "object_branch") == [1, 1, 2, 3, 4, 4]
    t = graph.transitions()
    # edges: O->P, P->x, P->y, x->S, y->S, S->T where x is the child first in raster order
    assert t["overlap_smaller"].values.tolist() == pytest.approx([0.875, 0.4, 0.4, 0.6, 0.6,
                                                                  80 / 90])
    # the tie goes to the lowest id (the first child in raster order), and it is marked
    assert t["is_dominant_for_source"].values.tolist() == [True, True, False, True, True, True]
    assert t["is_dominant_for_target"].values.tolist() == [True, True, True, True, False, True]
    assert t["dominant_tied"].values.tolist() == [False, True, False, True, False, False]


def test_with_the_largest_area_tie_rule_equal_areas_still_fall_to_the_lowest_id():
    policy = xhw.LinkPolicy(tie_rule="largest_area")
    t = lineage(TIE, policy=policy).transitions()
    assert t["is_dominant_for_source"].values.tolist() == [True, True, False, True, True, True]
    assert t["dominant_tied"].values.tolist() == [False, True, False, True, False, False]


# --- canonical transitions -------------------------------------------------------------------


CASES = {
    "1->1": (lay(2, {0: [(1, 0, 3)], 1: [(1, 1, 4)]}), ["birth", "death"], [1, 1]),
    "1->2": (lay(2, {0: [(0, 0, 4), (1, 0, 1), (2, 0, 4)], 1: [(0, 0, 3), (2, 0, 3)]}),
             ["birth", "death", "death"], [1, 2, 3]),
    "2->1": (lay(2, {0: [(0, 0, 3), (2, 0, 3)], 1: [(0, 0, 4), (1, 0, 1), (2, 0, 4)]}),
             ["birth", "birth", "death"], [1, 2, 3]),
    "2->2": (lay(3, {0: [(0, 0, 4), (2, 0, 4)], 1: [(0, 0, 4), (1, 1, 2), (2, 0, 4)],
                     2: [(0, 0, 4), (2, 0, 4)]}),
             ["birth", "birth", "complex", "death", "death"], [1, 2, 3, 4, 5]),
    "isolated": (lay(1, {0: [(1, 0, 3)]}), ["isolated"], [1]),
    "split in three": (lay(2, {0: [(0, 0, 9), (1, 0, 1), (2, 0, 9)],
                               1: [(0, 0, 2), (0, 5, 8), (2, 0, 3)]}),
                       ["birth", "death", "death", "death"], [1, 2, 3, 4]),
}


@pytest.mark.parametrize("name", list(CASES))
def test_canonical_transitions(name):
    exc, expected_types, expected_branches = CASES[name]
    graph = lineage(exc)
    assert types(graph) == expected_types
    assert column(graph, "object_branch") == expected_branches
    assert set(column(graph, "object_spacetime_event")) == {1}


def test_split_then_merge_is_an_undirected_cycle_kept_as_it_is():
    exc = lay(4, {0: [(0, 0, 6), (1, 0, 6), (2, 0, 6)], 1: [(0, 0, 3), (2, 0, 3)],
                  2: [(0, 0, 3), (2, 0, 3)], 3: [(0, 0, 6), (1, 0, 6), (2, 0, 6)]})
    graph = lineage(exc)
    assert column(graph, "is_split") == [True, False, False, False, False, False]
    assert column(graph, "is_merge") == [False, False, False, False, False, True]
    assert column(graph, "object_branch") == [1, 2, 3, 2, 3, 4]
    assert column(graph, "event_n_splits") == [1] and column(graph, "event_n_merges") == [1]
    assert column(graph, "event_n_branches") == [4]


# --- monitoring ------------------------------------------------------------------------------


def test_joining_events_open_a_new_event_by_default():
    graph = lineage(JOIN, mode="monitoring")
    # Nodes in raster order: A0 B0 A1 B1 A2 B2 merged0 merged1 merged2
    assert column(graph, "object_spacetime_event") == [1, 2, 1, 2, 1, 2, 3, 3, 3]
    assert column(graph, "event_merged_into") == [3, 3, 0]
    assert column(graph, "event_n_objects") == [3, 3, 9]          # C covers A and B
    assert column(graph, "event_duration_days") == [3, 3, 6]


def test_joining_events_keep_the_dominant_one_with_the_survivor_rule():
    # A -> M overlaps 60 / 70 = 0.86, B -> M 30 / 50 = 0.6: A survives (as in object tracking)
    graph = lineage(JOIN, mode="monitoring", merge_rule="survivor")
    assert column(graph, "object_spacetime_event") == [1, 2, 1, 2, 1, 2, 1, 1, 1]
    assert column(graph, "event_merged_into") == [0, 1]
    assert column(graph, "event_n_objects") == [9, 3]


def test_retrospective_events_are_one_component():
    graph = lineage(JOIN)
    assert column(graph, "object_spacetime_event") == [1] * 9
    assert column(graph, "event_merged_into") == [0]


def test_monitoring_marks_the_last_days_provisional():
    graph = lineage(JOIN, mode="monitoring", gap=2)
    # a patch of day 3 or later may still be bridged (gap 2, last day 5): days >= 5 - 2
    assert column(graph, "provisional") == [False] * 6 + [True] * 3


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("rule", ["new_event", "survivor"])
def test_monitoring_events_of_the_first_days_do_not_depend_on_later_days(seed, rule):
    rng = np.random.default_rng(seed)
    exc = rng.random((10, 6, 8)) < 0.35
    full = lineage(exc, mode="monitoring", merge_rule=rule, gap=1)
    cut = lineage(exc[:6], mode="monitoring", merge_rule=rule, gap=1)
    n = cut.dataset.sizes["object"]
    assert column(full, "object_spacetime_event")[:n] == column(cut, "object_spacetime_event")
    #: a reappearance after a gap never reassigns the branch of an earlier node
    assert column(full, "object_branch")[:n] == column(cut, "object_branch")
    # edges among the first days are the same
    first = [e for e in edges(full, linked_only=False) if e[1] < n]
    assert first == edges(cut, linked_only=False)


def test_with_the_survivor_rule_and_no_split_the_events_are_the_monitoring_tracks_of_m2():
    obj = objects(JOIN, mode="monitoring")
    graph = xhw.build_lineage(obj, merge_rule="survivor")
    np.testing.assert_array_equal(column(graph, "object_spacetime_event"),
                                  obj.dataset["object_track"].values)


# --- container -------------------------------------------------------------------------------


def test_round_trip_and_tampering():
    graph = lineage(NO_TIE)
    ds = graph.to_dataset()
    again = xhw.LineageGraph.from_dataset(ds)
    again.check_invariants()
    assert again.policy == graph.policy and again.analysis_mode == "retrospective"
    bad = ds.copy(deep=True)
    bad["in_degree"].values[0] = 5
    with pytest.raises(xhw.InvariantError, match="in_degree"):
        xhw.LineageGraph.from_dataset(bad).check_invariants()
    bad = ds.copy(deep=True)
    bad.attrs["policy_id"] = "sha256:" + "0" * 64
    with pytest.raises(xhw.SchemaError, match="policy_id"):
        xhw.LineageGraph.from_dataset(bad)


def test_check_against_rebuilds_the_graph_from_the_objects():
    obj = objects(NO_TIE)
    graph = xhw.build_lineage(obj)
    graph.check_against(obj)
    other = objects(TIE)
    with pytest.raises(xhw.ValidationError, match="run id"):
        graph.check_against(other)


def test_objects_without_a_track_step_use_the_default_policy():
    obj = objects(NO_TIE, track=False)
    graph = xhw.build_lineage(obj)
    assert graph.policy == xhw.LinkPolicy()
    assert len(edges(graph)) == 6


def test_objects_without_patches_are_rejected_and_empty_fields_give_empty_tables():
    mask = make_mask(np.zeros((3, 4, 4), bool), unit_measure=True)
    plain = xhw.build_objects(mask, recipe=xhw.SpatialRecipe(
        input_layer="exceedance", steps=(xhw.RecipeStep.persist(),)))
    with pytest.raises(xhw.ValidationError, match="no patches"):
        xhw.build_lineage(plain)
    graph = lineage(np.zeros((3, 4, 4), bool))
    assert graph.dataset.sizes.get("transition", 0) == 0
    assert graph.dataset.sizes.get("spacetime_event", 0) == 0


def test_ingested_objects_keep_their_backend_ids():
    exc = NO_TIE
    mask = make_mask(exc, unit_measure=True)
    raw = np.where(exc, 7, 0).astype(int)        # one external identity for every patch
    import xarray as xr
    labels = xr.DataArray(raw, dims=("time", "y", "x"),
                          coords={"time": mask.dataset["time"].values})
    source = xhw.SourceRef(dataset_id="external:demo", variable="mask", accessed="2026-10-06",
                           product_status="final", version="0.1")
    obj = xhw.ingest_objects(labels, mask=mask, kind="tracks", source=source, method="demo")
    graph = xhw.build_lineage(obj)
    graph.check_invariants()
    assert set(column(graph, "backend_id")) == {7}
