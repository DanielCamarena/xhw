# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""genealogy: the four rules of the track step.

``link_measure``, ``dominance_measure``, ``tie_rule`` and ``branch_rule`` default to the
object tracking behaviour (stored as None, so recipes keep their identifiers); each one is
checked on an exact field, and the monitoring guarantee (the first days do not depend on later
days) holds for all.
"""

import itertools
import json

import numpy as np
import pytest
from _object_fixtures import make_mask, recipe, seg
from test_tracking import ids, patches, tracks_of

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")

# Scientific contract and invariant.
EXPECTED_DEFAULT_ID = "sha256:5745db451acbbf0de23727a839240b9a29dfeb73c90cc814b862b98a53c3d265"
EXPECTED_TRACK_ID = "sha256:3738127dd42ae9d6b44850e642e60f633714e1d99580a6410e642c6e472e7f69"


def run(exc, *, mode="retrospective", gap=0, overlap=0.0, **rules):
    mask = make_mask(exc, unit_measure=True)
    step = xhw.RecipeStep.track(min_overlap=overlap, max_gap_days=gap, **rules)
    obj = xhw.build_objects(mask, recipe=recipe(seg(), step), analysis_mode=mode)
    obj.check_invariants()
    obj.check_against(mask)
    return obj


def column(obj, name):
    return obj.dataset[name].values.tolist()


# --- specification ----------------------------------------------------------------------------


def test_recipes_keep_the_registered_identifiers():
    assert xhw.SpatialRecipe.default().spec_id == EXPECTED_DEFAULT_ID
    tracked = xhw.SpatialRecipe(input_layer="event_membership", steps=(
        xhw.RecipeStep.segment(), xhw.RecipeStep.track(min_overlap=0.25, max_gap_days=2)))
    assert tracked.spec_id == EXPECTED_TRACK_ID


def test_a_rule_given_with_its_default_value_is_stored_as_none():
    explicit = xhw.RecipeStep.track(link_measure="overlap_smaller", dominance_measure=
                                    "overlap_smaller", tie_rule="lowest_id", branch_rule="inherit")
    assert explicit == xhw.RecipeStep.track()
    assert explicit.link_measure is None and explicit.rule("link_measure") == "overlap_smaller"
    assert explicit.spec_id == xhw.RecipeStep.track().spec_id


def test_a_non_default_rule_changes_the_identifier_and_round_trips():
    step = xhw.RecipeStep.track(link_measure="iou", dominance_measure="shared",
                                tie_rule="largest_area", branch_rule="terminate_and_create")
    assert step.spec_id != xhw.RecipeStep.track().spec_id
    assert xhw.RecipeStep.from_json(step.to_json()) == step
    assert step.rule("branch_rule") == "terminate_and_create"


def test_a_step_serialized_with_schema_0_1_reads_as_the_m2_behaviour():
    old = {"type": "RecipeStep", "schema_version": "0.1", "fields": {
        "op": "track", "footprint": None, "min_area": None, "min_duration_days": 1,
        "max_gap_days": 0, "join_rule": None, "min_overlap": 0.0}}
    assert xhw.RecipeStep.from_json(json.dumps(old)) == xhw.RecipeStep.track()


@pytest.mark.parametrize("bad", [dict(link_measure="shared"), dict(dominance_measure="area"),
                                 dict(tie_rule="random"), dict(branch_rule="split")])
def test_unknown_rule_values_are_rejected(bad):
    with pytest.raises(xhw.ValidationError):
        xhw.RecipeStep.track(**bad)


def test_the_rules_belong_to_the_track_step_only():
    with pytest.raises(xhw.ValidationError, match="does not use link_measure"):
        xhw.RecipeStep(op="segment", footprint=xhw.Footprint.connected(1),
                       min_area=xhw.MinArea.none(), link_measure="iou")


# --- link_measure -----------------------------------------------------------------------------

# 6 cells (cols 0-5) then 2 cells (cols 2-3) inside them: shared 2
INSIDE = patches(2, (0, 1, 0, 5), (1, 1, 2, 3))


@pytest.mark.parametrize("name, linked", [("overlap_smaller", True), ("iou", False),
                                          ("source_fraction", False), ("target_fraction", True)])
def test_min_overlap_thresholds_the_chosen_link_measure(name, linked):
    # overlap_smaller 2/2 = 1, iou 2/6, source_fraction 2/6, target_fraction 2/2
    obj = run(INSIDE, overlap=0.5, link_measure=name)
    assert (len(set(tracks_of(obj))) == 1) is linked


# --- dominance_measure and tie_rule -----------------------------------------------------------

# P (10 cells) splits into Q (2 cells, fully inside P) and R (10 cells, 6 of them inside P)
SPLIT_UNEQUAL = patches(2, (0, 1, 0, 9), (1, 1, 0, 1), (1, 1, 4, 13))


def test_the_dominance_measure_decides_which_child_inherits_in_monitoring():
    # overlap_smaller: Q 2/2 = 1.0 beats R 6/10; shared: R 6 beats Q 2
    default = run(SPLIT_UNEQUAL, mode="monitoring")
    assert tracks_of(default) == [1, 1, 2]
    by_shared = run(SPLIT_UNEQUAL, mode="monitoring", dominance_measure="shared")
    assert tracks_of(by_shared) == [1, 2, 1]
    assert column(by_shared, "track_parent") == [0, 1]


# P (8 cells) splits into Q (2 cells) and R (3 cells), both fully inside: overlap_smaller ties
SPLIT_TIE = patches(2, (0, 1, 0, 7), (1, 1, 0, 1), (1, 1, 4, 6))


def test_the_tie_rule_breaks_equal_dominance():
    assert tracks_of(run(SPLIT_TIE, mode="monitoring")) == [1, 1, 2]           # lowest id: Q
    assert tracks_of(run(SPLIT_TIE, mode="monitoring", tie_rule="largest_area")) == [1, 2, 1]


def test_the_dominance_measure_chooses_the_bridged_candidate():
    # A (2 cells) and B (4 cells) vanish on day 1; an 8-cell object reappears on day 2 over both:
    # overlap_smaller ties (1.0 each, lowest id A); shared prefers B (4 > 2)
    exc = patches(3, (0, 1, 0, 1), (0, 1, 4, 7), (2, 1, 0, 7))
    assert tracks_of(run(exc, gap=1)) == [1, 2, 1]
    assert tracks_of(run(exc, gap=1, dominance_measure="shared")) == [1, 2, 2]
    assert tracks_of(run(exc, gap=1, tie_rule="largest_area")) == [1, 2, 2]


# --- branch_rule ------------------------------------------------------------------------------

MERGE = patches(3, (0, 1, 0, 1), (0, 1, 4, 5), (1, 1, 0, 5), (2, 1, 0, 5))
SPLIT = patches(3, (0, 1, 0, 5), (1, 1, 0, 1), (1, 1, 4, 5), (2, 1, 0, 1), (2, 1, 4, 5))
MOVING = patches(4, (0, 1, 0, 1), (1, 1, 1, 2), (2, 1, 2, 3), (3, 1, 3, 4))


def test_terminate_and_create_gives_every_child_of_a_split_a_new_identity():
    obj = run(SPLIT, mode="monitoring", branch_rule="terminate_and_create")
    assert tracks_of(obj) == [1, 2, 3, 2, 3]
    assert column(obj, "track_parent") == [0, 1, 1]
    assert column(obj, "track_merged_into") == [0, 0, 0]


def test_terminate_and_create_opens_a_new_identity_on_a_merge():
    obj = run(MERGE, mode="monitoring", branch_rule="terminate_and_create")
    assert tracks_of(obj) == [1, 2, 3, 3]
    assert column(obj, "track_merged_into") == [3, 3, 0]
    assert column(obj, "track_parent") == [0, 0, 0]


def test_terminate_and_create_keeps_the_identity_of_a_continuation():
    obj = run(MOVING, mode="monitoring", branch_rule="terminate_and_create")
    assert tracks_of(obj) == [1, 1, 1, 1]


@pytest.mark.parametrize("rules", [dict(dominance_measure="shared"), dict(tie_rule="largest_area"),
                                   dict(branch_rule="terminate_and_create")])
def test_without_gaps_the_monitoring_rules_do_not_change_retrospective_tracks(rules):
    rng = np.random.default_rng(5)
    exc = rng.random((8, 6, 8)) < 0.4
    assert np.array_equal(ids(run(exc)), ids(run(exc, **rules)))


def test_the_branch_rule_never_changes_retrospective_tracks_even_with_gaps():
    # (with gaps the dominance and the tie rule choose the bridged candidate in both modes)
    rng = np.random.default_rng(6)
    exc = rng.random((10, 6, 8)) < 0.3
    assert np.array_equal(ids(run(exc, gap=2)),
                          ids(run(exc, gap=2, branch_rule="terminate_and_create")))


RULES = [dict(zip(("link_measure", "dominance_measure", "tie_rule", "branch_rule"), combo))
         for combo in itertools.product(("overlap_smaller", "iou"), ("overlap_smaller", "shared"),
                                        ("lowest_id", "largest_area"),
                                        ("inherit", "terminate_and_create"))]


@pytest.mark.parametrize("rules", RULES)
def test_monitoring_ids_of_the_first_days_do_not_depend_on_later_days_with_every_rule(rules):
    rng = np.random.default_rng(len(str(rules)))
    exc = rng.random((10, 6, 8)) < 0.35
    full = run(exc, mode="monitoring", gap=2, overlap=0.2, **rules)
    cut = run(exc[:6], mode="monitoring", gap=2, overlap=0.2, **rules)
    assert np.array_equal(ids(full)[:6], ids(cut))
