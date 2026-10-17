# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""object tracking specs: Footprint, MinArea, RecipeStep and SpatialRecipe."""

import pytest

import xhw


def seg(**overrides):
    params = dict(op="segment", footprint=xhw.Footprint.connected(1),
                  min_area=xhw.MinArea.default())
    params.update(overrides)
    return xhw.RecipeStep(**params)


def test_default_recipe_is_exceedance_segment_persist_with_five_percent_of_the_domain():
    recipe = xhw.SpatialRecipe.default()
    assert recipe.input_layer == "exceedance"
    step, persist = recipe.steps
    assert persist == xhw.RecipeStep.persist()       # inherits the definition's persistence
    assert step.op == "segment"
    assert step.footprint == xhw.Footprint(kind="connectivity", connectivity=1)
    assert step.min_area == xhw.MinArea(kind="domain_fraction", value=5.0)
    assert not recipe.has_track and recipe.has_segment


def test_recipe_round_trips_and_its_id_depends_on_the_order_of_the_steps():
    close = xhw.RecipeStep.morph("close", xhw.Footprint.disc(30.0))
    persist = xhw.RecipeStep.persist(max_gap_days=2)
    track = xhw.RecipeStep.track(min_overlap=0.1, max_gap_days=1, min_duration_days=3)
    area = xhw.MinArea(kind="percentile", value=5.0, floor_km2=500.0)
    steps = (persist, close, xhw.RecipeStep.segment(min_area=area), track)
    full = xhw.SpatialRecipe(input_layer="event_membership", steps=steps)
    assert xhw.SpatialRecipe.from_json(full.to_json()) == full
    assert full.spec_id == xhw.SpatialRecipe.from_dict(full.to_dict()).spec_id
    swapped = xhw.SpatialRecipe(input_layer="event_membership",
                                steps=(close, persist, *steps[2:]))
    assert swapped.spec_id != full.spec_id        # the filters do not commute
    assert full.has_track


@pytest.mark.parametrize("build", [
    lambda: xhw.Footprint(kind="connectivity"),
    lambda: xhw.Footprint(kind="connectivity", connectivity=4),
    lambda: xhw.Footprint(kind="connectivity", connectivity=1, radius_km=5.0),
    lambda: xhw.Footprint(kind="radius", radius_km=0.0),
    lambda: xhw.Footprint.explicit([(1, 0)]),                    # not closed under negation
    lambda: xhw.Footprint.explicit([(0, 0)]),                    # the centre is implicit
    lambda: xhw.Footprint.explicit([]),
    lambda: xhw.MinArea(kind="percentile"),
    lambda: xhw.MinArea(kind="percentile", value=100.0),
    lambda: xhw.MinArea(kind="domain_fraction", value=0.0),
    lambda: xhw.MinArea(kind="domain_fraction", value=101.0),
    lambda: xhw.MinArea(kind="absolute_km2", value=-1.0),
    lambda: xhw.MinArea(kind="none", value=1.0),
    lambda: xhw.MinArea(kind="domain_fraction", value=5.0, floor_km2=10.0),
    lambda: xhw.RecipeStep(op="open"),
    lambda: xhw.RecipeStep(op="track", min_overlap=0.0, max_gap_days=0),
    lambda: xhw.RecipeStep(op="track", min_overlap=1.5, max_gap_days=0, min_duration_days=1),
    lambda: xhw.RecipeStep(op="persist", footprint=xhw.Footprint.connected(1)),
    lambda: seg(min_overlap=0.5),
    lambda: xhw.RecipeStep(op="persist", min_duration_days=0),
    lambda: xhw.RecipeStep(op="persist", max_gap_days=-1),
    lambda: xhw.SpatialRecipe(input_layer="exceedance", steps=()),
    lambda: xhw.SpatialRecipe(input_layer="threshold", steps=(seg(),)),
    lambda: xhw.SpatialRecipe(input_layer="exceedance", steps=(xhw.RecipeStep.track(),)),
    lambda: xhw.SpatialRecipe(input_layer="exceedance", steps=(
        seg(), xhw.RecipeStep.track(), seg())),                  # track must be last
    lambda: xhw.SpatialRecipe(input_layer="exceedance", steps=(
        seg(), xhw.RecipeStep.persist(), xhw.RecipeStep.track())),   # and follow a segment
])
def test_invalid_specs_are_rejected(build):
    with pytest.raises(xhw.ValidationError):
        build()


def test_a_persist_step_may_inherit_everything_from_the_definition():
    step = xhw.RecipeStep.persist()
    assert (step.min_duration_days, step.max_gap_days, step.join_rule) == (None, None, None)


def test_offsets_may_be_listed_in_any_order_if_symmetric():
    fp = xhw.Footprint.explicit([(0, 1), (1, 0), (0, -1), (-1, 0)])
    assert fp.kind == "offsets" and len(fp.offsets) == 4
