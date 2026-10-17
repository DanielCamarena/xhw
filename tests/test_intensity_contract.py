# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Independent regressions for field identity and finite observation support."""

import numpy as np
import pytest
from _fixtures import baseline
from _lineage_fixtures import objects
from _object_fixtures import make_line_mask, make_mask, make_volume_mask

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")


@pytest.mark.parametrize("support", ["line", "surface", "curvilinear", "volume", "sigma"])
def test_field_identity_is_checked_before_values_are_used(support):
    exc = np.ones((2, 2, 2), bool)
    mask = {"line": lambda: make_line_mask(np.ones((2, 2), bool)),
            "surface": lambda: make_mask(exc),
            "curvilinear": lambda: make_mask(exc, curvilinear=True),
            "volume": lambda: make_volume_mask(np.ones((2, 2, 2, 2), bool)),
            "sigma": lambda: make_volume_mask(np.ones((2, 2, 2, 2), bool), sigma=(
                np.full((2, 2, 2), 10.), np.broadcast_to(
                    np.array([5., 15.])[:, None, None], (2, 2, 2))))}[support]()
    obj = objects(mask.dataset.exceedance.values, mask=mask, track=False)
    field = mask.intensity()
    graph = xhw.build_lineage(obj, field=field.transpose(*reversed(field.dims)))
    graph.check_invariants()
    dim = mask.grid.spatial_dims[-1]
    with pytest.raises(xhw.ValidationError, match="coordinate.*differs"):
        xhw.build_lineage(obj, field=field.isel({dim: slice(None, None, -1)}))
    role = mask.grid.coords.along if support == "line" else mask.grid.coords.lon
    with pytest.raises(xhw.ValidationError, match="coordinate.*missing"):
        xhw.build_lineage(obj, field=field.drop_vars(role))
    with pytest.raises(xhw.ValidationError, match="size"):
        xhw.build_lineage(obj, field=field.isel({dim: slice(0, 1)}))
    with pytest.raises(xhw.ValidationError, match="coordinate.*differs"):
        graph.maps(obj, spacetime_event=1, field=field.assign_coords({role: field[role] + 1}))


@pytest.mark.parametrize("weighted", [True, False])
def test_missing_days_are_not_zero_or_part_of_the_mean_denominator(weighted):
    exc = np.ones((2, 1, 1), bool)
    mask = make_mask(exc, unit_measure=True)
    obj = objects(exc, mask=mask, track=False)
    field = mask.intensity().copy(data=np.array([2., np.nan]).reshape(exc.shape))
    graph = xhw.build_lineage(obj, field=field, weighted=weighted)
    graph.check_invariants()
    graph.check_against(obj)
    assert graph.dataset.event_intensity_mean.item() == 2
    assert graph.maps(obj, spacetime_event=1, field=field).intensity_mean.item() == 2
    np.testing.assert_equal(graph.series(spacetime_event=1).intensity_mean.values, [2, np.nan])
    again = xhw.LineageGraph.from_dataset(graph.to_dataset())
    np.testing.assert_equal(again.series(spacetime_event=1).intensity_mean.values, [2, np.nan])


@pytest.mark.parametrize("weighted", [True, False])
def test_daily_mean_uses_valid_support_across_unequal_patches(weighted):
    # Two separate patches on day 1 join on day 2: three cells vs one,
    # with a missing observation inside the larger patch.
    exc = np.array([[[1, 1, 1, 0, 1]], [[1, 1, 1, 1, 1]]], bool)
    mask = make_mask(exc, unit_measure=True)
    obj = objects(exc, mask=mask, track=False)
    field = mask.intensity().copy(data=np.array([[[2, np.nan, 4, 0, 10]],
                                                [[0, 0, 0, 0, 0]]], float))
    graph = xhw.build_lineage(obj, field=field, weighted=weighted)
    graph.check_invariants()
    np.testing.assert_allclose(graph.series(spacetime_event=1).intensity_mean, [16/3, 0])
    assert graph.dataset.event_intensity_mean.item() == pytest.approx(16/8)


def test_valid_measure_controls_weighted_mean_with_nonuniform_cells():
    exc = np.ones((1, 2, 2), bool)
    mask = make_mask(exc, lat0=10., step=30.)
    obj = objects(exc, mask=mask, track=False)
    field = mask.intensity().copy(data=np.array([[[2, np.nan], [8, 12]]], float))
    area = mask.dataset.cell_area.values
    expected = (2*area[0, 0] + 8*area[1, 0] + 12*area[1, 1]) / (
        area[0, 0] + area[1, 0] + area[1, 1])
    graph = xhw.build_lineage(obj, field=field)
    assert graph.series(spacetime_event=1).intensity_mean.item() == pytest.approx(expected)
    assert graph.dataset.object_intensity_valid_cells.item() == 3


def test_positive_small_measure_is_not_replaced_by_a_denominator_floor():
    mask = make_mask(np.ones((1, 1, 1), bool))
    ds = mask.to_dataset()
    ds["cell_area"] = ds.cell_area.copy(data=np.full((1, 1), 1e-305))
    mask = xhw.XMask.from_dataset(ds)
    obj = objects(mask.dataset.exceedance.values, mask=mask, track=False)
    graph = xhw.build_lineage(obj, field=mask.intensity())
    assert graph.series(spacetime_event=1).intensity_mean.item() == pytest.approx(3.0)


def test_no_finite_intensity_is_missing_but_occupancy_is_preserved():
    exc = np.ones((2, 1, 1), bool)
    mask = make_mask(exc, unit_measure=True)
    obj = objects(exc, mask=mask, track=False)
    field = mask.intensity().copy(data=np.full(exc.shape, np.nan))
    graph = xhw.build_lineage(obj, field=field)
    graph.check_invariants()
    for suffix in ("mean", "max", "cumulative"):
        assert np.isnan(graph.dataset[f"event_intensity_{suffix}"].item())
        assert np.isnan(graph.dataset[f"object_intensity_{suffix}"].values).all()
    maps = graph.maps(obj, spacetime_event=1, field=field)
    assert maps.frequency.item() == 2
    assert np.isnan(maps.intensity_mean.item())
    assert np.isnan(maps.intensity_cumulative.item())
    old = graph.to_dataset().drop_vars("object_intensity_valid_measure")
    with pytest.raises(xhw.SchemaError, match="missing required intensity support variable"):
        xhw.LineageGraph.from_dataset(old)


def test_corrupted_valid_support_fails_invariants():
    exc = np.ones((1, 1, 1), bool)
    mask = make_mask(exc, unit_measure=True)
    graph = xhw.build_lineage(objects(exc, mask=mask, track=False), field=mask.intensity())
    ds = graph.to_dataset().copy(deep=True)
    ds.object_intensity_valid_cells.values[:] = 2
    with pytest.raises(xhw.InvariantError, match="count"):
        xhw.LineageGraph.from_dataset(ds).check_invariants()


def test_missing_required_serialized_fields_report_entity_and_nested_path():
    source = baseline().source.to_dict()
    del source['fields']['dataset_id']
    with pytest.raises(xhw.SchemaError, match="SourceRef.*dataset_id"):
        xhw.SourceRef.from_dict(source)
    spec = baseline().to_dict()
    del spec['fields']['source']['dataset_id']
    with pytest.raises(xhw.SchemaError, match=r"BaselineSpec.source:.*SourceRef.*dataset_id"):
        xhw.BaselineSpec.from_dict(spec)


def test_derived_intensity_preserves_physical_roles_stored_as_variables():
    mask = make_mask(np.ones((2, 2, 2), bool), unit_measure=True)
    mask = xhw.XMask.from_dataset(mask.to_dataset().reset_coords(['lat', 'lon']))
    obj = objects(mask.dataset.exceedance.values, mask=mask)
    field = mask.intensity()
    assert 'lat' in field.coords and 'lon' in field.coords
    xhw.build_lineage(obj, field=field).check_invariants()


def test_algebraic_maximum_and_explicit_cold_severity_have_different_peak_days():
    mask = make_mask(np.ones((2, 1, 1), bool), unit_measure=True)
    obj = objects(mask.dataset.exceedance.values, mask=mask)
    signed = mask.intensity().copy(data=np.array([-2., -6.]).reshape(2, 1, 1))
    anomaly = xhw.build_lineage(obj, field=signed)
    cold = (-signed).assign_attrs(signed.attrs, long_name='positive cold severity')
    severity = xhw.build_lineage(obj, field=cold)
    assert anomaly.dataset.event_intensity_max.item() == -2
    assert severity.dataset.event_intensity_max.item() == 6
    assert severity.dataset.event_intensity_max.attrs['units'] == signed.attrs['units']
    assert anomaly.dataset.event_peak_intensity_day.values[0] == np.datetime64('2000-01-01')
    assert severity.dataset.event_peak_intensity_day.values[0] == np.datetime64('2000-01-02')
