# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""genealogy: metrics, series, maps, intensity and the EventCatalog.

The hand-written numbers (events A and B that join in M) and are checked exactly; the
footprint is never the sum of the branches; the monitoring metrics of a joined event end equal
to the retrospective ones; intensity is off by default and its spatial mean is weighted unless
asked.
"""

import numpy as np
import pytest
import xarray as xr
from _lineage_fixtures import JOIN, NO_TIE, column, lay, lineage, objects
from _object_fixtures import make_mask

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
DAY3 = np.datetime64("2000-01-03")


def days(graph, name):
    return graph.dataset[name].values.astype("datetime64[D]").tolist()


# --- level 1 ---------------------------------------------------------------------------------


def test_joined_events_with_the_default_rule_reproduce_the_documented_case():
    graph = lineage(JOIN, mode="monitoring")                     # events A, B and the new C
    assert column(graph, "event_area_days") == [180.0, 120.0, 540.0]
    assert column(graph, "event_area_max") == [70.0, 50.0, 120.0]
    assert days(graph, "event_peak_area_day") == [DAY3] * 3     # C: 120 on day 3 (A 70 + B 50)
    assert column(graph, "event_footprint") == [70.0, 50.0, 130.0]


def test_joined_events_with_the_survivor_rule_recompute_the_survivor():
    graph = lineage(JOIN, mode="monitoring", merge_rule="survivor")
    assert column(graph, "event_area_days") == [540.0, 120.0]   # A over the union, B frozen
    assert column(graph, "event_area_max") == [120.0, 50.0]
    assert column(graph, "event_merged_into") == [0, 1]


def test_at_the_end_the_joined_event_has_the_retrospective_metrics():
    live = lineage(JOIN, mode="monitoring")
    retro = lineage(JOIN)
    for name in ("area_days", "area_max", "footprint", "net_displacement", "path_length",
                 "duration_days", "n_objects", "n_splits", "n_merges"):
        assert column(live, f"event_{name}")[-1] == pytest.approx(column(retro, f"event_{name}")[0])


def test_size_days_add_up_over_branches_but_footprints_do_not():
    graph = lineage(NO_TIE)
    assert column(graph, "branch_area_days") == [180.0, 60.0, 30.0, 140.0]
    assert column(graph, "event_area_days") == [410.0]                       # 180 + 60 + 30 + 140
    branches = column(graph, "branch_footprint")
    assert column(graph, "event_footprint") == [145.0] and sum(branches) == 275.0


def test_displacement_and_path_of_a_patch_that_moves_east_and_back():
    # 2 cells on row 1 (latitude -9): moves 3 cells east, then 1 back
    exc = lay(5, {0: [(1, 0, 2)], 1: [(1, 1, 3)], 2: [(1, 2, 4)], 3: [(1, 3, 5)],
                  4: [(1, 2, 4)]}, nx=8)
    graph = lineage(exc)
    s = graph.series(spacetime_event=1)
    lat, lon = np.deg2rad(s["centroid_lat"].values), np.deg2rad(s["centroid_lon"].values)

    def gc(i, j):
        h = (np.sin((lat[j] - lat[i]) / 2) ** 2
             + np.cos(lat[i]) * np.cos(lat[j]) * np.sin((lon[j] - lon[i]) / 2) ** 2)
        return 2 * 6_371_000.0 * np.arcsin(np.sqrt(h))

    assert column(graph, "event_path_length")[0] == pytest.approx(
        sum(gc(i, i + 1) for i in range(4)), rel=1e-12)
    assert column(graph, "event_net_displacement")[0] == pytest.approx(gc(0, 4), rel=1e-12)
    # one degree of longitude at latitude -9 (the spherical centroid sits a hair poleward)
    one = 2 * 6_371_000.0 * np.arcsin(np.cos(np.deg2rad(-9.0)) * np.sin(np.deg2rad(0.5)))
    assert column(graph, "event_path_length")[0] == pytest.approx(4 * one, rel=1e-5)
    assert column(graph, "event_net_displacement")[0] == pytest.approx(2 * one, rel=1e-5)


def test_branch_and_event_censoring_and_provisional_flags():
    graph = lineage(JOIN, mode="monitoring", gap=1)
    assert column(graph, "event_left_censored") == [True, True, True]
    assert column(graph, "event_right_censored") == [False, False, True]
    assert column(graph, "event_provisional") == [False, False, True]
    assert lineage(JOIN).dataset["event_provisional"].values.tolist() == [False]


# --- level 2 and level 3 ---------------------------------------------------------------------


def test_the_series_of_an_event_is_its_daily_size_and_centroid():
    graph = lineage(JOIN, mode="monitoring")
    series = graph.series(spacetime_event=3)
    assert series["area"].values.tolist() == [80.0, 100.0, 120.0, 100.0, 80.0, 60.0]
    assert series["n_objects"].values.tolist() == [2, 2, 2, 1, 1, 1]
    assert "centroid_lat" in series and "intensity_mean" not in series
    assert graph.series(spacetime_event=1)["area"].values.tolist() == [50.0, 60.0, 70.0]
    assert graph.series(branch=1)["area"].values.tolist() == [50.0, 60.0, 70.0]


def test_the_maps_of_an_event():
    obj = objects(JOIN, mode="monitoring")
    graph = xhw.build_lineage(obj)
    maps = graph.maps(obj, spacetime_event=3)
    assert int(maps["frequency"].sum()) == 540                  # unit cells: the size-days
    assert int((maps["frequency"] > 0).sum()) == 130            # the footprint
    assert maps.attrs["peak_area_day"] == "2000-01-03"
    assert int(maps["peak_area"].sum()) == 120
    first = maps["first_day"].values
    assert first[0, 0] == np.datetime64("2000-01-01") and np.isnat(first[1, 50])
    assert maps["last_day"].values[0, 0] == np.datetime64("2000-01-06")
    with pytest.raises(xhw.ValidationError, match="exactly one"):
        graph.maps(obj)
    with pytest.raises(xhw.ValidationError, match="built from"):
        graph.maps(objects(JOIN), spacetime_event=1)


# --- intensity -------------------------------------------------------------------------------


def field_like(mask, values):
    ds = mask.dataset
    return xr.DataArray(values, dims=("time", "y", "x"), coords=ds["temperature"].coords,
                        attrs={"units": "K"})


def test_intensity_is_off_by_default_and_exact_when_given():
    mask = make_mask(JOIN, unit_measure=True)
    obj = objects(JOIN, mask=mask)
    assert "event_intensity_max" not in xhw.build_lineage(obj).dataset
    values = np.zeros(JOIN.shape)
    values[:] = (np.arange(JOIN.shape[0]) + 1.0)[:, None, None]     # day d has intensity d + 1
    values[2, 0, :] = 9.0                                            # A on day 2 is hotter
    graph = xhw.build_lineage(obj, field=field_like(mask, values))
    graph.check_invariants()
    ds = graph.dataset
    assert ds["event_intensity_max"].values.tolist() == [9.0]
    assert days(graph, "event_peak_intensity_day") == [DAY3]
    # cumulative: sum of value x cell over the 540 cell-days
    area_by_day = np.array([80, 100, 120, 100, 80, 60], float)
    expect = float((area_by_day * (np.arange(6) + 1)).sum() + 70 * (9.0 - 3.0))
    assert ds["event_intensity_cumulative"].values.tolist() == [pytest.approx(expect)]
    assert ds["event_intensity_mean"].values.tolist() == [pytest.approx(expect / 540)]
    assert ds["event_intensity_cumulative"].attrs["units"] == "K 1 day"


def test_the_spatial_mean_is_weighted_by_cell_area_unless_disabled():
    # two cells of one patch at latitudes -10 and +50: the equatorward cell weighs more
    exc = np.zeros((1, 61, 1), bool)
    exc[0, :, 0] = True
    mask = make_mask(exc, lat0=-10.0)
    obj = objects(exc, mask=mask, track=False)
    values = np.zeros(exc.shape)
    values[0, :, 0] = np.arange(61.0)                          # warmer towards the pole
    weighted = xhw.build_lineage(obj, field=field_like(mask, values))
    plain = xhw.build_lineage(obj, field=field_like(mask, values), weighted=False)
    w_mean = weighted.dataset["event_intensity_mean"].values[0]
    p_mean = plain.dataset["event_intensity_mean"].values[0]
    assert p_mean == pytest.approx(30.0)
    assert w_mean < p_mean                                     # high latitudes have smaller cells
    area = mask.dataset["cell_area"].values[:, 0]
    assert w_mean == pytest.approx(float((area * np.arange(61.0)).sum() / area.sum()))
    assert "unweighted" in plain.dataset["event_intensity_mean"].attrs["cell_methods"]


def test_the_intensity_maps_and_series():
    mask = make_mask(JOIN, unit_measure=True)
    obj = objects(JOIN, mask=mask)
    values = np.broadcast_to((np.arange(6) + 1.0)[:, None, None], JOIN.shape).copy()
    field = field_like(mask, values)
    graph = xhw.build_lineage(obj, field=field)
    maps = graph.maps(obj, spacetime_event=1, field=field)
    assert maps["intensity_max"].values[0, 0] == 6.0             # cell (0, 0) occupied every day
    assert maps["intensity_cumulative"].values[0, 0] == 21.0     # 1 + 2 +... + 6
    assert maps["day_of_max_intensity"].values[0, 0] == np.datetime64("2000-01-06")
    assert maps.attrs["peak_intensity_day"] == "2000-01-06"
    series = graph.series(spacetime_event=1)
    assert series["intensity_max"].values.tolist() == [1, 2, 3, 4, 5, 6]


def test_a_field_on_other_days_is_rejected():
    mask = make_mask(JOIN, unit_measure=True)
    obj = objects(JOIN, mask=mask)
    field = field_like(mask, np.zeros(JOIN.shape)).isel(time=slice(0, 3))
    with pytest.raises(xhw.ValidationError, match="dimension|days"):
        xhw.build_lineage(obj, field=field)


# --- EventCatalog ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["retrospective", "monitoring"])
def test_the_events_as_an_object_tracked_catalog(mode):
    mask = make_mask(JOIN, unit_measure=True)
    obj = objects(JOIN, mask=mask, mode=mode)
    graph = xhw.build_lineage(obj)
    catalog = graph.to_event_catalog(mask)
    assert catalog.granularity == "object_tracked" and catalog.location_dims == ()
    catalog.check_against(mask)
    graph.check_catalog(catalog)
    again = xhw.EventCatalog.from_dataset(catalog.to_dataset())
    graph.check_catalog(again)
    rows = catalog.dataset
    if mode == "monitoring":                       # A and B overlap C in time: allowed here
        assert rows["event_id"].values.tolist() == [1, 2, 3]
        assert rows["merged_into"].values.tolist() == [3, 3, 0]
        assert rows["area_days"].values.tolist() == [180.0, 120.0, 540.0]
    assert rows["censor_reason"].values.tolist()[-1] == 3        # window start and end
    tampered = catalog.to_dataset().copy(deep=True)
    tampered["area_days"].values[0] = 1.0
    with pytest.raises(xhw.InvariantError, match="area_days"):
        graph.check_catalog(xhw.EventCatalog.from_dataset(tampered))


def test_the_catalog_needs_the_source_mask():
    graph = lineage(JOIN)
    with pytest.raises(xhw.ValidationError, match="source"):
        graph.to_event_catalog(make_mask(JOIN, unit_measure=True))


@pytest.mark.parametrize("rule", ["new_event", "survivor"])
@pytest.mark.parametrize("seed", range(3))
def test_the_footprint_propagated_through_joins_is_the_union_of_all_member_cells(seed, rule):
    # many joins in monitoring: the footprint is propagated up the merge forest, not expanded
    rng = np.random.default_rng(40 + seed)
    exc = rng.random((12, 6, 9)) < 0.45
    obj = objects(exc, mode="monitoring")
    graph = xhw.build_lineage(obj, merge_rule=rule)
    graph.check_invariants()
    assert (graph.dataset["event_merged_into"].values > 0).any()
    for e, value in enumerate(column(graph, "event_footprint"), start=1):
        maps = graph.maps(obj, spacetime_event=e)
        assert value == float((maps["frequency"] > 0).sum())        # unit cells
