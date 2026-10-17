# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""genealogy on every support of structured objects, and the graph against independent
references.

- The same abstract case (runs of cells along one axis) on a line, a surface row and a
  vertical column of a volume gives the same graph: node types, links, branches, events and
  measures.
- A patch across the periodic edge gives the graph of the same field rolled in longitude.
- The edges equal a brute-force count of shared cells; curvilinear and static sigma grids give
  the graph of their regular counterpart; physical vertical connectivity patches are followed
  as they are.
- The inheritance of object tracking monitoring is rebuilt from the dominance annotation.
"""

import numpy as np
import pytest
from _lineage_fixtures import column, edges
from _object_fixtures import (
    make_line_mask,
    make_mask,
    make_volume_mask,
    recipe,
    rotated_coordinates,
)

import xhw

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
N = 12

#: abstract cases: per day, half-open runs [a, b) of cells along one axis
CASES = {
    "1->1": [[(0, 3)], [(1, 4)]],
    "split": [[(0, 8)], [(0, 3), (5, 8)]],
    "merge": [[(0, 3), (5, 8)], [(0, 8)]],
    "complex": [[(0, 3), (5, 8)], [(0, 8)], [(0, 3), (5, 8)]],
    "cycle": [[(0, 8)], [(0, 3), (5, 8)], [(0, 8)]],
    "gap": [[(0, 3)], [], [(1, 4)]],
    "isolated": [[(0, 3)], [(6, 9)]],
}


def runs(case):
    out = np.zeros((len(case), N), bool)
    for t, day in enumerate(case):
        for a, b in day:
            out[t, a:b] = True
    return out


def build(mask, *, gap=0, mode="retrospective", footprint=None):
    step = xhw.RecipeStep.segment(footprint=footprint or xhw.Footprint.connected(1),
                                  min_area=xhw.MinArea.none())
    obj = xhw.build_objects(mask, recipe=recipe(step, xhw.RecipeStep.track(max_gap_days=gap)),
                            analysis_mode=mode)
    obj.check_invariants()
    graph = xhw.build_lineage(obj)
    graph.check_invariants()
    graph.check_against(obj)
    return obj, graph


def on_support(support, line_runs):
    if support == "line":
        return make_line_mask(line_runs)
    if support == "surface":
        return make_mask(line_runs[:, None, :], unit_measure=True)
    # a vertical column: the runs are along z, layers of 10 m
    return make_volume_mask(line_runs[:, :, None, None], dz=(10.0,) * N,
                            depth=tuple(5.0 + 10.0 * np.arange(N)))


TOPOLOGY = ("node_type", "is_split", "is_merge", "object_branch", "object_spacetime_event")


@pytest.mark.parametrize("name", list(CASES))
def test_the_same_case_gives_the_same_graph_on_a_line_a_surface_and_a_volume(name):
    exc = runs(CASES[name])
    graphs = {s: build(on_support(s, exc), gap=1)[1] for s in ("line", "surface", "volume")}
    ref = graphs["surface"]
    for support, graph in graphs.items():
        for var in TOPOLOGY:
            assert column(graph, var) == column(ref, var), (support, var)
        assert edges(graph, linked_only=False) == edges(ref, linked_only=False), support
        for var in ("overlap_smaller", "iou", "source_fraction", "target_fraction", "linked"):
            np.testing.assert_allclose(graph.transitions()[var].values,
                                       ref.transitions()[var].values, err_msg=f"{support} {var}")


def test_a_patch_across_the_periodic_edge_gives_the_graph_of_the_rolled_field():
    exc = np.zeros((3, 3, 12), bool)
    exc[0, 1, [10, 11, 0, 1]] = True                   # one patch across longitude 0
    exc[1, 1, [11, 0, 1, 2]] = True
    exc[2, 1, [11, 0]] = True
    exc[2, 1, [2, 3]] = False
    obj, graph = build(make_mask(exc, periodic=True))
    assert obj.dataset["object_time"].size == 3      # one patch a day: the edge is joined
    _, rolled = build(make_mask(np.roll(exc, 6, axis=2), periodic=True))
    for var in TOPOLOGY:
        assert column(graph, var) == column(rolled, var)
    np.testing.assert_allclose(graph.transitions()["shared"].values,
                               rolled.transitions()["shared"].values)
    np.testing.assert_allclose(column(graph, "event_path_length"),
                               column(rolled, "event_path_length"), rtol=1e-9)


def brute_force(obj):
    """(source, target, shared) of every pair of patches of consecutive days, by plain loops."""
    ds = obj.dataset
    g = obj.grid
    label = np.asarray(ds["label"].transpose(g.time_dim, *g.spatial_dims).values)
    w = (np.asarray(ds[g.cell_measure.var].values, float) if g.cell_measure.kind == "area"
         else np.ones(label.shape[1:]))
    day = np.searchsorted(ds["time"].values, ds["object_time"].values)
    number = ds["object_label"].values
    index = {(int(d), int(n)): i for i, (d, n) in enumerate(zip(day, number))}
    out = []
    for t in range(label.shape[0] - 1):
        for a in range(1, int(label[t].max()) + 1):
            for b in range(1, int(label[t + 1].max()) + 1):
                both = (label[t] == a) & (label[t + 1] == b)
                if both.any():
                    out.append((index[(t, a)], index[(t + 1, b)], float(w[both].sum())))
    return sorted(out)


@pytest.mark.parametrize("seed", range(4))
def test_the_edges_are_a_brute_force_count_of_shared_cells(seed):
    rng = np.random.default_rng(seed)
    exc = rng.random((6, 7, 9)) < 0.4
    obj, graph = build(make_mask(exc))                   # cell areas vary with latitude
    t = graph.transitions()
    got = sorted(zip(t["source_object"].values.tolist(), t["target_object"].values.tolist(),
                     t["shared"].values.tolist()))
    want = brute_force(obj)
    assert [g[:2] for g in got] == [w[:2] for w in want]
    np.testing.assert_allclose([g[2] for g in got], [w[2] for w in want], rtol=1e-12)


def test_a_curvilinear_grid_gives_the_graph_of_its_regular_counterpart():
    rng = np.random.default_rng(9)
    exc = rng.random((6, 5, 7)) < 0.4
    _, regular = build(make_mask(exc))
    _, curvi = build(make_mask(exc, curvilinear=True))
    for var in (*TOPOLOGY, "event_area_days", "event_footprint", "event_path_length"):
        np.testing.assert_allclose(column(curvi, var), column(regular, var), rtol=1e-12)
    _, rotated = build(make_mask(exc, coords2d=rotated_coordinates(5, 7)))
    for var in TOPOLOGY:                           # the topology does not depend on coordinates
        assert column(rotated, var) == column(regular, var)


def test_a_static_sigma_volume():
    rng = np.random.default_rng(4)
    exc = rng.random((5, 3, 4, 5)) < 0.4
    thickness = rng.uniform(2.0, 20.0, size=(3, 4, 5))
    depth = np.cumsum(thickness, axis=0) - thickness / 2
    _, graph = build(make_volume_mask(exc, sigma=(thickness, depth)))
    ds = graph.dataset
    assert ds["event_area_days"].attrs["units"] == "m3 day"
    assert (ds["event_depth_min"].values <= ds["event_depth_max"].values).all()
    assert (ds["event_depth_min"].values >= depth.min() - 1e-9).all()


def test_m2b_patches_are_followed_as_m2_forms_them():
    # layers 1, 1, 1, 100, 100 m: a 1 m empty layer is bridged by vertical_gap_m = 1
    uneven = dict(dz=(1.0, 1.0, 1.0, 100.0, 100.0), depth=(0.5, 1.5, 2.5, 53.0, 153.0))
    exc = np.zeros((3, 5, 3, 3), bool)
    exc[0, 0, 1, 1] = exc[0, 2, 1, 1] = True          # two cells with 1 m of water between
    exc[1, 2, 1, 1] = True
    exc[2, 2, 1, 1] = exc[2, 0, 1, 1] = True
    mask = make_volume_mask(exc, **uneven)
    obj, graph = build(mask, footprint=xhw.Footprint.connected(1, vertical_gap_m=1.0))
    assert obj.dataset["object_time"].size == 3       # one patch a day (gap bridged)
    assert column(graph, "node_type") == [1, 3, 2]    # birth, continuation, death
    obj0, graph0 = build(mask)                        # index connectivity: the level-0 cells
    assert obj0.dataset["object_time"].size == 5      # of days 0 and 2 are patches of their own
    assert not any(column(graph0, "is_merge"))
    assert sorted(column(graph0, "node_type")) == [0, 0, 1, 2, 3]   # 2 isolated, birth,...


# --- metrics on lines and volumes ------------------------------------------------------------


def test_a_column_that_deepens_has_a_vertical_displacement_and_no_horizontal_one():
    exc = np.zeros((3, 6, 1, 1), bool)
    exc[0, 0:2] = exc[1, 1:3] = exc[2, 2:4] = True      # centroid 10 m, 20 m, 30 m
    mask = make_volume_mask(exc, dz=(10.0,) * 6, depth=tuple(5.0 + 10.0 * np.arange(6)))
    _, graph = build(mask)
    assert column(graph, "event_net_vertical_displacement") == pytest.approx([20.0])
    assert column(graph, "event_vertical_path_length") == pytest.approx([20.0])
    assert column(graph, "event_net_displacement") == pytest.approx([0.0], abs=1e-6)
    assert column(graph, "event_depth_min") == [5.0] and column(graph, "event_depth_max") == [35.0]
    area = float(mask.dataset["cell_area"].values[0, 0])
    assert column(graph, "event_footprint_horizontal") == pytest.approx([area])
    assert column(graph, "event_footprint") == pytest.approx([4 * 10.0 * area])


def test_a_segment_that_moves_along_a_line():
    exc = runs([[(0, 2)], [(1, 3)], [(2, 4)], [(1, 3)]])   # centroids 500, 1500, 2500, 1500 m
    _, graph = build(make_line_mask(exc))
    assert column(graph, "event_net_displacement") == [1000.0]
    assert column(graph, "event_path_length") == [3000.0]
    assert graph.dataset["event_path_length"].attrs["units"] == "m"
    assert column(graph, "event_footprint") == [4000.0]


# --- the monitoring identities of object tracking from the dominance annotation
# ---------------------


def rebuild_m2_monitoring(graph):
    """track_id of object tracking monitoring from the transition table: a node inherits from the
    parents whose dominant child it is, the best of them by overlap (ties: lowest track, then
    node); otherwise it opens a new track.
    """
    t = graph.transitions()
    n = graph.dataset.sizes["object"]
    keep = t["linked"].values
    src, dst = t["source_object"].values[keep], t["target_object"].values[keep]
    score = t["overlap_smaller"].values[keep]
    heir = t["is_dominant_for_source"].values[keep]
    track, count = np.zeros(n, int), 0
    for obj in range(n):
        chosen = [(score[i], int(src[i])) for i in np.flatnonzero((dst == obj) & heir)]
        if chosen:
            best = min(chosen, key=lambda so: (-so[0], track[so[1]], so[1]))
            track[obj] = track[best[1]]
        else:
            count += 1
            track[obj] = count
    return track


@pytest.mark.parametrize("seed", range(8))
def test_the_dominance_annotation_rebuilds_the_monitoring_tracks_of_m2(seed):
    rng = np.random.default_rng(100 + seed)
    exc = rng.random((10, 6, 8)) < 0.4
    mask = make_mask(exc, unit_measure=True)
    obj, graph = build(mask, mode="monitoring", gap=1)
    np.testing.assert_array_equal(rebuild_m2_monitoring(graph),
                                  obj.dataset["object_track"].values)
