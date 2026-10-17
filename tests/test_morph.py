# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""object tracking numerical core: neighbourhoods, integer morphology, labelling, temporal
operator.
"""

import numpy as np
import pytest

from xhw import Footprint
from xhw._detect import persist_array
from xhw._morph import (
    _label_graph,  # noqa: PLC2701 (the reference path of the equivalence test)
    build_offsets,
    dilate,
    erode,
    label_field,
    morphology,
    neighbour_fraction,
    relabel_first,
)


def offsets(footprint, ny, nx, periodic=False):
    return build_offsets(footprint, np.arange(ny, dtype=float), np.arange(nx, dtype=float),
                         periodic)


def labels(cells, shape, rank=1, periodic=False):
    field = np.zeros((1, *shape), bool)
    for y, x in cells:
        field[0, y, x] = True
    fp = Footprint.connected(rank)
    return label_field(field, fp, offsets(fp, *shape, periodic), periodic)[0][0]


# --- connectivity -----------------------------------------------------------------------------


def test_diagonal_cells_join_with_eight_neighbours_only():
    four, eight = (labels([(0, 0), (1, 1)], (3, 3), rank=r) for r in (1, 2))
    assert four.max() == 2 and eight.max() == 1


def test_all_neighbours_rank_is_the_eight_neighbourhood_in_2d():
    cells = [(0, 0), (1, 1), (2, 2), (4, 0)]
    assert np.array_equal(labels(cells, (5, 5), rank=2), labels(cells, (5, 5), rank=3))


def test_a_ring_encloses_a_hole_and_is_one_patch():
    ring = [(y, x) for y in range(3) for x in range(3) if (y, x) != (1, 1)]
    lab = labels(ring, (3, 3))
    assert lab.max() == 1 and lab[1, 1] == 0


def test_longitude_wraps_only_when_periodic_and_latitude_never_does():
    cells = [(1, 0), (1, 5)]
    assert labels(cells, (4, 6), periodic=False).max() == 2
    assert labels(cells, (4, 6), periodic=True).max() == 1
    assert labels([(0, 2), (3, 2)], (4, 6), periodic=True).max() == 2     # top and bottom rows


def test_a_diagonal_across_the_seam_needs_both_periodicity_and_eight_neighbours():
    cells = [(0, 5), (1, 0)]
    assert labels(cells, (4, 6), rank=1, periodic=True).max() == 2
    assert labels(cells, (4, 6), rank=2, periodic=True).max() == 1
    assert labels(cells, (4, 6), rank=2, periodic=False).max() == 2


def test_labels_are_numbered_by_first_cell_in_raster_order_and_empty_days_stay_empty():
    field = np.zeros((3, 4, 4), bool)
    field[0, 3, 3] = field[0, 0, 2] = True
    field[2, 1, 1] = True
    fp = Footprint.connected(1)
    lab, counts = label_field(field, fp, offsets(fp, 4, 4), False)
    assert lab[0, 0, 2] == 1 and lab[0, 3, 3] == 2
    assert counts.tolist() == [2, 0, 1] and not lab[1].any()


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("rank", [1, 2])
def test_scipy_path_and_graph_path_give_the_same_labels(rank, periodic):
    rng = np.random.default_rng(rank + 10 * periodic)
    fp = Footprint.connected(rank)
    off = offsets(fp, 12, 15, periodic)
    for density in (0.2, 0.4, 0.55):
        field = rng.random((4, 12, 15)) < density
        fast, _ = label_field(field, fp, off, periodic)
        graph = np.stack([_label_graph(day, off, periodic) for day in field])
        assert np.array_equal(fast, graph)
        assert np.array_equal(fast, np.stack([relabel_first(day) for day in fast]))


def test_a_disc_footprint_reproduces_the_rank_connectivities_near_the_equator():
    lat, lon = np.linspace(-2, 2, 5), np.arange(12) * 1.0          # 1 degree, about 111 km
    rng = np.random.default_rng(3)
    field = rng.random((3, 5, 12)) < 0.45
    for radius, rank in ((1.2 * 111.2, 1), (1.5 * 111.2, 2)):
        disc = Footprint.disc(radius)
        got, _ = label_field(field, disc, build_offsets(disc, lat, lon, False), False)
        want, _ = label_field(field, Footprint.connected(rank),
                              build_offsets(Footprint.connected(rank), lat, lon, False), False)
        assert np.array_equal(got, want)


def test_a_disc_smaller_than_the_cell_spacing_connects_nothing():
    lat, lon = np.arange(4) * 1.0, np.arange(4) * 1.0
    disc = Footprint.disc(50.0)
    assert build_offsets(disc, lat, lon, False) == []
    field = np.ones((1, 4, 4), bool)
    lab, counts = label_field(field, disc, [], False)
    assert counts[0] == 16 and sorted(np.unique(lab)) == list(range(1, 17))


def test_a_disc_in_kilometres_is_wider_in_cells_at_high_latitude():
    lon = np.arange(0, 20, 1.0)
    low, high = np.array([0.0, 1.0]), np.array([70.0, 71.0])
    disc = Footprint.disc(250.0)
    reach = lambda lat: max(  # noqa: E731
        dx for dy, dx, rows in build_offsets(disc, lat, lon, False) if dy == 0)
    assert reach(low) == 2 and reach(high) >= 6          # cos(70 deg) = 0.34


def test_explicit_offsets_define_the_connectivity():
    horizontal = Footprint.explicit([(0, 1), (0, -1)])
    field = np.zeros((1, 3, 4), bool)
    field[0, 0, 0] = field[0, 0, 1] = field[0, 1, 0] = True
    lab, counts = label_field(field, horizontal, build_offsets(horizontal, np.arange(3.),
                                                               np.arange(4.), False), False)
    assert counts[0] == 2 and lab[0, 0, 0] == lab[0, 0, 1] != lab[0, 1, 0]


# --- integer morphology -----------------------------------------------------------------------


def everywhere(shape):
    return np.ones(shape, bool)


def test_closing_fills_a_hole_and_never_a_cell_outside_the_valid_ones():
    fp = Footprint.connected(2)
    field = np.ones((1, 5, 5), bool)
    field[0, 2, 2] = False
    valid = everywhere(field.shape)
    off = offsets(fp, 5, 5)
    assert morphology("close", field, valid, off, False).all()
    valid[0, 2, 2] = False                       # the hole is land: it must stay empty
    closed = morphology("close", field & valid, valid, off, False)
    assert not closed[0, 2, 2] and closed.sum() == 24


def test_opening_removes_an_isolated_cell_and_keeps_a_block():
    fp = Footprint.connected(2)
    field = np.zeros((1, 9, 9), bool)
    field[0, 1:4, 1:4] = True                    # a 3 x 3 block
    field[0, 7, 7] = True                        # an isolated cell
    opened = morphology("open", field, everywhere(field.shape), offsets(fp, 9, 9), False)
    assert opened[0, 1:4, 1:4].all() and not opened[0, 7, 7] and opened.sum() == 9


def test_the_edge_of_the_grid_and_land_do_not_erode_a_patch():
    fp = Footprint.connected(2)
    field = np.zeros((1, 6, 6), bool)
    field[0, 0:3, 0:3] = True                    # a block in the corner of the grid
    valid = everywhere(field.shape)
    valid[0, :, 5] = False                       # a land column on the other side
    off = offsets(fp, 6, 6)
    assert np.array_equal(morphology("open", field, valid, off, False), field)
    eroded = erode(field, valid, off, False)
    assert eroded[0, 0, 0] and eroded[0, 1, 1]   # a zero-border erosion would delete the corner


@pytest.mark.parametrize("periodic", [False, True])
def test_opening_is_inside_the_field_closing_contains_it_and_opening_is_idempotent(periodic):
    rng = np.random.default_rng(5)
    fp = Footprint.connected(2)
    off = offsets(fp, 10, 12, periodic)
    land = rng.random((10, 12)) < 0.1
    valid = np.broadcast_to(~land, (3, 10, 12))
    for _ in range(30):
        field = (rng.random((3, 10, 12)) < 0.5) & valid
        opened = morphology("open", field, valid, off, periodic)
        closed = morphology("close", field, valid, off, periodic)
        assert not (opened & ~field).any() and not (field & ~closed).any()
        assert not (closed & ~valid).any()
        assert np.array_equal(morphology("open", opened, valid, off, periodic), opened)


def test_dilation_crosses_the_seam_only_when_periodic():
    field = np.zeros((1, 3, 6), bool)
    field[0, 1, 0] = True
    valid = everywhere(field.shape)
    fp = Footprint.connected(1)
    assert dilate(field, valid, offsets(fp, 3, 6, True), True)[0, 1, 5]
    assert not dilate(field, valid, offsets(fp, 3, 6, False), False)[0, 1, 5]


def test_neighbour_fraction_is_normalised_by_the_valid_weight():
    fp = Footprint.connected(2)
    field = np.zeros((1, 3, 3), bool)
    field[0, 1, 1] = True
    valid = everywhere(field.shape)
    weights = np.ones((3, 3))
    off = offsets(fp, 3, 3)
    frac = neighbour_fraction(field, valid, weights, off, False)[0]
    assert frac[1, 1] == pytest.approx(1 / 9) and frac[0, 0] == pytest.approx(1 / 4)
    valid[0, 0, 1] = False                       # land beside the corner: does not lower it
    frac = neighbour_fraction(field, valid, weights, off, False)[0]
    assert frac[0, 0] == pytest.approx(1 / 3)
    weights[1, 1] = 3.0                          # area weights; land at (0, 1) still out
    frac = neighbour_fraction(field, valid, weights, off, False)[0]
    assert frac[1, 1] == pytest.approx(3 / 10)
    none_valid = np.zeros_like(valid)
    assert np.isnan(neighbour_fraction(field, none_valid, weights, off, False)).all()


# --- the temporal operator on a processed field -----------------------------------------------


def test_persist_array_applies_duration_and_gap_joining_on_the_first_axis():
    series = np.array([1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1], bool)
    field = series[:, None, None]
    ok = np.ones_like(field)
    member, joined = persist_array(field, ok, min_days=1, max_gap=2, rule="qualify_then_join")
    assert member[:, 0, 0].astype(int).tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
    assert joined[:, 0, 0].astype(int).tolist() == [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
    member, _ = persist_array(field, ok, min_days=2, max_gap=0, rule="qualify_then_join")
    assert member[:, 0, 0].astype(int).tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    blocked = ok.copy()
    blocked[2] = False                            # an ineligible day cannot be bridged
    member, _ = persist_array(field, blocked, min_days=1, max_gap=2, rule="qualify_then_join")
    assert not member[2, 0, 0] and member[4, 0, 0]
