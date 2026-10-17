# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""physical vertical connectivity: physical vertical connectivity (``Footprint.vertical_gap_m``)
in volumes.

Two cells are vertical neighbours when the layers between them add up to at most the gap, in
both columns; adjacent layers always touch. Checked against a brute-force reading of that
definition, against the index connectivity (when no gap can be bridged), on uneven and sigma
layers, with morphology, ingestion and the specification.
"""

import numpy as np
import pytest
import xarray as xr
from _object_fixtures import make_mask, make_volume_mask, recipe, seg

import xhw
from xhw import _morph

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")
SOURCE = xhw.SourceRef(dataset_id="external:demo", variable="mask", accessed="2026-10-06",
                       product_status="final", version="0.1")


def segment(rank, gap):
    return xhw.RecipeStep.segment(footprint=xhw.Footprint.connected(rank, vertical_gap_m=gap),
                                  min_area=xhw.MinArea.none())


def build(mask, *steps):
    obj = xhw.build_objects(mask, recipe=recipe(*steps))
    obj.check_invariants()
    obj.check_against(mask)
    return obj


def labels(obj):
    return np.asarray(obj.dataset["label"].transpose("time", "z", "y", "x").values)


def column(*levels, nz=5, shape=(1, 6, 6)):
    out = np.zeros((shape[0], nz, *shape[1:]), bool)
    for z in levels:
        out[0, z, 2, 2] = True
    return out


UNEVEN = dict(dz=(1.0, 1.0, 1.0, 100.0, 100.0), depth=(0.5, 1.5, 2.5, 53.0, 153.0))


def test_a_gap_of_thin_layers_is_bridged_and_a_gap_of_thick_layers_is_not():
    thin = make_volume_mask(column(0, 2), **UNEVEN)          # level 1 (1 m) is empty
    assert labels(build(thin, seg(1))).max() == 2              # the index connectivity splits them
    assert labels(build(thin, segment(1, 1.0))).max() == 1     # 1 m of empty water is bridged
    assert labels(build(thin, segment(1, 0.5))).max() == 2
    thick = make_volume_mask(column(2, 4), **UNEVEN)          # level 3 (100 m) is empty
    assert labels(build(thick, segment(1, 50.0))).max() == 2
    assert labels(build(thick, segment(1, 100.0))).max() == 1


def test_adjacent_layers_always_touch_however_thick():
    mask = make_volume_mask(column(3, 4), **UNEVEN)          # two 100 m layers, face to face
    assert labels(build(mask, segment(1, 0.001))).max() == 1


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("periodic", [False, True])
def test_a_gap_smaller_than_every_layer_is_the_index_connectivity(rank, periodic):
    rng = np.random.default_rng(rank + 10 * periodic)
    exc = rng.random((3, 3, 6, 6)) < 0.35
    mask = make_volume_mask(exc, periodic=periodic)          # layers of 10, 20 and 30 m
    assert (labels(build(mask, seg(rank))) == labels(build(mask, segment(rank, 5.0)))).all()


def brute_force(thickness, z, y, x, rank, gap, periodic):
    """Neighbours of (z, y, x) read directly from the definition."""
    nz, ny, nx = thickness.shape

    def between(col_y, col_x, a, b):
        lo, hi = min(a, b), max(a, b)
        return thickness[lo + 1:hi, col_y, col_x].sum()

    out = set()
    for bz, by, bx in _morph._rank_offsets_3d(rank):
        ty, tx = y + by, x + bx
        if periodic:
            tx %= nx
        if not (0 <= ty < ny and 0 <= tx < nx):
            continue
        if bz == 0:
            out.add((z, ty, tx))
            continue
        for k in range(1, nz):
            tz = z + bz * k
            if not 0 <= tz < nz:
                break
            if between(y, x, z, tz) <= gap and between(ty, tx, z, tz) <= gap:
                out.add((tz, ty, tx))
    return out


def from_offsets(offsets, shape, z, y, x, periodic):
    nz, ny, nx = shape
    out = set()
    for offset in offsets:
        dz, dy, dx, rows = _morph.unpack(offset)
        tz, ty, tx = z + dz, y + dy, x + dx
        if periodic:
            tx %= nx
        if not (0 <= tz < nz and 0 <= ty < ny and 0 <= tx < nx):
            continue
        if rows is None or rows[z, y, x]:              # masks are (nz, ny, nx) here
            out.add((tz, ty, tx))
    return out


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("gap", [0.0, 4.0, 15.0])
def test_the_offsets_on_sigma_layers_are_the_definition(rank, gap):
    rng = np.random.default_rng(7)
    thickness = rng.uniform(1.0, 12.0, size=(6, 5, 5))       # sigma-like: varies by column
    fp = xhw.Footprint.connected(rank, vertical_gap_m=gap)
    for periodic in (False, True):
        lat, lon = np.arange(5.0), np.arange(5.0) * 72.0
        offsets = _morph.build_offsets(fp, lat, lon, periodic, volume=True,
                                       thickness=thickness)
        for z, y, x in [(0, 0, 0), (2, 2, 2), (5, 4, 4), (3, 0, 4), (1, 4, 0)]:
            assert (from_offsets(offsets, thickness.shape, z, y, x, periodic)
                    == brute_force(thickness, z, y, x, rank, gap, periodic)), (z, y, x, periodic)


def test_the_neighbourhood_is_symmetric_and_the_morphology_consistent():
    rng = np.random.default_rng(3)
    thickness = rng.uniform(1.0, 20.0, size=(6, 5, 7))
    fp = xhw.Footprint.connected(2, vertical_gap_m=10.0)
    offsets = _morph.build_offsets(fp, np.arange(5.0), np.arange(7.0), False, volume=True,
                                   thickness=thickness)
    shape = thickness.shape
    cells = [(z, y, x) for z in range(6) for y in range(5) for x in range(7)]
    pairs = {(c, d) for c in cells for d in from_offsets(offsets, shape, *c, False)}
    assert all((d, c) in pairs for c, d in pairs)
    field = rng.random((2, *shape)) < 0.4
    valid = np.ones_like(field)
    opened = _morph.morphology("open", field, valid, offsets, False)
    closed = _morph.morphology("close", field, valid, offsets, False)
    assert (opened <= field).all() and (field <= closed).all()
    assert (_morph.morphology("open", opened, valid, offsets, False) == opened).all()


def test_sigma_columns_bridge_differently():
    thickness = np.ones((3, 6, 6)) * 1.0
    thickness[1, :, 3:] = 50.0                    # the middle layer is thick on the east side
    depth = np.cumsum(thickness, axis=0) - thickness / 2
    exc = np.zeros((1, 3, 6, 6), bool)
    exc[0, 0, 2, 1] = exc[0, 2, 2, 1] = True      # west column: 1 m between them
    exc[0, 0, 2, 4] = exc[0, 2, 2, 4] = True      # east column: 50 m between them
    mask = make_volume_mask(exc, sigma=(thickness, depth))
    got = build(mask, segment(1, 2.0))
    assert got.dataset["object_cells"].values.tolist() == [2, 1, 1]


def test_the_ingestion_checks_a_physical_vertical_connectivity():
    mask = make_volume_mask(column(0, 2), **UNEVEN)
    raw = np.zeros((1, 5, 6, 6), int)
    raw[0, 0, 2, 2] = raw[0, 2, 2, 2] = 4            # one label across the thin empty layer
    labels_da = xr.DataArray(raw, dims=("time", "z", "y", "x"),
                             coords={"time": mask.dataset["time"].values})
    kw = dict(mask=mask, kind="patches", source=SOURCE, method="demo")
    xhw.ingest_objects(labels_da, connectivity=xhw.Footprint.connected(1, vertical_gap_m=1.0), **kw)
    with pytest.raises(xhw.ValidationError, match="connected patch"):
        xhw.ingest_objects(labels_da, connectivity=xhw.Footprint.connected(1), **kw)


def test_neighbor_fraction_counts_the_bridged_layers():
    mask = make_volume_mask(column(0), **UNEVEN)
    index = xhw.neighbor_fraction(mask, footprint=xhw.Footprint.connected(1))
    bridged = xhw.neighbor_fraction(mask, footprint=xhw.Footprint.connected(1, vertical_gap_m=1.0))
    assert index.values[0, 2, 2, 2] == 0.0          # level 2 does not see level 0 by index
    assert bridged.values[0, 2, 2, 2] > 0.0         # but does across 1 m of water


def test_the_specification_of_the_vertical_gap():
    fp = xhw.Footprint.connected(2, vertical_gap_m=5)
    assert fp.vertical_gap_m == 5.0 and fp.physical_vertical
    assert xhw.Footprint.from_json(fp.to_json()) == fp
    old = {"type": "Footprint", "schema_version": "0.1",
           "fields": {"kind": "connectivity", "connectivity": 1, "radius_km": None,
                      "offsets": None}}
    assert xhw.Footprint.from_dict(old) == xhw.Footprint.connected(1)
    assert xhw.Footprint.connected(1).spec_id == xhw.Footprint.from_dict(old).spec_id
    assert xhw.Footprint.connected(1).spec_id != xhw.Footprint.connected(1, 0.5).spec_id
    for bad in (dict(kind="radius", radius_km=10.0, vertical_gap_m=1.0),
                dict(kind="connectivity", connectivity=1, vertical_gap_m=-1.0)):
        with pytest.raises(xhw.ValidationError):
            xhw.Footprint(**bad)
    surface = make_mask(np.ones((1, 4, 4), bool), unit_measure=True)
    with pytest.raises(xhw.NotSupportedError, match="volume"):
        xhw.build_objects(surface, recipe=recipe(segment(1, 2.0)))
