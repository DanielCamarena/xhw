# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Neighbourhoods, integer morphology and connected-component labelling of object tracking.

Everything works on boolean fields (..., y, x) with the validity of each cell. Morphology is
exact integer arithmetic (counts of neighbours, no FFT). Land, invalid cells and the edge of
the grid never join a patch and never receive a cell. They are treated differently by the two
operators, always on the side of not inventing evidence: an *opening* does not erode a patch
at the edge or next to land (erosion keeps a cell when every *valid* neighbour in the
footprint is set), and a *closing* only fills a gap that is surrounded by valid cells within
the footprint (never against the edge or land, where the gap is not between two parts of the
field). Longitude is periodic only when the grid says so (``periodic_x``); latitude never is
(declared numerical rules, audit finding 3).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ._errors import NotSupportedError
from ._recipe import Footprint
from ._schema import require

EARTH_RADIUS_KM = 6371.0
MAX_OFFSETS = 2500

Offset = tuple  # (dy, dx, rows[, dz]) with rows: None (every cell), a boolean (ny,) mask (the
# radius varies with latitude) or a boolean (ny, nx) mask (curvilinear grids);
# dz (volumes only) is the shift along the vertical axis, 0 if absent


def unpack(offset) -> tuple[int, int, int, object]:
    """``(dz, dy, dx, rows)`` of an offset (dz is 0 for the surface offsets)."""
    return (offset[3] if len(offset) > 3 else 0), offset[0], offset[1], offset[2]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def is_volume(grid) -> bool:
    return grid.support == "volume"


def is_line(grid) -> bool:
    return grid.support == "line"


def axes(ds, grid):
    """Latitude and longitude of a surface or a volume: both 1-D (along y and along x) or both
    2-D (curvilinear grids, dims y and x). A line has no horizontal coordinates: ``(None,
    None)`` (its coordinate is ``coords.along``).
    """
    require(grid.support in ("line", "surface", "volume"),
            "objects need a line, a surface or a volume support", NotSupportedError)
    if grid.support == "line":
        return None, None
    y_dim, x_dim = grid.spatial_dims[-2:]
    lat, lon = ds[grid.coords.lat], ds[grid.coords.lon]
    regular = lat.dims == (y_dim,) and lon.dims == (x_dim,)
    curvilinear = (set(lat.dims) == {y_dim, x_dim} and set(lon.dims) == {y_dim, x_dim}
                   and lat.ndim == lon.ndim == 2)
    if not (regular or curvilinear):
        raise NotSupportedError("objects need latitude along y and longitude along x (1-D), or "
                                "2-D latitude and longitude over (y, x) (curvilinear)")
    if curvilinear:
        lat, lon = lat.transpose(y_dim, x_dim), lon.transpose(y_dim, x_dim)
    return np.asarray(lat.values, dtype=float), np.asarray(lon.values, dtype=float)


def cell_coords(lat: np.ndarray, lon: np.ndarray, nz: int = 1):
    """Latitude and longitude of every cell of the flat view ``(nz * ny, nx)``."""
    if lat.ndim == 1:
        lat2 = np.repeat(lat[:, None], lon.size, axis=1)
        lon2 = np.repeat(lon[None, :], lat.size, axis=0)
    else:
        lat2, lon2 = lat, lon
    return np.tile(lat2, (nz, 1)), np.tile(lon2, (nz, 1))


# --- neighbourhoods ------------------------------------------------------------------------

_RANK_OFFSETS = {
    1: ((-1, 0), (1, 0), (0, -1), (0, 1)),
    2: tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)),
}
_RANK_OFFSETS[3] = _RANK_OFFSETS[2]   # in 2-D, "all neighbours" is the 8-neighbourhood


def _rank_offsets_3d(rank: int) -> list[tuple[int, int, int]]:
    """(dz, dy, dx) offsets of the 6 / 18 / 26 neighbourhoods (ranks 1, 2, 3)."""
    return [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if 0 < abs(dz) + abs(dy) + abs(dx) <= rank]


def build_offsets(footprint: Footprint, lat, lon, periodic_x: bool, volume: bool = False,
                  line: bool = False, thickness: np.ndarray | None = None) -> list[Offset]:
    """Symmetric neighbour offsets ``(dy, dx, rows[, dz])`` (both signs); the centre is implicit.
    A line works as one row (offsets ``(0, dx)``); in a volume the offsets carry ``dz``. The
    radius footprint needs horizontal coordinates and is a surface footprint. A connectivity
    with ``vertical_gap_m`` (physical vertical connectivity, physical vertical connectivity)
    needs a volume and the layer `thickness` (z, y, x) in metres.
    """
    if footprint.physical_vertical:
        require(volume, "vertical_gap_m needs a volume support", NotSupportedError)
        require(thickness is not None, "vertical_gap_m needs the layer thickness")
        return _vertical_gap_offsets(footprint.connectivity, footprint.vertical_gap_m,
                                     np.asarray(thickness, dtype=float), periodic_x)
    if line:
        require(footprint.kind != "radius", "a radius footprint needs horizontal coordinates "
                "(not available on a line: use a connectivity or explicit (dx,) offsets)",
                NotSupportedError)
        if footprint.kind == "connectivity":
            return [(0, -1, None), (0, 1, None)]
        require(all(len(item) == 1 for item in footprint.offsets),
                "the explicit offsets of a line must be (dx,)")
        return [(0, dx, None) for (dx,) in footprint.offsets]
    if volume:
        require(footprint.kind != "radius", "a radius footprint is not available in a volume "
                "(use a connectivity or explicit (dz, dy, dx) offsets)", NotSupportedError)
        if footprint.kind == "connectivity":
            return [(dy, dx, None, dz) for dz, dy, dx in _rank_offsets_3d(footprint.connectivity)]
        require(all(len(item) == 3 for item in footprint.offsets),
                "the explicit offsets of a volume must be (dz, dy, dx)")
        return [(dy, dx, None, dz) for dz, dy, dx in footprint.offsets]
    if footprint.kind == "connectivity":
        return [(dy, dx, None) for dy, dx in _RANK_OFFSETS[footprint.connectivity]]
    require(footprint.kind == "radius" or all(len(item) == 2 for item in footprint.offsets),
            "the explicit offsets of a surface must be (dy, dx)")
    if footprint.kind == "offsets":
        return [(dy, dx, None) for dy, dx in footprint.offsets]
    if lat.ndim == 2:
        return _radius_offsets_2d(footprint.radius_km, lat, lon, periodic_x)
    return _radius_offsets_1d(footprint.radius_km, lat, lon, periodic_x)


def _extrapolate(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    """`values` at integer positions that may lie outside the axis: linear extrapolation with the
    spacing of the nearest edge (the mask of a radius counts the neighbours that fall outside
    the grid, as the rank connectivities do).
    """
    n = values.size
    lo = values[1] - values[0] if n > 1 else 0.0
    hi = values[-1] - values[-2] if n > 1 else 0.0
    inside = values[np.clip(index, 0, n - 1)]
    return np.where(index < 0, values[0] + index * lo,
                    np.where(index >= n, values[-1] + (index - n + 1) * hi, inside))


def _pad_linear(a: np.ndarray, pad: int, axis: int, wrap: bool = False) -> np.ndarray:
    """Pad an array by `pad` cells on both sides of `axis`: periodic copies if `wrap`, else
    linear extrapolation with the edge spacing.
    """
    if pad == 0:
        return a
    if wrap:
        index = np.arange(-pad, a.shape[axis] + pad) % a.shape[axis]
        return np.take(a, index, axis=axis)
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    lo_step = a[1] - a[0] if n > 1 else np.zeros_like(a[0])
    hi_step = a[-1] - a[-2] if n > 1 else np.zeros_like(a[0])
    k = np.arange(pad, 0, -1).reshape((-1,) + (1,) * (a.ndim - 1))
    lo = a[0] - k * lo_step
    hi = a[-1] + k[::-1] * hi_step
    return np.moveaxis(np.concatenate([lo, a, hi], axis=0), 0, axis)


def _vertical_gap_offsets(rank: int, gap_m: float, thickness: np.ndarray,
                          periodic_x: bool) -> list[Offset]:
    """Offsets of a rank connectivity in a volume whose vertical reach is physical: an offset
    ``(dz, dy, dx)`` of the rank structure with ``|dz| = 1`` becomes the offsets ``dz = +-k``
    whose intermediate layers (the k - 1 layers between the two cells) add up to at most
    `gap_m` metres, both in the column of the cell and in that of its neighbour (so the
    relation is symmetric). Layers beyond the top and the bottom are extrapolated with the
    edge thickness: those neighbours belong to the footprint but never exist, as with the
    index connectivity.
    """
    nz = thickness.shape[0]
    pad = np.concatenate([np.repeat(thickness[:1], nz, axis=0), thickness,
                          np.repeat(thickness[-1:], nz, axis=0)])
    top = np.concatenate([np.zeros((1,) + thickness.shape[1:]), np.cumsum(pad, axis=0)])
    levels = np.arange(nz) + nz                          # index of each level in `pad`

    def gap(k: int, sign: int) -> np.ndarray:
        """Thickness of the layers strictly between level z and level z + sign * k."""
        if sign > 0:
            return top[levels + k] - top[levels + 1]
        return top[levels] - top[levels - k + 1]

    found: list[Offset] = []
    for dz, dy, dx in _rank_offsets_3d(rank):
        if dz == 0:
            found.append((dy, dx, None, 0))
            continue
        for k in range(1, nz):
            source = gap(k, dz)
            if float(source.min()) > gap_m:
                break
            target = shifted(source, dy, dx, periodic_x, np.nan)
            target = np.where(np.isnan(target), source, target)
            ok = (source <= gap_m) & (target <= gap_m)
            if ok.any():
                found.append((dy, dx, None if ok.all() else ok, dz * k))
    require(len(found) <= MAX_OFFSETS,
            f"vertical_gap_m={gap_m:g} m needs more than {MAX_OFFSETS} neighbour offsets on this "
            "grid; use a smaller gap", NotSupportedError)
    return found


def _radius_offsets_1d(r: float, lat: np.ndarray, lon: np.ndarray,
                       periodic_x: bool) -> list[Offset]:
    """Regular longitude and latitude axes: the cells within `r` km, mask per latitude row.
    Neighbours beyond the edge of the grid belong to the footprint (their latitude is
    extrapolated); they simply never exist, so a closing cannot fill against the edge.
    """
    ny, nx = lat.size, lon.size
    steps = np.abs(np.diff(lon))
    if nx > 1:
        require(bool(np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9)),
                "a radius footprint needs a regular longitude axis", NotSupportedError)
    dlon = float(steps[0]) if nx > 1 else 1.0
    dlat = float(np.min(np.abs(np.diff(lat)))) if ny > 1 else 1.0
    dy_max = min(ny - 1, int(np.ceil(r / (dlat * np.pi / 180 * EARTH_RADIUS_KM))))
    edge = float(np.max(np.abs(lat)))
    dx_km = float(haversine_km(edge, 0.0, edge, dlon))
    dx_cap = nx // 2 if periodic_x else nx - 1
    dx_max = min(dx_cap, int(np.ceil(r / max(dx_km, 1e-9))))
    require((2 * dy_max + 1) * (2 * dx_max + 1) <= MAX_OFFSETS,
            f"a radius of {r:g} km needs more than {MAX_OFFSETS} neighbour offsets on this grid "
            "(high latitudes); use a smaller radius or explicit offsets", NotSupportedError)
    found: list[Offset] = []
    rows = np.arange(ny)
    for dy in range(-dy_max, dy_max + 1):
        t_lat = np.clip(_extrapolate(lat, rows + dy), -90.0, 90.0)
        for dx in range(-dx_max, dx_max + 1):
            if (dy, dx) == (0, 0):
                continue
            if periodic_x and nx % 2 == 0 and dx == -nx // 2:
                continue    # the same cell as +nx/2
            ok = haversine_km(lat, 0.0, t_lat, dx * dlon) <= r
            if ok.any():
                found.append((dy, dx, None if ok.all() else ok))
    return found


def _radius_offsets_2d(r: float, lat: np.ndarray, lon: np.ndarray,
                       periodic_x: bool) -> list[Offset]:
    """Curvilinear grid: the cells whose centre is within `r` km (great circle), as a mask per
    cell for each index offset. The offset range comes from the smallest distance between
    adjacent cells of the grid, so a grid that folds on itself needs explicit offsets.
    """
    ny, nx = lat.shape
    along_y = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]) if ny > 1 else np.full(1, r)
    along_x = haversine_km(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:]) if nx > 1 else \
        np.full(1, r)
    if periodic_x and nx > 2:
        along_x = np.concatenate([along_x.ravel(),
                                  haversine_km(lat[:, -1], lon[:, -1], lat[:, 0], lon[:, 0])])
    dy_min, dx_min = float(along_y[along_y > 0].min()), float(along_x[along_x > 0].min())
    dx_cap = nx // 2 if periodic_x else nx - 1
    dy_max, dx_max = min(ny - 1, int(np.ceil(r / dy_min))), min(dx_cap, int(np.ceil(r / dx_min)))
    require((2 * dy_max + 1) * (2 * dx_max + 1) <= MAX_OFFSETS,
            f"a radius of {r:g} km needs more than {MAX_OFFSETS} neighbour offsets on this grid "
            "(small cells); use a smaller radius or explicit offsets", NotSupportedError)
    found: list[Offset] = []
    # coordinates beyond the edge, extrapolated, so that the footprint counts the neighbours that
    # fall outside the grid (they never exist: a closing cannot fill against the edge)
    ext_lat = _pad_linear(_pad_linear(lat, dy_max, 0), dx_max, 1, wrap=periodic_x)
    ext_lon = _pad_linear(_pad_linear(lon, dy_max, 0), dx_max, 1, wrap=periodic_x)
    ext_lat = np.clip(ext_lat, -90.0, 90.0)
    for dy in range(-dy_max, dy_max + 1):
        for dx in range(-dx_max, dx_max + 1):
            if (dy, dx) == (0, 0):
                continue
            if periodic_x and nx % 2 == 0 and dx == -nx // 2:
                continue    # the same cell as +nx/2
            window = (slice(dy_max + dy, dy_max + dy + ny), slice(dx_max + dx, dx_max + dx + nx))
            ok = haversine_km(lat, lon, ext_lat[window], ext_lon[window]) <= r
            if ok.any():
                found.append((dy, dx, None if ok.all() else ok))
    return found


def forward(offsets: list[Offset]) -> list[Offset]:
    """One offset of every neighbour pair (the relation is symmetric)."""
    return [o for o in offsets if (unpack(o)[0], o[0], o[1]) > (0, 0, 0)]


# --- integer morphology --------------------------------------------------------------------


def shifted(a: np.ndarray, dy: int, dx: int, periodic_x: bool, fill, dz: int = 0) -> np.ndarray:
    """``out[..., z, y, x] = a[..., z + dz, y + dy, x + dx]``; `fill` outside (x wraps if
    periodic). `dz` needs a vertical axis (the third from the end).
    """
    out = np.full_like(a, fill)
    to, frm = [slice(None)] * a.ndim, [slice(None)] * a.ndim
    for axis, d in ((-3, dz), (-2, dy), (-1, 0 if periodic_x else dx)):
        if d:
            n = a.shape[axis]
            to[axis] = slice(max(0, -d), max(0, min(n, n - d)))
            frm[axis] = slice(max(0, d), max(0, min(n, n + d)))
    source = np.roll(a, -dx, axis=-1) if periodic_x and dx else a
    out[tuple(to)] = source[tuple(frm)]
    return out


def rows_mask(rows):
    """A footprint mask as something that broadcasts against (..., ny, nx)."""
    if rows is None:
        return True
    return rows[:, None] if rows.ndim == 1 else rows      # (ny, nx) or (nz, ny, nx)


def neighbour_counts(field: np.ndarray, valid: np.ndarray, offsets: list[Offset],
                     periodic_x: bool) -> tuple[np.ndarray, np.ndarray]:
    """Number of set cells and of valid cells among the neighbours of each cell (int16)."""
    n_set = np.zeros(field.shape, dtype=np.int16)
    n_valid = np.zeros(field.shape, dtype=np.int16)
    for offset in offsets:
        dz, dy, dx, rows = unpack(offset)
        mask = rows_mask(rows)
        n_set += shifted(field, dy, dx, periodic_x, False, dz) & mask
        n_valid += shifted(valid, dy, dx, periodic_x, False, dz) & mask
    return n_set, n_valid


def dilate(field: np.ndarray, valid: np.ndarray, offsets: list[Offset], periodic_x: bool):
    n_set, _ = neighbour_counts(field, valid, offsets, periodic_x)
    return (field | (n_set > 0)) & valid


def footprint_size(offsets: list[Offset], ny: int) -> np.ndarray:
    """Number of neighbours of the footprint of each cell, shaped (ny, 1) (the radius varies with
    latitude) or (ny, nx) (curvilinear grids).
    """
    size = np.zeros((ny, 1), dtype=np.int16)
    for offset in offsets:
        rows = offset[2]
        size = size + (1 if rows is None else rows_mask(rows).astype(np.int16))
    return size


def erode(field: np.ndarray, valid: np.ndarray, offsets: list[Offset], periodic_x: bool,
          strict: bool = False):
    """Keep a set cell when all its neighbours are set. ``strict=False``: only the valid
    neighbours count (land and the edge do not erode). ``strict=True``: every neighbour of the
    footprint must exist and be set (used by the closing).
    """
    n_set, n_valid = neighbour_counts(field, valid, offsets, periodic_x)
    if strict:
        return field & (n_set == footprint_size(offsets, field.shape[-2]))
    return field & (n_set == n_valid)


def morphology(op: str, field: np.ndarray, valid: np.ndarray, offsets: list[Offset],
               periodic_x: bool) -> np.ndarray:
    """Binary opening (erosion, dilation) or closing (dilation, strict erosion, united with the
    field), exact and bounded by the valid cells: opening never adds cells, closing never
    removes any.
    """
    field = field & valid
    if op == "open":
        return dilate(erode(field, valid, offsets, periodic_x), valid, offsets, periodic_x)
    require(op == "close", f"unknown morphology {op!r}")
    grown = dilate(field, valid, offsets, periodic_x)
    return field | erode(grown, valid, offsets, periodic_x, strict=True)


def neighbour_fraction(field: np.ndarray, valid: np.ndarray, weights: np.ndarray,
                       offsets: list[Offset], periodic_x: bool) -> np.ndarray:
    """Weighted fraction of set cells among the valid cells of the footprint *including the cell
    itself*, normalised by the valid weight (land never lowers it); NaN where the cell has no
    valid cell in its footprint.
    """
    num = np.where(field & valid, weights, 0.0)
    den = np.where(valid, weights, 0.0)
    for offset in offsets:
        dz, dy, dx, rows = unpack(offset)
        keep = rows_mask(rows)
        num = num + shifted(np.where(field & valid, weights, 0.0), dy, dx, periodic_x, 0.0,
                            dz) * keep
        den = den + shifted(np.where(valid, weights, 0.0), dy, dx, periodic_x, 0.0, dz) * keep
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# --- connected components ------------------------------------------------------------------


def relabel_first(labels: np.ndarray) -> np.ndarray:
    """Renumber the labels 1..k in order of their first cell in raster order (deterministic)."""
    flat = labels.ravel()
    positive = flat > 0
    if not positive.any():
        return labels.astype(np.int32, copy=False)
    values, first = np.unique(flat[positive], return_index=True)
    order = np.argsort(first, kind="stable")
    lut = np.zeros(int(values.max()) + 1, dtype=np.int32)
    lut[values[order]] = np.arange(1, values.size + 1, dtype=np.int32)
    return lut[labels]


def _label_rank(day: np.ndarray, rank: int, periodic_x: bool) -> np.ndarray:
    structure = ndi.generate_binary_structure(day.ndim, rank)
    nx = day.shape[-1]
    if not periodic_x or nx < 2:
        return ndi.label(day, structure=structure)[0].astype(np.int32)
    padded = np.concatenate([day[..., -1:], day, day[..., :1]], axis=-1)
    lab, n = ndi.label(padded, structure=structure)
    pairs = [(lab[..., 0], lab[..., nx]), (lab[..., nx + 1], lab[..., 1])]
    a = np.concatenate([p[0][(p[0] > 0) & (p[1] > 0)] for p in pairs])
    b = np.concatenate([p[1][(p[0] > 0) & (p[1] > 0)] for p in pairs])
    if a.size:
        graph = coo_matrix((np.ones(a.size), (a, b)), shape=(n + 1, n + 1))
        comp = connected_components(graph, directed=False)[1]
        lab = np.where(lab > 0, comp[lab] + 1, 0)
    return relabel_first(lab[..., 1:nx + 1])


def _label_graph(day: np.ndarray, offsets: list[Offset], periodic_x: bool,
                 same_value: bool = False) -> np.ndarray:
    """Components of a field (y, x) or (z, y, x) under an explicit neighbourhood. With
    `same_value` two neighbouring cells join only if they hold the same value (labels).
    """
    volume = day.ndim == 3
    ny, nx = day.shape[-2:]
    nz = day.shape[0] if volume else 1
    coords = np.nonzero(day)
    zs = coords[0] if volume else np.zeros(coords[0].size, dtype=np.int64)
    ys, xs = coords[-2], coords[-1]
    m = ys.size
    out = np.zeros(day.shape, dtype=np.int32)
    if m == 0:
        return out
    values = day if volume else day[None]
    index = np.full((nz, ny, nx), -1, dtype=np.int64)
    index[zs, ys, xs] = np.arange(m)
    src, dst = [], []
    for offset in forward(offsets):
        dz, dy, dx, rows = unpack(offset)
        tz, ty, tx = zs + dz, ys + dy, xs + dx
        ok = (ty >= 0) & (ty < ny) & (tz >= 0) & (tz < nz)
        if periodic_x:
            tx = tx % nx
        else:
            ok &= (tx >= 0) & (tx < nx)
        if rows is not None:
            ok &= (rows[ys] if rows.ndim == 1 else rows[ys, xs] if rows.ndim == 2
                   else rows[zs, ys, xs])
        sel = np.flatnonzero(ok)
        other = index[tz[sel], ty[sel], tx[sel]]
        hit = other >= 0
        if same_value:
            hit &= values[tz[sel], ty[sel], tx[sel]] == values[zs[sel], ys[sel], xs[sel]]
        src.append(sel[hit])
        dst.append(other[hit])
    src = np.concatenate(src) if src else np.zeros(0, dtype=np.int64)
    dst = np.concatenate(dst) if dst else np.zeros(0, dtype=np.int64)
    graph = coo_matrix((np.ones(src.size), (src, dst)), shape=(m, m))
    comp = connected_components(graph, directed=False)[1]
    out[coords] = comp + 1
    return relabel_first(out)


def split_labels(day: np.ndarray, offsets: list[Offset], periodic_x: bool) -> dict[int, int]:
    """Labels of `day` ((y, x) or (z, y, x), 0 = background) that do not form a single connected
    patch under the neighbourhood, with their number of patches ({} if every label is one).
    """
    present = np.unique(day[day > 0])
    if present.size == 0:
        return {}
    comp = _label_graph(day, offsets, periodic_x, same_value=True)
    pairs = np.unique(np.stack([day[day > 0], comp[day > 0]], axis=1), axis=0)
    count = np.bincount(np.searchsorted(present, pairs[:, 0]), minlength=present.size)
    return {int(v): int(c) for v, c in zip(present, count) if c > 1}


def touching_labels(day: np.ndarray, offsets: list[Offset], periodic_x: bool) -> np.ndarray:
    """Pairs ``(a, b)`` of different labels of `day` ((y, x) or (z, y, x)) that have neighbouring
    cells under the footprint, as an array of shape (n, 2) with a < b.
    """
    pairs = []
    for offset in forward(offsets):
        dz, dy, dx, rows = unpack(offset)
        other = shifted(day, dy, dx, periodic_x, 0, dz)
        hit = (day > 0) & (other > 0) & (day != other) & rows_mask(rows)
        if hit.any():
            a, b = day[hit], other[hit]
            pairs.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    if not pairs:
        return np.zeros((0, 2), dtype=day.dtype)
    return np.unique(np.concatenate(pairs), axis=0)


def label_field(field: np.ndarray, footprint: Footprint, offsets: list[Offset],
                periodic_x: bool) -> tuple[np.ndarray, np.ndarray]:
    """Daily connected patches of `field` (T, y, x) or (T, z, y, x): int32 labels (0 =
    background, 1..n per day in order of the first cell) and the number of patches of each
    day. Rank connectivities use scipy.ndimage; radius and explicit footprints use a sparse
    graph; both give the same labels where they overlap (tested).
    """
    labels = np.zeros(field.shape, dtype=np.int32)
    counts = np.zeros(field.shape[0], dtype=np.int64)
    for t in np.flatnonzero(field.any(axis=tuple(range(1, field.ndim)))):
        if footprint.kind == "connectivity" and not footprint.physical_vertical:
            labels[t] = _label_rank(field[t], footprint.connectivity, periodic_x)
        else:
            labels[t] = _label_graph(field[t], offsets, periodic_x)
        counts[t] = labels[t].max()
    return labels, counts
