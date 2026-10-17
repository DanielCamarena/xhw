# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Synthetic XMasks for the object tracking object tests: exact boolean fields, no detector
involved.
"""

from __future__ import annotations

import uuid

import numpy as np
import xarray as xr
from _fixtures import baseline, grid, hobday, mask_dataset, provenance

import xhw

EARTH_RADIUS = 6_371_000.0


def cell_areas(lat: np.ndarray, dlat: float, dlon: float) -> np.ndarray:
    """Area (object tracking) of a lat/lon cell of size dlat x dlon degrees centred on `lat` (per
    row).
    """
    lo, hi = np.deg2rad(lat - dlat / 2), np.deg2rad(lat + dlat / 2)
    return EARTH_RADIUS ** 2 * np.deg2rad(dlon) * (np.sin(hi) - np.sin(lo))


def make_mask(exc, *, mem=None, domain=None, step: float = 1.0, lat0: float = -10.0,
              periodic: bool = False, unit_measure: bool = False, definition=None,
              base=None, start: str = "2000-01-01", run_id: str | None = None,
              curvilinear: bool = False, coords2d=None) -> xhw.XMask:
    """XMask whose ``exceedance`` is `exc` (T, ny, nx); ``event_membership`` is `mem` (default:
    equal to `exc`, i.e. min_duration 1 and no gaps). `domain` (ny, nx) marks the cells that
    are valid (the rest is outside_domain). With ``periodic=True`` the longitude covers 360
    degrees in nx cells. ``curvilinear=True`` stores the same latitude and longitude as 2-D
    arrays over (y, x); `coords2d` = ``(lat2, lon2)`` replaces them with other 2-D coordinates
    (e.g. a rotated grid).
    """
    exc = np.asarray(exc, bool)
    T, ny, nx = exc.shape
    domain = np.ones((ny, nx), bool) if domain is None else np.asarray(domain, bool)
    lat = lat0 + step * np.arange(ny)
    lon = (360.0 / nx) * np.arange(nx) if periodic else -80.0 + step * np.arange(nx)
    dlon = 360.0 / nx if periodic else step
    area = np.outer(cell_areas(lat, step, dlon), np.ones(nx))
    exc = exc & domain
    mem = exc.copy() if mem is None else np.asarray(mem, bool) & domain
    reason = np.broadcast_to(np.where(domain, 0, 1).astype(np.uint8), exc.shape).copy()
    temp = np.where(exc, 13.0, 10.0)
    times = (np.datetime64(start) + np.arange(T)).astype("datetime64[ns]")
    definition = definition or hobday(min_duration_days=1, max_gap_days=0)
    statics = {"domain_mask": (("y", "x"), domain)}
    if coords2d is not None:
        coords = {"lat": (("y", "x"), np.asarray(coords2d[0], float)),
                  "lon": (("y", "x"), np.asarray(coords2d[1], float))}
    elif curvilinear:
        coords = {"lat": (("y", "x"), np.repeat(lat[:, None], nx, axis=1)),
                  "lon": (("y", "x"), np.repeat(lon[None, :], ny, axis=0))}
    else:
        coords = {"lat": ("y", lat), "lon": ("x", lon)}
    if unit_measure:
        measure = xhw.CellMeasure(kind="unit")
    else:
        statics["cell_area"] = (("y", "x"), area, {"units": "m2"})
        measure = xhw.CellMeasure(kind="area", var="cell_area", units="m2")
    ds = mask_dataset(times, ("time", "y", "x"), temp, 12.0, 10.0, exc, mem,
                      np.zeros_like(exc), reason, definition.variable,
                      coords, statics)
    g = grid(support="surface", topology="structured", spatial_dims=("y", "x"),
             coords=xhw.CoordRoles(lat="lat", lon="lon"), cell_measure=measure,
             periodic_x=periodic)
    return xhw.XMask.from_dataset(ds, grid=g, definition=definition,
                                        baseline=base or baseline(),
                                        provenance=provenance(run_id or str(uuid.uuid4())))


def field(shape, *cells) -> np.ndarray:
    """Boolean field with the listed ``(t, y, x)`` cells set."""
    out = np.zeros(shape, bool)
    for t, y, x in cells:
        out[t, y, x] = True
    return out


def block(out: np.ndarray, t, ys, xs) -> np.ndarray:
    out[t, ys, xs] = True
    return out


def recipe(*steps, layer: str = "exceedance") -> xhw.SpatialRecipe:
    return xhw.SpatialRecipe(input_layer=layer, steps=tuple(steps))


def seg(rank: int = 1, area=None) -> xhw.RecipeStep:
    return xhw.RecipeStep.segment(footprint=xhw.Footprint.connected(rank),
                                  min_area=area or xhw.MinArea.none())


def dataset_equal(a: xr.Dataset, b: xr.Dataset) -> bool:
    return a.identical(b)


def make_volume_mask(exc, *, dz=(10.0, 20.0, 30.0), depth=(5.0, 20.0, 45.0), periodic: bool = False,
                     unit_measure: bool = False, domain=None, definition=None,
                     start: str = "2000-01-01", sigma=None) -> xhw.XMask:
    """XMask of a volume (time, z, y, x) whose ``exceedance`` (= ``event_membership``) is `exc`;
    layers of thickness `dz` (m) at depth `depth` (m); cells of 1 degree. `sigma` =
    ``(thickness, depth)`` arrays over (z, y, x) makes a static sigma grid (the vertical
    coordinate is then the 3-D variable ``depth``).
    """
    exc = np.asarray(exc, bool)
    T, nz, ny, nx = exc.shape
    domain = np.ones((nz, ny, nx), bool) if domain is None else np.asarray(domain, bool)
    lat = -10.0 + np.arange(ny)
    lon = (360.0 / nx) * np.arange(nx) if periodic else -80.0 + np.arange(nx)
    area = np.outer(cell_areas(lat, 1.0, 360.0 / nx if periodic else 1.0), np.ones(nx))
    exc = exc & domain
    reason = np.broadcast_to(np.where(domain, 0, 1).astype(np.uint8), exc.shape).copy()
    times = (np.datetime64(start) + np.arange(T)).astype("datetime64[ns]")
    definition = definition or hobday(min_duration_days=1, max_gap_days=0)
    statics = {"domain_mask": (("z", "y", "x"), domain),
               "dz": ("z", np.asarray(dz[:nz], float), {"units": "m"})}
    vertical_name = "z"
    z_coord = {"z": ("z", np.asarray(depth[:nz], float), {"units": "m", "positive": "down"})}
    if sigma is not None:
        statics["dz"] = (("z", "y", "x"), np.asarray(sigma[0], float), {"units": "m"})
        z_coord = {"depth": (("z", "y", "x"), np.asarray(sigma[1], float),
                             {"units": "m", "positive": "down"})}
        vertical_name = "depth"
    if unit_measure:
        measure = xhw.CellMeasure(kind="unit")
    else:
        statics["cell_area"] = (("y", "x"), area, {"units": "m2"})
        measure = xhw.CellMeasure(kind="area", var="cell_area", units="m2")
    ds = mask_dataset(times, ("time", "z", "y", "x"), np.where(exc, 13.0, 10.0), 12.0, 10.0, exc,
                      exc.copy(), np.zeros_like(exc), reason, definition.variable,
                      {**z_coord, "lat": ("y", lat), "lon": ("x", lon)}, statics)
    g = grid(support="volume", topology="structured", spatial_dims=("z", "y", "x"),
             coords=xhw.CoordRoles(lat="lat", lon="lon", depth=vertical_name),
             cell_measure=measure,
             vertical=xhw.VerticalSpec(coordinate=vertical_name, thickness="dz",
                                       thickness_units="m",
                                       time_dependent=False), periodic_x=periodic)
    return xhw.XMask.from_dataset(ds, grid=g, definition=definition, baseline=baseline(),
                                        provenance=provenance(str(uuid.uuid4())))


def make_line_mask(exc, *, length=None, along=None, definition=None,
                   start: str = "2000-01-01") -> xhw.XMask:
    """XMask of a line (time, along); cells of `length` m (default 1000) at `along` m."""
    exc = np.asarray(exc, bool)
    T, n = exc.shape
    length = np.full(n, 1000.0) if length is None else np.asarray(length, float)
    along = 1000.0 * np.arange(n) if along is None else np.asarray(along, float)
    times = (np.datetime64(start) + np.arange(T)).astype("datetime64[ns]")
    definition = definition or hobday(min_duration_days=1, max_gap_days=0)
    ds = mask_dataset(times, ("time", "along"), np.where(exc, 13.0, 10.0), 12.0, 10.0, exc,
                      exc.copy(), np.zeros_like(exc), np.zeros(exc.shape, np.uint8),
                      definition.variable, {"along": ("along", along, {"units": "m"})},
                      {"domain_mask": ("along", np.ones(n, bool)),
                       "segment_length": ("along", length, {"units": "m"})})
    g = grid(support="line", topology="transect", spatial_dims=("along",),
             coords=xhw.CoordRoles(along="along"),
             cell_measure=xhw.CellMeasure(kind="length", var="segment_length", units="m"))
    return xhw.XMask.from_dataset(ds, grid=g, definition=definition, baseline=baseline(),
                                        provenance=provenance(str(uuid.uuid4())))


def rotated_coordinates(ny: int, nx: int, angle_deg: float = 25.0, step: float = 1.0,
                        lat0: float = -3.0, lon0: float = -80.0):
    """2-D latitude and longitude of a lattice rotated by `angle_deg` (curvilinear test grid)."""
    j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    a = np.deg2rad(angle_deg)
    return (lat0 + step * (j * np.cos(a) + i * np.sin(a)),
            lon0 + step * (i * np.cos(a) - j * np.sin(a)))
