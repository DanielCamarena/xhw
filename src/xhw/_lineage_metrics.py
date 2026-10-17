# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Metrics of branches and spacetime events (genealogy, and of declared numerical rules).

Level 1 (scalars, stored): size maximum and the day of that peak, size-days (the sum of daily
sizes, which is **not** the footprint), the footprint (union of occupied cells, computed on
the label field, never by adding branches), the net displacement and the path length of the
daily centroid (horizontal great-circle distance on surfaces and volumes, kept apart from the
vertical one in volumes; along-line distance on lines), and in volumes the horizontal
footprint (columns occupied at some level) and the depth range. "Size" is a length, an area or
a volume according to the support. Level 2 (series) is the node table grouped by branch or
event. Level 3 (maps) is computed on demand. Intensity (mean, maximum, cumulative, the day of
its peak) needs a `field` and is off by default; the daily spatial mean is weighted by the
cell measure unless ``weighted=False`` (the plain mean of cells, an approximation).

A group is given by ``(node, group)`` pairs (0-based group), so a node may count in several
groups: in monitoring an event covers the nodes of the events it absorbed.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS = 6_371_000.0      # m, as in the object fixtures and the object tracking areas
_CHUNK = 64                     # days per pass over the label field


def great_circle(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distance (m) between points in degrees (haversine)."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp, dl = p2 - p1, np.deg2rad(np.asarray(lon2) - np.asarray(lon1))
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def cell_passes(labels, table, pn, pg, values=None):
    """Yield, per chunk of days, ``(group, cell, day, value, node)`` for every occupied cell of
    every member node (cells are flat indices of the (z*y, x) view); a node in several groups
    yields its cells once per group.
    """
    T, _, nx = labels.shape
    order = np.argsort(pn, kind="stable")
    counts = np.bincount(pn, minlength=table.n)
    starts = np.concatenate(([0], np.cumsum(counts)))[:-1]
    groups = pg[order].astype(np.int64)
    for t0 in range(0, T, _CHUNK):
        tt, yy, xx = np.nonzero(labels[t0:t0 + _CHUNK])
        tt = tt + t0
        gid = table.base[tt] + labels[tt, yy, xx] - 1
        cell = yy.astype(np.int64) * nx + xx
        rep = counts[gid]
        offs = np.arange(int(rep.sum())) - np.repeat(np.cumsum(rep) - rep, rep)
        group = groups[np.repeat(starts[gid], rep) + offs]
        val = None if values is None else np.repeat(values[tt, yy, xx], rep)
        yield group, np.repeat(cell, rep), np.repeat(tt, rep), val, np.repeat(gid, rep)


def _depth_order(parent: np.ndarray) -> np.ndarray:
    """Groups ordered so that every group comes after the groups merged into it (deepest first)."""
    k = parent.size
    depth = np.full(k, -1, np.int64)
    up = parent.tolist()
    for g in range(k):                          # each group is walked once (memoised)
        path, e = [], g
        while e >= 0 and depth[e] < 0:
            path.append(e)
            e = up[e]
        d = depth[e] if e >= 0 else -1
        for node in reversed(path):
            d += 1
            depth[node] = d
    return np.argsort(-depth, kind="stable")


def footprint_metrics(labels, weights, table, pn, pg, k, *, parent=None, columns=None,
                      column_weight=None, depth=None) -> dict:
    """Footprint (union of cells, in the cell measure) of each group; in volumes also the
    horizontal footprint (union of columns) and the depth range of the occupied cells.

    `pn`, `pg` give each node once (its own group). `parent` (0-based, -1 for none) is the
    group each group was merged into (monitoring): the footprint of a group is then the union
    of its own cells and of the footprints of the groups merged into it, propagated up the
    merge forest so that a node is never expanded once per absorbing group (each set has at
    most the spatial cells of the grid).
    """
    flat_w = weights.ravel()
    ncell = flat_w.size
    ncol = int(columns.max()) + 1 if columns is not None and columns.size else 1
    keys = np.zeros(0, np.int64)
    ckeys = np.zeros(0, np.int64)
    dmin, dmax = np.full(k, np.inf), np.full(k, -np.inf)
    for group, cell, _, _, _ in cell_passes(labels, table, pn, pg):
        keys = np.union1d(keys, group * ncell + cell)
        if columns is not None:
            ckeys = np.union1d(ckeys, group * ncol + columns[cell])
        if depth is not None and group.size:
            np.minimum.at(dmin, group, depth[cell])
            np.maximum.at(dmax, group, depth[cell])
    if parent is None or not (np.asarray(parent) >= 0).any():
        out = {"footprint": np.bincount(keys // ncell, weights=flat_w[keys % ncell],
                                        minlength=k)}
        if columns is not None:
            out["footprint_horizontal"] = np.bincount(ckeys // ncol,
                                                      weights=column_weight[ckeys % ncol],
                                                      minlength=k)
    else:
        parent = np.asarray(parent)
        own = np.split(keys % ncell, np.searchsorted(keys // ncell, np.arange(1, k)))
        cown = (np.split(ckeys % ncol, np.searchsorted(ckeys // ncol, np.arange(1, k)))
                if columns is not None else None)
        pending: dict[int, list] = {}
        cpending: dict[int, list] = {}
        foot, cfoot = np.zeros(k), np.zeros(k)
        for g in _depth_order(parent).tolist():
            cells = np.unique(np.concatenate([own[g], *pending.pop(g, [])]))
            foot[g] = float(flat_w[cells].sum())
            if columns is not None:
                cols = np.unique(np.concatenate([cown[g], *cpending.pop(g, [])]))
                cfoot[g] = float(column_weight[cols].sum())
            p = int(parent[g])
            if p >= 0:
                pending.setdefault(p, []).append(cells)
                if columns is not None:
                    cpending.setdefault(p, []).append(cols)
                dmin[p] = min(dmin[p], dmin[g])
                dmax[p] = max(dmax[p], dmax[g])
        out = {"footprint": foot}
        if columns is not None:
            out["footprint_horizontal"] = cfoot
    if depth is not None:
        out["depth_min"] = np.where(np.isfinite(dmin), dmin, np.nan)
        out["depth_max"] = np.where(np.isfinite(dmax), dmax, np.nan)
    return out


def node_intensity(labels, weights, table, values, weighted: bool) -> dict:
    """Intensity of every node: maximum, spatial mean (weighted by the cell measure or not) and
    cumulative (sum of value x cell measure, one day). Cells with a missing value are skipped.
    """
    n = table.n
    out = {"max": np.full(n, np.nan), "mean": np.full(n, np.nan), "cumulative": np.zeros(n),
           "weight": np.zeros(n), "count": np.zeros(n)}
    nx = labels.shape[2]
    flat_w = weights.ravel()
    total = np.zeros(n)
    for t0 in range(0, labels.shape[0], _CHUNK):
        tt, yy, xx = np.nonzero(labels[t0:t0 + _CHUNK])
        tt = tt + t0
        gid = table.base[tt] + labels[tt, yy, xx] - 1
        v = values[tt, yy, xx].astype(float)
        ok = np.isfinite(v)
        gid, v, w = gid[ok], v[ok], flat_w[(yy * nx + xx)[ok]]
        np.fmax.at(out["max"], gid, v)
        out["cumulative"] += np.bincount(gid, weights=v * w, minlength=n)
        out["weight"] += np.bincount(gid, weights=w, minlength=n)
        out["count"] += np.bincount(gid, minlength=n)
        total += np.bincount(gid, minlength=n, weights=v)
    if weighted:
        out["mean"] = np.where(out["weight"] > 0, out["cumulative"]
                               / np.where(out["weight"] > 0, out["weight"], 1), np.nan)
    else:
        out["mean"] = np.where(out["count"] > 0, total / np.where(out["count"] > 0,
                                                                  out["count"], 1), np.nan)
    return out


# --- daily tables propagated through the merge forest ------------------------------------------
#
# Every metric of a group comes from its daily table: one row per day with a patch, holding sums
# over the nodes of that day (size, the size-weighted centroid vector, counts and, with a field,
# intensity sums and the daily maximum). A group's table is built from its own nodes and, in
# monitoring, the tables of the groups merged into it are added day by day, deepest first. Each
# table has at most one row per day, so nothing grows with the length of a merge chain (the first
# version expanded every node once per absorbing event: 31 million pairs in a large domain like
# the North Pacific).

_MAX_COLUMNS = ("imax",)


def _reduce(g, d, cols: dict, T: int):
    """Rows with the same (group, day) summed (maxima for `_MAX_COLUMNS`), sorted by both."""
    key = g.astype(np.int64) * T + d
    rows, inv = np.unique(key, return_inverse=True)
    out = {}
    for name, v in cols.items():
        if name in _MAX_COLUMNS:
            m = np.full(rows.size, np.nan)
            np.fmax.at(m, inv, v)
            out[name] = m
        else:
            out[name] = np.bincount(inv, weights=v, minlength=rows.size)
    return rows // T, rows % T, out


def node_columns(table, support, coords, node_int=None, weighted=True, flags=None) -> dict:
    """Per-node columns of the daily tables."""
    size = table.area
    cols = {"size": size.astype(float), "n": np.ones(table.n)}
    if support == "line":
        cols["along"] = size * coords["along"]
    else:
        phi, lam = np.deg2rad(coords["lat"]), np.deg2rad(coords["lon"])
        cols["vx"] = size * np.cos(phi) * np.cos(lam)
        cols["vy"] = size * np.cos(phi) * np.sin(lam)
        cols["vz"] = size * np.sin(phi)
    if support == "volume":
        cols["depth"] = size * coords["depth"]
    for name, values in (flags or {}).items():
        cols[name] = np.asarray(values, float)
    if node_int is not None:
        w = node_int["weight"] if weighted else node_int["count"]
        cols["imax"] = node_int["max"]
        cols["icum"] = node_int["cumulative"]
        cols["inum"] = np.nan_to_num(node_int["mean"]) * w
        cols["iden"] = w
    return cols


def group_tables(group, day, cols: dict, k: int, T: int, parent=None):
    """Final daily tables of the groups: (group, day, columns), sorted by group then day."""
    g, d, c = _reduce(group, day, cols, T)
    if parent is None or not (np.asarray(parent) >= 0).any():
        return g, d, c
    parent = np.asarray(parent)
    bounds = np.searchsorted(g, np.arange(k + 1))
    pending: dict[int, list] = {}
    final: list = [None] * k
    names = list(c)
    for e in _depth_order(parent).tolist():
        lo, hi = bounds[e], bounds[e + 1]
        parts = [(d[lo:hi], {n: c[n][lo:hi] for n in names})] + pending.pop(e, [])
        if len(parts) == 1:
            dd, cc = parts[0]
        else:
            days = np.concatenate([p[0] for p in parts])
            merged = {n: np.concatenate([p[1][n] for p in parts]) for n in names}
            _, dd, cc = _reduce(np.zeros(days.size, np.int64), days, merged, T)
        final[e] = (dd, cc)
        if parent[e] >= 0:
            pending.setdefault(int(parent[e]), []).append((dd, cc))
    g_out = np.concatenate([np.full(f[0].size, e, np.int64) for e, f in enumerate(final)]
                           + [np.zeros(0, np.int64)])
    d_out = np.concatenate([f[0] for f in final] + [np.zeros(0, np.int64)])
    c_out = {n: np.concatenate([f[1][n] for f in final] + [np.zeros(0)]) for n in names}
    return g_out, d_out, c_out


def union_counts(group, item, k: int, parent=None) -> np.ndarray:
    """Number of distinct items of each group, the items of merged groups included."""
    pairs = np.unique(np.stack([np.asarray(group, np.int64), np.asarray(item, np.int64)], 1)
                      .reshape(-1, 2), axis=0)
    if parent is None or not (np.asarray(parent) >= 0).any():
        return np.bincount(pairs[:, 0], minlength=k)
    bounds = np.searchsorted(pairs[:, 0], np.arange(k + 1))
    pending: dict[int, list] = {}
    out = np.zeros(k, np.int64)
    for e in _depth_order(np.asarray(parent)).tolist():
        items = np.unique(np.concatenate([pairs[bounds[e]:bounds[e + 1], 1],
                                          *pending.pop(e, [])]))
        out[e] = items.size
        if parent[e] >= 0:
            pending.setdefault(int(parent[e]), []).append(items)
    return out


def table_metrics(g, d, c: dict, k: int, support: str) -> dict:
    """Level-1 metrics of each group from its final daily table."""
    out = {"start": np.zeros(k, np.int64), "end": np.zeros(k, np.int64),
           "active": np.bincount(g, minlength=k), "size_max": np.zeros(k),
           "peak_day": np.zeros(k, np.int64),
           "size_days": np.bincount(g, weights=c["size"], minlength=k)}
    for name in ("n", "split", "merge", "provisional"):
        if name in c:
            out[name] = np.bincount(g, weights=c[name], minlength=k)
    if g.size:
        first = np.ones(g.size, bool)
        first[1:] = g[1:] != g[:-1]
        last = np.ones(g.size, bool)
        last[:-1] = g[:-1] != g[1:]
        out["start"][g[first]] = d[first]
        out["end"][g[last]] = d[last]
        order = np.lexsort((d, -c["size"], g))
        win = order[np.concatenate(([True], g[order][1:] != g[order][:-1]))]
        out["size_max"][g[win]] = c["size"][win]
        out["peak_day"][g[win]] = d[win]
    else:
        first = last = np.zeros(0, bool)
    weight = np.where(c["size"] > 0, c["size"], 1.0)
    same = g[1:] == g[:-1]
    if support == "line":
        along = c["along"] / weight
        step = np.abs(np.diff(along))
        net = np.abs(along[last] - along[first])
    else:
        lat = np.rad2deg(np.arctan2(c["vz"], np.hypot(c["vx"], c["vy"])))
        lon = np.rad2deg(np.arctan2(c["vy"], c["vx"]))
        step = great_circle(lat[:-1], lon[:-1], lat[1:], lon[1:])
        net = great_circle(lat[first], lon[first], lat[last], lon[last])
    out["path_length"] = np.bincount(g[1:][same], weights=step[same], minlength=k)
    out["net_displacement"] = np.zeros(k)
    out["net_displacement"][g[first]] = net
    if support == "volume":
        depth = c["depth"] / weight
        dstep = np.abs(np.diff(depth))
        out["vertical_path_length"] = np.bincount(g[1:][same], weights=dstep[same], minlength=k)
        out["net_vertical_displacement"] = np.zeros(k)
        out["net_vertical_displacement"][g[first]] = depth[last] - depth[first]
    if "imax" in c:
        out["intensity_max"] = np.full(k, np.nan)
        np.fmax.at(out["intensity_max"], g, c["imax"])
        out["intensity_cumulative"] = np.bincount(g, weights=c["icum"], minlength=k)
        num = np.bincount(g, weights=c["inum"], minlength=k)
        den = np.bincount(g, weights=c["iden"], minlength=k)
        out["intensity_mean"] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        out["intensity_cumulative"] = np.where(den > 0, out["intensity_cumulative"], np.nan)
        out["peak_intensity_day"] = np.full(k, -1, np.int64)
        ok = np.isfinite(c["imax"])
        if ok.any():
            gi, di, vi = g[ok], d[ok], c["imax"][ok]
            order = np.lexsort((di, -vi, gi))
            win = order[np.concatenate(([True], gi[order][1:] != gi[order][:-1]))]
            out["peak_intensity_day"][gi[win]] = di[win]
    return out
