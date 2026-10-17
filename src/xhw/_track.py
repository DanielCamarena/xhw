# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Overlap tracking of daily patches: links, object gaps and track identities.

Objects are the daily patches of `label_field`, numbered globally in raster order (day,
label). Two objects of consecutive days are linked when their overlap area is positive and
their ``link_measure`` (default: shared area over the area of the *smaller* object) is at
least ``min_overlap`` (``>=``). With ``max_gap_days = g`` an object that has no continuation
may reappear up to ``g`` empty days later and still be the same object, if the reappearing
object overlaps the last footprint it had; each reappearing object is bridged to one
candidate, the one with the largest ``dominance_measure`` (ties: ``tie_rule``).

``retrospective``: the connected components of the link graph are the tracks (a merge gives A
and B the same identity from their first day: later data rewrite the past). ``monitoring``:
identities are assigned forward in time and never rewritten. With ``branch_rule="inherit"``
(the object tracking default) on a merge the object keeps the identity of the dominant parent
(ties: lowest identity) and the other track ends, on a split the dominant child inherits and
the others are born as new tracks; with ``"terminate_and_create"`` no child inherits on a
split and a merge opens a new identity. Splits and merges are recorded, not resolved: the
lineage graph records their genealogy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Table:
    """Objects (daily patches) in raster order (day, label)."""

    day: np.ndarray        # (n,) time index
    label: np.ndarray      # (n,) label within the day
    area: np.ndarray       # (n,) summed cell measure
    cells: np.ndarray      # (n,) number of cells
    first_cell: np.ndarray  # (n,) flat index (y * nx + x) of the first cell of the day
    base: np.ndarray       # (T + 1,) global id of the first object of each day

    @property
    def n(self) -> int:
        return int(self.day.size)


@dataclass(frozen=True)
class Links:
    src: np.ndarray     # earlier object
    dst: np.ndarray     # later object
    ratio: np.ndarray   # value of the link measure (default: shared / area of the smaller)
    gap: np.ndarray     # empty days between the two (0 = consecutive days)
    shared: np.ndarray  # shared cell measure


def measure(name: str, shared, area_src, area_dst):
    """Overlap measure of pairs: ``overlap_smaller``, ``iou``, ``source_fraction``,
    ``target_fraction`` or ``shared``.
    """
    shared = np.asarray(shared, dtype=float)
    if name == "shared":
        return shared
    if name == "overlap_smaller":
        return shared / np.minimum(area_src, area_dst)
    if name == "iou":
        return shared / (area_src + area_dst - shared)
    if name == "source_fraction":
        return shared / area_src
    if name == "target_fraction":
        return shared / area_dst
    raise ValueError(f"unknown overlap measure {name!r}")


def dominance(links: Links, table: Table, name: str) -> np.ndarray:
    """Dominance score of every link (larger = more dominant)."""
    return measure(name, links.shared, table.area[links.src], table.area[links.dst])


def overlap_pairs(la: np.ndarray, lb: np.ndarray, weights: np.ndarray):
    """Labels of `la` and `lb` that share cells and the shared weight of each pair."""
    both = (la > 0) & (lb > 0)
    if not both.any():
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0)
    nb = int(lb.max()) + 1
    key = la[both].astype(np.int64) * nb + lb[both]
    unique, inverse = np.unique(key, return_inverse=True)
    return unique // nb, unique % nb, np.bincount(inverse, weights=weights[both])


def _pairs(labels, weights, table: Table, s: int, d: int, min_overlap: float,
           link_measure: str = "overlap_smaller"):
    a, b, shared = overlap_pairs(labels[s], labels[d], weights)
    if not a.size:
        return a, b, shared, shared
    ga, gb = table.base[s] + a - 1, table.base[d] + b - 1
    value = measure(link_measure, shared, table.area[ga], table.area[gb])
    keep = (shared > 0) & (value >= min_overlap)
    return ga[keep], gb[keep], value[keep], shared[keep]


def build_links(labels: np.ndarray, weights: np.ndarray, table: Table, *, min_overlap: float,
                max_gap_days: int, link_measure: str = "overlap_smaller",
                dominance_measure: str = "overlap_smaller", tie_rule: str = "lowest_id") -> Links:
    """Consecutive-day links plus, if `max_gap_days` > 0, the links across empty days."""
    days = labels.shape[0]
    counts = np.diff(table.base)
    src, dst, ratio, gap, shared = [], [], [], [], []
    for t in range(days - 1):
        if counts[t] and counts[t + 1]:
            ga, gb, r, sh = _pairs(labels, weights, table, t, t + 1, min_overlap, link_measure)
            src.append(ga), dst.append(gb), ratio.append(r), shared.append(sh)
            gap.append(np.zeros(ga.size, int))
    s_all = np.concatenate(src) if src else np.zeros(0, dtype=np.int64)
    d_all = np.concatenate(dst) if dst else np.zeros(0, dtype=np.int64)
    if max_gap_days > 0:
        forward_linked = np.zeros(table.n, bool)
        backward_linked = np.zeros(table.n, bool)
        forward_linked[s_all], backward_linked[d_all] = True, True
        used = np.zeros(table.n, bool)
        for d in range(2, days):
            if not counts[d] or backward_linked[table.base[d]:table.base[d + 1]].all():
                continue
            found = []
            for k in range(2, min(max_gap_days + 1, d) + 1):
                s = d - k
                if not counts[s]:
                    continue
                ga, gb, r, sh = _pairs(labels, weights, table, s, d, min_overlap, link_measure)
                ok = ~forward_linked[ga] & ~used[ga] & ~backward_linked[gb]
                found.append((r[ok], ga[ok], gb[ok], np.full(int(ok.sum()), k - 1), sh[ok]))
            if not found:
                continue
            r, ga, gb, gp, sh = (np.concatenate(x) for x in zip(*found))
            score = measure(dominance_measure, sh, table.area[ga], table.area[gb])
            keys = (ga, gb, -score)                       # best first; ties: lowest ids
            if tie_rule == "largest_area":
                keys = (ga, gb, -(table.area[ga] + table.area[gb]), -score)
            for i in np.lexsort(keys):
                if not used[ga[i]] and not backward_linked[gb[i]]:
                    used[ga[i]], backward_linked[gb[i]] = True, True
                    src.append(ga[i:i + 1]), dst.append(gb[i:i + 1])
                    ratio.append(r[i:i + 1]), gap.append(gp[i:i + 1])
                    shared.append(sh[i:i + 1])
    if not src:
        z = np.zeros(0, dtype=np.int64)
        return Links(z, z, np.zeros(0), z, np.zeros(0))
    return Links(*(np.concatenate(x) for x in (src, dst, ratio, gap, shared)))


def retrospective_tracks(table: Table, links: Links, cells_per_day: int) -> np.ndarray:
    """Track number (1-based) of every object: connected components of the link graph, numbered
    by first day and then by the position of the first cell (deterministic).
    """
    n = table.n
    if n == 0:
        return np.zeros(0, dtype=np.int32)
    graph = coo_matrix((np.ones(links.src.size), (links.src, links.dst)), shape=(n, n))
    components = connected_components(graph, directed=False)[1]
    key = table.day.astype(np.int64) * cells_per_day + table.first_cell
    best = np.full(components.max() + 1, np.iinfo(np.int64).max)
    np.minimum.at(best, components, key)
    number = np.empty(best.size, dtype=np.int32)
    number[np.argsort(best, kind="stable")] = np.arange(1, best.size + 1, dtype=np.int32)
    return number[components]


def monitoring_tracks(table: Table, links: Links, *, dominance_measure: str = "overlap_smaller",
                      tie_rule: str = "lowest_id", branch_rule: str = "inherit"):
    """Forward identities that are never rewritten. Returns (track per object, parent track by
    split per track, track it merged into per track), 1-based, 0 = none.
    """
    n = table.n
    area = table.area
    track = np.zeros(n, dtype=np.int32)
    parents: list[int] = [0]
    merged: list[int] = [0]
    score_all = dominance(links, table, dominance_measure)
    order = np.lexsort((links.src, links.dst))
    dst_sorted, src_sorted, score_sorted = links.dst[order], links.src[order], score_all[order]
    start = np.searchsorted(dst_sorted, np.arange(n + 1))
    children: dict[int, list[tuple[float, int]]] = {}
    for s, d, r in zip(links.src.tolist(), links.dst.tolist(), score_all.tolist()):
        children.setdefault(s, []).append((r, d))
    by_area = tie_rule == "largest_area"

    def child_key(kd):                      # dominant child of a parent
        return (-kd[0], -area[kd[1]], kd[1]) if by_area else (-kd[0], kd[1])

    def parent_key(ro):                     # dominant parent of a child: (score, object)
        return ((-ro[0], -area[ro[1]], track[ro[1]], ro[1]) if by_area
                else (-ro[0], track[ro[1]], ro[1]))

    heir = {s: min(kids, key=child_key)[1] for s, kids in children.items()}

    def new_track(parent: int) -> int:
        parents.append(parent)
        merged.append(0)
        return len(parents) - 1

    for obj in range(n):
        lo, hi = start[obj], start[obj + 1]
        preds = list(zip(score_sorted[lo:hi].tolist(), src_sorted[lo:hi].tolist()))
        if branch_rule == "terminate_and_create":
            if len(preds) == 1 and len(children[preds[0][1]]) == 1:
                track[obj] = track[preds[0][1]]            # 1 -> 1 keeps the identity
            elif len(preds) == 1:                          # a child of a split: new identity
                track[obj] = new_track(int(track[preds[0][1]]))
            else:                                          # birth, merge or 2 -> 2
                track[obj] = new_track(0)
                for _, o in preds:
                    if merged[track[o]] == 0:
                        merged[track[o]] = int(track[obj])
            continue
        chosen = [(r, o) for r, o in preds if heir[o] == obj]
        if chosen:
            best = min(chosen, key=parent_key)
            track[obj] = track[best[1]]
            for _, o in chosen:
                if track[o] != track[obj] and merged[track[o]] == 0:
                    merged[track[o]] = int(track[obj])
        else:
            track[obj] = new_track(int(track[min(preds, key=parent_key)[1]]) if preds else 0)
    return track, np.array(parents, dtype=np.int32), np.array(merged, dtype=np.int32)


def multi_overlap(table: Table, links: Links) -> np.ndarray:
    """Objects linked to more than one object of an adjacent day (a split or a merge)."""
    n = table.n
    return (np.bincount(links.src, minlength=n) > 1) | (np.bincount(links.dst, minlength=n) > 1)


def track_summary(table: Table, track: np.ndarray, flagged: np.ndarray, days: int,
                  min_duration_days: int) -> dict[str, np.ndarray]:
    """Per-track figures for the tracks 1..max(track), indexed by track number - 1."""
    k = int(track.max()) if track.size else 0
    out: dict[str, np.ndarray] = {}
    if not k:
        return {name: np.zeros(0) for name in (
            "start", "end", "duration", "active", "bridged", "area_max", "area_mean",
            "area_days", "n_multi_overlap_days", "n_objects")}
    sel = track > 0
    t, day, area = track[sel] - 1, table.day[sel], table.area[sel]
    start = np.full(k, days, dtype=np.int64)
    end = np.full(k, -1, dtype=np.int64)
    np.minimum.at(start, t, day)
    np.maximum.at(end, t, day)
    per_day = np.unique(t.astype(np.int64) * days + day, return_inverse=True)
    day_area = np.bincount(per_day[1], weights=area)
    day_track = per_day[0] // days
    active = np.bincount(day_track, minlength=k)
    area_max = np.zeros(k)
    np.maximum.at(area_max, day_track, day_area)
    area_days = np.bincount(day_track, weights=day_area, minlength=k)
    flagged_days = np.unique(t[flagged[sel]].astype(np.int64) * days + day[flagged[sel]])
    out["start"], out["end"] = start, end
    out["duration"] = end - start + 1
    out["active"] = active
    out["bridged"] = out["duration"] - active
    out["area_max"] = area_max
    out["area_mean"] = area_days / active
    out["area_days"] = area_days
    out["n_multi_overlap_days"] = np.bincount(flagged_days // days, minlength=k)
    out["n_objects"] = np.bincount(t, minlength=k)
    return out
