# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Independent identity rule and node-to-cell mapping for stored lineage comparisons."""
from __future__ import annotations

from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "reference" / "lineage"
NAMES = ("blobs1", "blobs2", "blobs3", "blobs4")

def deeptrack_rule(objects, graph) -> np.ndarray:
    """The identities of DeepTrack's ``track_objects_with_splitting`` rebuilt from the xHW
    transition table: on the first day the local labels; then each patch takes the smallest
    identity of its linked parents of the previous day (splits share it, the smaller one wins
    a merge), and patches without parents get new identities in label order.
    """
    ds = graph.dataset
    times = objects.dataset[objects.grid.time_dim].values
    day = np.searchsorted(times, ds["object_time"].values)
    label = ds["object_label"].values
    t = graph.transitions()
    keep = t["linked"].values & (t["gap_days"].values == 0)
    parents: dict[int, list[int]] = {}
    for s, d in zip(t["source_object"].values[keep].tolist(),
                    t["target_object"].values[keep].tolist()):
        parents.setdefault(d, []).append(s)
    ident = np.zeros(ds.sizes["object"], np.int64)
    next_id = 1
    for obj in np.flatnonzero(day == 0):
        ident[obj] = label[obj]
        next_id = max(next_id, int(label[obj]) + 1)
    for d in range(1, times.size):
        new = []
        for obj in np.flatnonzero(day == d):
            found = parents.get(int(obj))
            if found:
                ident[obj] = min(ident[p] for p in found)
            else:
                new.append(obj)
        for obj in new:
            ident[obj] = next_id
            next_id += 1
    return ident

def node_field(objects, values) -> np.ndarray:
    """A value per node spread on the cells of its patch, on (time, *spatial dims)."""
    g = objects.grid
    lab = np.asarray(objects.dataset["label"].transpose(g.time_dim, *g.spatial_dims).values)
    day = np.searchsorted(objects.dataset[g.time_dim].values, objects.dataset["object_time"].values)
    base = np.concatenate(([0], np.cumsum(np.bincount(day, minlength=lab.shape[0]))))
    out = np.zeros(lab.shape, np.int64)
    idx = np.nonzero(lab)
    out[idx] = np.asarray(values)[base[idx[0]] + lab[idx] - 1]
    return out
