# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Exact fields for the genealogy tests: hand-written split/merge cases and.

Patches are runs of unit cells on three rows (row 1 joins rows 0 and 2 when a patch spans
them); with 4-neighbour connectivity two runs on rows 0 and 2 of the same day are separate
patches. Every area and overlap below is the number of cells.
"""

from __future__ import annotations

import numpy as np
from _object_fixtures import make_mask, recipe, seg

import xhw

NX = 140


def lay(days: int, spec, ny: int = 3, nx: int = NX) -> np.ndarray:
    """Boolean field; spec is {day: [(row, first, stop),...]} with half-open column ranges."""
    exc = np.zeros((days, ny, nx), bool)
    for t, runs in spec.items():
        for row, a, b in runs:
            exc[t, row, a:b] = True
    return exc


def flip(exc: np.ndarray) -> np.ndarray:
    """The same case with rows reversed: patches that were first in raster order come last."""
    return exc[:, ::-1, :].copy()


# without a tie: areas O=80, P=100, Q=60, R=30, S=70, T=70; overlaps O∩P=70, P∩Q=50,
# P∩R=10, Q∩S=55, R∩S=5, S∩T=65.
NO_TIE = lay(5, {
    0: [(1, 0, 50), (0, 0, 30)],                                  # O
    1: [(0, 0, 50), (1, 0, 40), (2, 0, 10)],                      # P
    2: [(0, 0, 60), (2, 0, 30)],                                  # Q (row 0), R (row 2)
    3: [(0, 5, 60), (1, 25, 35), (2, 25, 30)],                    # S
    4: [(0, 5, 65), (1, 25, 35)],                                 # T
})

# with a tie: O=80, P=100, Q=50, R=50, S=90, T=90; O∩P=70, P∩Q=P∩R=20, Q∩S=R∩S=30, S∩T=80.
TIE = lay(5, {
    0: [(1, 0, 70), (0, 0, 10)],                                  # O
    1: [(0, 0, 20), (1, 0, 60), (2, 0, 20)],                      # P
    2: [(0, 0, 50), (2, 0, 50)],                                  # Q (row 0), R (row 2)
    3: [(0, 20, 50), (1, 20, 50), (2, 20, 50)],                   # S
    4: [(0, 20, 50), (1, 20, 60), (2, 30, 50)],                   # T
})

#: events A (50, 60, 70) and B (30, 40, 50) join on day 3 in M (100, then 80 and 60).
JOIN = lay(6, {
    0: [(0, 0, 50), (2, 0, 30)],
    1: [(0, 0, 60), (2, 0, 40)],
    2: [(0, 0, 70), (2, 0, 50)],
    3: [(0, 0, 60), (1, 0, 10), (2, 0, 30)],
    4: [(0, 0, 50), (1, 0, 10), (2, 0, 20)],
    5: [(0, 0, 50), (1, 0, 10)],
})


def objects(exc, *, mode="retrospective", track=True, gap=0, overlap=0.0, mask=None, **rules):
    """ObjectField of an exact field: 4-neighbour patches, no minimum area, unit cells."""
    mask = mask or make_mask(exc, unit_measure=True)
    steps = [seg()]
    if track:
        steps.append(xhw.RecipeStep.track(min_overlap=overlap, max_gap_days=gap, **rules))
    obj = xhw.build_objects(mask, recipe=recipe(*steps), analysis_mode=mode)
    obj.check_invariants()
    return obj


def lineage(exc, *, mode="retrospective", policy=None, merge_rule="new_event", **kw):
    obj = objects(exc, mode=mode, **kw)
    graph = xhw.build_lineage(obj, policy=policy, merge_rule=merge_rule)
    graph.check_invariants()
    return graph


def edges(graph, linked_only: bool = True) -> list[tuple[int, int]]:
    ds = graph.dataset
    keep = ds["linked"].values if linked_only else np.ones(ds.sizes["transition"], bool)
    return list(zip(ds["source_object"].values[keep].tolist(),
                    ds["target_object"].values[keep].tolist()))


def column(graph, name):
    return graph.dataset[name].values.tolist()
