# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""genealogy: the lineage of xHW against Ocetrac 0.1.7 and marEx 4.1.2.

Reads versioned outputs (neither tool is needed here): ``reference/objects/labels.npz``
(object tracking) and ``reference/lineage/marex.npz``.

- **Ocetrac** ``label_temporal_neighbor(contain_thresh=t)``: the retrospective spacetime
  events of xHW (8-neighbour patches, ``LinkPolicy(threshold=t)``) are the same partition of
  the cells, for every threshold, as for the tracks of object tracking (the event is the
  connected component of the links).
- **marEx** with merging, ``overlap_threshold=0.5`` and its nearest-neighbour partition, on
  the field it tracked (``ID_field > 0``): ``xhw.partition_merges`` (8 neighbours, periodic x,
  the same threshold on the parts of the previous day) is compared merge by merge. A marEx
  merge is a patch whose cells carry two or more marEx ids (each part keeps the id of its
  parent). Equivalent on ``blobs4`` (same merges, same parents, 479 of 486 cells grouped
  alike). On the other masks every marEx merge is also an xHW merge (same day, same cells),
  but xHW finds more merges or more parents in some of them. **Deviation, documented, cause
  not found**: the area of the parents (the part, not the whole patch it came from) was tested
  as an explanation and agrees worse; marEx consolidates its ids while it partitions (declared
  numerical rules), which can change which parts of the previous day are parents. Measured
  numbers are pinned so that changes show.
- The marEx labels read back with `ingest_objects` keep marEx's ids as the backend ids of the
  nodes of the lineage.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from _lineage_case import NAMES, REF
from _object_fixtures import make_mask, make_volume_mask, recipe, seg  # noqa: E402
from _reference import load_npz

import xhw

OBJECTS_REF = Path(__file__).parent / "reference" / "objects"

pytestmark = pytest.mark.filterwarnings("ignore::xhw.XHWDataQualityWarning")

#: measured on the versioned reference (deviation, documented): marEx merges, xHW merges, merges
#: at the same patch with a different number of parents, events of marEx and of xHW
MAREX = {"blobs1": (12, 12, 2, 16, 16), "blobs2": (7, 9, 4, 14, 14),
         "blobs3": (2, 8, 1, 12, 11), "blobs4": (6, 6, 0, 20, 19)}


def load(path: Path, manifest: Path, key: str):
    if not manifest.exists() or not path.exists():
        pytest.fail(f"no reference yet: {path.name}")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["files"][key]
    return np.load(path)


@pytest.fixture(scope="module")
def ocetrac():
    return load(OBJECTS_REF / "labels.npz", OBJECTS_REF / "manifest.json", "labels.npz")


@pytest.fixture(scope="module")
def marex():
    return load(REF / "marex.npz", REF / "manifest.json", "marex.npz")


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    if not np.array_equal(a > 0, b > 0):
        return False
    pairs = np.unique(np.stack([a[a > 0], b[b > 0]], axis=1), axis=0)
    return (np.unique(pairs[:, 0]).size == pairs.shape[0]
            and np.unique(pairs[:, 1]).size == pairs.shape[0])


def event_field(obj: xhw.ObjectField, graph: xhw.LineageGraph) -> np.ndarray:
    """The spacetime event of every cell."""
    label = np.asarray(obj.dataset["label"].transpose("time", "y", "x").values)
    day = np.searchsorted(obj.dataset["time"].values, obj.dataset["object_time"].values)
    base = np.concatenate(([0], np.cumsum(np.bincount(day, minlength=label.shape[0]))))
    event = np.asarray(graph.dataset["object_spacetime_event"].values)
    out = np.zeros_like(label)
    tt, yy, xx = np.nonzero(label)
    out[tt, yy, xx] = event[base[tt] + label[tt, yy, xx] - 1]
    return out


@pytest.mark.parametrize("thr", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("name", ["blobs1", "blobs2", "blobs3", "blobs4"])
def test_events_are_the_components_of_ocetrac(ocetrac, name, thr):
    mask = ocetrac[f"in_{name}"]
    obj = xhw.build_objects(make_mask(mask, unit_measure=True), recipe=recipe(seg(rank=2)))
    graph = xhw.build_lineage(obj, policy=xhw.LinkPolicy(threshold=thr))
    graph.check_invariants()
    assert same_partition(event_field(obj, graph), ocetrac[f"ocetrac_{name}_{thr}"])


def merges(ids: np.ndarray, label: np.ndarray, parts: np.ndarray):
    """Merges of marEx and of xHW: {(day, patch): number of parts}."""
    out_m, out_x = {}, {}
    for t in range(label.shape[0]):
        for patch in range(1, int(label[t].max()) + 1):
            cells = label[t] == patch
            k_m, k_x = np.unique(ids[t][cells]).size, np.unique(parts[t][cells]).size
            if k_m > 1:
                out_m[(t, patch)] = k_m
            if k_x > 1:
                out_x[(t, patch)] = k_x
    return out_m, out_x


@pytest.mark.parametrize("name", NAMES)
def test_merges_against_marex(marex, name):
    ids = marex[f"id_{name}"]
    mask = make_mask(ids > 0, unit_measure=True, periodic=True)
    obj = xhw.build_objects(mask, recipe=recipe(seg(rank=2), xhw.RecipeStep.track(
        min_overlap=0.5)))
    parts = xhw.partition_merges(obj, mask)
    parts.check_invariants()
    parts.check_against(mask)
    label = np.asarray(obj.dataset["label"].transpose("time", "y", "x").values)
    split = np.asarray(parts.dataset["label"].transpose("time", "y", "x").values)
    found_m, found_x = merges(ids, label, split)
    assert set(found_m) <= set(found_x)          # every marEx merge is an xHW merge
    differ = sum(found_m[k] != found_x[k] for k in found_m)
    graph = xhw.build_lineage(parts)
    n_events = int(graph.dataset.sizes["spacetime_event"])
    n_marex = int(np.unique(ids[ids > 0]).size)
    assert (len(found_m), len(found_x), differ, n_marex, n_events) == MAREX[name]
    if name == "blobs4":                          # equivalent merges and partition
        assert found_m == found_x
        alike = total = 0
        for (t, patch) in found_m:
            cells = label[t] == patch
            pairs, counts = np.unique(np.stack([ids[t][cells], split[t][cells]], 1), axis=0,
                                      return_counts=True)
            best: dict[int, int] = {}
            for (a, _), c in zip(pairs.tolist(), counts.tolist()):
                best[a] = max(best.get(a, 0), c)
            alike += sum(best.values())
            total += int(cells.sum())
        assert (alike, total) == (479, 486)


def test_marex_labels_ingested_keep_their_ids_as_backend_ids(marex):
    ids = marex["id_blobs4"]
    mask = make_mask(ids > 0, unit_measure=True, periodic=True)
    labels = xr.DataArray(ids, dims=("time", "y", "x"),
                          coords={"time": mask.dataset["time"].values})
    source = xhw.SourceRef(dataset_id="marEx:4.1.2", variable="ID_field", accessed="2026-10-06",
                           product_status="final", version="4.1.2")
    obj = xhw.ingest_objects(labels, mask=mask, kind="tracks", source=source,
                             method="marEx tracker, overlap 0.5, nn partitioning")
    graph = xhw.build_lineage(obj, policy=xhw.LinkPolicy(threshold=0.5))
    graph.check_invariants()
    backend = np.asarray(graph.dataset["backend_id"].values)
    assert set(backend.tolist()) == set(np.unique(ids[ids > 0]).tolist())


# --- Ocetrac DeepTrack (volumes): track_objects_with_splitting ---------------------------------
#
# DeepTrack links a patch of day t to an identity of day t-1 when max(|A&B|/|A|, |A&B|/|B|) > t
# (strict) and takes the smallest identity of its parents; split children share it. With
# threshold 0 "linked" means "shares a cell", as in xHW, so its identities are rebuilt exactly from
# the xHW transition table with that rule. Above 0 it differs by design: the comparison is strict
# (xHW uses >=) and B is the union of all the patches of an identity on day t-1 (xHW compares
# patch by patch). The measured differences are pinned.

#: per seed: (DeepTrack identities, identities of the rule from xHW) at thresholds 0.3 and 0.5
DEEPTRACK_DEVIATION = {100: ((89, 88), (99, 92)), 101: ((94, 94), (106, 97)),
                       102: ((97, 96), (114, 98)), 103: ((104, 104), (115, 106))}


def deeptrack_volume(seed: int) -> np.ndarray:
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(30, 5, 14, 16))
    return ndi.gaussian_filter(noise, sigma=(1.5, 0.8, 1.0, 1.0)) > 0.15


@pytest.fixture(scope="module")
def deeptrack():
    return load_npz("deeptrack", "outputs.npz")


def deeptrack_case(deeptrack, mask4d, thr):
    from _lineage_case import deeptrack_rule, node_field
    nz = mask4d.shape[1]
    vm = make_volume_mask(mask4d, dz=(10.0,) * nz, depth=tuple(5.0 + 10.0 * np.arange(nz)),
                          unit_measure=True)
    obj = xhw.build_objects(vm, recipe=recipe(seg(rank=1)))
    seed = next(s for s in DEEPTRACK_DEVIATION
                if np.array_equal(mask4d, deeptrack[f"input_{s}"]))
    daily = deeptrack[f"daily_{seed}"]
    label = np.asarray(obj.dataset["label"].transpose("time", "z", "y", "x").values)
    tracked = deeptrack[f"tracked_{seed}_{thr}"]
    graph = xhw.build_lineage(obj, policy=xhw.LinkPolicy(threshold=thr))
    mine = node_field(obj, deeptrack_rule(obj, graph))
    return daily, label, tracked, mine, graph


@pytest.mark.parametrize("seed", sorted(DEEPTRACK_DEVIATION))
def test_deeptrack_identities_are_rebuilt_from_the_transition_table(deeptrack, seed):
    for thr in (0.0, 0.1):
        daily, label, tracked, mine, graph = deeptrack_case(deeptrack, deeptrack_volume(seed),
                                                            thr)
        assert np.array_equal(daily, label)               # the same daily 6-neighbour patches
        assert graph.dataset["is_merge"].values.sum() > 5  # with merges to decide
        assert np.array_equal(tracked, mine)


@pytest.mark.parametrize("seed", sorted(DEEPTRACK_DEVIATION))
def test_deeptrack_deviation_above_zero_is_pinned(deeptrack, seed):
    got = []
    for thr in (0.3, 0.5):
        _, _, tracked, mine, _ = deeptrack_case(deeptrack, deeptrack_volume(seed), thr)
        got.append((int(tracked.max()), int(mine.max())))
    assert tuple(got) == DEEPTRACK_DEVIATION[seed]
