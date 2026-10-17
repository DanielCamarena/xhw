# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Construction must not compute (time x space) data; check_invariants must."""

import pytest
from _fixtures import f5_surface

import xhw

dask = pytest.importorskip("dask")  # dask is in the `dev` extra; a skip here is a setup error

LAZY_VARS = ("temperature", "threshold", "reference", "exceedance", "event_membership",
             "joined_gap", "valid", "invalid_reason")


class ComputeForbidden(AssertionError):
    pass


def forbid_compute(*args, **kwargs):
    raise ComputeForbidden("dask computation during a lazy-only operation")


def lazy_surface():
    """F5 with (time x space) variables as dask arrays; coordinates and statics stay numpy."""
    fixture = f5_surface()
    ds = fixture.mask.dataset.copy(deep=True)
    for name in LAZY_VARS:
        ds[name] = ds[name].chunk({"time": 2})
    return fixture.mask, ds


def build(mask, ds):
    return xhw.XMask.from_dataset(ds, grid=mask.grid, definition=mask.definition,
                                        baseline=mask.baseline, provenance=mask.provenance)


def test_construction_access_and_serialization_do_not_compute():
    mask, ds = lazy_surface()
    with dask.config.set(scheduler=forbid_compute):
        lazy = build(mask, ds)
        out = lazy.to_dataset()
        intensity = lazy.intensity()
    assert all(lazy.dataset[name].chunks is not None for name in LAZY_VARS)
    assert out["exceedance"].chunks is not None and intensity.chunks is not None


def test_forbidding_scheduler_really_detects_computation():
    mask, ds = lazy_surface()
    with dask.config.set(scheduler=forbid_compute), pytest.raises(ComputeForbidden):
        ds["temperature"].values  # noqa: B018 - the access itself is the test


def test_check_invariants_computes_explicitly():
    mask, ds = lazy_surface()
    lazy = build(mask, ds)
    with dask.config.set(scheduler=forbid_compute), pytest.raises(ComputeForbidden):
        lazy.check_invariants()
    lazy.check_invariants()  # default scheduler: explicit computation succeeds
