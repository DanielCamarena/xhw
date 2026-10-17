# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""SSA against stored ssalib outputs for fixed trend and seasonal eigentriple groups.

Reconstruction error is bounded by 1e-8; nearly degenerate noise groups are excluded.
"""

import hashlib
import json

import _ssa_case as case
import numpy as np
import pandas as pd
import pytest

from xhw import _ssa


def reference(name):
    path = case.REF / name
    if not (case.REF / "manifest.json").exists() or not path.exists():
        pytest.fail("missing SSA reference files: reinstall the complete source distribution")
    manifest = json.loads((case.REF / "manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][name], name
    return path


def test_inputs_are_the_registered_series():
    np.testing.assert_allclose(np.loadtxt(reference("inputs.txt")), case.series(), atol=1e-12)


def test_signal_groups_equal_ssalib():
    ref = pd.read_csv(reference("ssalib.csv"))
    x = case.series()[None]
    sigma, u, v = _ssa.leading_triples(x, case.WINDOW, case.RANK)
    np.testing.assert_allclose(sigma[0, :5], ref["sigma"].to_numpy()[:5], rtol=1e-10)
    for name, members in case.GROUPS.items():
        weights = np.isin(np.arange(case.RANK), members)[None].astype(float)
        ours = _ssa.reconstruct(sigma, u, v, weights, x.shape[1])[0]
        diff = np.abs(ours - ref[name].to_numpy())
        print(f"{name}: max |xHW - ssalib| {diff.max():.1e}")
        assert diff.max() < 1e-8
