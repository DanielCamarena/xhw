# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Synthetic signal and fixed eigentriple groups for stored SSA comparisons."""
from __future__ import annotations

from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "reference" / "ssa"
SEED = 20261003
N, WINDOW, RANK = 366 * 20, 1830, 12
GROUPS = {"trend": [0], "annual": [1, 2], "semiannual": [3, 4]}

def series() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    t = np.arange(N)
    return (15 + 0.0002 * t + 3 * np.cos(2 * np.pi * t / 366)
            + 0.8 * np.sin(4 * np.pi * t / 366) + rng.normal(0, 0.4, N))
