# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Deterministic synthetic SST series for equivalence benchmarks (no real data)."""

from __future__ import annotations

from datetime import date

import numpy as np


def sst_series(seed: int, start: str = "1982-01-01", end: str = "2021-12-31") -> tuple:
    """Seasonal cycle + AR(1) noise + weak trend; daily, gap-free, includes leap years."""
    times = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + np.timedelta64(1, "D"))
    rng = np.random.default_rng(seed)
    n = times.size
    doy = (times - times.astype("datetime64[Y]")).astype(int)
    noise = np.empty(n)
    noise[0] = rng.normal(0, 0.8)
    shocks = rng.normal(0, 0.8 * np.sqrt(1 - 0.9 ** 2), n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + shocks[i]
    years = np.arange(n) / 365.25
    values = 18.0 + 3.0 * np.sin(2 * np.pi * (doy - 80) / 365.25) + 0.02 * years + noise
    ordinals = np.array([date.fromisoformat(str(t)).toordinal() for t in times])
    return times, ordinals, values
