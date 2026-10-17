# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Calendar-following synthetic series for stored STL comparisons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REF = Path(__file__).parent / "reference" / "stl"
SEED = 20261003
TREND_WINDOW_DAYS = 3653
CASES = {"periodic": ("periodic", 0), "window7": (7, 0), "periodic_robust": ("periodic", 1)}

def dates() -> pd.DatetimeIndex:
    return pd.date_range("2005-01-01", "2019-12-31", freq="D")

def series() -> pd.Series:
    d = dates()
    rng = np.random.default_rng(SEED)
    peak = pd.DatetimeIndex([pd.Timestamp(year=x.year, month=7, day=15) for x in d])
    cycle = 3 * np.cos(2 * np.pi * (d - peak).days.to_numpy() / 365.25)
    red = np.zeros(d.size)
    for i in range(1, d.size):
        red[i] = 0.8 * red[i - 1] + rng.normal(0, 0.4)
    return pd.Series(17 + 0.03 * (d - d[0]).days.to_numpy() / 365.25 + cycle + red, index=d)

def grid_index(d: pd.DatetimeIndex) -> np.ndarray:
    """Same rule as xhw._dynamic: (year - first year) * 366 + calendar position - 1."""
    from xhw._baseline import doy_positions

    pos = doy_positions(d.to_numpy(), "hobday_366_interp")
    slots = (d.year.to_numpy() - d.year[0]) * 366 + pos - 1
    return slots - slots[0]

def on_grid() -> np.ndarray:
    from xhw._dynamic import _to_grid

    return _to_grid(series().to_numpy()[None, :], grid_index(dates()), interpolate=True)[0]
