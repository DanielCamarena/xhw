# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""aggregate_daily: sub-daily samples to a daily statistic on a declared day.

Each input time stamp belongs to exactly one day of the target clock. ``instantaneous``
samples (e.g. ERA5 ``t2m``) of day D are the stamps in [start, end); ``period_ending`` samples
(ERA5 ``mx2t``/``mn2t``: extreme of the period ending at the stamp) are the stamps in (start,
end]. Days with fewer valid samples than required are NaN; nothing is extrapolated and units
are never converted.

The processing step is written in the attributes of a companion variable ``<name>_n_samples``
(valid samples per day), not in the temperature attributes. `fit_baseline` and `detect` use it
only as a consistency check against the DefinitionSpec (atmospheric detection-5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ._schema import check_choice, check_nonempty_str, is_int, require
from ._specs import ProcessingStep, clock_offset

STEP_ATTR = "xhw_step"
STEP_NAME = "aggregate_daily"
_CF_METHODS = {"max": "maximum", "min": "minimum", "mean": "mean"}
_DAY = np.timedelta64(1, "D")
_STEP_KEYS = ("source_variable", "statistic", "sampling", "input_clock", "day_definition",
              "sample_seconds", "samples_per_day", "min_samples_per_day")


def companion_name(name: str) -> str:
    """Name of the variable holding valid samples per day and the aggregation step."""
    return f"{name}_n_samples"


def _sample_step(times: np.ndarray, time_dim: str) -> np.timedelta64:
    require(times.dtype.kind == "M", f"{time_dim!r} must be datetime64")
    require(times.size >= 2, "need at least two time stamps to infer the sampling frequency")
    steps = np.diff(times)
    step = steps[0]
    require(bool(np.all(steps == step)) and step > np.timedelta64(0, "ns"),
            "input time axis is not regular (duplicates, gaps or irregular steps); reindex "
            "explicitly and mark added stamps as missing")
    require(_DAY % step == np.timedelta64(0, "ns"),
            f"sampling step {step} does not divide one day")
    return step


def aggregate_daily(ds: xr.Dataset, *, variable: str, statistic: str, sampling: str,
                    input_clock: str, day_definition: str, time_dim: str = "time",
                    min_samples_per_day: int | None = None, name: str | None = None
                    ) -> xr.Dataset:
    """Aggregate regularly sampled data to one value per day of `day_definition`.

    `statistic` is ``"max"``, ``"min"`` or ``"mean"``. `sampling` is ``"instantaneous"`` or
    ``"period_ending"``. `input_clock` is the clock of the input stamps and `day_definition`
    the target day, both ``"UTC"`` or ``"offset:+HH:MM"``. `min_samples_per_day` defaults to a
    complete day for the sampling frequency. Returns the static variables of `ds`, the daily
    variable `name` (default `variable`) labelled with the local date at 00:00, and
    ``<name>_n_samples``. Lazy for dask input.
    """
    require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")
    check_nonempty_str("variable", variable)
    require(variable in ds.data_vars, f"variable {variable!r} not found")
    check_choice("statistic", statistic, tuple(_CF_METHODS))
    check_choice("sampling", sampling, ("instantaneous", "period_ending"))
    input_offset = clock_offset("input_clock", input_clock)
    day_offset = clock_offset("day_definition", day_definition)
    require(time_dim in ds.coords, f"time coordinate {time_dim!r} not found")
    da = ds[variable]
    require(time_dim in da.dims, f"{variable!r} has no {time_dim!r} dimension")
    times = np.asarray(ds[time_dim].values)
    step = _sample_step(times, time_dim)
    per_day = int(_DAY // step)
    if min_samples_per_day is None:
        min_samples_per_day = per_day
    require(is_int(min_samples_per_day) and 1 <= min_samples_per_day <= per_day,
            f"min_samples_per_day must be an int in [1, {per_day}]")
    name = variable if name is None else name
    check_nonempty_str("name", name)
    static = ds.drop_dims(time_dim)
    count_name = companion_name(name)
    for out_name in (name, count_name):
        require(out_name not in static.variables, f"output name {out_name!r} already in ds")

    # Stamp -> day of the target clock: floor of the shifted stamp; a period-ending stamp is
    # moved to the start of its period first, so 00:00 belongs to the previous day.
    shift = (day_offset - input_offset).astype("timedelta64[ns]")
    if sampling == "period_ending":
        shift = shift - step
    shifted = da.assign_coords({time_dim: times + shift})
    counts = shifted.notnull().resample({time_dim: "1D"}).sum()
    values = getattr(shifted.resample({time_dim: "1D"}), statistic)(skipna=True)
    values = values.where(counts >= min_samples_per_day)
    days = pd.DatetimeIndex(values[time_dim].values)
    require(bool(np.all(days == days.normalize())), "internal error: day labels not at 00:00")

    step_attrs = {
        STEP_ATTR: STEP_NAME, "source_variable": variable, "statistic": statistic,
        "sampling": sampling, "input_clock": input_clock, "day_definition": day_definition,
        "sample_seconds": int(step // np.timedelta64(1, "s")), "samples_per_day": per_day,
        "min_samples_per_day": int(min_samples_per_day),
        "long_name": "valid input samples per day", "units": "1",
    }
    value_attrs = dict(da.attrs)
    value_attrs.update({"cell_methods": f"{time_dim}: {_CF_METHODS[statistic]}",
                        "ancillary_variables": count_name})
    return static.assign({
        name: values.assign_attrs(value_attrs),
        count_name: counts.astype(np.int32).assign_attrs(step_attrs),
    })


def _plain(value):
    """NetCDF attributes come back as numpy scalars; provenance stores plain JSON values."""
    return value.item() if isinstance(value, np.generic) else value


def daily_steps(ds: xr.Dataset, variables: tuple[str, ...], statistics: tuple[str, ...],
                day_definition: str) -> list[ProcessingStep]:
    """Consistency of daily inputs with the declared day and statistic (atmospheric detection-5).

    Aggregation step present and matching -> recorded; present and different -> error; absent
    -> the declaration is accepted and recorded as not verified (only for a declared,
    non-native day definition).
    """
    steps = []
    for variable, statistic in zip(variables, statistics):
        companion = companion_name(variable)
        attrs = ds[companion].attrs if companion in ds.variables else {}
        if attrs.get(STEP_ATTR) == STEP_NAME:
            found = (attrs.get("day_definition"), attrs.get("statistic"))
            require(found == (day_definition, statistic),
                    f"{variable!r} was aggregated with day_definition={found[0]!r}, "
                    f"statistic={found[1]!r}, but the DefinitionSpec declares "
                    f"day_definition={day_definition!r}, daily_statistic={statistic!r}")
            params = (("variable", variable),
                      *((k, _plain(attrs[k])) for k in _STEP_KEYS if k in attrs))
            steps.append(ProcessingStep(name=STEP_NAME, params=params))
        elif day_definition != "native":
            steps.append(ProcessingStep(name="daily_input_declared", params=(
                ("variable", variable), ("day_definition", day_definition),
                ("daily_statistic", statistic), ("verified", False))))
    return steps
