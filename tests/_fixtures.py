# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Hand-built data contracts fixtures F1-F9.

Masks and catalogs are written by hand from docs/VALIDATION.md; nothing here calls a detector.
Constant synthetic baseline: C=10, Q90=12, Q10=8 (degC), strict comparison unless a fixture
states otherwise.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import xarray as xr

import xhw

SOURCE = xhw.SourceRef(dataset_id="synthetic:xhw-fixtures", variable="temperature",
                       accessed="2026-09-23", product_status="final")
NATIVE = xhw.NativeGrid(product="synthetic fixture grid", resolution="n/a", is_native=True)
POINT_COORDS = xhw.CoordRoles(lat="lat", lon="lon")
NONE_MEASURE = xhw.CellMeasure(kind="none")
# F7: lag_start = start_M - start_A = 1 - 4 days (VALIDATION, compound). Documented for declarative
# association,
#: not computed by xHW in data contracts.
EXPECTED_LAG_START_DAYS = -3


class Fixture(NamedTuple):
    mask: xhw.XMask
    catalog: xhw.EventCatalog


def provenance(run_id: str, parents: tuple[str, ...] = ()) -> xhw.Provenance:
    return xhw.Provenance(run_id=run_id, created_utc="2026-09-23T00:00:00+00:00",
                          software=(("xhw", "fixture"),), inputs=(SOURCE,), parents=parents)


def baseline(**overrides) -> xhw.BaselineSpec:
    params = dict(method="fixed_climatology", computed_on="temperature",
                  reference_period=("1970-01-01", "1999-12-31"), source=SOURCE,
                  doy_window_days=11, smoothing_window_days=31,
                  percentile_estimator="hyndman_fan_7", leap_day_policy="hobday_366_interp",
                  min_samples=30, fit_mode="retrospective", available_until=None)
    params.update(overrides)
    return xhw.BaselineSpec(**params)


def hobday(**overrides) -> xhw.DefinitionSpec:
    params = dict(profile="hobday2016", phenomenon="mhw", variable="sea_surface_temperature",
                  combination="single", daily_statistic="native", day_definition="native",
                  tail="upper", comparator=">", threshold_kind="percentile", percentile=90.0,
                  absolute_threshold=None, min_duration_days=5, max_gap_days=2,
                  join_rule="qualify_then_join", missing_policy="break", season_months=None,
                  category_scheme="hobday2018")
    params.update(overrides)
    return xhw.DefinitionSpec(**params)


def perkins2013(statistic: str = "max", **overrides) -> xhw.DefinitionSpec:
    """Profiles perkins2013_tx90 / perkins2013_tn90."""
    kind = "tx" if statistic == "max" else "tn"
    params = dict(profile=f"perkins2013_{kind}90", phenomenon="ahw", variable=f"{kind}_daily",
                  combination="single", daily_statistic=statistic,
                  day_definition="offset:-05:00", tail="upper", comparator=">",
                  threshold_kind="percentile", percentile=90.0, absolute_threshold=None,
                  min_duration_days=3, max_gap_days=0, join_rule="qualify_then_join",
                  missing_policy="as_non_exceedance", season_months=None, category_scheme=None)
    params.update(overrides)
    return xhw.DefinitionSpec(**params)


def txtn90(tn_lag: int = 0, **overrides) -> xhw.DefinitionSpec:
    """Profiles xhw_txtn90 (same day) and xhw_tx_then_tn90 (TN of the next day)."""
    profile = "xhw_txtn90" if tn_lag == 0 else "xhw_tx_then_tn90"
    params = dict(profile=profile, phenomenon="ahw", variable="tx_daily&tn_daily",
                  combination="and_daily", daily_statistic="native",
                  day_definition="offset:-05:00", tail="upper", comparator=">",
                  threshold_kind="percentile", percentile=None, absolute_threshold=None,
                  min_duration_days=3, max_gap_days=0, join_rule="qualify_then_join",
                  missing_policy="as_non_exceedance", season_months=None, category_scheme=None,
                  components=(xhw.DailyComponent(variable="tx_daily", daily_statistic="max",
                                                 percentile=90.0, lag_days=0),
                              xhw.DailyComponent(variable="tn_daily", daily_statistic="min",
                                                 percentile=90.0, lag_days=tn_lag)))
    params.update(overrides)
    return xhw.DefinitionSpec(**params)


def ahw_baseline(**overrides) -> xhw.BaselineSpec:
    """Baseline of the AHW profiles: 15-day window, type 8, no smoothing."""
    params = dict(doy_window_days=15, smoothing_window_days=None,
                  percentile_estimator="hyndman_fan_8")
    params.update(overrides)
    return baseline(**params)


def grid(**overrides) -> xhw.GridSpec:
    params = dict(support="point", topology="none", time_dim="time", spatial_dims=(),
                  coords=POINT_COORDS, calendar="standard", frequency="1D", time_axes="single",
                  periodic_x=False, cell_measure=NONE_MEASURE, vertical=None, domain="ocean",
                  domain_mask="domain_mask", domain_fraction=None, native=NATIVE)
    params.update(overrides)
    return xhw.GridSpec(**params)


def flags(n: int, *days: int) -> np.ndarray:
    out = np.zeros(n, dtype=bool)
    out[list(days)] = True
    return out


def days(first: int, last: int) -> list[int]:
    return list(range(first, last + 1))


def temperature_attrs(variable: str) -> dict:
    return {"units": "degC", "units_metadata": "temperature: on_scale", "standard_name": variable}


def mask_dataset(times, dims, temp, threshold, reference, exc, mem, gap, invalid_reason,
                 variable, coords, statics) -> xr.Dataset:
    """Assemble an XMask dataset; `valid` is written as invalid_reason == 0."""
    valid = np.asarray(invalid_reason) == 0
    shape = np.shape(temp)
    ds = xr.Dataset(
        {
            "temperature": (dims, np.asarray(temp, dtype=float), temperature_attrs(variable)),
            "threshold": (dims, np.broadcast_to(float(threshold), shape).copy(),
                          temperature_attrs(variable) | {"standard_name": "threshold"}),
            "reference": (dims, np.broadcast_to(float(reference), shape).copy(),
                          temperature_attrs(variable) | {"standard_name": "reference"}),
            "exceedance": (dims, np.asarray(exc, dtype=bool)),
            "event_membership": (dims, np.asarray(mem, dtype=bool)),
            "joined_gap": (dims, np.asarray(gap, dtype=bool)),
            "valid": (dims, valid),
            "invalid_reason": (dims, np.asarray(invalid_reason, dtype=np.uint8)),
            **statics,
        },
        coords={"time": times, **coords},
    )
    for name in ("threshold", "reference"):
        ds[name].attrs.pop("standard_name")
    return ds


def point_mask(temp, exc, mem, gap, *, definition, start="2000-01-01", invalid_reason=None,
               threshold=12.0, reference=10.0, grid_spec=None, run_id="mask", extra=None,
               extra_coords=None, base=None) -> xhw.XMask:
    n = len(temp)
    first = np.datetime64(start, "D")
    times = np.arange(first, first + np.timedelta64(n, "D")).astype("datetime64[ns]")
    reason = np.zeros(n, np.uint8) if invalid_reason is None else invalid_reason
    ds = mask_dataset(times, ("time",), temp, threshold, reference, exc, mem, gap, reason,
                      definition.variable, {"lat": -12.0, "lon": -77.5, **(extra_coords or {})},
                      {"domain_mask": ((), True)})
    for name, value in (extra or {}).items():
        ds[name] = value
    return xhw.XMask.from_dataset(ds, grid=grid_spec or grid(), definition=definition,
                                        baseline=base or baseline(),
                                        provenance=provenance(run_id))


def catalog(mask, rows, *, mode="retrospective", run_id=None) -> xhw.EventCatalog:
    """rows: (loc, start_idx, end_idx, duration, n_exc, n_gap, left, right, reason, qualified,
    contains_forecast) — all written by hand.
    """
    times = mask.dataset[mask.grid.time_dim].values
    loc_dims = mask.grid.spatial_dims
    n = len(rows)
    col = list(zip(*rows)) if rows else [[] for _ in range(11)]
    data = {
        "event_id": ("event", np.arange(1, n + 1, dtype=np.int64)),
        "start": ("event", times[list(col[1])]),
        "end": ("event", times[list(col[2])]),
        "duration_days": ("event", np.array(col[3], dtype=np.int64)),
        "n_exceedance_days": ("event", np.array(col[4], dtype=np.int64)),
        "n_joined_gap_days": ("event", np.array(col[5], dtype=np.int64)),
        "left_censored": ("event", np.array(col[6], dtype=bool)),
        "right_censored": ("event", np.array(col[7], dtype=bool)),
        "censor_reason": ("event", np.array(col[8], dtype=np.uint8)),
        "qualified": ("event", np.array(col[9], dtype=bool)),
        "contains_forecast": ("event", np.array(col[10], dtype=bool)),
    }
    for k, dim in enumerate(loc_dims):
        data[f"index_{dim}"] = ("event", np.array([loc[k] for loc in col[0]], dtype=np.int64))
    run = run_id or f"{mask.provenance.run_id}-catalog-{mode}"
    return xhw.EventCatalog.from_dataset(
        xr.Dataset(data), mask=mask, granularity="point_temporal", analysis_mode=mode,
        provenance=provenance(run, parents=(mask.provenance.run_id,)),
    )


def constant_baseline(mask, *, threshold, reference, definition=None, grid_spec=None):
    """A BaselineResult written by hand: constant C and Q on every day of year."""
    g = grid_spec or mask.grid
    ds = mask.dataset
    domain = ds[g.domain_mask]
    shape = (366, *(ds.sizes[d] for d in g.spatial_dims))
    dims = ("dayofyear", *g.spatial_dims)
    inside = np.broadcast_to(domain.transpose(*g.spatial_dims).values, shape)
    attrs = {"units": "degC", "units_metadata": "temperature: on_scale"}
    data = xr.Dataset(
        {"threshold": (dims, np.where(inside, threshold, np.nan), attrs),
         "reference": (dims, np.where(inside, reference, np.nan), attrs),
         "sample_count": (dims, np.full(shape, 999, np.int32)),
         "invalid_reason": (dims, np.where(inside, 0, 1).astype(np.uint8)),
         g.domain_mask: domain},
        coords={"dayofyear": np.arange(1, 367)},
    )
    return xhw.BaselineResult.from_dataset(
        data, grid=g, baseline=mask.baseline, definition=definition or mask.definition,
        provenance=provenance("constant-baseline"))


# --- F1: point MHW, gap of 2 (VALIDATION "Gap de 2") -------------------------------------

def f1_point_mhw() -> Fixture:
    temp = [10, 13, 13, 13, 13, 13, 10, 10, 13, 13, 13, 13, 13, 10]
    n = len(temp)
    mask = point_mask(temp, flags(n, *days(1, 5), *days(8, 12)), flags(n, *days(1, 12)),
                      flags(n, 6, 7), definition=hobday(), run_id="F1")
    return Fixture(mask, catalog(mask, [((), 1, 12, 12, 10, 2, False, False, 0, True, False)]))


# --- F2: point MCS, symmetric (VALIDATION "symmetric MCS") -------------------------------

def f2_point_mcs() -> Fixture:
    temp = [10, 7, 7, 7, 7, 7, 10]
    definition = hobday(phenomenon="mcs", tail="lower", comparator="<", percentile=10.0,
                        category_scheme=None)
    mask = point_mask(temp, flags(7, *days(1, 5)), flags(7, *days(1, 5)), flags(7),
                      definition=definition, threshold=8.0, run_id="F2")
    return Fixture(mask, catalog(mask, [((), 1, 5, 5, 5, 0, False, False, 0, True, False)]))


# --- F3: point AHW TX90 over land, 3-day profile ----------------------

def f3_point_ahw() -> Fixture:
    temp = [26, 31, 31, 31, 26, 31, 31, 26]  # days 5-6: interior short run, never recorded
    definition = hobday(profile="tx90_perkins2013", phenomenon="ahw", variable="air_temperature",
                        daily_statistic="max", day_definition="offset:-05:00",
                        min_duration_days=3, max_gap_days=0, category_scheme=None)
    mask = point_mask(temp, flags(8, 1, 2, 3, 5, 6), flags(8, 1, 2, 3), flags(8),
                      definition=definition, threshold=30.0, reference=25.0,
                      grid_spec=grid(domain="land"), run_id="F3")
    return Fixture(mask, catalog(mask, [((), 1, 3, 3, 3, 0, False, False, 0, True, False)]))


# --- F4: transect with a missing day inside a run (missing_policy="break") ---------------

F4_CANDIDATE = ((1,), 7, 9, 3, 3, 0, True, True, 4 | 2, False, False)


def f4_line(mode: str = "retrospective") -> Fixture:
    t, a = 10, 3
    temp = np.full((t, a), 10.0)
    temp[1:6, 0] = 13.0
    temp[1:6, 1] = 13.0
    temp[6, 1] = np.nan
    temp[7:10, 1] = 13.0
    exc = np.zeros((t, a), bool)
    exc[1:6, 0] = exc[1:6, 1] = exc[7:10, 1] = True
    mem = np.zeros((t, a), bool)
    mem[1:6, 0] = mem[1:6, 1] = True
    reason = np.zeros((t, a), np.uint8)
    reason[6, 1] = 2  # missing_data
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-11")).astype(
        "datetime64[ns]")
    definition = hobday()
    ds = mask_dataset(times, ("time", "along"), temp, 12.0, 10.0, exc, mem, np.zeros_like(exc),
                      reason, definition.variable,
                      {"along": ("along", [0.0, 5000.0, 10000.0], {"units": "m"})},
                      {"domain_mask": ("along", np.ones(a, bool)),
                       "segment_length": ("along", np.full(a, 5000.0), {"units": "m"})})
    spec = grid(support="line", topology="transect", spatial_dims=("along",),
                coords=xhw.CoordRoles(along="along"),
                cell_measure=xhw.CellMeasure(kind="length", var="segment_length", units="m"))
    mask = xhw.XMask.from_dataset(ds, grid=spec, definition=definition,
                                        baseline=baseline(), provenance=provenance("F4"))
    rows = [((0,), 1, 5, 5, 5, 0, False, False, 0, True, False),
            ((1,), 1, 5, 5, 5, 0, False, True, 4, True, False)]
    if mode == "monitoring":
        rows.append(F4_CANDIDATE)
    return Fixture(mask, catalog(mask, rows, mode=mode))


# --- F5: 7x7 surface with land column and partial coastal cells --------------------------

def surface_dataset(temp, exc, mem, reason, definition, times) -> xr.Dataset:
    ny, nx = 7, 7
    lat = np.linspace(-15.0, -9.0, ny)
    lon = np.linspace(-80.0, -74.0, nx)
    ocean = np.ones((ny, nx), bool)
    ocean[:, 0] = False
    fraction = np.ones((ny, nx))
    fraction[:, 0], fraction[:, 1] = 0.0, 0.5
    area = np.outer(np.cos(np.deg2rad(lat)), np.full(nx, 1.2e10))
    dims = ("time", "y", "x")
    return mask_dataset(times, dims, temp, 12.0, 10.0, exc, mem, np.zeros_like(exc), reason,
                        definition.variable, {"lat": ("y", lat), "lon": ("x", lon)},
                        {"domain_mask": (("y", "x"), ocean),
                         "domain_fraction": (("y", "x"), fraction),
                         "cell_area": (("y", "x"), area, {"units": "m2"})})


def surface_grid() -> xhw.GridSpec:
    return grid(support="surface", topology="structured", spatial_dims=("y", "x"),
                coords=xhw.CoordRoles(lat="lat", lon="lon"),
                cell_measure=xhw.CellMeasure(kind="area", var="cell_area", units="m2"),
                domain_fraction="domain_fraction")


def f5_surface() -> Fixture:
    t = 6
    temp = np.full((t, 7, 7), 10.0)
    temp[0:5, 3, 3] = 13.0
    temp[:, :, 0] = 20.0  # extreme values over land: invalid, never exceedance (VALIDATION)
    exc = np.zeros((t, 7, 7), bool)
    exc[0:5, 3, 3] = True
    mem = exc.copy()
    reason = np.zeros((t, 7, 7), np.uint8)
    reason[:, :, 0] = 1  # outside_domain
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-07")).astype(
        "datetime64[ns]")
    definition = hobday()
    ds = surface_dataset(temp, exc, mem, reason, definition, times)
    mask = xhw.XMask.from_dataset(ds, grid=surface_grid(), definition=definition,
                                        baseline=baseline(), provenance=provenance("F5"))
    return Fixture(mask, catalog(mask, [((3, 3), 0, 4, 5, 5, 0, True, False, 1, True, False)]))


# --- F6: volume with two layers of different thickness -----------------------------------

def f6_volume() -> Fixture:
    t = 6
    temp = np.full((t, 2, 2, 2), 10.0)
    temp[0:5, 0, 0, 0] = 13.0
    exc = np.zeros(temp.shape, bool)
    exc[0:5, 0, 0, 0] = True
    times = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-07")).astype(
        "datetime64[ns]")
    definition = hobday()
    ds = mask_dataset(times, ("time", "z", "y", "x"), temp, 12.0, 10.0, exc, exc.copy(),
                      np.zeros_like(exc), np.zeros(temp.shape, np.uint8), definition.variable,
                      {"z": ("z", [5.0, 20.0], {"units": "m", "positive": "down"}),
                       "lat": ("y", [-12.0, -11.0]), "lon": ("x", [-78.0, -77.0])},
                      {"domain_mask": (("z", "y", "x"), np.ones((2, 2, 2), bool)),
                       "dz": ("z", [10.0, 20.0], {"units": "m"}),
                       "cell_area": (("y", "x"), np.full((2, 2), 1.2e10), {"units": "m2"})})
    spec = grid(support="volume", topology="structured", spatial_dims=("z", "y", "x"),
                coords=xhw.CoordRoles(lat="lat", lon="lon", depth="z"),
                cell_measure=xhw.CellMeasure(kind="area", var="cell_area", units="m2"),
                vertical=xhw.VerticalSpec(coordinate="z", thickness="dz", thickness_units="m",
                                          time_dependent=False))
    mask = xhw.XMask.from_dataset(ds, grid=spec, definition=definition,
                                        baseline=baseline(), provenance=provenance("F6"))
    return Fixture(mask, catalog(mask, [((0, 0, 0), 0, 4, 5, 5, 0, True, False, 1, True,
                                         False)]))


# --- F7: coastal pair AHW (land) - MHW (ocean) with CouplingSpec -------------------------

class CouplingFixture(NamedTuple):
    ahw: Fixture
    mhw: Fixture
    coupling: xhw.CouplingSpec


def f7_coastal_pair() -> CouplingFixture:
    n, start = 12, "1999-12-31"  # index k = day k of VALIDATION (AHW 4-6, MHW 1-10)
    mhw_temp = [13.0 if 1 <= k <= 10 else 10.0 for k in range(n)]
    mhw_mask = point_mask(mhw_temp, flags(n, *days(1, 10)), flags(n, *days(1, 10)), flags(n),
                          definition=hobday(), start=start, run_id="F7-mhw")
    ahw_def = hobday(profile="tx90_perkins2013", phenomenon="ahw", variable="air_temperature",
                     daily_statistic="max", day_definition="offset:-05:00",
                     min_duration_days=3, max_gap_days=0, category_scheme=None)
    ahw_temp = [31.0 if 4 <= k <= 6 else 26.0 for k in range(n)]
    ahw_mask = point_mask(ahw_temp, flags(n, 4, 5, 6), flags(n, 4, 5, 6), flags(n),
                          definition=ahw_def, start=start, threshold=30.0, reference=25.0,
                          grid_spec=grid(domain="land"), run_id="F7-ahw")
    coupling = xhw.CouplingSpec(
        strategy="event_first", phenomena=("ahw", "mhw"), spatial_rule="coastal_pairs",
        spatial_params=(("pairs_table", "fixture:coastal-pairs-F7"), ("max_distance_km", 50.0)),
        lag_window_days=(-5, 5), membership_basis="event_membership", min_phase_days=1,
        matching="many_to_many")
    return CouplingFixture(
        Fixture(ahw_mask, catalog(ahw_mask, [((), 4, 6, 3, 3, 0, False, False, 0, True, False)])),
        Fixture(mhw_mask, catalog(mhw_mask, [((), 1, 10, 10, 10, 0, False, False, 0, True,
                                              False)])),
        coupling,
    )


# --- F8: forecast slice (one emission, one member), monitoring ---------------------------

def f8_forecast() -> Fixture:
    temp = [10, 10, 13, 13, 13, 13, 13]  # 2026-09-17..23; forecast from 2026-09-21
    segment = ("time", np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int8))
    mask = point_mask(temp, flags(7, *days(2, 6)), flags(7, *days(2, 6)), flags(7),
                      definition=hobday(), start="2026-09-17",
                      grid_spec=grid(time_axes="forecast"), run_id="F8",
                      extra={"segment": segment},
                      extra_coords={"forecast_reference_time": np.datetime64("2026-09-21", "ns"),
                                    "member": 0})
    rows = [((), 2, 6, 5, 5, 0, False, True, 8, True, True)]
    return Fixture(mask, catalog(mask, rows, mode="monitoring"))


# --- F9: pure exceedance, marEx-like profile --------------------------

def f9_pure_exceedance() -> Fixture:
    temp = [10, 12, 10, 12.5, 12, 10]  # 12 >= 12 is an exceedance with ">="
    definition = hobday(profile="marex_like_pure_exceedance", comparator=">=", percentile=95.0,
                        min_duration_days=1, max_gap_days=0, category_scheme=None)
    base = baseline(method="shifting", computed_on="anomaly", smoothing_window_days=None,
                    leap_day_policy="calendar_doy", params=(("window_years", 15),))
    mask = point_mask(temp, flags(6, 1, 3, 4), flags(6, 1, 3, 4), flags(6),
                      definition=definition, base=base, run_id="F9")
    rows = [((), 1, 1, 1, 1, 0, False, False, 0, True, False),
            ((), 3, 4, 2, 2, 0, False, False, 0, True, False)]
    return Fixture(mask, catalog(mask, rows))


ALL = {
    "F1": f1_point_mhw, "F2": f2_point_mcs, "F3": f3_point_ahw, "F4": f4_line,
    "F5": f5_surface, "F6": f6_volume, "F8": f8_forecast, "F9": f9_pure_exceedance,
}
