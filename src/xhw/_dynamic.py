# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Dynamic baselines: reference = trend + seasonal; threshold = reference + residual percentile.

Methods are detrended, shifting, STL, state space, GAM and SSA. Seasonal phase follows
calendar dates. Retrospective and causal fitting record history and availability; residual
sampling excludes filled observations. Baseline fitting requires whole time series.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr

from ._baseline import (
    BASELINE_UNDEFINED,
    COMPONENT,
    DOY,
    N_DOY,
    OUTSIDE_DOMAIN,
    YEAR,
    _fit_core,
    _fit_years_core,
    _quality_warnings,
    _reference_mask,
    _window_indices,
    _year_sets,
    doy_positions,
)
from ._mask import ON_SCALE
from ._schema import parse_iso_utc, require
from ._specs import NUMPY_PERCENTILE_METHOD, ProcessingStep, Provenance, dynamic_params

#: variables of a dynamic BaselineResult
TIME_VARS = ("trend", "seasonal", "reference", "threshold", "invalid_reason")
DOY_VARS = ("residual_threshold", "sample_count", "residual_invalid_reason")
_DAYS_PER_DECADE = 3652.5
_IMPLEMENTED = ("detrended", "shifting", "stl", "state_space", "gam", "ssa")


# --- time plan (shared by every cell) ---------------------------------------------------------


def _day_numbers(times) -> np.ndarray:
    return np.asarray(times, dtype="datetime64[D]").astype(np.int64)


def _date_number(text: str) -> int:
    return int(np.datetime64(parse_iso_utc("date", text).date(), "D").astype(np.int64))


def fit_dates(baseline, days: np.ndarray):
    """Which fit serves each date. Returns (groups, covered): ``groups`` maps the last day of
    data a fit may use (-1 = all data, retrospective) to the indices of the dates it serves;
    ``covered`` is False for dates without a usable fit (too little history, or beyond
    ``max_extrapolation_days``).
    """
    p = dynamic_params(baseline)
    if baseline.fit_mode == "retrospective":
        return {-1: np.arange(days.size)}, np.ones(days.size, bool)
    lag = p["lag_days"]
    horizon = p["max_extrapolation_days"]
    last = _date_number(baseline.available_until) - lag
    valid = np.zeros(days.size, bool)
    groups: dict[int, list] = {}
    if p["causal_scheme"] == "cutoff":
        valid = days <= last + horizon
        return {last: np.flatnonzero(valid)}, valid
    step = p["refit_every_days"]
    first = days[0] + _min_history_days(baseline, p) - 1
    cuts = list(range(first, last + 1, step))
    if not cuts or cuts[-1] != last:
        cuts.append(last)
    cuts = np.array(cuts)
    for i, day in enumerate(days):
        usable = cuts[cuts <= min(day - 1 - lag, last)]
        if usable.size and day - usable[-1] <= horizon:
            groups.setdefault(int(usable[-1]), []).append(i)
            valid[i] = True
    return {c: np.array(v) for c, v in groups.items()}, valid


def _min_history_days(baseline, p) -> int:
    return int(round(p.get("min_fit_years", p.get("min_years")) * 365.25))


# --- methods (per batch of cells; values (N, T)) ---------------------------------------------


def _harmonic_design(days, phase, center, degree, n_harmonics):
    tau = (days - center) / _DAYS_PER_DECADE
    trend = np.stack([tau ** k for k in range(degree + 1)], axis=1)
    angle = 2 * np.pi * phase
    seasonal = np.concatenate([np.stack([np.cos(h * angle), np.sin(h * angle)], axis=1)
                               for h in range(1, n_harmonics + 1)], axis=1)
    return trend, seasonal


def _lstsq(design, values, rows, min_rows):
    """Least squares on observed rows, column by column when the gaps differ (NaN-tolerant)."""
    n = values.shape[0]
    coef = np.full((design.shape[1], n), np.nan)
    sub, a = values[:, rows].T, design[rows]
    finite = np.isfinite(sub)
    full = finite.all(axis=0) & (sub.shape[0] >= min_rows)
    if full.any():
        coef[:, full] = np.linalg.lstsq(a, sub[:, full], rcond=None)[0]
    for i in np.flatnonzero(~full):
        ok = finite[:, i]
        if ok.sum() >= min_rows:
            coef[:, i] = np.linalg.lstsq(a[ok], sub[ok, i], rcond=None)[0]
    return coef


def _detrended(values, plan, rows, targets):
    """Trend and seasonal at `targets` from a fit on `rows` (boolean over the time axis)."""
    p = plan["p"]
    days, phase = plan["days"], plan["phase"]
    degree, n_harm = p["trend_degree"], p["n_harmonics"]
    center = days[rows].mean() if rows.any() else 0.0
    trend_d, seas_d = _harmonic_design(days, phase, center, degree, n_harm)
    design = np.concatenate([trend_d, seas_d], axis=1)
    coef = _lstsq(design, values, rows, plan["min_rows"])
    k = trend_d.shape[1]
    trend = trend_d[targets] @ coef[:k]
    if p["seasonal"] == "harmonic":
        seasonal = seas_d[targets] @ coef[k:]
    # day-of-year climatology of the detrended series on the fit days (temporal detection pooling)
    else:
        detrended = values - (trend_d @ coef[:k]).T
        detrended[:, ~rows] = np.nan
        clim = _climatology(detrended, plan, rows)
        seasonal = clim[:, plan["hobday_pos"][targets] - 1].T
    return trend.T, seasonal.T


def _climatology(values, plan, rows):
    """Pooled day-of-year mean on the calendar positions, 29 Feb interpolated, smoothed (temporal
    detection).
    """
    b = plan["baseline"]
    sets = _window_indices(plan["hobday_pos"], rows, b.doy_window_days // 2)
    _, ref, _, _ = _fit_core(values, sets=sets, doy_mode=True, feb29=True, pct=50.0,
                             method=plan["method"], smooth=b.smoothing_window_days,
                             min_samples=1, absolute=None)
    return ref


def _shifting(values, plan, rows, targets):
    """Day-of-year climatology of a window of years for the year of each target date."""
    b, p = plan["baseline"], plan["p"]
    width, min_years = p["window_years"], p["min_years"]
    years, days = plan["years"], plan["days"]
    causal = b.fit_mode == "causal"
    n = values.shape[0]
    trend = np.full((n, targets.size), np.nan)
    seasonal = np.full((n, targets.size), np.nan)
    observed = np.isfinite(values) & rows
    year_list = np.unique(years)
    for year in np.unique(years[targets]):
        window = (np.arange(year - width, year) if causal
                  else np.arange(year - (width - 1) // 2, year + width // 2 + 1))
        in_window = np.isin(years, window) & rows
        if causal:  # whole previous years only, before this year's first date minus the lag
            start = days[years == year].min()
            lag = p["lag_days"]
            in_window &= days <= start - 1 - lag
            ends = {y: days[years == y].max() for y in window if y in year_list}
            complete = [y for y, end in ends.items() if end <= start - 1 - lag]
            in_window &= np.isin(years, complete)
        # a year counts if at least half of its days are observed (*)
        counted = np.zeros(n, int)
        for y in window:
            sel = in_window & (years == y)
            if sel.any():
                counted += observed[:, sel].sum(axis=1) >= 0.5 * (years == y).sum()
        clim = _climatology(np.where(in_window, values, np.nan), plan, in_window)
        enough = counted >= min_years
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # cells without data -> NaN
            level = np.where(enough, np.nanmean(clim, axis=1), np.nan)
        here = np.flatnonzero(years[targets] == year)
        pos = plan["hobday_pos"][targets[here]] - 1
        trend[:, here] = level[:, None]
        seasonal[:, here] = np.where(enough[:, None], clim[:, pos] - level[:, None], np.nan)
    return trend, seasonal


# --- 366-position calendar grid -----------------------------------------------


def _to_grid(values, index, interpolate):
    """(n, t) on dates -> (n, L) on the grid of 366 positions per year. The only empty slots are
    29 Feb of common years: the mean of 28 Feb and 1 Mar (``interpolate``) or NaN.
    """
    out = np.full((values.shape[0], index[-1] + 1), np.nan)
    out[:, index] = values
    if interpolate:
        missing = np.setdiff1d(np.arange(out.shape[1]), index)
        out[:, missing] = 0.5 * (out[:, missing - 1] + out[:, missing + 1])
    return out


def _odd(value: float) -> int:
    n = int(np.ceil(value))
    return n if n % 2 else n + 1


def _stl_fit(grid, options, robust, engine):
    """Trend and seasonal of complete series (M, L): statsmodels cell by cell, or the vectorised
    engine (all cells at once, non-robust only).
    """
    if engine == "vectorized":
        from ._stl import stl

        return stl(grid, low_pass=_odd(N_DOY + 1), trend_deg=1, low_pass_deg=1, **options)
    from statsmodels.tsa.seasonal import STL

    inner, outer = (1, 15) if robust else (2, 0)          # iterations as in R stl (*)
    trend, seasonal = np.empty_like(grid), np.empty_like(grid)
    for i, y in enumerate(grid):
        fit = STL(y, robust=robust, **options).fit(inner_iter=inner, outer_iter=outer)
        trend[i], seasonal[i] = fit.trend, fit.seasonal
    return trend, seasonal


def _on_grid(values, plan, rows, targets, decompose):
    """Run `decompose(grid (M, L)) -> (trend, seasonal)` on the 366-position grid of the complete
    series fitted on `rows` (a prefix of the dates; methods that need complete series, C.6).
    Dates after the fit (causal ``cutoff``): trend extended with the slope of its last year,
    seasonal of the same position in the last fitted year (*).
    """
    p, index = plan["p"], plan["grid_index"]
    n = values.shape[0]
    trend = np.full((n, targets.size), np.nan)
    seasonal = np.full((n, targets.size), np.nan)
    grid = _to_grid(values[:, rows], index[rows], interpolate=True)
    length = grid.shape[1]
    complete = np.flatnonzero(np.isfinite(grid).all(axis=1))
    if length < p["min_fit_years"] * N_DOY or not complete.size:
        return trend, seasonal
    tr, se = decompose(grid[complete])
    inside = rows[targets]
    trend[np.ix_(complete, np.flatnonzero(inside))] = tr[:, index[targets[inside]]]
    seasonal[np.ix_(complete, np.flatnonzero(inside))] = se[:, index[targets[inside]]]
    slots_out = index[targets[~inside]]
    if slots_out.size:
        span = min(N_DOY, length - 1)
        slope = (tr[:, -1] - tr[:, -1 - span]) / span
        ahead = slots_out - (length - 1)
        out = np.flatnonzero(~inside)
        trend[np.ix_(complete, out)] = tr[:, -1:] + slope[:, None] * ahead
        back = slots_out - N_DOY * np.ceil(ahead / N_DOY).astype(int)
        seasonal[np.ix_(complete, out)] = se[:, back]
    return trend, seasonal


def _stl(values, plan, rows, targets):
    """STL (statsmodels or the vectorised engine) on the calendar grid."""
    p = plan["p"]
    periodic = p["seasonal_window"] == "periodic"

    def decompose(grid):
        length = grid.shape[1]
        options = dict(period=N_DOY, trend=_odd(p["trend_window_days"] * N_DOY / 365.25),
                       seasonal=10 * length + 1 if periodic else p["seasonal_window"],
                       seasonal_deg=0 if periodic else 1)
        tr, se = _stl_fit(grid, options, bool(p["robust"]), p["engine"])
        if periodic:  # exactly periodic, as R: mean seasonal of each position
            position = (np.arange(length) + plan["grid_offset"]) % N_DOY
            counts = np.maximum(np.bincount(position, minlength=N_DOY), 1)
            means = np.stack([np.bincount(position, row, N_DOY) for row in se]) / counts
            se = means[:, position]
        return tr, se

    return _on_grid(values, plan, rows, targets, decompose)


def _ssa(values, plan, rows, targets):
    """Singular spectrum analysis (`_ssa`) on the calendar grid; the window is shortened to half
    the fitted length when the series is too short for it.
    """
    from . import _ssa as ssa

    p = plan["p"]

    def decompose(grid):
        window = min(int(round(p["window_days"] * N_DOY / 365.25)), grid.shape[1] // 2)
        return ssa.decompose(grid, window=window, rank=p["n_components"], period=N_DOY,
                             trend_period=p["trend_period_days"] * N_DOY / 365.25,
                             n_harmonics=p["n_harmonics"])

    return _on_grid(values, plan, rows, targets, decompose)


def _state_space_model(y, p):
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    model = UnobservedComponents(y, level="local linear trend",
                                 freq_seasonal=[{"period": N_DOY,
                                                 "harmonics": p["n_harmonics"]}])
    slope = float(p["trend_timescale_days"]) ** -4   # ratio to the irregular variance
    params = [1.0, p["level_ratio"], slope, p["seasonal_ratio"]]
    return model, params


def _matrix(model, name):
    value = model.ssm[name]
    return value[..., 0] if value.ndim == 3 else value


def _components(model, state):
    level = state[0]
    return level, _matrix(model, "design")[0] @ state - level


def _state_space(values, plan, rows, targets):
    """Kalman smoother on the grid with the data after `rows` hidden: inside the fit it is the
    two-sided smoother; after it, the forecast from the last fitted day.
    """
    p, index = plan["p"], plan["grid_index"]
    n = values.shape[0]
    trend = np.full((n, targets.size), np.nan)
    seasonal = np.full((n, targets.size), np.nan)
    hidden = np.where(rows, values, np.nan)
    grid = _to_grid(hidden, index, interpolate=False)
    slots = index[targets]
    for i in range(n):
        if np.isfinite(hidden[i]).sum() < plan["min_rows"]:
            continue
        model, params = _state_space_model(grid[i], p)
        level, seas = _components(model, model.smooth(params).smoothed_state)
        trend[i], seasonal[i] = level[slots], seas[slots]
    return trend, seasonal


def _state_space_expanding(values, plan):
    """Causal by construction: E[state(t) | data up to t - 1 - lag] in one Kalman pass, with the
    data after ``available_until - lag`` hidden (one-step predictions, moved ``lag`` steps
    ahead with the transition matrix).
    """
    p, index, days = plan["p"], plan["grid_index"], plan["days"]
    lag = p["lag_days"]
    n, t = values.shape
    trend = np.full((n, t), np.nan)
    seasonal = np.full((n, t), np.nan)
    hidden = np.where(days <= plan["last_data_day"], values, np.nan)
    grid = _to_grid(hidden, index, interpolate=False)
    for i in range(n):
        if np.isfinite(hidden[i]).sum() < plan["min_rows"]:
            continue
        model, params = _state_space_model(grid[i], p)
        predicted = model.filter(params).predicted_state[:, :-1]
        if lag:
            step = np.linalg.matrix_power(_matrix(model, "transition"), lag)
            moved = np.full_like(predicted, np.nan)
            moved[:, lag:] = step @ predicted[:, :-lag]
            predicted = moved
        level, seas = _components(model, predicted)
        trend[i], seasonal[i] = level[index], seas[index]
    return trend, seasonal


def _gam(values, plan, rows, targets):
    """Penalised B-splines (`_gam`), fitted on the observed `rows`; after the fit the time
    splines continue linearly (causal ``cutoff``).
    """
    from . import _gam as gam

    p, days = plan["p"], plan["days"]
    fit_days = days[rows]
    years = (fit_days[-1] - fit_days[0] + 1) / 365.25
    n_trend = max(4, int(np.ceil(p["trend_splines_per_decade"] * years / 10)))
    design = gam.Design(fit_days, n_trend, p["seasonal_splines"], p["trend_lam"],
                        p["seasonal_lam"])
    phase = plan["phase"]
    centre = (np.arange(1, N_DOY + 1) - 0.5) / N_DOY
    if p["engine"] == "pygam":
        return _gam_pygam(values, plan, rows, targets, design, n_trend, centre)
    coef = gam.fit(design, design.matrix(fit_days, phase[rows]), values[:, rows],
                   plan["min_rows"])
    return design.components(coef, days[targets], phase[targets], centre)


def _gam_pygam(values, plan, rows, targets, design, n_trend, centre):
    """Reference engine: pygam's LinearGAM cell by cell (its periodic penalty differs from the
    cyclic second difference of the vectorised engine, ).
    """
    from pygam import LinearGAM, s

    p, days, pos = plan["p"], plan["days"], plan["hobday_pos"]
    n = values.shape[0]
    trend = np.full((n, targets.size), np.nan)
    seasonal = np.full((n, targets.size), np.nan)
    x_fit = np.column_stack([days[rows], pos[rows]]).astype(float)
    x_out = np.column_stack([days[targets], pos[targets]]).astype(float)
    x_year = np.column_stack([np.full(N_DOY, days[rows][0]), np.arange(1, N_DOY + 1)])
    for i in range(n):
        y = values[i, rows]
        ok = np.isfinite(y)
        if ok.sum() < plan["min_rows"]:
            continue
        model = LinearGAM(s(0, n_splines=n_trend, lam=p["trend_lam"],
                            edge_knots=[days[rows][0], days[rows][-1]])
                          + s(1, n_splines=p["seasonal_splines"], basis="cp",
                              lam=p["seasonal_lam"], edge_knots=[0.5, N_DOY + 0.5]))
        model.fit(x_fit[ok], y[ok])
        mean = model.partial_dependence(term=1, X=x_year).mean()
        trend[i] = model.coef_[-1] + model.partial_dependence(term=0, X=x_out) + mean
        seasonal[i] = model.partial_dependence(term=1, X=x_out) - mean
    return trend, seasonal


_METHODS = {"detrended": _detrended, "shifting": _shifting, "stl": _stl,
            "state_space": _state_space, "gam": _gam, "ssa": _ssa}
#: methods that need a complete series and decompose the filled one (C.6)
_USES_FILLED = ("stl", "ssa")


def _dynamic_core(values, filled, domain, *, plan):
    """Trend, seasonal and residual percentile of each series (time last). Only the cells of the
    domain are decomposed: the others are masked afterwards anyway, and fitting them costs
    time and, for the methods that hold every cell at once (SSA), memory.
    """
    lead = values.shape[:-1]
    x = np.asarray(values, dtype=float).reshape(-1, values.shape[-1])
    gaps = np.broadcast_to(np.asarray(filled, bool), values.shape).reshape(x.shape)
    observed = np.where(gaps, np.nan, x)
    cells = np.flatnonzero(np.broadcast_to(np.asarray(domain, bool), lead).reshape(-1))
    n, t = x.shape
    trend = np.full((n, t), np.nan)
    seasonal = np.full((n, t), np.nan)
    days = plan["days"]
    name = plan["baseline"].method
    source = (x if name in _USES_FILLED else observed)[cells]
    if not cells.size:
        pass
    elif plan["one_pass"]:
        trend[cells], seasonal[cells] = _state_space_expanding(source, plan)
    else:
        method = _METHODS[name]
        for cutoff, targets in plan["groups"].items():
            rows = np.ones(t, bool) if cutoff == -1 else days <= cutoff
            tr, se = method(source, plan, rows, targets)
            columns = np.arange(t)[targets]
            trend[np.ix_(cells, columns)], seasonal[np.ix_(cells, columns)] = tr, se
    residual = observed - trend - seasonal  # filled days never enter the percentile
    kwargs = plan["q_kwargs"]
    q, _, count, reason = _fit_core(residual, sets=plan["q_sets"], **kwargs)
    outs = [trend.reshape(lead + (t,)), seasonal.reshape(lead + (t,)),
            q.reshape(lead + (N_DOY,)), count.reshape(lead + (N_DOY,)),
            reason.reshape(lead + (N_DOY,))]
    if plan["year_sets"] is not None:
        qy, cy, ry = _fit_years_core(residual, year_sets=plan["year_sets"], **kwargs)
        shape = lead + (len(plan["year_sets"]), N_DOY)
        outs += [qy.reshape(shape), cy.reshape(shape), ry.reshape(shape)]
    return tuple(outs)


# --- fit_baseline entry point -----------------------------------------------------------------


def fit_dynamic(ds, *, names, units, steps, grid, baseline, definition, provenance=None,
                replica=None):
    from ._baseline import BaselineResult

    require(baseline.method in _IMPLEMENTED, f"method {baseline.method!r} is not implemented "
            "yet ")
    require(definition.threshold_kind == "percentile",
            "dynamic baselines need a percentile threshold ")
    require(baseline.leap_day_policy != "not_applicable",
            "dynamic baselines need a day-of-year axis for the residual percentile ")
    p = dynamic_params(baseline)
    tdim = grid.time_dim
    times = ds[tdim].values
    days = _day_numbers(times)
    require(np.all(np.diff(days) > 0), "dynamic baselines need strictly increasing daily times")
    groups, covered = fit_dates(baseline, days)
    hobday_pos = doy_positions(times, "hobday_366_interp")
    in_ref = _reference_mask(times, baseline)
    require(bool(in_ref.any()), "no data inside baseline.reference_period")
    q_pos = doy_positions(times, baseline.leap_day_policy)
    q_sets = _window_indices(q_pos, in_ref, baseline.doy_window_days // 2)
    years = pd.DatetimeIndex(times).year.to_numpy()
    year_sets = None
    if baseline.in_base_correction != "none":
        year_sets = _year_sets(q_sets, years, baseline.reference_years,
                               baseline.in_base_correction, replica)
    slots = (years - years[0]) * N_DOY + hobday_pos - 1
    grid_index = slots - slots[0]
    empty = np.setdiff1d(np.arange(grid_index[-1] + 1), grid_index)
    require(bool(np.all((empty + slots[0]) % N_DOY == 59)),
            "dynamic baselines need consecutive daily dates (only 29 Feb of common years "
            "may be absent from the calendar grid)")
    one_pass = (baseline.method == "state_space" and baseline.fit_mode == "causal"
                and p["causal_scheme"] == "expanding")
    plan_base = dict(
        baseline=baseline, p=p, days=days, years=years, hobday_pos=hobday_pos,
        phase=(hobday_pos - 0.5) / N_DOY, groups=groups, q_sets=q_sets, year_sets=year_sets,
        method=NUMPY_PERCENTILE_METHOD[baseline.percentile_estimator],
        min_rows=(_min_history_days(baseline, p)
                  if baseline.method in ("detrended", "state_space", "gam") else 1),
        grid_index=grid_index, grid_offset=int(slots[0] % N_DOY), one_pass=one_pass,
        last_data_day=(_date_number(baseline.available_until) - p["lag_days"]
                       if baseline.fit_mode == "causal" else int(days[-1])))
    pcts = ([c.percentile for c in definition.components]
            if definition.combination == "and_daily" else [definition.percentile])
    domain = ds[grid.domain_mask]
    fitted = []
    for name, pct in zip(names, pcts):
        filled = ds["filled"] if "filled" in ds else xr.zeros_like(ds[name], dtype=bool)
        plan = dict(plan_base, q_kwargs=dict(
            doy_mode=True, feb29=baseline.leap_day_policy == "hobday_366_interp", pct=pct,
            method=plan_base["method"], smooth=baseline.smoothing_window_days,
            min_samples=baseline.min_samples, absolute=None))
        n_out = 5 + (3 if year_sets is not None else 0)
        core_dims = [[tdim], [tdim], [DOY], [DOY], [DOY]] + [[YEAR, DOY]] * (n_out - 5)
        sizes = {DOY: N_DOY, **({YEAR: len(year_sets)} if year_sets is not None else {})}
        out = xr.apply_ufunc(
            _dynamic_core, ds[name], filled, domain, input_core_dims=[[tdim], [tdim], []],
            output_core_dims=core_dims,
            dask="parallelized", kwargs=dict(plan=plan),
            output_dtypes=[float, float, float, np.int32, np.uint8]
            + ([float, np.int32, np.uint8] if n_out > 5 else []),
            dask_gufunc_kwargs={"output_sizes": sizes, "allow_rechunk": True})
        count = out[3]
        if count.chunks is None:
            inside = domain.broadcast_like(count).transpose(*count.dims).values
            _quality_warnings(baseline, definition, pct, in_ref, True, count.values, inside)
        else:
            _quality_warnings(baseline, definition, pct, in_ref, True, None, None)
        fitted.append(out)
    combined = definition.combination == "and_daily"
    if combined:
        labels = [c.variable for c in definition.components]
        parts = tuple(xr.concat([f[k] for f in fitted], dim=COMPONENT).assign_coords(
            {COMPONENT: labels}) for k in range(len(fitted[0])))
    else:
        parts = fitted[0]
    trend, seasonal, q, count, q_reason = parts[:5]
    attrs = {"units": ds[names[0]].attrs["units"], "units_metadata": ON_SCALE}
    covered_da = xr.DataArray(covered, dims=tdim)
    trend = trend.where(domain & covered_da)
    seasonal = seasonal.where(domain & covered_da)
    reference = trend + seasonal
    q_reason = xr.where(domain, q_reason, OUTSIDE_DOMAIN).astype(np.uint8)
    at_t = xr.DataArray(q_pos - 1, dims=tdim)
    q_t = q.where(domain).isel({DOY: at_t})
    reason = xr.where(~domain, OUTSIDE_DOMAIN,
                      xr.where(reference.isnull(), BASELINE_UNDEFINED, 0)
                      | q_reason.isel({DOY: at_t})).astype(np.uint8)
    threshold = (reference + q_t).where(reason == 0)
    out = xr.Dataset({
        "trend": trend.assign_attrs(attrs, long_name="trend component"),
        "seasonal": seasonal.assign_attrs(attrs, long_name="seasonal component"),
        "reference": reference.where(reason == 0).assign_attrs(
            attrs, long_name="reference climatology (trend + seasonal)"),
        "threshold": threshold.assign_attrs(attrs, long_name="detection threshold"),
        "invalid_reason": reason,
        "residual_threshold": q.where(domain).assign_attrs(
            attrs, long_name="residual percentile per day of year"),
        "sample_count": count.assign_attrs(long_name="observed residuals per day of year"),
        "residual_invalid_reason": q_reason,
        grid.domain_mask: domain,
    })
    if year_sets is not None:
        qy, cy, ry = parts[5:]
        out["residual_threshold_by_year"] = qy.where(domain).assign_attrs(
            attrs, long_name=f"residual percentile of each reference year "
            f"({baseline.in_base_correction})")
        out["sample_count_by_year"] = cy
        out["invalid_reason_by_year"] = xr.where(domain, ry, OUTSIDE_DOMAIN).astype(np.uint8)
        out = out.assign_coords({YEAR: np.array(baseline.reference_years, dtype=np.int64)})
    out = out.assign_coords({tdim: ds[tdim].values, DOY: np.arange(1, N_DOY + 1)})
    order = [d for d in (YEAR, COMPONENT, tdim, DOY) if d in out.dims]
    out = out.transpose(*order, *grid.spatial_dims)
    if provenance is None:
        # effective parameters, defaults included: the Spec stores them as given
        params = (("variable", ",".join(names)), ("method", baseline.method),
                  ("excluded_filled_days", "filled" in ds), *sorted(p.items()))
        steps.append(ProcessingStep(name="fit_baseline", params=params))
        provenance = Provenance.new(inputs=(baseline.source,), steps=tuple(steps))
    return BaselineResult.from_dataset(out, grid=grid, baseline=baseline, definition=definition,
                                       provenance=provenance)
