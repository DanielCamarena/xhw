# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Vectorised GAM: penalised B-splines, the model of pygam's LinearGAM.

x(t) = intercept + sum_i b_i B_i(time) + sum_j c_j C_j(phase) + noise

with cubic B-splines on uniform knots over the fit period (linear extrapolation beyond it),
cyclic cubic B-splines over the year (period 366 positions), second-difference penalties lam *
D'D (cyclic for the seasonal term; Eilers and Marx 1996), an unpenalised intercept and a ridge
sqrt(eps) on every coefficient that resolves the intercept/spline collinearity, as pygam does.
Reimplemented from that description (pygam is Apache-2.0; nothing is copied). The normal
equations share one matrix among cells without gaps, so they are solved for all cells at once;
cells with gaps get their own (small) system.
"""

from __future__ import annotations

import numpy as np

ORDER = 3
_RIDGE = np.sqrt(np.finfo(np.float64).eps)


def _knots(n_splines: int) -> np.ndarray:
    """Augmented uniform knots on [0, 1] for `n_splines` cubic B-splines."""
    inner = np.linspace(0.0, 1.0, 1 + n_splines - ORDER)
    step = inner[1] - inner[0]
    extra = np.arange(1, ORDER + 1) * step
    knots = np.r_[-extra[::-1], inner, 1.0 + extra]
    knots[-1] += 1e-9  # last knot inclusive
    return knots


def _cox_de_boor(x: np.ndarray, knots: np.ndarray, order: int) -> np.ndarray:
    """B-spline basis of degree `order` at x in [0, 1] (columns = len(knots) - order - 1)."""
    x = x[:, None]
    basis = ((x >= knots[:-1]) & (x < knots[1:])).astype(float)   # x = 1 lies in [1, 1 + step)
    for k in range(1, order + 1):
        left_den = knots[k:-1] - knots[:-k - 1]
        right_den = knots[k + 1:] - knots[1:-k]
        left = (x - knots[:-k - 1]) / left_den * basis[:, :-1]
        right = (knots[k + 1:] - x) / right_den * basis[:, 1:]
        basis = left + right
    return basis


def spline_basis(x: np.ndarray, n_splines: int) -> np.ndarray:
    """Cubic B-splines on [0, 1]; outside, each function continues linearly."""
    knots = _knots(n_splines)
    x = np.asarray(x, dtype=float)
    inside = (x >= 0) & (x <= 1)
    out = np.empty((x.size, n_splines))
    out[inside] = _cox_de_boor(x[inside], knots, ORDER)
    if (~inside).any():
        ends = _cox_de_boor(np.array([0.0, 1.0]), knots, ORDER)
        lower = _cox_de_boor(np.array([0.0, 1.0]), knots, ORDER - 1)
        den_l = knots[ORDER:-1] - knots[:-ORDER - 1]
        den_r = knots[ORDER + 1:] - knots[1:-ORDER]
        grads = ORDER * (lower[:, :-1] / den_l - lower[:, 1:] / den_r)
        for row, mask, origin in ((0, x < 0, 0.0), (1, x > 1, 1.0)):
            out[mask] = ends[row] + grads[row] * (x[mask, None] - origin)
    return out


def cyclic_basis(phase: np.ndarray, n_splines: int) -> np.ndarray:
    """Cyclic cubic B-splines with period 1 and `n_splines` uniform knots."""
    knots = _knots(n_splines + ORDER)
    full = _cox_de_boor(np.mod(np.asarray(phase, float), 1.0), knots, ORDER)
    full[:, :ORDER] = np.maximum(full[:, :ORDER], full[:, -ORDER:])   # wrap the period
    return full[:, :-ORDER]


def difference_penalty(n: int, cyclic: bool) -> np.ndarray:
    """D'D for second differences of the coefficients (wrapped when cyclic)."""
    if cyclic:
        eye = np.eye(n)
        d = np.roll(eye, -2, axis=1) - 2 * np.roll(eye, -1, axis=1) + eye
    else:
        d = np.diff(np.eye(n), n=2, axis=0)
    return d.T @ d


class Design:
    """Shared design of one fit: bases at the fit dates and at any target date."""

    def __init__(self, fit_days, n_trend, n_seasonal, lam_trend, lam_seasonal):
        self.origin, self.span = float(fit_days[0]), float(max(fit_days[-1] - fit_days[0], 1))
        self.n_trend, self.n_seasonal = n_trend, n_seasonal
        m = n_trend + n_seasonal + 1
        self.penalty = np.zeros((m, m))
        self.penalty[:n_trend, :n_trend] = lam_trend * difference_penalty(n_trend, False)
        s = slice(n_trend, n_trend + n_seasonal)
        self.penalty[s, s] = lam_seasonal * difference_penalty(n_seasonal, True)
        self.penalty += _RIDGE * np.eye(m)

    def matrix(self, days, phase):
        trend = spline_basis((np.asarray(days, float) - self.origin) / self.span, self.n_trend)
        seasonal = cyclic_basis(phase, self.n_seasonal)
        return np.hstack([trend, seasonal, np.ones((len(trend), 1))])

    def components(self, coef, days, phase, centre_phase):
        """Trend (intercept + time splines) and seasonal (cyclic splines, zero mean over the
        year: its mean moves to the trend) at the targets; coef is (m, N).
        """
        x = self.matrix(days, phase)
        nt = self.n_trend
        seasonal_coef = coef[nt:nt + self.n_seasonal]
        mean = cyclic_basis(centre_phase, self.n_seasonal).mean(axis=0) @ seasonal_coef
        trend = x[:, :nt] @ coef[:nt] + coef[-1] + mean
        seasonal = x[:, nt:nt + self.n_seasonal] @ seasonal_coef - mean
        return trend.T, seasonal.T


def fit(design: Design, x: np.ndarray, y: np.ndarray, min_rows: int) -> np.ndarray:
    """Coefficients (m, N) for the series y (N, n) observed on the design matrix x (n, m)."""
    n_cells = y.shape[0]
    coef = np.full((x.shape[1], n_cells), np.nan)
    finite = np.isfinite(y)
    complete = finite.all(axis=1) & (y.shape[1] >= min_rows)
    if complete.any():
        gram = x.T @ x + design.penalty
        coef[:, complete] = np.linalg.solve(gram, x.T @ y[complete].T)
    for i in np.flatnonzero(~complete):
        ok = finite[i]
        if ok.sum() >= min_rows:
            xi = x[ok]
            coef[:, i] = np.linalg.solve(xi.T @ xi + design.penalty, xi.T @ y[i, ok])
    return coef
