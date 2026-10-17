# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Vectorised STL: Cleveland et al. (1990), as ``stats::stl``/statsmodels.

Reimplemented from the published algorithm (inner loop of ``stl``: detrending, cycle-subseries
LOESS with one extrapolated value at each end, low-pass filter of moving averages and LOESS,
trend LOESS), not copied from any implementation. Without robustness weights every LOESS
smoother is a linear operator that does not depend on the series, so it is built once and
applied to all cells at once: rows whose window is centred share one kernel (FFT convolution),
the rest are explicit weight rows (matrix products in blocks). Jumps are 1 (exact LOESS).
Robust STL (series-dependent weights) stays with the statsmodels engine.
"""

from __future__ import annotations

import numpy as np

_BLOCK = 512  # rows of explicit weights per matrix product (bounds memory)


def _weights(xs, nleft, nright, n, length, degree):
    """LOESS weights at positions `xs` (1-based) over the window [nleft, nright] (stlest).
    Returns (weights (rows, width), ok (rows,)).
    """
    xs = np.asarray(xs, dtype=float)[:, None]
    j = np.arange(nleft, nright + 1, dtype=float)[None, :]
    h = np.maximum(xs - nleft, nright - xs)
    if length > n:
        h = h + (length - n) // 2
    r = np.abs(j - xs)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(r <= 0.001 * h, 1.0,
                     np.where(r <= 0.999 * h, (1.0 - (r / h) ** 3) ** 3, 0.0))
    total = w.sum(axis=1, keepdims=True)
    ok = total[:, 0] > 0
    w = np.where(ok[:, None], w / np.where(total > 0, total, 1.0), 0.0)
    if degree > 0:
        a = (w * j).sum(axis=1, keepdims=True)
        c = (w * (j - a) ** 2).sum(axis=1, keepdims=True)
        linear = (h > 0) & (np.sqrt(c) > 0.001 * (n - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            b = (xs - a) / c
        w = np.where(linear, w * (b * (j - a) + 1.0), w)
    return w, ok


def _apply_rows(y, xs, nleft, nright, length, degree, out):
    """out[:, xs-1] = LOESS at `xs` with one shared window; not-ok rows keep y (stless)."""
    n = y.shape[1]
    for start in range(0, len(xs), _BLOCK):
        rows = np.asarray(xs[start:start + _BLOCK])
        w, ok = _weights(rows, nleft, nright, n, length, degree)
        values = y[:, nleft - 1:nright] @ w.T
        out[:, rows - 1] = np.where(ok[None, :], values, y[:, rows - 1])


def loess(y, length, degree):
    """STL's ``stless`` with jump 1 and no robustness weights, for every row of `y` (N, n)."""
    n = y.shape[1]
    out = np.empty_like(y)
    if length >= n:
        _apply_rows(y, np.arange(1, n + 1), 1, n, length, degree, out)
        return out
    half = (length + 1) // 2
    xs = np.arange(1, n + 1)
    nleft = np.clip(xs - half + 1, 1, n - length + 1)   # window of stless for each row
    nleft[xs <= half] = 1
    centred = (xs - nleft) == (nleft + length - 1 - xs)
    first, last = xs[~centred & (xs <= half)], xs[~centred & (xs > half)]
    if first.size:
        _apply_rows(y, first, 1, length, length, degree, out)
    if last.size:
        _apply_rows(y, last, n - length + 1, n, length, degree, out)
    inner = xs[centred]
    if inner.size:
        k = (length - 1) // 2
        kernel, ok = _weights([k + 1], 1, length, n, length, degree)
        if ok[0]:
            size = n + length - 1
            fft = np.fft.rfft(y, size, axis=1) * np.fft.rfft(kernel[0][::-1], size)
            full = np.fft.irfft(fft, size, axis=1)
            out[:, inner - 1] = full[:, inner - 1 + k]
        else:
            out[:, inner - 1] = y[:, inner - 1]
    return out


def _subseries_operator(k, length, degree):
    """(k + 2, k) operator of the cycle-subseries smoothing: LOESS at 1..k plus the values
    extrapolated at 0 and k + 1 (stlss).
    """
    eye = np.eye(k)
    op = np.empty((k + 2, k))
    op[1:k + 1] = loess(eye, length, degree).T
    for row, xs, nleft, nright, neighbour in ((0, 0, 1, min(length, k), 1),
                                              (k + 1, k + 1, max(1, k - length + 1), k, k)):
        w, ok = _weights([xs], nleft, nright, k, length, degree)
        op[row] = 0.0
        if ok[0]:
            op[row, nleft - 1:nright] = w[0]
        else:
            op[row] = op[neighbour]
    return op


def _cycle_subseries(work, period, length, degree):
    n_cells, n = work.shape
    out = np.empty((n_cells, n + 2 * period))
    counts = (n - 1 - np.arange(period)) // period + 1
    for k in np.unique(counts):
        positions = np.flatnonzero(counts == k)
        op = _subseries_operator(int(k), length, degree)
        index = positions[None, :] + period * np.arange(k)[:, None]           # (k, P)
        smoothed = np.einsum("mk,nkp->nmp", op, work[:, index])              # (N, k+2, P)
        target = positions[None, :] + period * np.arange(k + 2)[:, None]
        out[:, target] = smoothed
    return out


def _moving_average(x, width):
    csum = np.cumsum(np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1), axis=1)
    return (csum[:, width:] - csum[:, :-width]) / width


def stl(y, *, period, seasonal, trend, low_pass, seasonal_deg, trend_deg, low_pass_deg,
        inner_iter=2):
    """Trend and seasonal of every row of `y` (N, n) without robustness (outer loop 0)."""
    n = y.shape[1]
    trend_part = np.zeros_like(y)
    seasonal_part = np.zeros_like(y)
    for _ in range(inner_iter):
        cycle = _cycle_subseries(y - trend_part, period, seasonal, seasonal_deg)
        smooth = _moving_average(_moving_average(_moving_average(cycle, period), period), 3)
        low = loess(smooth, low_pass, low_pass_deg)
        seasonal_part = cycle[:, period:period + n] - low
        trend_part = loess(y - seasonal_part, trend, trend_deg)
    return trend_part, seasonal_part
