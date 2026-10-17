# Copyright 2026 Daniel Camarena
# SPDX-License-Identifier: Apache-2.0

"""Vectorised singular spectrum analysis.

Basic SSA (Golyandina and Zhigljavsky): the trajectory (Hankel) matrix of window L of each
series is decomposed into its leading eigentriples, the eigentriples are grouped and each
group is turned back into a series by diagonal averaging. Implemented for many cells at once
without forming any trajectory matrix: products with the Hankel matrix and the diagonal
averaging are FFT convolutions, and the leading singular triples come from a randomised SVD
with power iterations (Halko, Martinsson and Tropp 2011; fixed seed, so results are
reproducible).

Grouping is automatic and deterministic: each eigentriple goes to the trend if the dominant
frequency of its left vector is below 1 / ``trend_period``, to the seasonal part if it lies
within one frequency bin (1 / L) of a harmonic k / period (k = 1..``n_harmonics``), and
otherwise to the residual (as Rssa's periodogram grouping, re-implemented).
"""

from __future__ import annotations

import numpy as np

_OVERSAMPLE = 10
_POWER_ITERATIONS = 4
_SEED = 20261003
_BLOCK_BYTES = 128 * 2**20


def _fft_size(n: int) -> int:
    size = 1
    while size < n:
        size *= 2
    return size


class _Hankel:
    """Products with the trajectory matrices (L x K) of the rows of x (N_cells, N)."""

    def __init__(self, x, window):
        self.cells, self.n = x.shape
        self.window, self.k = window, self.n - window + 1
        self.size = _fft_size(self.n + max(self.window, self.k))
        self.x_fft = np.fft.rfft(x, self.size, axis=1)[:, :, None]

    def times(self, v):      # (cells, K, r) -> (cells, L, r): (X v)_i = sum_j x[i + j] v[j]
        full = np.fft.irfft(self.x_fft * np.fft.rfft(v[:, ::-1], self.size, axis=1),
                            self.size, axis=1)
        return full[:, self.k - 1:self.k - 1 + self.window]

    def transposed_times(self, u):   # (cells, L, r) -> (cells, K, r)
        full = np.fft.irfft(self.x_fft * np.fft.rfft(u[:, ::-1], self.size, axis=1),
                            self.size, axis=1)
        return full[:, self.window - 1:self.window - 1 + self.k]


def leading_triples(x, window, rank):
    """Leading `rank` singular triples (sigma (cells, r), U (cells, L, r), V (cells, K, r))."""
    hankel = _Hankel(x, window)
    width = rank + _OVERSAMPLE
    omega = np.random.default_rng(_SEED).standard_normal((hankel.k, width))
    y = hankel.times(np.broadcast_to(omega, (hankel.cells,) + omega.shape))
    q = np.linalg.qr(y)[0]
    for _ in range(_POWER_ITERATIONS):
        z = np.linalg.qr(hankel.transposed_times(q))[0]
        q = np.linalg.qr(hankel.times(z))[0]
    b = np.swapaxes(hankel.transposed_times(q), 1, 2)                # (cells, width, K)
    u_small, sigma, vt = np.linalg.svd(b, full_matrices=False)
    u = q @ u_small
    return sigma[:, :rank], u[:, :, :rank], np.swapaxes(vt, 1, 2)[:, :, :rank]


def reconstruct(sigma, u, v, weights, n):
    """Diagonal averaging of sum_k weights[c, k] sigma[c, k] u_k v_k' for each cell c."""
    size = _fft_size(n)
    product = np.fft.rfft(u, size, axis=1) * np.fft.rfft(v, size, axis=1)  # (cells, f, r)
    series = np.fft.irfft((product * (weights * sigma)[:, None, :]).sum(axis=2), size,
                          axis=1)[:, :n]
    window, k = u.shape[1], v.shape[1]
    t = np.arange(n)
    counts = np.minimum.reduce([t + 1, np.full(n, window), np.full(n, k), n - t])
    return series / counts


def dominant_frequency(u):
    """Frequency (cycles per step) of the periodogram maximum of each left vector."""
    power = np.abs(np.fft.rfft(u, axis=1)) ** 2
    return np.argmax(power, axis=1) / u.shape[1]


def block_cells(n, window, rank):
    """Cells per block so that one spectrum of the products (cells, frequencies, rank +
    oversampling, complex) stays within _BLOCK_BYTES; the peak is a few such arrays.
    """
    size = _fft_size(n + max(window, n - window + 1))
    per_cell = (size // 2 + 1) * (rank + _OVERSAMPLE) * 16
    return max(1, _BLOCK_BYTES // per_cell)


def decompose(x, *, window, rank, period, trend_period, n_harmonics, cells_per_block=None):
    """Trend and seasonal parts of every complete row of x (N_cells, N), in blocks of cells (each
    cell is independent, so blocking changes memory, not results).
    """
    step = cells_per_block or block_cells(x.shape[1], window, rank)
    parts = [_decompose_block(x[i:i + step], window=window, rank=rank, period=period,
                              trend_period=trend_period, n_harmonics=n_harmonics)
             for i in range(0, x.shape[0], step)]
    return tuple(np.concatenate([part[k] for part in parts]) for k in range(2))


def _decompose_block(x, *, window, rank, period, trend_period, n_harmonics):
    sigma, u, v = leading_triples(x, window, rank)
    freq = dominant_frequency(u)                                          # (cells, r)
    harmonics = np.arange(1, n_harmonics + 1) / period
    seasonal = (np.abs(freq[..., None] - harmonics).min(axis=-1) <= 1.0 / window)
    trend = (freq < 1.0 / trend_period) & ~seasonal
    n = x.shape[1]
    return (reconstruct(sigma, u, v, trend.astype(float), n),
            reconstruct(sigma, u, v, seasonal.astype(float), n))
