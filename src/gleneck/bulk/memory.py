from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MemoryTerms:
    """Equilibrium memory kernel and colored-noise filter on the training grid."""

    memory: np.ndarray
    noise_filter: np.ndarray


def resample_memory_kernel(memory: np.ndarray, l_max: int, orig_interval: float = 1.0) -> np.ndarray:
    """Resample a fitted memory kernel onto integer GLE lag indices."""
    if l_max <= 0:
        raise ValueError("l_max must be positive.")
    if orig_interval <= 0:
        raise ValueError("orig_interval must be positive.")

    values = np.asarray(memory, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("memory kernel is empty.")

    t_coarse = np.arange(values.size, dtype=float) * orig_interval
    t_fine = np.arange(l_max, dtype=float)
    return np.interp(t_fine, t_coarse, values, left=0.0, right=0.0)


def colored_noise_filter(memory_resampled: np.ndarray, floor: float = 1e-20) -> np.ndarray:
    """Construct the FFT-based colored-noise filter used by the legacy notebook."""
    if floor <= 0:
        raise ValueError("floor must be positive.")

    memory = np.asarray(memory_resampled, dtype=float).reshape(-1)
    if memory.size < 2:
        raise ValueError("memory_resampled must contain at least two lags.")

    even_kernel = np.concatenate([memory, memory[-2:0:-1]])
    spectrum = np.fft.rfft(even_kernel)
    positive_spectrum = np.maximum(np.real(spectrum), floor)
    sqrt_spectrum = np.sqrt(positive_spectrum)
    return np.fft.irfft(sqrt_spectrum, n=even_kernel.shape[0])[: memory.size]


def prepare_memory_terms(memory: np.ndarray, l_max: int, orig_interval: float = 1.0) -> MemoryTerms:
    """Prepare the memory and colored-noise terms consumed by GLE-NECK training."""
    resampled = resample_memory_kernel(memory, l_max=l_max, orig_interval=orig_interval)
    return MemoryTerms(memory=resampled, noise_filter=colored_noise_filter(resampled))


def estimate_memory_kernel_from_vacf(
    time_lag: np.ndarray,
    vacf: np.ndarray,
    alpha: float = 1e-10,
) -> tuple[np.ndarray, float]:
    """Estimate a memory kernel from a normalized VACF using a Volterra solve.

    The discretization follows the legacy notebook equation

    dC(t) / dt = - integral_0^t M(s) C(t - s) ds

    with a Tikhonov-regularized lower-triangular convolution solve.  The
    returned kernel is on the same time grid as ``time_lag``.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    time = np.asarray(time_lag, dtype=float).reshape(-1)
    corr = np.asarray(vacf, dtype=float).reshape(-1)
    if time.size != corr.size:
        raise ValueError("time_lag and vacf must have matching lengths.")
    if time.size < 2:
        raise ValueError("At least two VACF points are required.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(corr)):
        raise ValueError("time_lag and vacf must be finite.")
    if corr[0] == 0:
        raise ValueError("VACF first value must be nonzero for normalization.")

    corr = corr / corr[0]
    diffs = np.diff(time)
    dt = float(np.median(diffs))
    if dt <= 0:
        raise ValueError("time_lag must be strictly increasing.")
    if np.max(np.abs(diffs - dt)) > max(1e-12, 1e-6 * dt):
        raise ValueError("time_lag must be uniformly spaced.")

    n_points = corr.size
    convolution = np.zeros((n_points, n_points), dtype=float)
    for row in range(n_points):
        convolution[row, : row + 1] = dt * corr[row::-1]

    rhs = np.empty(n_points, dtype=float)
    rhs[0] = 0.0
    rhs[1:] = -(corr[1:] - corr[:-1]) / dt

    normal = convolution.T @ convolution
    if alpha:
        normal = normal + alpha * np.eye(n_points)
    kernel = np.linalg.solve(normal, convolution.T @ rhs)
    return kernel, dt


def gaussian_smooth(values: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Small dependency-free Gaussian smoothing helper for fitted memory artifacts."""
    data = np.asarray(values, dtype=float).reshape(-1)
    if sigma <= 0:
        return data.copy()
    radius = int(truncate * sigma + 0.5)
    if radius < 1:
        return data.copy()
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(data, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")
