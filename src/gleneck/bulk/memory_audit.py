from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .memory import estimate_memory_kernel_from_vacf, prepare_memory_terms
from .units import internal_time_to_ps, memory_internal_to_per_ps2


@dataclass(frozen=True)
class DecaySummary:
    peak: float
    peak_time: float
    initial: float
    t_1e: float | None
    t_10pct: float | None
    integral: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def first_threshold_time(tau: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    """Return the first time where ``values`` fall at or below ``threshold``."""
    x = np.asarray(tau, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("tau and values must have matching lengths.")
    if x.size == 0:
        return None
    hits = np.flatnonzero(y <= threshold)
    if hits.size == 0:
        return None
    return float(x[int(hits[0])])


def decay_summary(tau: np.ndarray, values: np.ndarray) -> DecaySummary:
    """Summarize decay times for a memory-like curve on its explicit time grid."""
    x = np.asarray(tau, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("tau and values must have matching lengths.")
    if x.size == 0:
        raise ValueError("tau and values cannot be empty.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("tau and values must be finite.")
    if np.any(np.diff(x) < 0):
        raise ValueError("tau must be monotonically non-decreasing.")

    peak_index = int(np.argmax(y))
    initial = float(y[0])
    t_1e = None
    t_10pct = None
    if initial > 0:
        t_1e = first_threshold_time(x, y, initial / np.e)
        t_10pct = first_threshold_time(x, y, 0.1 * initial)
    return DecaySummary(
        peak=float(y[peak_index]),
        peak_time=float(x[peak_index]),
        initial=initial,
        t_1e=t_1e,
        t_10pct=t_10pct,
        integral=float(np.trapezoid(y, x)),
    )


def alpha_for_time_unit_conversion(alpha_internal: float, ps_per_internal: float) -> float:
    """Scale Volterra Tikhonov regularization when converting time units.

    If ``t_ps = ps_per_internal * t_internal``, then the memory kernel scales as
    ``M_ps = M_internal / ps_per_internal**2``.  The normal-equation
    regularizer must scale as ``alpha_ps = alpha_internal * ps_per_internal**2``
    for the refit to be mathematically equivalent.
    """
    if alpha_internal < 0:
        raise ValueError("alpha_internal must be non-negative.")
    if ps_per_internal <= 0:
        raise ValueError("ps_per_internal must be positive.")
    return float(alpha_internal) * float(ps_per_internal) ** 2


def volterra_unit_scaling_audit(
    time_internal: np.ndarray,
    vacf: np.ndarray,
    alpha_internal: float,
    max_lags: int | None = None,
) -> dict[str, Any]:
    """Refit a VACF in internal and ps time units and compare memory scaling."""
    time_i = np.asarray(time_internal, dtype=float).reshape(-1)
    corr = np.asarray(vacf, dtype=float).reshape(-1)
    if max_lags is not None:
        if max_lags < 2:
            raise ValueError("max_lags must be at least 2.")
        time_i = time_i[:max_lags]
        corr = corr[:max_lags]

    time_ps = np.asarray(internal_time_to_ps(time_i), dtype=float)
    ps_per_internal = float(time_ps[1] / time_i[1]) if time_i.size > 1 and time_i[1] else float(
        internal_time_to_ps(1.0)
    )
    alpha_ps = alpha_for_time_unit_conversion(alpha_internal, ps_per_internal)

    memory_internal, dt_internal = estimate_memory_kernel_from_vacf(time_i, corr, alpha=alpha_internal)
    memory_ps, dt_ps = estimate_memory_kernel_from_vacf(time_ps, corr, alpha=alpha_ps)
    converted_internal = np.asarray(memory_internal_to_per_ps2(memory_internal), dtype=float)
    denominator = np.maximum(1.0, np.abs(memory_ps))
    abs_error = np.abs(converted_internal - memory_ps)
    rel_error = abs_error / denominator

    return {
        "lags_used": int(time_i.size),
        "dt_internal": float(dt_internal),
        "dt_ps": float(dt_ps),
        "ps_per_internal": ps_per_internal,
        "alpha_internal": float(alpha_internal),
        "alpha_ps_equivalent": float(alpha_ps),
        "max_abs_error_ps2": float(np.max(abs_error)),
        "max_relative_error_ps2": float(np.max(rel_error)),
        "rmse_ps2": float(np.sqrt(np.mean((converted_internal - memory_ps) ** 2))),
    }


def memory_resampling_audit(
    memory_internal: np.ndarray,
    source_dt_internal: float,
    gle_dt_internal: float,
    l_max: int,
) -> dict[str, Any]:
    """Audit raw-memory and GLE-resampled time grids."""
    if source_dt_internal <= 0 or gle_dt_internal <= 0:
        raise ValueError("source_dt_internal and gle_dt_internal must be positive.")
    if l_max <= 0:
        raise ValueError("l_max must be positive.")

    values = np.asarray(memory_internal, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("memory_internal cannot be empty.")

    orig_interval = float(source_dt_internal / gle_dt_internal)
    terms = prepare_memory_terms(values, l_max=l_max, orig_interval=orig_interval)
    raw_tau_ps = np.asarray(internal_time_to_ps(np.arange(values.size) * source_dt_internal), dtype=float)
    fine_tau_ps = np.asarray(internal_time_to_ps(np.arange(l_max) * gle_dt_internal), dtype=float)
    mistaken_tau_ps = np.asarray(internal_time_to_ps(np.arange(values.size) * gle_dt_internal), dtype=float)
    raw_ps2 = np.asarray(memory_internal_to_per_ps2(values), dtype=float)
    resampled_ps2 = np.asarray(memory_internal_to_per_ps2(terms.memory), dtype=float)

    return {
        "source_dt_internal": float(source_dt_internal),
        "source_dt_ps": float(internal_time_to_ps(source_dt_internal)),
        "gle_dt_internal": float(gle_dt_internal),
        "gle_dt_ps": float(internal_time_to_ps(gle_dt_internal)),
        "memory_orig_interval": orig_interval,
        "raw_points": int(values.size),
        "l_max": int(l_max),
        "raw_decay": decay_summary(raw_tau_ps, raw_ps2).to_dict(),
        "resampled_decay": decay_summary(fine_tau_ps, resampled_ps2).to_dict(),
        "mistaken_raw_as_fine_decay": decay_summary(mistaken_tau_ps, raw_ps2).to_dict(),
    }
