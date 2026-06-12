from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import BulkModelConfig
from .gle import make_force_tables
from .io import load_cg_potential_npz
from .memory import prepare_memory_terms
from .units import effective_dt, effective_dt_ps, effective_temperature, notebook_unit_scalars


def memory_validation_report(memory: np.ndarray, config: BulkModelConfig) -> dict[str, Any]:
    """Summarize fitted memory and the FFT colored-noise construction."""
    terms = prepare_memory_terms(memory, l_max=config.l_max, orig_interval=config.memory_orig_interval)
    kernel = np.asarray(terms.memory, dtype=float)
    even_kernel = np.concatenate([kernel, kernel[-2:0:-1]])
    spectrum = np.real(np.fft.rfft(even_kernel))
    clipped_spectrum = np.maximum(spectrum, 1e-20)
    clipped_kernel = np.fft.irfft(clipped_spectrum, n=even_kernel.size)[: kernel.size]
    denom = max(float(np.linalg.norm(kernel)), 1e-30)
    return {
        "finite_memory": bool(np.all(np.isfinite(kernel))),
        "finite_noise_filter": bool(np.all(np.isfinite(terms.noise_filter))),
        "l_max": int(config.l_max),
        "effective_dt": effective_dt(config),
        "effective_dt_internal": effective_dt(config),
        "effective_dt_ps": effective_dt_ps(config),
        "memory_min": float(np.min(kernel)),
        "memory_max": float(np.max(kernel)),
        "memory_integral": float(np.sum(kernel) * effective_dt(config)),
        "memory_integral_internal": float(np.sum(kernel) * effective_dt(config)),
        "negative_spectrum_count": int(np.sum(spectrum < 0.0)),
        "negative_spectrum_min": float(np.min(spectrum)),
        "clipped_kernel_relative_error": float(np.linalg.norm(clipped_kernel - kernel) / denom),
        "noise_filter_min": float(np.min(terms.noise_filter)),
        "noise_filter_max": float(np.max(terms.noise_filter)),
    }


def potential_validation_report(potential_path: Path, config: BulkModelConfig) -> dict[str, Any]:
    """Summarize retained-solute CG potential and force tables."""
    potential = load_cg_potential_npz(potential_path)
    r_bins, u_table, force_table = make_force_tables(potential, config)
    return {
        "path": str(potential_path),
        "finite_potential": bool(np.all(np.isfinite(u_table))),
        "finite_force": bool(np.all(np.isfinite(force_table))),
        "r_min": float(r_bins[0]),
        "r_max": float(r_bins[-1]),
        "potential_min": float(np.min(u_table)),
        "potential_max": float(np.max(u_table)),
        "force_min": float(np.min(force_table)),
        "force_max": float(np.max(force_table)),
        "hard_core_potential": float(np.max(u_table[:, :, : max(1, int(np.searchsorted(r_bins, config.r_onset)))])),
    }


def unit_validation_report(config: BulkModelConfig) -> dict[str, Any]:
    units = notebook_unit_scalars()
    return {
        "notebook_dt": config.dt,
        "effective_dt": effective_dt(config),
        "effective_dt_internal": effective_dt(config),
        "effective_dt_ps": effective_dt_ps(config),
        "notebook_temperature": config.temperature,
        "effective_temperature": effective_temperature(config),
        "unit_scalars": {
            "distance": units.distance,
            "time": units.time,
            "energy": units.energy,
            "mass": units.mass,
            "temperature": units.temperature,
        },
    }


def validation_issues(report: dict[str, Any]) -> list[str]:
    """Return high-signal issues from a validation report."""
    issues: list[str] = []
    memory = report.get("memory", {})
    potential = report.get("potential", {})
    zero_field = report.get("zero_field", {})
    if not memory.get("finite_memory", False):
        issues.append("memory_kernel_has_nonfinite_values")
    if not memory.get("finite_noise_filter", False):
        issues.append("noise_filter_has_nonfinite_values")
    if memory.get("negative_spectrum_count", 0) > 0:
        issues.append("memory_kernel_requires_psd_clipping_for_noise")
    if not potential.get("finite_potential", False):
        issues.append("cg_potential_has_nonfinite_values")
    if not potential.get("finite_force", False):
        issues.append("cg_force_table_has_nonfinite_values")
    if zero_field.get("diagnostic_warnings"):
        issues.extend(f"zero_field_{item}" for item in zero_field["diagnostic_warnings"])
    return issues
