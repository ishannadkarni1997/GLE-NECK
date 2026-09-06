from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from gleneck.artifacts import read_numeric_table

from .config import DEFAULT_BULK_MODEL, DEFAULT_MPT_TRAINING, BulkModelConfig, BulkTrainingConfig
from .io import default_bulk_input_paths, load_cg_potential, load_cg_potential_npz, write_cg_potential_npz
from .memory import prepare_memory_terms
from .neck import initialize_kernel_params, require_jax_stack
from .units import (
    effective_dt,
    effective_dt_ps,
    memory_internal_to_per_ps2,
    memory_per_ps2_to_internal,
    effective_mass,
    effective_temperature,
    notebook_unit_scalars,
)


@dataclass(frozen=True)
class MobilityCurve:
    fields: np.ndarray
    drifts: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {"fields": self.fields.tolist(), "drifts": self.drifts.tolist()}


def load_mobility_curve(path: Path) -> MobilityCurve:
    """Load a two-column mobility curve with or without a header."""
    rows: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"No numeric mobility rows found in {path}.")
    data = np.asarray(rows, dtype=float)
    return MobilityCurve(fields=data[:, 0], drifts=data[:, 1])


def write_mobility_curve(path: Path, fields: np.ndarray, drifts: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([fields, drifts]),
        delimiter=",",
        header="external_field,drift_velocity",
        comments="",
    )
    return path


def model_config_with_overrides(
    config: BulkModelConfig = DEFAULT_BULK_MODEL,
    *,
    dt: float | None = None,
    l_max: int | None = None,
    memory_orig_interval: float | None = None,
    kernel_model: str | None = None,
    poly_degree: int | None = None,
    kernel_field_scale: float | None = None,
    kernel_tau_prior_ps: float | None = None,
    kernel_exp_basis_taus_ps: tuple[float, ...] | None = None,
    kernel_shape_regularization_weight: float | None = None,
    kernel_l2_regularization_weight: float | None = None,
    kernel_time_moment_regularization_weight: float | None = None,
    kernel_derivative_regularization_weight: float | None = None,
    kernel_second_derivative_regularization_weight: float | None = None,
    memory_history_scaling: str | None = None,
    learning_rate: float | None = None,
    init_velocity_scale: float | None = None,
    noise_scale: float | None = None,
) -> BulkModelConfig:
    """Return a model config with runtime-only GLE stability overrides."""
    updates: dict[str, Any] = {}
    if dt is not None:
        if dt <= 0:
            raise ValueError("--dt must be positive.")
        updates["dt"] = float(dt)
    if l_max is not None:
        if l_max <= 1:
            raise ValueError("--l-max must be greater than 1.")
        updates["l_max"] = int(l_max)
    if memory_orig_interval is not None:
        if memory_orig_interval <= 0:
            raise ValueError("--memory-orig-interval must be positive.")
        updates["memory_orig_interval"] = float(memory_orig_interval)
    if kernel_model is not None:
        if kernel_model not in {"asym_softmax_poly", "asym_exp_basis", "e2_direct", "e2_lag_embed", "e2_tau_mlp", "e2_film_tau"}:
            raise ValueError(
                "--kernel-model must be 'asym_softmax_poly', 'asym_exp_basis', 'e2_direct', "
                "'e2_lag_embed', 'e2_tau_mlp', or 'e2_film_tau'."
            )
        updates["kernel_model"] = kernel_model
    if poly_degree is not None:
        if poly_degree < 1:
            raise ValueError("--poly-degree must be at least 1.")
        updates["poly_degree"] = int(poly_degree)
    if kernel_field_scale is not None:
        if kernel_field_scale <= 0:
            raise ValueError("--kernel-field-scale must be positive.")
        updates["kernel_field_scale"] = float(kernel_field_scale)
    if kernel_tau_prior_ps is not None:
        if kernel_tau_prior_ps < 0:
            raise ValueError("--kernel-tau-prior-ps must be non-negative.")
        updates["kernel_tau_prior_ps"] = float(kernel_tau_prior_ps)
    if kernel_exp_basis_taus_ps is not None:
        if not kernel_exp_basis_taus_ps:
            raise ValueError("--kernel-exp-basis-taus-ps must contain at least one value.")
        if any(tau <= 0 for tau in kernel_exp_basis_taus_ps):
            raise ValueError("--kernel-exp-basis-taus-ps values must be positive.")
        updates["kernel_exp_basis_taus_ps"] = tuple(float(tau) for tau in kernel_exp_basis_taus_ps)
    if kernel_shape_regularization_weight is not None:
        if kernel_shape_regularization_weight < 0:
            raise ValueError("--kernel-shape-regularization-weight must be non-negative.")
        updates["kernel_shape_regularization_weight"] = float(kernel_shape_regularization_weight)
    if kernel_l2_regularization_weight is not None:
        if kernel_l2_regularization_weight < 0:
            raise ValueError("--kernel-l2-regularization-weight must be non-negative.")
        updates["kernel_l2_regularization_weight"] = float(kernel_l2_regularization_weight)
    if kernel_time_moment_regularization_weight is not None:
        if kernel_time_moment_regularization_weight < 0:
            raise ValueError("--kernel-time-moment-regularization-weight must be non-negative.")
        updates["kernel_time_moment_regularization_weight"] = float(kernel_time_moment_regularization_weight)
    if kernel_derivative_regularization_weight is not None:
        if kernel_derivative_regularization_weight < 0:
            raise ValueError("--kernel-derivative-regularization-weight must be non-negative.")
        updates["kernel_derivative_regularization_weight"] = float(kernel_derivative_regularization_weight)
    if kernel_second_derivative_regularization_weight is not None:
        if kernel_second_derivative_regularization_weight < 0:
            raise ValueError("--kernel-second-derivative-regularization-weight must be non-negative.")
        updates["kernel_second_derivative_regularization_weight"] = float(kernel_second_derivative_regularization_weight)
    if memory_history_scaling is not None:
        if memory_history_scaling not in {"force-units", "legacy-training"}:
            raise ValueError("--memory-history-scaling must be 'force-units' or 'legacy-training'.")
        updates["memory_history_scaling"] = memory_history_scaling
    if learning_rate is not None:
        if learning_rate <= 0:
            raise ValueError("--learning-rate must be positive.")
        updates["learning_rate"] = float(learning_rate)
    if init_velocity_scale is not None:
        if init_velocity_scale < 0:
            raise ValueError("--init-velocity-scale must be non-negative.")
        updates["init_velocity_scale"] = float(init_velocity_scale)
    if noise_scale is not None:
        if noise_scale < 0:
            raise ValueError("--noise-scale must be non-negative.")
        updates["noise_scale"] = float(noise_scale)
    return replace(config, **updates) if updates else config


def write_kernel_evolution_tables(
    output_dir: Path,
    base_memory: np.ndarray,
    kernel_history: np.ndarray,
    kernel_epochs: np.ndarray,
    config: BulkModelConfig,
    suffix: str = "",
) -> tuple[Path, Path]:
    """Write corrective and total memory-kernel evolution tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_internal = np.asarray(base_memory, dtype=float)[: config.l_max]
    kernels_internal = np.asarray(kernel_history, dtype=float)
    base = memory_internal_to_per_ps2(base_internal)
    kernels = memory_internal_to_per_ps2(kernels_internal)
    epochs = np.asarray(kernel_epochs, dtype=int)
    if kernels.ndim != 2 or kernels.shape[1] != base.shape[0]:
        raise ValueError(f"kernel history shape {kernels.shape} does not match base kernel length {base.shape[0]}.")
    lag_index = np.arange(base.shape[0], dtype=float)
    tau_internal = lag_index * effective_dt(config)
    tau_ps = lag_index * effective_dt_ps(config)
    epoch_columns = [f"epoch_{int(epoch)}" for epoch in epochs]

    correction_header = ",".join(["lag_index", "tau_internal", "tau_ps", "original_kernel_ps2", *epoch_columns])
    correction_table = np.column_stack([lag_index, tau_internal, tau_ps, base, kernels.T])
    correction_path = output_dir / f"corrective_kernel_evolution{suffix}.csv"
    np.savetxt(correction_path, correction_table, delimiter=",", header=correction_header, comments="")

    total_header = ",".join(["lag_index", "tau_internal", "tau_ps", "original_kernel_ps2", *epoch_columns])
    total_table = np.column_stack([lag_index, tau_internal, tau_ps, base, (base[None, :] + kernels).T])
    total_path = output_dir / f"total_kernel_evolution{suffix}.csv"
    np.savetxt(total_path, total_table, delimiter=",", header=total_header, comments="")
    return correction_path, total_path


def write_kernel_snapshot_table(
    output_dir: Path,
    base_memory: np.ndarray,
    corrective_kernel: np.ndarray,
    config: BulkModelConfig,
    suffix: str,
) -> Path:
    """Write one corrective-kernel snapshot for a selected model state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base_internal = np.asarray(base_memory, dtype=float)[: config.l_max]
    correction_internal = np.asarray(corrective_kernel, dtype=float)[: config.l_max]
    base = memory_internal_to_per_ps2(base_internal)
    correction = memory_internal_to_per_ps2(correction_internal)
    lag_index = np.arange(base.shape[0], dtype=float)
    tau_internal = lag_index * effective_dt(config)
    tau_ps = lag_index * effective_dt_ps(config)
    table = np.column_stack(
        [
            lag_index,
            tau_internal,
            tau_ps,
            base_internal,
            correction_internal,
            base_internal + correction_internal,
            base,
            correction,
            base + correction,
        ]
    )
    path = output_dir / f"corrective_kernel_snapshot{suffix}.csv"
    np.savetxt(
        path,
        table,
        delimiter=",",
        header=(
            "lag_index,tau_internal,tau_ps,"
            "original_kernel_internal,corrective_kernel_internal,total_kernel_internal,"
            "original_kernel_ps2,corrective_kernel_ps2,total_kernel_ps2"
        ),
        comments="",
    )
    return path


def kernel_shape_diagnostics(
    kernels_by_field: np.ndarray,
    fields: np.ndarray,
    config: BulkModelConfig,
    *,
    label: str,
    late_lag_threshold_ps: float = 0.5,
) -> list[dict[str, Any]]:
    """Summarize where each field-conditioned corrective kernel is concentrated."""
    kernels = np.asarray(kernels_by_field, dtype=float)
    field_values = np.asarray(fields, dtype=float)
    tau_ps = np.arange(config.l_max, dtype=float) * effective_dt_ps(config)
    if kernels.ndim != 2:
        raise ValueError(f"kernels_by_field must have shape (field, lag), got {kernels.shape}.")
    if kernels.shape[0] != field_values.shape[0]:
        raise ValueError(f"kernel field count {kernels.shape[0]} does not match fields {field_values.shape[0]}.")
    diagnostics: list[dict[str, Any]] = []
    for field, kernel in zip(field_values, kernels):
        finite = bool(np.all(np.isfinite(kernel)))
        if finite and kernel.size:
            argmax_index = int(np.nanargmax(kernel))
            argmax_tau = float(tau_ps[argmax_index])
            max_value = float(kernel[argmax_index])
            first_value = float(kernel[0])
            last_value = float(kernel[-1])
        else:
            argmax_index = -1
            argmax_tau = float("nan")
            max_value = float("nan")
            first_value = float("nan")
            last_value = float("nan")
        warnings: list[str] = []
        if not finite:
            warnings.append("nonfinite_corrective_kernel")
        if finite and argmax_tau > late_lag_threshold_ps:
            warnings.append(f"argmax_tau_after_{late_lag_threshold_ps:g}ps")
        diagnostics.append(
            {
                "label": label,
                "field": float(field),
                "finite": finite,
                "argmax_index": argmax_index,
                "argmax_tau_ps": argmax_tau,
                "max_value": max_value,
                "first_value": first_value,
                "last_value": last_value,
                "last_over_first_abs": float(abs(last_value) / max(abs(first_value), 1.0e-300)),
                "late_lag_threshold_ps": late_lag_threshold_ps,
                "warnings": warnings,
            }
        )
    return diagnostics


def kernel_shape_warnings(diagnostics: list[dict[str, Any]]) -> list[str]:
    """Collect unique field-labelled kernel-shape warning labels."""
    warnings: list[str] = []
    for item in diagnostics:
        field = float(item["field"])
        label = str(item["label"])
        for warning in item["warnings"]:
            warnings.append(f"{label}_E{field:.6g}_{warning}")
    return sorted(set(warnings))


def field_slug(field: float) -> str:
    """Return a filename-safe field label such as E1 or E0p5."""
    return f"E{float(field):.6g}".replace("-", "m").replace(".", "p")


def load_or_create_potential(root: Path, potential_path: Path | None = None) -> tuple[dict[tuple[int, int], np.ndarray], Path]:
    if potential_path is not None:
        return load_cg_potential_npz(potential_path), potential_path
    processed = root / "data" / "processed" / "bulk" / "ibi_potentials.npz"
    if processed.exists():
        return load_cg_potential_npz(processed), processed
    raise FileNotFoundError(f"Missing CG potential archive: {processed}")


def load_memory_kernel_for_training(path: Path) -> np.ndarray:
    if path.suffix == ".csv":
        table = read_numeric_table(path)
        table.require("fitted_memory_ps2")
        return np.asarray(memory_per_ps2_to_internal(table.columns["fitted_memory_ps2"]), dtype=float)
    return np.load(path)


def generated_retained_positions(stack: Any, config: BulkModelConfig = DEFAULT_BULK_MODEL):
    """Create deterministic low-overlap retained-solute positions in the bulk box."""
    jnp = stack.jnp
    box = jnp.asarray(config.box_size, dtype=jnp.float64)
    n_axis = math.ceil(config.n_cg ** (1.0 / 3.0))
    axes = [
        jnp.linspace(0.0, float(box[dim]), n_axis, endpoint=False, dtype=jnp.float64)
        + float(box[dim]) / (2.0 * n_axis)
        for dim in range(3)
    ]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([axis.reshape(-1) for axis in mesh], axis=1)[: config.n_cg]


def generated_retained_state(stack: Any, seed: int = 0, config: BulkModelConfig = DEFAULT_BULK_MODEL):
    """Generate retained-solute positions and Maxwell-Boltzmann velocities."""
    jax, jnp = stack.jax, stack.jnp
    positions = generated_retained_positions(stack, config)
    _, mass = make_species_and_mass(stack, config)
    key = jax.random.PRNGKey(seed)
    velocities = (
        jax.random.normal(key, (config.n_cg, 3), dtype=jnp.float64)
        * jnp.sqrt(effective_temperature(config) / mass)
        * config.init_velocity_scale
    )
    velocities = velocities - jnp.mean(velocities, axis=0, keepdims=True)
    return positions, velocities


def lag_order_velocity_history(velocity: np.ndarray, l_max: int) -> np.ndarray:
    """Convert a chronological velocity trajectory into lag order: current, previous, ..."""
    values = np.asarray(velocity, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"velocity history must have shape (T, N, 3), got {values.shape}.")
    if values.shape[0] == 0:
        raise ValueError("velocity history is empty.")
    history = values[-l_max:][::-1].copy()
    if history.shape[0] < l_max:
        pad = np.repeat(history[-1:,:,:], l_max - history.shape[0], axis=0)
        history = np.concatenate([history, pad], axis=0)
    return history


def load_retained_state(
    stack: Any,
    root: Path,
    private_root: Path | None = None,
    init_mode: str = "generated",
    seed: int = 0,
    config: BulkModelConfig = DEFAULT_BULK_MODEL,
):
    """Load legacy retained state or generate a fresh thermal state."""
    jnp = stack.jnp
    paths = default_bulk_input_paths(root, private_root)
    if init_mode == "generated":
        positions, velocities = generated_retained_state(stack, seed=seed, config=config)
        return (positions, velocities, None), "generated"
    if init_mode != "legacy":
        raise ValueError("--init-mode must be 'generated' or 'legacy'.")
    if not paths.trajectory.exists() or not paths.velocity.exists():
        raise FileNotFoundError(
            "Retained-solute trajectory inputs are missing. Stage traj_cg.npy and vel_cg.npy, "
            "or use --init-mode generated.\n"
            f"{paths.trajectory}\n{paths.velocity}"
        )
    trajectory = np.load(paths.trajectory)
    velocity = np.load(paths.velocity)
    history = lag_order_velocity_history(velocity, config.l_max) if velocity.shape[0] >= config.l_max else None
    state_source = "legacy" if history is not None else f"legacy_random_history_from_{velocity.shape[0]}_velocity_frames"
    return (
        jnp.asarray(trajectory[-1], dtype=jnp.float64),
        jnp.asarray(velocity[-1], dtype=jnp.float64),
        None if history is None else jnp.asarray(history, dtype=jnp.float64),
    ), state_source


def make_species_and_mass(stack: Any, config: BulkModelConfig = DEFAULT_BULK_MODEL):
    jnp = stack.jnp
    species = jnp.concatenate(
        [
            jnp.zeros(config.n_a, dtype=jnp.int32),
            jnp.ones(config.n_b, dtype=jnp.int32),
        ]
    )
    mass = jnp.concatenate(
        [
            jnp.full(config.n_a, effective_mass(config.mass_a), dtype=jnp.float64),
            jnp.full(config.n_b, effective_mass(config.mass_b), dtype=jnp.float64),
        ]
    )[:, None]
    return species, mass


def make_force_tables(potential: dict[tuple[int, int], np.ndarray], config: BulkModelConfig = DEFAULT_BULK_MODEL):
    """Build tabulated potential and smoothed radial-force tables for retained dynamics."""
    r_bins = np.linspace(config.r_min, config.r_max, config.num_bins)
    u_aa = np.asarray(potential[(0, 0)], dtype=float)
    u_ab = np.asarray(potential[(0, 1)], dtype=float)
    u_bb = np.asarray(potential[(1, 1)], dtype=float)
    u_table = np.asarray([[u_aa, u_ab], [u_ab, u_bb]], dtype=float)
    switch, switch_derivative = smooth_cutoff_np(r_bins, config)
    force_table = -(switch * np.gradient(u_table, r_bins, axis=-1) + switch_derivative * u_table)
    return r_bins, u_table, force_table


def smooth_cutoff_np(radii: np.ndarray, config: BulkModelConfig = DEFAULT_BULK_MODEL) -> tuple[np.ndarray, np.ndarray]:
    """JAX-MD multiplicative isotropic cutoff and radial derivative."""
    r = np.asarray(radii, dtype=float)
    r_cutoff_sq = float(config.r_cutoff) ** 2
    r_onset_sq = float(config.r_onset) ** 2
    r_sq = r * r
    denominator = (r_cutoff_sq - r_onset_sq) ** 3
    inner = (r_cutoff_sq - r_sq) ** 2 * (r_cutoff_sq + 2.0 * r_sq - 3.0 * r_onset_sq) / denominator
    switch = np.where(r < config.r_onset, 1.0, np.where(r < config.r_cutoff, inner, 0.0))
    derivative = 12.0 * r * (r_cutoff_sq - r_sq) * (r_onset_sq - r_sq) / denominator
    derivative = np.where((r >= config.r_onset) & (r < config.r_cutoff), derivative, 0.0)
    return switch, derivative


def smooth_cutoff_jax(stack: Any, radii: Any, config: BulkModelConfig = DEFAULT_BULK_MODEL):
    """JAX-MD multiplicative isotropic cutoff and radial derivative."""
    jnp = stack.jnp
    r_cutoff_sq = jnp.asarray(config.r_cutoff * config.r_cutoff, dtype=radii.dtype)
    r_onset_sq = jnp.asarray(config.r_onset * config.r_onset, dtype=radii.dtype)
    r_sq = radii * radii
    denominator = (r_cutoff_sq - r_onset_sq) ** 3
    inner = (r_cutoff_sq - r_sq) ** 2 * (r_cutoff_sq + 2.0 * r_sq - 3.0 * r_onset_sq) / denominator
    switch = jnp.where(radii < config.r_onset, 1.0, jnp.where(radii < config.r_cutoff, inner, 0.0))
    derivative = 12.0 * radii * (r_cutoff_sq - r_sq) * (r_onset_sq - r_sq) / denominator
    derivative = jnp.where((radii >= config.r_onset) & (radii < config.r_cutoff), derivative, 0.0)
    return switch, derivative


def interpolate_tabulated_potential_and_slope(stack: Any, radii: Any, r_bins: Any, potential_table: Any):
    """Match ``jnp.interp`` values and its piecewise-linear radial slope."""
    jnp = stack.jnp
    flat_radii = radii.reshape(-1)
    flat_table = potential_table.reshape((-1, potential_table.shape[-1]))
    idx_hi = jnp.searchsorted(r_bins, flat_radii, side="right")
    idx_hi = jnp.clip(idx_hi, 1, r_bins.shape[0] - 1)
    idx_lo = idx_hi - 1
    r_lo = r_bins[idx_lo]
    r_hi = r_bins[idx_hi]
    u_lo = jnp.take_along_axis(flat_table, idx_lo[:, None], axis=1)[:, 0]
    u_hi = jnp.take_along_axis(flat_table, idx_hi[:, None], axis=1)[:, 0]
    slope = (u_hi - u_lo) / (r_hi - r_lo)
    values = u_lo + slope * (flat_radii - r_lo)
    below = flat_radii <= r_bins[0]
    above = flat_radii >= r_bins[-1]
    values = jnp.where(below, flat_table[:, 0], jnp.where(above, flat_table[:, -1], values))
    slope = jnp.where(below | above, 0.0, slope)
    return values.reshape(radii.shape), slope.reshape(radii.shape)


def pairwise_tabulated_force(stack: Any, positions: Any, species: Any, r_bins: Any, potential_table: Any, config: BulkModelConfig):
    """All-pairs periodic force matching the legacy JAX-MD tabulated potential."""
    jnp = stack.jnp
    box = jnp.asarray(config.box_size, dtype=jnp.float64)
    delta = positions[:, None, :] - positions[None, :, :]
    delta = delta - box * jnp.round(delta / box)
    squared_distances = jnp.sum(delta * delta, axis=-1)
    valid_pair = squared_distances > 1e-24
    distances = jnp.sqrt(jnp.where(valid_pair, squared_distances, 1.0))
    pair_potential_table = potential_table[species[:, None], species[None, :]]
    potential_value, potential_slope = interpolate_tabulated_potential_and_slope(
        stack,
        distances,
        r_bins,
        pair_potential_table,
    )
    switch, switch_derivative = smooth_cutoff_jax(stack, distances, config)
    radial_force = -(switch * potential_slope + switch_derivative * potential_value)
    mask = valid_pair & (distances < config.r_cutoff)
    radial_force = jnp.where(mask, radial_force, 0.0)
    unit = jnp.where(valid_pair[..., None], delta / distances[..., None], 0.0)
    return jnp.sum(radial_force[..., None] * unit, axis=1)


def shift_periodic(stack: Any, positions: Any, displacement: Any, config: BulkModelConfig):
    jnp = stack.jnp
    box = jnp.asarray(config.box_size, dtype=jnp.float64)
    return (positions + displacement) % box


def _mass_array_np(config: BulkModelConfig = DEFAULT_BULK_MODEL) -> np.ndarray:
    return np.concatenate(
        [
            np.full(config.n_a, effective_mass(config.mass_a), dtype=float),
            np.full(config.n_b, effective_mass(config.mass_b), dtype=float),
        ]
    )


def _minimum_pair_distance_np(positions: np.ndarray, config: BulkModelConfig = DEFAULT_BULK_MODEL) -> float:
    if positions.shape[0] < 2:
        return float("nan")
    box = np.asarray(config.box_size, dtype=float)
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= box * np.round(delta / box)
    distances = np.linalg.norm(delta, axis=-1)
    distances[np.diag_indices_from(distances)] = np.inf
    return float(np.min(distances))


def stability_diagnostics(
    positions: Any,
    velocities: Any,
    label: str,
    config: BulkModelConfig = DEFAULT_BULK_MODEL,
    hard_core_threshold: float | None = None,
    speed_threshold: float = 10.0,
    temperature_factor: float = 10.0,
    low_temperature_factor: float = 0.1,
) -> dict[str, Any]:
    """Summarize GLE trajectory stability without touching differentiable code paths."""
    pos = np.asarray(positions, dtype=float)
    vel = np.asarray(velocities, dtype=float)
    if pos.ndim == 2:
        pos = pos[None, :, :]
    if vel.ndim == 2:
        vel = vel[None, :, :]
    if pos.shape != vel.shape:
        raise ValueError(f"Position and velocity shapes differ: {pos.shape} vs {vel.shape}.")

    finite = bool(np.all(np.isfinite(pos)) and np.all(np.isfinite(vel)))
    speeds = np.linalg.norm(vel, axis=-1)
    masses = _mass_array_np(config)
    target_temperature = effective_temperature(config)
    kinetic_temperature = np.sum(masses[None, :] * np.sum(vel * vel, axis=-1), axis=1) / (3.0 * config.n_cg)
    frame_drift_velocity = np.mean(vel, axis=1, keepdims=True)
    peculiar_velocity = vel - frame_drift_velocity
    peculiar_temperature = (
        np.sum(masses[None, :] * np.sum(peculiar_velocity * peculiar_velocity, axis=-1), axis=1)
        / (3.0 * config.n_cg)
    )
    min_distances = np.asarray([_minimum_pair_distance_np(frame, config) for frame in pos], dtype=float)
    drift_x = np.mean(vel[:, :, 0], axis=1)
    hard_core = config.r_onset if hard_core_threshold is None else hard_core_threshold

    warnings: list[str] = []
    if not finite:
        warnings.append("nonfinite_position_or_velocity")
    if np.nanmin(min_distances) < hard_core:
        warnings.append(f"minimum_pair_distance_below_{hard_core:g}")
    if np.nanmax(speeds) > speed_threshold:
        warnings.append(f"max_speed_above_{speed_threshold:g}")
    if np.nanmax(peculiar_temperature) > temperature_factor * target_temperature:
        warnings.append(f"temperature_above_{temperature_factor:g}x_target")
    if np.nanmean(peculiar_temperature) < low_temperature_factor * target_temperature:
        warnings.append(f"temperature_below_{low_temperature_factor:g}x_target")

    return {
        "label": label,
        "shape": list(pos.shape),
        "finite": finite,
        "warnings": warnings,
        "min_pair_distance": float(np.nanmin(min_distances)),
        "final_min_pair_distance": float(min_distances[-1]),
        "max_speed": float(np.nanmax(speeds)),
        "rms_speed": float(np.sqrt(np.nanmean(speeds * speeds))),
        "temperature_mean": float(np.nanmean(kinetic_temperature)),
        "temperature_min": float(np.nanmin(kinetic_temperature)),
        "temperature_max": float(np.nanmax(kinetic_temperature)),
        "temperature_final": float(kinetic_temperature[-1]),
        "peculiar_temperature_mean": float(np.nanmean(peculiar_temperature)),
        "peculiar_temperature_min": float(np.nanmin(peculiar_temperature)),
        "peculiar_temperature_max": float(np.nanmax(peculiar_temperature)),
        "peculiar_temperature_final": float(peculiar_temperature[-1]),
        "target_temperature": target_temperature,
        "drift_x_mean": float(np.nanmean(drift_x)),
        "drift_x_final": float(drift_x[-1]),
    }


def diagnostic_warnings(diagnostics: list[dict[str, Any]]) -> list[str]:
    """Collect unique stability warning labels from diagnostics."""
    return sorted({warning for item in diagnostics for warning in item["warnings"]})


def diagnostic_status(diagnostics: list[dict[str, Any]]) -> str:
    return "unstable" if diagnostic_warnings(diagnostics) else "completed"


def run_gle(
    stack: Any,
    positions: Any,
    velocities: Any,
    potential: dict[tuple[int, int], np.ndarray],
    memory: np.ndarray,
    noise_filter: np.ndarray,
    field: float,
    steps: int,
    seed: int = 0,
    corrective_kernel: Any | None = None,
    history_velocities: Any | None = None,
    initial_noise_buffer: Any | None = None,
    return_final_state: bool = False,
    return_noise_buffer: bool = False,
    config: BulkModelConfig = DEFAULT_BULK_MODEL,
):
    """Run the retained-solute GLE integrator used for baseline and NECK training."""
    jax, jnp = stack.jax, stack.jnp
    r_bins_np, potential_table_np, _ = make_force_tables(potential, config)
    r_bins = jnp.asarray(r_bins_np, dtype=jnp.float64)
    potential_table = jnp.asarray(potential_table_np, dtype=jnp.float64)
    species, mass = make_species_and_mass(stack, config)
    inv_mass = 1.0 / mass
    base_memory = jnp.asarray(memory, dtype=jnp.float64)
    noise_filter_jax = jnp.asarray(noise_filter, dtype=jnp.float64)
    kernel = base_memory if corrective_kernel is None else base_memory + corrective_kernel
    dt = jnp.asarray(effective_dt(config), dtype=jnp.float64)
    temp = jnp.asarray(effective_temperature(config), dtype=jnp.float64)
    key = jax.random.PRNGKey(seed)
    if history_velocities is None:
        key, history_key = jax.random.split(key)
        hist_v = (
            jax.random.normal(history_key, (config.l_max, config.n_cg, 3), dtype=jnp.float64)
            * jnp.sqrt(temp * inv_mass)
            * config.init_velocity_scale
        )
    else:
        hist_v = jnp.asarray(history_velocities, dtype=jnp.float64)
        if hist_v.shape[0] < config.l_max:
            pad = jnp.tile(hist_v[-1:, :, :], (config.l_max - hist_v.shape[0], 1, 1))
            hist_v = jnp.concatenate([hist_v, pad], axis=0)
        hist_v = hist_v[: config.l_max]
    key, buffer_key, series_key = jax.random.split(key, 3)
    if initial_noise_buffer is None:
        wn_buf = jax.random.normal(buffer_key, (config.l_max, config.n_cg, 3), dtype=jnp.float64)
    else:
        wn_buf = jnp.asarray(initial_noise_buffer, dtype=jnp.float64)
        if wn_buf.shape[0] < config.l_max:
            pad = jnp.tile(wn_buf[-1:, :, :], (config.l_max - wn_buf.shape[0], 1, 1))
            wn_buf = jnp.concatenate([wn_buf, pad], axis=0)
        wn_buf = wn_buf[: config.l_max]
    xi_series = (
        jax.random.normal(series_key, (steps, config.n_cg, 3), dtype=jnp.float64)
        * config.noise_scale
        * jnp.sqrt(3.0 * temp / inv_mass)
    )
    unit_scalars = notebook_unit_scalars()
    field_unit = unit_scalars.energy / unit_scalars.distance
    field_force = jnp.stack(
        [
            jnp.asarray(field, dtype=jnp.float64) * jnp.asarray(field_unit, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        ]
    )

    def step(carry, xi_t):
        pos, vel, hist_v, wn_buf = carry
        conservative = pairwise_tabulated_force(stack, pos, species, r_bins, potential_table, config)
        memory_convolution = jnp.tensordot(kernel[1:], hist_v[:-1], axes=(0, 0))
        if config.memory_history_scaling == "force-units":
            # Force-units convention used by the notebook baseline:
            # convert the history convolution to force-like units before the velocity update.
            friction = -memory_convolution * (dt / inv_mass)
        elif config.memory_history_scaling == "legacy-training":
            # Final legacy training cells multiply the history term by dt / mass before
            # adding it to the same total-force accumulator. Keep this as an explicit
            # diagnostic mode because it is not algebraically equivalent to force-units.
            friction = -memory_convolution * (dt * inv_mass)
        else:
            raise ValueError(f"Unknown memory_history_scaling: {config.memory_history_scaling!r}")
        wn_buf = jnp.concatenate([xi_t[None], wn_buf[:-1]], axis=0)
        noise = jnp.tensordot(noise_filter_jax, wn_buf, axes=(0, 0))
        total_force = conservative + friction + noise + field_force
        denom = 1.0 + dt * kernel[0] * inv_mass
        vel_new = (vel + dt * total_force * inv_mass) / denom
        pos_new = shift_periodic(stack, pos, 0.5 * (vel + vel_new) * dt, config)
        hist_v = jnp.concatenate([vel_new[None], hist_v[:-1]], axis=0)
        return (pos_new, vel_new, hist_v, wn_buf), (pos_new, vel_new)

    (pos, vel, hist_v, wn_buf), (sampled_positions_array, sampled_velocities_array) = jax.lax.scan(
        jax.checkpoint(step),
        (positions, velocities, hist_v, wn_buf),
        xi_series,
        _split_transpose=True,
    )
    if return_final_state:
        if return_noise_buffer:
            return sampled_positions_array, sampled_velocities_array, (pos, vel, hist_v, wn_buf)
        return sampled_positions_array, sampled_velocities_array, (pos, vel, hist_v)
    return sampled_positions_array, sampled_velocities_array


def drift_from_velocities(stack: Any, velocities: Any, window: int) -> Any:
    return stack.jnp.mean(velocities[-window:, :, 0])


def warmup_retained_state(
    stack: Any,
    positions: Any,
    velocities: Any,
    potential: dict[tuple[int, int], np.ndarray],
    memory: np.ndarray,
    noise_filter: np.ndarray,
    warmup_steps: int,
    history_velocities: Any | None = None,
    seed: int = 0,
    config: BulkModelConfig = DEFAULT_BULK_MODEL,
):
    """Run a zero-field GLE warmup and return the final state."""
    if warmup_steps <= 0:
        return positions, velocities, history_velocities
    sampled_positions, sampled_velocities, final_state = run_gle(
        stack,
        positions,
        velocities,
        potential,
        memory,
        noise_filter,
        field=0.0,
        steps=warmup_steps,
        seed=seed,
        history_velocities=history_velocities,
        return_final_state=True,
        config=config,
    )
    return final_state


def run_mobility_curve(
    root: Path,
    fields: np.ndarray,
    output_dir: Path,
    private_root: Path | None = None,
    potential_path: Path | None = None,
    memory_path: Path | None = None,
    init_mode: str = "generated",
    warmup_steps: int = 0,
    steps: int = 100,
    drift_window: int | None = None,
    seed: int = 0,
    config: BulkModelConfig = DEFAULT_BULK_MODEL,
) -> dict[str, Any]:
    started = time.time()
    stack = require_jax_stack()
    potential, potential_path = load_or_create_potential(root, potential_path)
    (positions, velocities, history_velocities), state_source = load_retained_state(stack, root, private_root, init_mode, seed, config)
    diagnostics = [stability_diagnostics(positions, velocities, "initial_state", config)]
    memory_path = memory_path or (root / "data" / "processed" / "bulk" / "memory_kernel.csv")
    terms = prepare_memory_terms(load_memory_kernel_for_training(memory_path), l_max=config.l_max, orig_interval=config.memory_orig_interval)
    drift_window = drift_window or config.drift_window
    positions, velocities, history_velocities = warmup_retained_state(
        stack,
        positions,
        velocities,
        potential,
        terms.memory,
        terms.noise_filter,
        warmup_steps=warmup_steps,
        history_velocities=history_velocities,
        seed=seed + 10_000,
        config=config,
    )
    diagnostics.append(stability_diagnostics(positions, velocities, "post_warmup_state", config))

    drifts = []
    for index, field in enumerate(fields):
        traj, vels = run_gle(
            stack,
            positions,
            velocities,
            potential,
            terms.memory,
            terms.noise_filter,
            field=field,
            steps=steps,
            seed=seed + index,
            history_velocities=history_velocities,
            config=config,
        )
        drifts.append(float(drift_from_velocities(stack, vels, drift_window)))
        diagnostics.append(stability_diagnostics(traj, vels, f"field_{float(field):.6g}", config))

    output_dir.mkdir(parents=True, exist_ok=True)
    mobility_path = write_mobility_curve(output_dir / "GLE_baseline_mobility.csv", np.asarray(fields, dtype=float), np.asarray(drifts))
    diagnostics_path = output_dir / "GLE_stability_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    warnings = diagnostic_warnings(diagnostics)
    report = {
        "status": "unstable" if warnings else "completed",
        "elapsed_seconds": time.time() - started,
        "state_source": state_source,
        "warmup_steps": warmup_steps,
        "dt": config.dt,
        "effective_dt": effective_dt(config),
        "effective_dt_internal": effective_dt(config),
        "effective_dt_ps": effective_dt_ps(config),
        "l_max": config.l_max,
        "memory_orig_interval": config.memory_orig_interval,
        "memory_kernel_unit": "internal_time^-2",
        "memory_history_scaling": config.memory_history_scaling,
        "init_velocity_scale": config.init_velocity_scale,
        "noise_scale": config.noise_scale,
        "temperature": config.temperature,
        "effective_temperature": effective_temperature(config),
        "potential_path": str(potential_path),
        "memory_path": str(memory_path),
        "steps": steps,
        "drift_window": drift_window,
        "mobility_output": str(mobility_path),
        "diagnostics_output": str(diagnostics_path),
        "diagnostic_warnings": warnings,
        "fields": [float(field) for field in fields],
        "drifts": drifts,
    }
    (output_dir / "GLE_baseline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def resolve_training_config(mode: str) -> BulkTrainingConfig:
    from .config import DEFAULT_SPT_HIGH_TRAINING, DEFAULT_SPT_LOW_TRAINING

    if mode == "mpt":
        return DEFAULT_MPT_TRAINING
    if mode == "spt-low":
        return DEFAULT_SPT_LOW_TRAINING
    if mode == "spt-high":
        return DEFAULT_SPT_HIGH_TRAINING
    raise ValueError(f"Unknown mode: {mode}")


def train_gleneck_legacy_targets(
    root: Path,
    output_dir: Path,
    mode: str = "mpt",
    private_root: Path | None = None,
    potential_path: Path | None = None,
    memory_path: Path | None = None,
    training_fields: np.ndarray | None = None,
    training_target_drifts: np.ndarray | None = None,
    training_name: str | None = None,
    init_mode: str = "generated",
    warmup_steps: int = 0,
    epochs: int = 5,
    steps_per_loss: int = 50,
    drift_window: int | None = None,
    eval_fields: np.ndarray | None = None,
    loss_weighting: str = "absolute",
    optimizer: str = "adam",
    lbfgs_epochs: int = 0,
    lbfgs_memory_size: int = 10,
    checkpoint_stride: int = 0,
    seed: int = 0,
    config: BulkModelConfig = DEFAULT_BULK_MODEL,
) -> dict[str, Any]:
    """Train the corrective kernel against drift targets."""
    started = time.time()
    stack = require_jax_stack()
    jax, jnp = stack.jax, stack.jnp
    notebook_unit_scalars()
    if training_fields is None:
        training = resolve_training_config(mode)
        fields_np = np.asarray(training.fields, dtype=float)
    else:
        fields_np = np.asarray(training_fields, dtype=float)
    if training_target_drifts is None:
        training = resolve_training_config(mode)
        targets_np = np.asarray(training.target_drifts, dtype=float)
    else:
        targets_np = np.asarray(training_target_drifts, dtype=float)
    if fields_np.shape != targets_np.shape:
        raise ValueError("training_fields and training_target_drifts must have matching shapes.")
    if fields_np.ndim != 1 or fields_np.size == 0:
        raise ValueError("At least one one-dimensional training field is required.")
    if optimizer not in {"adam", "adam-lbfgs"}:
        raise ValueError("optimizer must be 'adam' or 'adam-lbfgs'.")
    if lbfgs_epochs < 0:
        raise ValueError("lbfgs_epochs must be non-negative.")
    if optimizer == "adam-lbfgs" and lbfgs_epochs < 1:
        raise ValueError("optimizer='adam-lbfgs' requires lbfgs_epochs > 0.")
    if lbfgs_memory_size < 1:
        raise ValueError("lbfgs_memory_size must be positive.")
    if checkpoint_stride < 0:
        raise ValueError("checkpoint_stride must be non-negative.")
    output_dir.mkdir(parents=True, exist_ok=True)
    training = BulkTrainingConfig(
        name=training_name or f"custom_{mode}",
        mode=mode,
        fields=tuple(float(field) for field in fields_np),
        target_drifts=tuple(float(drift) for drift in targets_np),
        epochs=epochs,
    )

    potential, potential_path = load_or_create_potential(root, potential_path)
    (positions, velocities, history_velocities), state_source = load_retained_state(stack, root, private_root, init_mode, seed, config)
    diagnostics = [stability_diagnostics(positions, velocities, "initial_state", config)]
    memory_path = memory_path or (root / "data" / "processed" / "bulk" / "memory_kernel.csv")
    terms = prepare_memory_terms(load_memory_kernel_for_training(memory_path), l_max=config.l_max, orig_interval=config.memory_orig_interval)
    positions, velocities, history_velocities = warmup_retained_state(
        stack,
        positions,
        velocities,
        potential,
        terms.memory,
        terms.noise_filter,
        warmup_steps=warmup_steps,
        history_velocities=history_velocities,
        seed=seed + 10_000,
        config=config,
    )
    diagnostics.append(stability_diagnostics(positions, velocities, "post_warmup_state", config))
    transformed, params, tx, opt_state = initialize_kernel_params(config)
    drift_window = drift_window or config.drift_window
    fields = jnp.asarray(fields_np, dtype=jnp.float64)
    targets = jnp.asarray(targets_np, dtype=jnp.float64)
    eval_fields_np = np.asarray(eval_fields if eval_fields is not None else fields_np, dtype=float)
    eval_fields_jax = jnp.asarray(eval_fields_np, dtype=jnp.float64)
    shape_regularization_weight = float(config.kernel_shape_regularization_weight)
    l2_regularization_weight = float(config.kernel_l2_regularization_weight)
    time_moment_regularization_weight = float(config.kernel_time_moment_regularization_weight)
    derivative_regularization_weight = float(config.kernel_derivative_regularization_weight)
    second_derivative_regularization_weight = float(config.kernel_second_derivative_regularization_weight)
    equilibrium_shape_kernel = jnp.asarray(terms.memory[: config.l_max], dtype=jnp.float64)
    lag_times_ps = jnp.arange(config.l_max, dtype=jnp.float64) * effective_dt_ps(config)

    def positive_distribution(values):
        positive = jnp.maximum(jnp.asarray(values, dtype=jnp.float64), 0.0)
        return positive / jnp.maximum(jnp.sum(positive), 1.0e-12)

    equilibrium_memory_distribution = positive_distribution(equilibrium_shape_kernel)

    def run_one(params_inner, field):
        correction = transformed.apply(params_inner, None, field)
        _, vels = run_gle(
            stack,
            positions,
            velocities,
            potential,
            terms.memory,
            terms.noise_filter,
            field=field,
            steps=steps_per_loss,
            seed=seed,
            corrective_kernel=correction,
            history_velocities=history_velocities,
            config=config,
        )
        return drift_from_velocities(stack, vels, drift_window)

    def loss_fn(params_inner):
        predictions = jnp.asarray([run_one(params_inner, field) for field in fields])
        residuals = predictions - targets
        if loss_weighting == "absolute":
            weighted_residuals = residuals
        elif loss_weighting == "relative":
            weighted_residuals = residuals / jnp.maximum(jnp.abs(targets), 1.0e-6)
        else:
            raise ValueError("--loss-weighting must be 'absolute' or 'relative'.")
        drift_loss = jnp.mean(weighted_residuals**2)
        return drift_loss + kernel_regularization_loss(params_inner), predictions

    def single_field_loss_fn(params_inner, field, target):
        prediction = run_one(params_inner, field)
        residual = prediction - target
        if loss_weighting == "absolute":
            weighted_residual = residual
        elif loss_weighting == "relative":
            weighted_residual = residual / jnp.maximum(jnp.abs(target), 1.0e-6)
        else:
            raise ValueError("--loss-weighting must be 'absolute' or 'relative'.")
        return weighted_residual**2, prediction

    def kernel_shape_regularization_unweighted(params_inner):
        if shape_regularization_weight <= 0.0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        penalties = []
        for field in fields:
            correction = transformed.apply(params_inner, None, field)[: config.l_max]
            correction_distribution = positive_distribution(correction)
            penalties.append(jnp.sum((correction_distribution - equilibrium_memory_distribution) ** 2))
        return jnp.mean(jnp.asarray(penalties, dtype=jnp.float64))

    def kernel_shape_regularization_loss(params_inner):
        return shape_regularization_weight * kernel_shape_regularization_unweighted(params_inner)

    def kernel_l2_regularization_unweighted(params_inner):
        if l2_regularization_weight <= 0.0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        penalties = []
        for field in fields:
            correction = jnp.asarray(transformed.apply(params_inner, None, field)[: config.l_max], dtype=jnp.float64)
            penalties.append(jnp.mean(correction**2))
        return jnp.mean(jnp.asarray(penalties, dtype=jnp.float64))

    def kernel_time_moment_regularization_unweighted(params_inner):
        if time_moment_regularization_weight <= 0.0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        penalties = []
        for field in fields:
            correction = jnp.asarray(transformed.apply(params_inner, None, field)[: config.l_max], dtype=jnp.float64)
            penalties.append(jnp.mean((lag_times_ps**2) * (correction**2)))
        return jnp.mean(jnp.asarray(penalties, dtype=jnp.float64))

    def kernel_derivative_regularization_unweighted(params_inner):
        if derivative_regularization_weight <= 0.0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        penalties = []
        dt_ps = jnp.asarray(effective_dt_ps(config), dtype=jnp.float64)
        for field in fields:
            correction = jnp.asarray(transformed.apply(params_inner, None, field)[: config.l_max], dtype=jnp.float64)
            derivative = jnp.diff(correction) / dt_ps
            penalties.append(jnp.mean(derivative**2))
        return jnp.mean(jnp.asarray(penalties, dtype=jnp.float64))

    def kernel_second_derivative_regularization_unweighted(params_inner):
        if second_derivative_regularization_weight <= 0.0:
            return jnp.asarray(0.0, dtype=jnp.float64)
        penalties = []
        dt_ps = jnp.asarray(effective_dt_ps(config), dtype=jnp.float64)
        for field in fields:
            correction = jnp.asarray(transformed.apply(params_inner, None, field)[: config.l_max], dtype=jnp.float64)
            second_derivative = jnp.diff(correction, n=2) / (dt_ps**2)
            penalties.append(jnp.mean(second_derivative**2))
        return jnp.mean(jnp.asarray(penalties, dtype=jnp.float64))

    def kernel_regularization_loss(params_inner):
        return (
            kernel_shape_regularization_loss(params_inner)
            + l2_regularization_weight * kernel_l2_regularization_unweighted(params_inner)
            + time_moment_regularization_weight * kernel_time_moment_regularization_unweighted(params_inner)
            + derivative_regularization_weight * kernel_derivative_regularization_unweighted(params_inner)
            + second_derivative_regularization_weight
            * kernel_second_derivative_regularization_unweighted(params_inner)
        )

    def field_losses_from_predictions(predictions_inner):
        residuals = predictions_inner - targets
        if loss_weighting == "absolute":
            weighted_residuals = residuals
        elif loss_weighting == "relative":
            weighted_residuals = residuals / jnp.maximum(jnp.abs(targets), 1.0e-6)
        else:
            raise ValueError("--loss-weighting must be 'absolute' or 'relative'.")
        return weighted_residuals**2

    def tree_l2_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return jnp.asarray(0.0, dtype=jnp.float64)
        return jnp.sqrt(
            sum(jnp.sum(jnp.asarray(leaf, dtype=jnp.float64) ** 2) for leaf in leaves)
        )

    def loss_scalar_fn(params_inner):
        return loss_fn(params_inner)[0]

    def record_training_state(phase_index, loss_value, predictions_np, field_losses_np, field_grad_norms_np, grad_norm_value):
        losses.append(loss_value)
        optimizer_phase_history.append(phase_index)
        prediction_history.append(predictions_np)
        field_loss_history.append(field_losses_np)
        field_grad_norm_history.append(field_grad_norms_np)
        grad_norm_history.append(grad_norm_value)
        epoch_kernels = [
            np.asarray(transformed.apply(params, None, field)[: config.l_max], dtype=float)
            for field in fields
        ]
        kernel_history_by_field.append(np.asarray(epoch_kernels, dtype=float))

    losses = []
    optimizer_phase_history = []
    prediction_history = []
    field_loss_history = []
    field_grad_norm_history = []
    grad_norm_history = []
    kernel_history_by_field = []
    best_loss = float("inf")
    best_epoch = -1
    best_params = None
    best_predictions: np.ndarray | None = None

    def tree_to_numpy(tree):
        return jax.tree_util.tree_map(lambda value: np.asarray(value), tree)

    checkpoint_manifest: dict[str, Any] = {
        "checkpoint_stride": checkpoint_stride,
        "latest": None,
        "best": None,
        "history": [],
    }

    def write_checkpoint(
        name: str,
        global_epoch: int,
        phase: str,
        params_snapshot,
        opt_state_snapshot,
        loss_value: float,
        predictions_np: np.ndarray,
        field_losses_np: np.ndarray,
        grad_norm_value: float,
        numbered: bool = False,
    ) -> Path | None:
        if checkpoint_stride <= 0:
            return None
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"checkpoint_{name}.pkl"
        if numbered:
            path = checkpoint_dir / f"checkpoint_epoch_{global_epoch + 1:05d}_{name}.pkl"
        payload = {
            "epoch_zero_indexed": int(global_epoch),
            "epoch_one_indexed": int(global_epoch + 1),
            "phase": phase,
            "params": tree_to_numpy(params_snapshot),
            "optimizer_state": tree_to_numpy(opt_state_snapshot),
            "loss": float(loss_value),
            "predictions": np.asarray(predictions_np, dtype=float),
            "field_losses": np.asarray(field_losses_np, dtype=float),
            "grad_l2": float(grad_norm_value),
            "fields": fields_np.copy(),
            "target_drifts": targets_np.copy(),
            "mode": mode,
            "training_name": training_name or training.name,
            "optimizer": optimizer,
            "learning_rate": config.learning_rate,
            "steps_per_loss": steps_per_loss,
            "warmup_steps": warmup_steps,
            "drift_window": drift_window,
            "config": config.to_dict(),
            "params_position": "before_optimizer_update_for_recorded_loss",
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def update_checkpoint_manifest(kind: str, path: Path | None, global_epoch: int, loss_value: float) -> None:
        if path is None:
            return
        item = {
            "kind": kind,
            "path": str(path),
            "epoch_zero_indexed": int(global_epoch),
            "epoch_one_indexed": int(global_epoch + 1),
            "loss": float(loss_value),
        }
        checkpoint_manifest[kind] = item
        checkpoint_manifest["history"].append(item)
        (output_dir / "checkpoint_manifest.json").write_text(json.dumps(checkpoint_manifest, indent=2, sort_keys=True) + "\n")

    for epoch in range(epochs):
        if fields.shape[0] == 1:
            (field_loss, field_prediction), field_grads = jax.value_and_grad(single_field_loss_fn, has_aux=True)(
                params,
                fields[0],
                targets[0],
            )
            regularization_loss, regularization_grads = jax.value_and_grad(kernel_regularization_loss)(params)
            loss = field_loss + regularization_loss
            predictions = jnp.asarray([field_prediction])
            grads = jax.tree_util.tree_map(lambda left, right: left + right, field_grads, regularization_grads)
            field_losses = jnp.asarray([field_loss])
            field_grad_norms = jnp.asarray([tree_l2_norm(field_grads)])
        else:
            field_losses = []
            field_predictions = []
            field_grad_norms = []
            accumulated_grads = None
            for field, target in zip(fields, targets, strict=True):
                (field_loss, field_prediction), field_grads = jax.value_and_grad(single_field_loss_fn, has_aux=True)(
                    params,
                    field,
                    target,
                )
                field_losses.append(field_loss)
                field_predictions.append(field_prediction)
                field_grad_norms.append(tree_l2_norm(field_grads))
                if accumulated_grads is None:
                    accumulated_grads = field_grads
                else:
                    accumulated_grads = jax.tree_util.tree_map(
                        lambda left, right: left + right,
                        accumulated_grads,
                        field_grads,
                    )
            loss = jnp.mean(jnp.asarray(field_losses))
            predictions = jnp.asarray(field_predictions)
            data_grads = jax.tree_util.tree_map(lambda value: value / fields.shape[0], accumulated_grads)
            regularization_loss, regularization_grads = jax.value_and_grad(kernel_regularization_loss)(params)
            loss = loss + regularization_loss
            grads = jax.tree_util.tree_map(lambda left, right: left + right, data_grads, regularization_grads)
        loss_value = float(loss)
        predictions_np = np.asarray(predictions, dtype=float)
        field_losses_np = np.asarray(field_losses, dtype=float)
        field_grad_norms_np = np.asarray(field_grad_norms, dtype=float)
        grad_norm_value = float(tree_l2_norm(grads))
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch
            best_predictions = predictions_np.copy()
            best_params = jax.tree_util.tree_map(lambda value: value.copy(), params)
            update_checkpoint_manifest(
                "best",
                write_checkpoint("best", epoch, "adam", params, opt_state, loss_value, predictions_np, field_losses_np, grad_norm_value),
                epoch,
                loss_value,
            )
        if checkpoint_stride > 0 and ((epoch + 1) % checkpoint_stride == 0 or epoch + 1 == epochs):
            update_checkpoint_manifest(
                "latest",
                write_checkpoint(
                    "latest",
                    epoch,
                    "adam",
                    params,
                    opt_state,
                    loss_value,
                    predictions_np,
                    field_losses_np,
                    grad_norm_value,
                ),
                epoch,
                loss_value,
            )
            update_checkpoint_manifest(
                "latest_numbered",
                write_checkpoint(
                    "latest",
                    epoch,
                    "adam",
                    params,
                    opt_state,
                    loss_value,
                    predictions_np,
                    field_losses_np,
                    grad_norm_value,
                    numbered=True,
                ),
                epoch,
                loss_value,
            )
        updates, opt_state = tx.update(grads, opt_state)
        params = stack.optax.apply_updates(params, updates)
        record_training_state(
            0,
            loss_value,
            predictions_np,
            field_losses_np,
            field_grad_norms_np,
            grad_norm_value,
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"adam_epoch={epoch + 1}/{epochs} loss={losses[-1]:.8g} "
                f"predictions={prediction_history[-1].tolist()}",
                flush=True,
            )

    if optimizer == "adam-lbfgs":
        lbfgs_tx = stack.optax.lbfgs(memory_size=lbfgs_memory_size)
        lbfgs_state = lbfgs_tx.init(params)
        value_and_grad = stack.optax.value_and_grad_from_state(loss_scalar_fn)
        for lbfgs_epoch in range(lbfgs_epochs):
            loss, grads = value_and_grad(params, state=lbfgs_state)
            _, predictions = loss_fn(params)
            field_losses = field_losses_from_predictions(predictions)
            loss_value = float(loss)
            predictions_np = np.asarray(predictions, dtype=float)
            field_losses_np = np.asarray(field_losses, dtype=float)
            field_grad_norms_np = np.full(fields_np.shape, np.nan, dtype=float)
            grad_norm_value = float(tree_l2_norm(grads))
            global_epoch = len(losses)
            if loss_value < best_loss:
                best_loss = loss_value
                best_epoch = global_epoch
                best_predictions = predictions_np.copy()
                best_params = jax.tree_util.tree_map(lambda value: value.copy(), params)
                update_checkpoint_manifest(
                    "best",
                    write_checkpoint(
                        "best",
                        global_epoch,
                        "lbfgs",
                        params,
                        lbfgs_state,
                        loss_value,
                        predictions_np,
                        field_losses_np,
                        grad_norm_value,
                    ),
                    global_epoch,
                    loss_value,
                )
            if checkpoint_stride > 0 and ((lbfgs_epoch + 1) % checkpoint_stride == 0 or lbfgs_epoch + 1 == lbfgs_epochs):
                update_checkpoint_manifest(
                    "latest",
                    write_checkpoint(
                        "latest",
                        global_epoch,
                        "lbfgs",
                        params,
                        lbfgs_state,
                        loss_value,
                        predictions_np,
                        field_losses_np,
                        grad_norm_value,
                    ),
                    global_epoch,
                    loss_value,
                )
                update_checkpoint_manifest(
                    "latest_numbered",
                    write_checkpoint(
                        "latest",
                        global_epoch,
                        "lbfgs",
                        params,
                        lbfgs_state,
                        loss_value,
                        predictions_np,
                        field_losses_np,
                        grad_norm_value,
                        numbered=True,
                    ),
                    global_epoch,
                    loss_value,
                )
            updates, lbfgs_state = lbfgs_tx.update(
                grads,
                lbfgs_state,
                params,
                value=loss,
                grad=grads,
                value_fn=loss_scalar_fn,
            )
            params = stack.optax.apply_updates(params, updates)
            record_training_state(
                1,
                loss_value,
                predictions_np,
                field_losses_np,
                field_grad_norms_np,
                grad_norm_value,
            )
            if lbfgs_epoch == 0 or (lbfgs_epoch + 1) % 5 == 0 or lbfgs_epoch + 1 == lbfgs_epochs:
                print(
                    f"lbfgs_epoch={lbfgs_epoch + 1}/{lbfgs_epochs} loss={losses[-1]:.8g} "
                    f"predictions={prediction_history[-1].tolist()} grad_l2={grad_norm_history[-1]:.8g}",
                    flush=True,
                )

    if best_params is None or best_predictions is None:
        best_params = params
        best_epoch = len(losses) - 1
        best_loss = losses[-1] if losses else float("nan")
        best_predictions = prediction_history[-1] if prediction_history else np.asarray([], dtype=float)

    final_eval_predictions = []
    for field in eval_fields_jax:
        correction = transformed.apply(params, None, field)
        traj, vels = run_gle(
            stack,
            positions,
            velocities,
            potential,
            terms.memory,
            terms.noise_filter,
            field=field,
            steps=steps_per_loss,
            seed=seed + 20_000,
            corrective_kernel=correction,
            history_velocities=history_velocities,
            config=config,
        )
        final_eval_predictions.append(float(drift_from_velocities(stack, vels, drift_window)))
        diagnostics.append(stability_diagnostics(traj, vels, f"final_eval_field_{float(field):.6g}", config))

    best_eval_predictions = []
    for field in eval_fields_jax:
        correction = transformed.apply(best_params, None, field)
        traj, vels = run_gle(
            stack,
            positions,
            velocities,
            potential,
            terms.memory,
            terms.noise_filter,
            field=field,
            steps=steps_per_loss,
            seed=seed + 30_000,
            corrective_kernel=correction,
            history_velocities=history_velocities,
            config=config,
        )
        best_eval_predictions.append(float(drift_from_velocities(stack, vels, drift_window)))
        diagnostics.append(stability_diagnostics(traj, vels, f"best_eval_field_{float(field):.6g}", config))

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "training_loss.csv", np.column_stack([np.arange(len(losses)), losses]), delimiter=",", header="epoch,loss", comments="")
    np.savetxt(
        output_dir / "drift_predictions.csv",
        np.column_stack([np.arange(len(prediction_history)), np.asarray(prediction_history)]),
        delimiter=",",
        header="epoch," + ",".join(f"E_{float(field):.5g}" for field in fields_np),
        comments="",
    )
    field_loss_history_array = np.asarray(field_loss_history, dtype=float)
    field_grad_norm_history_array = np.asarray(field_grad_norm_history, dtype=float)
    np.savetxt(
        output_dir / "per_field_training_diagnostics.csv",
        np.column_stack(
            [
                np.arange(len(losses)),
                np.asarray(optimizer_phase_history, dtype=float),
                np.asarray(losses, dtype=float),
                np.asarray(grad_norm_history, dtype=float),
                np.asarray(prediction_history, dtype=float),
                field_loss_history_array,
                field_grad_norm_history_array,
            ]
        ),
        delimiter=",",
        header=(
            "epoch,phase,total_loss,total_grad_l2,"
            + ",".join(f"prediction_E_{float(field):.5g}" for field in fields_np)
            + ","
            + ",".join(f"loss_E_{float(field):.5g}" for field in fields_np)
            + ","
            + ",".join(f"grad_l2_E_{float(field):.5g}" for field in fields_np)
        ),
        comments="",
    )
    kernel_history_by_field_array = np.asarray(kernel_history_by_field, dtype=float)
    kernel_history = (
        kernel_history_by_field_array[:, -1, :]
        if kernel_history_by_field_array.size
        else np.empty((0, config.l_max), dtype=float)
    )
    best_kernel_by_field = np.asarray(
        [np.asarray(transformed.apply(best_params, None, field)[: config.l_max], dtype=float) for field in fields],
        dtype=float,
    )
    final_shape_regularization_unweighted = float(kernel_shape_regularization_unweighted(params))
    best_shape_regularization_unweighted = float(kernel_shape_regularization_unweighted(best_params))
    final_l2_regularization_unweighted = float(kernel_l2_regularization_unweighted(params))
    best_l2_regularization_unweighted = float(kernel_l2_regularization_unweighted(best_params))
    final_time_moment_regularization_unweighted = float(kernel_time_moment_regularization_unweighted(params))
    best_time_moment_regularization_unweighted = float(kernel_time_moment_regularization_unweighted(best_params))
    final_derivative_regularization_unweighted = float(kernel_derivative_regularization_unweighted(params))
    best_derivative_regularization_unweighted = float(kernel_derivative_regularization_unweighted(best_params))
    final_second_derivative_regularization_unweighted = float(kernel_second_derivative_regularization_unweighted(params))
    best_second_derivative_regularization_unweighted = float(kernel_second_derivative_regularization_unweighted(best_params))
    final_kernel_by_field = (
        kernel_history_by_field_array[-1]
        if kernel_history_by_field_array.size
        else np.empty((fields_np.shape[0], config.l_max), dtype=float)
    )
    final_kernel_shape_diagnostics = kernel_shape_diagnostics(
        final_kernel_by_field,
        fields_np,
        config,
        label="final",
    )
    best_kernel_shape_diagnostics = kernel_shape_diagnostics(
        best_kernel_by_field,
        fields_np,
        config,
        label="best",
    )
    kernel_shape_diagnostic_items = final_kernel_shape_diagnostics + best_kernel_shape_diagnostics
    kernel_shape_warning_items = kernel_shape_warnings(kernel_shape_diagnostic_items)
    np.savez_compressed(
        output_dir / "corrective_kernel_history.npz",
        epochs=np.arange(len(kernel_history)),
        kernels=kernel_history,
        kernels_by_field=kernel_history_by_field_array,
        fields=fields_np,
        target_drifts=targets_np,
        kernel_history_field=np.asarray([fields_np[-1]], dtype=float),
        kernel_history_fields=fields_np,
        tau_internal=np.arange(config.l_max, dtype=float) * effective_dt(config),
        tau_ps=np.arange(config.l_max, dtype=float) * effective_dt_ps(config),
        original_kernel=np.asarray(terms.memory, dtype=float)[: config.l_max],
        original_kernel_per_ps2=memory_internal_to_per_ps2(np.asarray(terms.memory, dtype=float)[: config.l_max]),
        kernel_model=np.asarray([config.kernel_model]),
        kernel_field_scale=np.asarray([config.kernel_field_scale], dtype=float),
        kernel_tau_prior_ps=np.asarray([config.kernel_tau_prior_ps], dtype=float),
        kernel_exp_basis_taus_ps=np.asarray(config.kernel_exp_basis_taus_ps, dtype=float),
        kernel_shape_regularization_weight=np.asarray([config.kernel_shape_regularization_weight], dtype=float),
        kernel_l2_regularization_weight=np.asarray([config.kernel_l2_regularization_weight], dtype=float),
        kernel_time_moment_regularization_weight=np.asarray([config.kernel_time_moment_regularization_weight], dtype=float),
        kernel_derivative_regularization_weight=np.asarray([config.kernel_derivative_regularization_weight], dtype=float),
        kernel_second_derivative_regularization_weight=np.asarray(
            [config.kernel_second_derivative_regularization_weight], dtype=float
        ),
        best_shape_regularization_unweighted=np.asarray([best_shape_regularization_unweighted], dtype=float),
        final_shape_regularization_unweighted=np.asarray([final_shape_regularization_unweighted], dtype=float),
        best_l2_regularization_unweighted=np.asarray([best_l2_regularization_unweighted], dtype=float),
        final_l2_regularization_unweighted=np.asarray([final_l2_regularization_unweighted], dtype=float),
        best_time_moment_regularization_unweighted=np.asarray([best_time_moment_regularization_unweighted], dtype=float),
        final_time_moment_regularization_unweighted=np.asarray([final_time_moment_regularization_unweighted], dtype=float),
        best_derivative_regularization_unweighted=np.asarray([best_derivative_regularization_unweighted], dtype=float),
        final_derivative_regularization_unweighted=np.asarray([final_derivative_regularization_unweighted], dtype=float),
        best_second_derivative_regularization_unweighted=np.asarray(
            [best_second_derivative_regularization_unweighted], dtype=float
        ),
        final_second_derivative_regularization_unweighted=np.asarray(
            [final_second_derivative_regularization_unweighted], dtype=float
        ),
        memory_history_scaling=np.asarray([config.memory_history_scaling]),
        learning_rate=np.asarray([config.learning_rate], dtype=float),
        optimizer=np.asarray([optimizer]),
        optimizer_phase=np.asarray(optimizer_phase_history, dtype=int),
        lbfgs_epochs=np.asarray([lbfgs_epochs], dtype=int),
        lbfgs_memory_size=np.asarray([lbfgs_memory_size], dtype=int),
        per_field_losses=field_loss_history_array,
        per_field_grad_l2=field_grad_norm_history_array,
        total_grad_l2=np.asarray(grad_norm_history, dtype=float),
        best_epoch=np.asarray([best_epoch], dtype=int),
        best_loss=np.asarray([best_loss], dtype=float),
        best_training_predictions=np.asarray(best_predictions, dtype=float),
        best_kernels_by_field=best_kernel_by_field,
        final_kernels_by_field=final_kernel_by_field,
        best_kernels_by_field_per_ps2=memory_internal_to_per_ps2(best_kernel_by_field),
        final_kernels_by_field_per_ps2=memory_internal_to_per_ps2(final_kernel_by_field),
        kernels_per_ps2=memory_internal_to_per_ps2(kernel_history),
        kernels_by_field_per_ps2=memory_internal_to_per_ps2(kernel_history_by_field_array),
    )
    correction_table_path, total_table_path = write_kernel_evolution_tables(
        output_dir,
        np.asarray(terms.memory, dtype=float),
        np.asarray(kernel_history, dtype=float),
        np.arange(len(kernel_history)),
        config,
    )
    correction_tables_by_field: dict[str, str] = {}
    total_tables_by_field: dict[str, str] = {}
    for field_index, field in enumerate(fields_np):
        correction_field_path, total_field_path = write_kernel_evolution_tables(
            output_dir,
            np.asarray(terms.memory, dtype=float),
            kernel_history_by_field_array[:, field_index, :],
            np.arange(len(kernel_history)),
            config,
            suffix=f"_{field_slug(float(field))}",
        )
        correction_tables_by_field[f"{float(field):.12g}"] = str(correction_field_path)
        total_tables_by_field[f"{float(field):.12g}"] = str(total_field_path)
    eval_mobility_path = write_mobility_curve(
        output_dir / "GLENECK_eval_mobility.csv",
        eval_fields_np,
        np.asarray(final_eval_predictions, dtype=float),
    )
    best_eval_mobility_path = write_mobility_curve(
        output_dir / "GLENECK_best_eval_mobility.csv",
        eval_fields_np,
        np.asarray(best_eval_predictions, dtype=float),
    )
    best_kernel_snapshot_outputs: dict[str, str] = {}
    for field_index, field in enumerate(fields_np):
        best_snapshot_path = write_kernel_snapshot_table(
            output_dir,
            np.asarray(terms.memory, dtype=float),
            best_kernel_by_field[field_index],
            config,
            suffix=f"_best_epoch_{best_epoch + 1}_{field_slug(float(field))}",
        )
        best_kernel_snapshot_outputs[f"{float(field):.12g}"] = str(best_snapshot_path)
    diagnostics_path = output_dir / "GLENECK_stability_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n")
    warnings = diagnostic_warnings(diagnostics)
    report = {
        "status": "unstable" if warnings else "completed",
        "elapsed_seconds": time.time() - started,
        "mode": mode,
        "training_name": training_name or training.name,
        "state_source": state_source,
        "warmup_steps": warmup_steps,
        "dt": config.dt,
        "effective_dt": effective_dt(config),
        "effective_dt_internal": effective_dt(config),
        "effective_dt_ps": effective_dt_ps(config),
        "l_max": config.l_max,
        "memory_orig_interval": config.memory_orig_interval,
        "memory_kernel_unit": "internal_time^-2",
        "exported_kernel_unit": "ps^-2",
        "kernel_model": config.kernel_model,
        "poly_degree": config.poly_degree,
        "kernel_field_scale": config.kernel_field_scale,
        "kernel_tau_prior_ps": config.kernel_tau_prior_ps,
        "kernel_exp_basis_taus_ps": list(config.kernel_exp_basis_taus_ps),
        "kernel_shape_regularization_weight": config.kernel_shape_regularization_weight,
        "kernel_l2_regularization_weight": config.kernel_l2_regularization_weight,
        "kernel_time_moment_regularization_weight": config.kernel_time_moment_regularization_weight,
        "kernel_derivative_regularization_weight": config.kernel_derivative_regularization_weight,
        "kernel_second_derivative_regularization_weight": config.kernel_second_derivative_regularization_weight,
        "best_shape_regularization_unweighted": best_shape_regularization_unweighted,
        "best_shape_regularization_weighted": config.kernel_shape_regularization_weight * best_shape_regularization_unweighted,
        "final_shape_regularization_unweighted": final_shape_regularization_unweighted,
        "final_shape_regularization_weighted": config.kernel_shape_regularization_weight * final_shape_regularization_unweighted,
        "best_l2_regularization_unweighted": best_l2_regularization_unweighted,
        "best_l2_regularization_weighted": config.kernel_l2_regularization_weight * best_l2_regularization_unweighted,
        "final_l2_regularization_unweighted": final_l2_regularization_unweighted,
        "final_l2_regularization_weighted": config.kernel_l2_regularization_weight * final_l2_regularization_unweighted,
        "best_time_moment_regularization_unweighted": best_time_moment_regularization_unweighted,
        "best_time_moment_regularization_weighted": (
            config.kernel_time_moment_regularization_weight * best_time_moment_regularization_unweighted
        ),
        "final_time_moment_regularization_unweighted": final_time_moment_regularization_unweighted,
        "final_time_moment_regularization_weighted": (
            config.kernel_time_moment_regularization_weight * final_time_moment_regularization_unweighted
        ),
        "best_derivative_regularization_unweighted": best_derivative_regularization_unweighted,
        "best_derivative_regularization_weighted": (
            config.kernel_derivative_regularization_weight * best_derivative_regularization_unweighted
        ),
        "final_derivative_regularization_unweighted": final_derivative_regularization_unweighted,
        "final_derivative_regularization_weighted": (
            config.kernel_derivative_regularization_weight * final_derivative_regularization_unweighted
        ),
        "best_second_derivative_regularization_unweighted": best_second_derivative_regularization_unweighted,
        "best_second_derivative_regularization_weighted": (
            config.kernel_second_derivative_regularization_weight
            * best_second_derivative_regularization_unweighted
        ),
        "final_second_derivative_regularization_unweighted": final_second_derivative_regularization_unweighted,
        "final_second_derivative_regularization_weighted": (
            config.kernel_second_derivative_regularization_weight
            * final_second_derivative_regularization_unweighted
        ),
        "memory_history_scaling": config.memory_history_scaling,
        "learning_rate": config.learning_rate,
        "kernel_history_field": float(fields_np[-1]),
        "init_velocity_scale": config.init_velocity_scale,
        "noise_scale": config.noise_scale,
        "temperature": config.temperature,
        "effective_temperature": effective_temperature(config),
        "config": config.to_dict(),
        "potential_path": str(potential_path),
        "memory_path": str(memory_path),
        "epochs": epochs,
        "total_optimizer_iterations": len(losses),
        "optimizer": optimizer,
        "adam_epochs": epochs,
        "lbfgs_epochs": lbfgs_epochs,
        "lbfgs_memory_size": lbfgs_memory_size,
        "checkpoint_stride": checkpoint_stride,
        "checkpoint_manifest": str(output_dir / "checkpoint_manifest.json") if checkpoint_stride > 0 else None,
        "checkpoint_latest": checkpoint_manifest["latest"]["path"] if checkpoint_manifest["latest"] else None,
        "checkpoint_best": checkpoint_manifest["best"]["path"] if checkpoint_manifest["best"] else None,
        "nominal_epochs": training.epochs,
        "loss_weighting": loss_weighting,
        "steps_per_loss": steps_per_loss,
        "drift_window": drift_window,
        "fields": fields_np.tolist(),
        "eval_fields": eval_fields_np.tolist(),
        "target_drifts": targets_np.tolist(),
        "final_loss": losses[-1] if losses else None,
        "final_predictions": prediction_history[-1].tolist() if prediction_history else [],
        "final_eval_predictions": final_eval_predictions,
        "best_epoch": best_epoch,
        "best_epoch_one_indexed": best_epoch + 1,
        "best_loss": best_loss,
        "best_predictions": best_predictions.tolist(),
        "best_eval_predictions": best_eval_predictions,
        "eval_mobility_output": str(eval_mobility_path),
        "best_eval_mobility_output": str(best_eval_mobility_path),
        "per_field_diagnostics_output": str(output_dir / "per_field_training_diagnostics.csv"),
        "final_field_losses": field_loss_history[-1].tolist() if field_loss_history else [],
        "final_field_grad_l2": field_grad_norm_history[-1].tolist() if field_grad_norm_history else [],
        "final_total_grad_l2": grad_norm_history[-1] if grad_norm_history else None,
        "corrective_kernel_evolution_output": str(correction_table_path),
        "total_kernel_evolution_output": str(total_table_path),
        "corrective_kernel_evolution_outputs_by_field": correction_tables_by_field,
        "total_kernel_evolution_outputs_by_field": total_tables_by_field,
        "best_kernel_snapshot_outputs_by_field": best_kernel_snapshot_outputs,
        "diagnostics_output": str(diagnostics_path),
        "diagnostic_warnings": warnings,
        "kernel_shape_diagnostics": kernel_shape_diagnostic_items,
        "kernel_shape_warnings": kernel_shape_warning_items,
    }
    (output_dir / "GLENECK_training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
