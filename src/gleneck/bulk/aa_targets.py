from __future__ import annotations

import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import BulkAATargetConfig, DEFAULT_BULK_AA_TARGET
from .memory import gaussian_smooth
from .neck import require_jax_stack
from .units import effective_dt, effective_field, effective_mass, effective_temperature, internal_time_to_ps


def parse_fields(text: str | None, default: tuple[float, ...] = DEFAULT_BULK_AA_TARGET.default_fields) -> tuple[float, ...]:
    """Parse a comma-separated field list."""
    if text is None or text.strip() == "":
        return default
    fields = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not fields:
        raise ValueError("At least one field is required.")
    return fields


def config_from_args(args: Any) -> BulkAATargetConfig:
    """Build an AA target config from argparse-like attributes."""
    return replace(
        DEFAULT_BULK_AA_TARGET,
        n_solvent=args.n_solvent,
        n_a=args.n_a,
        n_b=args.n_b,
        box_size=tuple(args.box_size),
        temperature=args.temperature,
        dt=args.dt,
        gamma=args.gamma,
        thermostat=args.thermostat,
        langevin_mode=getattr(args, "langevin_mode", DEFAULT_BULK_AA_TARGET.langevin_mode),
        force_mode=args.force_mode,
        drift_reference=args.drift_reference,
        r_cutoff=args.r_cutoff,
        dr_threshold=args.dr_threshold,
        rdf_cutoff=args.rdf_cutoff,
        rdf_dr=args.rdf_dr,
        seed=args.seed,
    )


def make_initial_positions(config: BulkAATargetConfig) -> np.ndarray:
    """Create non-overlapping mixed initial positions on a randomized grid."""
    rng = np.random.default_rng(config.seed)
    box = np.asarray(config.box_size, dtype=float)
    volume_per_particle = np.prod(box) / config.n_total
    spacing = volume_per_particle ** (1.0 / 3.0)
    grid_shape = np.maximum(np.floor(box / spacing).astype(int), 1)

    while int(np.prod(grid_shape)) < config.n_total:
        dim = int(np.argmax(box / grid_shape))
        grid_shape[dim] += 1

    axes = [
        np.linspace(0.0, box[dim], grid_shape[dim], endpoint=False) + box[dim] / (2.0 * grid_shape[dim])
        for dim in range(3)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    positions = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    rng.shuffle(positions)
    positions = positions[: config.n_total]

    jitter_scale = min(0.05 * spacing, 0.1)
    positions += rng.normal(0.0, jitter_scale, size=positions.shape)
    positions %= box
    return positions.astype(np.float64)


def make_species(config: BulkAATargetConfig) -> np.ndarray:
    """Return species IDs in the legacy-compatible order W, A, B."""
    return np.concatenate(
        [
            np.zeros(config.n_solvent, dtype=np.int32),
            np.ones(config.n_a, dtype=np.int32),
            2 * np.ones(config.n_b, dtype=np.int32),
        ]
    )


def retained_indices(config: BulkAATargetConfig) -> np.ndarray:
    return np.arange(config.n_solvent, config.n_total, dtype=np.int32)


def retained_species(config: BulkAATargetConfig) -> np.ndarray:
    return np.concatenate([np.zeros(config.n_a, dtype=np.int32), np.ones(config.n_b, dtype=np.int32)])


def normalize_force_mode(force_mode: str) -> str:
    normalized = force_mode.strip().lower().replace("_", "-")
    aliases = {
        "solute": "solute",
        "solute-only": "solute",
        "retained": "solute",
        "retained-only": "solute",
        "ions": "solute",
        "ion-only": "solute",
        "all": "all-mobile",
        "all-mobile": "all-mobile",
        "liquid": "all-mobile",
        "mobile": "all-mobile",
        "solute-counter-solvent": "solute-counter-solvent",
        "counter-solvent": "solute-counter-solvent",
        "zero-net": "solute-counter-solvent",
        "zero-net-force": "solute-counter-solvent",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown AA force mode {force_mode!r}; expected 'solute', 'all-mobile', or 'solute-counter-solvent'.")
    return aliases[normalized]


def normalize_drift_reference(reference: str) -> str:
    normalized = reference.strip().lower().replace("_", "-")
    aliases = {
        "lab": "lab",
        "absolute": "lab",
        "solute": "lab",
        "solvent": "solvent",
        "relative": "solvent",
        "relative-solvent": "solvent",
        "solvent-relative": "solvent",
        "bath": "solvent",
        "all": "all-com",
        "all-com": "all-com",
        "com": "all-com",
        "center-of-mass": "all-com",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown drift reference {reference!r}; expected 'lab', 'solvent', or 'all-com'.")
    return aliases[normalized]


def normalize_langevin_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("_", "-")
    aliases = {
        "all": "all",
        "all-components": "all",
        "solvent": "solvent",
        "solvent-only": "solvent",
        "solvent-yz": "solvent-yz",
        "bath-yz": "solvent-yz",
        "transverse-solvent": "solvent-yz",
        "all-yz": "all-yz",
        "yz": "all-yz",
        "transverse": "all-yz",
        "none": "none",
        "off": "none",
    }
    if normalized not in aliases:
        raise ValueError(f"Unknown Langevin mode {mode!r}; expected 'all', 'solvent', 'solvent-yz', 'all-yz', or 'none'.")
    return aliases[normalized]


def langevin_thermostat_mask(species: np.ndarray, mode: str) -> np.ndarray:
    """Return component mask for Langevin thermostatting.

    The production NEMD default is solvent-yz: thermostat only bath
    degrees of freedom transverse to the driven x direction.
    """
    species_array = np.asarray(species)
    normalized = normalize_langevin_mode(mode)
    mask = np.zeros((species_array.size, 3), dtype=float)
    if normalized == "none":
        return mask
    if normalized == "all":
        mask[:, :] = 1.0
    elif normalized == "solvent":
        mask[species_array == 0, :] = 1.0
    elif normalized == "all-yz":
        mask[:, 1:] = 1.0
    elif normalized == "solvent-yz":
        mask[species_array == 0, 1:] = 1.0
    return mask


def group_peculiar_velocity_np(velocity: np.ndarray, mass: np.ndarray, species: np.ndarray) -> np.ndarray:
    """Subtract mass-weighted species/group streaming velocity from each particle."""
    velocity_array = np.asarray(velocity, dtype=float)
    mass_array = np.asarray(mass, dtype=float).reshape((-1, 1))
    species_array = np.asarray(species, dtype=np.int32)
    peculiar = np.empty_like(velocity_array)
    for group in np.unique(species_array):
        mask = species_array == group
        group_mass = np.sum(mass_array[mask])
        if group_mass <= 0:
            peculiar[mask] = velocity_array[mask]
            continue
        group_velocity = np.sum(mass_array[mask] * velocity_array[mask], axis=0, keepdims=True) / group_mass
        peculiar[mask] = velocity_array[mask] - group_velocity
    return peculiar


def kinetic_temperature_np(velocity: np.ndarray, mass: np.ndarray) -> float:
    velocity_array = np.asarray(velocity, dtype=float)
    if velocity_array.shape[0] == 0:
        return float("nan")
    mass_array = np.asarray(mass, dtype=float)
    return float(np.sum(mass_array * velocity_array**2) / (3.0 * velocity_array.shape[0]))


def external_force_weights(species: np.ndarray, force_mode: str) -> np.ndarray:
    """Return per-particle external-force weights for AA transport tests."""
    mode = normalize_force_mode(force_mode)
    species_array = np.asarray(species)
    if mode == "solute":
        return (species_array > 0).astype(float)
    if mode == "solute-counter-solvent":
        solute = species_array > 0
        solvent = ~solute
        weights = solute.astype(float)
        if np.any(solvent):
            weights[solvent] = -float(np.sum(solute)) / float(np.sum(solvent))
        return weights
    return np.ones_like(species_array, dtype=float)


def external_force_mask(species: np.ndarray, force_mode: str) -> np.ndarray:
    """Return the particles directly acted on by the external-force protocol."""
    return (np.abs(external_force_weights(species, force_mode)) > 0).astype(float)


def force_mode_counts(config: BulkAATargetConfig, force_mode: str | None = None) -> dict[str, int]:
    species = make_species(config)
    mask = external_force_mask(species, force_mode or config.force_mode)
    solvent = species == 0
    solute = species > 0
    return {
        "total_particles": int(species.size),
        "forced_particles": int(np.sum(mask > 0)),
        "forced_solvent_particles": int(np.sum((mask > 0) & solvent)),
        "forced_solute_particles": int(np.sum((mask > 0) & solute)),
        "solvent_particles": int(np.sum(solvent)),
        "solute_particles": int(np.sum(solute)),
        "net_external_force_weight": float(np.sum(external_force_weights(species, force_mode or config.force_mode))),
    }


def _minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.round(delta / box)


def _pair_distances(positions: np.ndarray, indices_i: np.ndarray, indices_j: np.ndarray, box: np.ndarray, same: bool) -> np.ndarray:
    pos_i = positions[indices_i]
    pos_j = positions[indices_j]
    if same:
        if len(indices_i) < 2:
            return np.empty(0, dtype=float)
        delta = pos_i[:, None, :] - pos_i[None, :, :]
        delta = _minimum_image(delta, box)
        tri = np.triu_indices(len(indices_i), k=1)
        return np.linalg.norm(delta[tri], axis=1)
    delta = pos_i[:, None, :] - pos_j[None, :, :]
    delta = _minimum_image(delta, box)
    return np.linalg.norm(delta.reshape(-1, 3), axis=1)


def compute_retained_rdfs(
    retained_positions: np.ndarray,
    config: BulkAATargetConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute A-A, B-B, and A-B RDFs for retained solutes."""
    centers, histograms, n_frames = accumulate_retained_rdf_histograms(retained_positions, config)
    return centers, normalize_retained_rdf_histograms(histograms, n_frames, config)


def retained_rdf_grid(config: BulkAATargetConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return RDF bin edges and centers for the retained solutes."""
    edges = np.arange(0.0, config.rdf_cutoff + config.rdf_dr, config.rdf_dr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def empty_retained_rdf_histograms(config: BulkAATargetConfig) -> dict[str, np.ndarray]:
    """Create empty A-A, B-B, and A-B RDF histograms."""
    _, centers = retained_rdf_grid(config)
    return {
        "g_r_AA": np.zeros_like(centers),
        "g_r_BB": np.zeros_like(centers),
        "g_r_AB": np.zeros_like(centers),
    }


def accumulate_retained_rdf_histograms(
    retained_positions: np.ndarray,
    config: BulkAATargetConfig,
    histograms: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """Accumulate retained-solute RDF pair histograms without storing all samples."""
    box = np.asarray(config.box_size, dtype=float)
    edges, centers = retained_rdf_grid(config)
    species = retained_species(config)
    a_idx = np.where(species == 0)[0]
    b_idx = np.where(species == 1)[0]
    positions = np.asarray(retained_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("retained_positions must have shape (T, N, 3).")
    n_frames = int(positions.shape[0])
    if histograms is None:
        histograms = empty_retained_rdf_histograms(config)
    for frame in positions:
        for key, i_idx, j_idx, same in (
            ("g_r_AA", a_idx, a_idx, True),
            ("g_r_BB", b_idx, b_idx, True),
            ("g_r_AB", a_idx, b_idx, False),
        ):
            distances = _pair_distances(frame, i_idx, j_idx, box, same=same)
            histograms[key] += np.histogram(distances, bins=edges)[0]
    return centers, histograms, n_frames


def normalize_retained_rdf_histograms(
    histograms: dict[str, np.ndarray],
    n_frames: int,
    config: BulkAATargetConfig,
) -> dict[str, np.ndarray]:
    """Normalize retained-solute RDF histograms into g(r)."""
    if n_frames <= 0:
        raise ValueError("n_frames must be positive.")
    edges, centers = retained_rdf_grid(config)
    shell_volumes = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(np.prod(np.asarray(config.box_size, dtype=float)))

    rho_a = config.n_a / volume
    rho_b = config.n_b / volume
    normalizers = {
        "g_r_AA": n_frames * config.n_a * rho_a * shell_volumes / 2.0,
        "g_r_BB": n_frames * config.n_b * rho_b * shell_volumes / 2.0,
        "g_r_AB": n_frames * config.n_a * rho_b * shell_volumes,
    }
    return {
        key: np.divide(
            np.asarray(histograms[key], dtype=float),
            normalizers[key],
            out=np.zeros_like(centers),
            where=normalizers[key] > 0,
        )
        for key in ("g_r_AA", "g_r_BB", "g_r_AB")
    }


def compute_same_species_rdf(
    sampled_positions: np.ndarray,
    box_size: tuple[float, float, float],
    rdf_cutoff: float,
    rdf_dr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a same-species RDF from sampled positions using shell normalization."""
    positions = np.asarray(sampled_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("sampled_positions must have shape (T, N, 3).")
    box = np.asarray(box_size, dtype=float)
    edges = np.arange(0.0, rdf_cutoff + rdf_dr, rdf_dr)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_volumes = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(np.prod(box))
    n_frames, n_particles = positions.shape[:2]
    hist = np.zeros_like(centers)
    indices = np.arange(n_particles)
    for frame in positions:
        distances = _pair_distances(frame, indices, indices, box, same=True)
        hist += np.histogram(distances, bins=edges)[0]
    density = n_particles / volume
    normalizer = n_frames * n_particles * density * shell_volumes / 2.0
    rdf = np.divide(hist, normalizer, out=np.zeros_like(hist), where=normalizer > 0)
    return centers, rdf


def smooth_and_tail_normalize_rdfs(
    rdfs: dict[str, np.ndarray],
    sigma: float = 2.0,
    tail_start: int = 80,
    tail_end: int = 100,
) -> dict[str, np.ndarray]:
    """Smooth RDF curves and normalize their long-range tail to one."""
    smoothed: dict[str, np.ndarray] = {}
    for key, values in rdfs.items():
        curve = gaussian_smooth(np.asarray(values, dtype=float), sigma)
        start = max(0, min(int(tail_start), curve.size - 1))
        end = max(start + 1, min(int(tail_end), curve.size))
        tail = curve[start:end]
        finite_tail = tail[np.isfinite(tail)]
        scale = float(np.mean(finite_tail)) if finite_tail.size else 1.0
        if np.isfinite(scale) and scale > 0.0:
            curve = curve / scale
        smoothed[key] = curve
    return smoothed


def write_full_rdf_csv(path: Path, centers: np.ndarray, rdfs: dict[str, np.ndarray]) -> Path:
    """Write solute and optional solvent RDF columns with a stable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["r_distance", "g_r_AA", "g_r_BB", "g_r_AB", "g_r_WW"]
    table_columns = [np.asarray(centers, dtype=float)]
    for key in columns[1:]:
        if key in rdfs:
            table_columns.append(np.asarray(rdfs[key], dtype=float))
        else:
            table_columns.append(np.full_like(centers, np.nan, dtype=float))
    np.savetxt(path, np.column_stack(table_columns), delimiter=",", header=",".join(columns), comments="")
    return path


def compute_vacf(velocities: np.ndarray, dt: float, window: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Compute drift-subtracted normalized VACF for retained solute velocities."""
    if velocities.ndim != 3 or velocities.shape[-1] != 3:
        raise ValueError("velocities must have shape (T, N, 3).")
    centered = velocities - np.mean(velocities, axis=(0, 1), keepdims=True)
    n_frames = centered.shape[0]
    if window is None:
        window = min(1000, n_frames)
    window = max(1, min(window, n_frames))
    vacf = np.empty(window, dtype=float)
    for tau in range(window):
        products = centered[: n_frames - tau] * centered[tau:]
        vacf[tau] = float(np.mean(np.sum(products, axis=-1)))
    if vacf[0] != 0:
        vacf = vacf / vacf[0]
    return np.arange(window, dtype=float) * dt, vacf


def write_rdf_csv(path: Path, centers: np.ndarray, rdfs: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = np.column_stack([centers, rdfs["g_r_AA"], rdfs["g_r_BB"], rdfs["g_r_AB"]])
    np.savetxt(path, table, delimiter=",", header="r_distance,g_r_AA,g_r_BB,g_r_AB", comments="")
    return path


def write_vacf_csv(path: Path, time_lag: np.ndarray, vacf: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_lag = np.asarray(time_lag, dtype=float)
    np.savetxt(
        path,
        np.column_stack([time_lag, time_lag, internal_time_to_ps(time_lag), vacf]),
        delimiter=",",
        header="time_lag,time_lag_internal,time_lag_ps,vacf_solute_norm",
        comments="",
    )
    return path


def write_drift_trace_csv(path: Path, trace: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample_index",
        "time",
        "solute_vx_mean",
        "solvent_vx_mean",
        "all_vx_mean",
        "all_com_vx_mean",
        "solute_relative_solvent_vx_mean",
        "solute_relative_all_com_vx_mean",
        "solute_temperature",
        "all_temperature",
        "solvent_peculiar_temperature",
        "solute_peculiar_temperature",
        "all_peculiar_temperature",
    ]
    default = np.full_like(trace["sample_index"], np.nan, dtype=float)
    table_columns = [trace[key] if key in trace else default for key in columns]
    table = np.column_stack(table_columns)
    np.savetxt(path, table, delimiter=",", header=",".join(columns), comments="")
    return path


def drift_observable(trace: dict[str, np.ndarray], reference: str = "solvent") -> np.ndarray:
    """Return the solute drift observable in the requested reference frame."""
    mode = normalize_drift_reference(reference)
    solute_vx = np.asarray(trace["solute_vx_mean"], dtype=float)
    if mode == "lab":
        return solute_vx
    if mode == "solvent":
        if "solute_relative_solvent_vx_mean" in trace:
            return np.asarray(trace["solute_relative_solvent_vx_mean"], dtype=float)
        return solute_vx - np.asarray(trace["solvent_vx_mean"], dtype=float)
    if "solute_relative_all_com_vx_mean" in trace:
        return np.asarray(trace["solute_relative_all_com_vx_mean"], dtype=float)
    return solute_vx - np.asarray(trace["all_vx_mean"], dtype=float)


def summarize_drift_trace(
    trace: dict[str, np.ndarray],
    burn: int,
    reference: str = "solvent",
    *,
    absolute_tolerance: float = 0.002,
    relative_tolerance: float = 0.05,
    window_fraction: float = 0.2,
    max_all_temperature: float | None = None,
) -> dict[str, float | str | list[str]]:
    drift = drift_observable(trace, reference)
    burn = int(max(0, min(len(drift) - 1, burn)))
    steady = drift[burn:]
    window = max(1, int(round(len(drift) * window_fraction)))
    tail = drift[-window:]
    previous = drift[-2 * window : -window] if len(drift) >= 2 * window else drift[:window]
    tail_mean = float(np.mean(tail))
    previous_mean = float(np.mean(previous))
    tail_delta = tail_mean - previous_mean
    tolerance = max(absolute_tolerance, relative_tolerance * abs(tail_mean))
    warnings: list[str] = []
    if abs(tail_delta) > tolerance:
        warnings.append("tail_mean_shift_exceeds_tolerance")
    if not np.all(np.isfinite(drift)):
        warnings.append("nonfinite_drift_trace")
    if max_all_temperature is not None and "all_temperature" in trace:
        all_temperature = np.asarray(trace["all_temperature"], dtype=float)
        if np.any(all_temperature[-window:] > max_all_temperature):
            warnings.append("all_temperature_guard_exceeded")
    return {
        "burn_samples": burn,
        "drift_reference": normalize_drift_reference(reference),
        "steady_mean": float(np.mean(steady)),
        "steady_std": float(np.std(steady)),
        "tail_mean": tail_mean,
        "previous_tail_mean": previous_mean,
        "tail_delta": float(tail_delta),
        "tail_tolerance": float(tolerance),
        "status": "needs_longer_run" if warnings else "steady_by_tail_check",
        "warnings": warnings,
    }


def _sanitize_field_label(field: float) -> str:
    return f"E_{field:.6g}".replace("-", "m").replace(".", "p")


def _build_reference_system(stack: Any, config: BulkAATargetConfig, external_field: float):
    jax, jnp = stack.jax, stack.jnp
    from jax_md import energy, partition, quantity, space

    box = jnp.asarray(config.box_size, dtype=jnp.float64)
    displacement, shift = space.periodic(box)
    species_np = make_species(config)
    species = jnp.asarray(species_np, dtype=jnp.int32)

    sigma_base = jnp.asarray([config.sigma_w, config.sigma_a, config.sigma_b], dtype=jnp.float64)
    epsilon_base = jnp.asarray([config.epsilon_w, config.epsilon_a, config.epsilon_b], dtype=jnp.float64)
    sigma = 0.5 * (sigma_base[:, None] + sigma_base[None, :])
    epsilon = jnp.sqrt(epsilon_base[:, None] * epsilon_base[None, :])

    neighbor_fn, energy_fn = energy.lennard_jones_neighbor_list(
        displacement,
        box,
        species=species,
        sigma=sigma,
        epsilon=epsilon,
        r_cutoff=config.r_cutoff,
        format=partition.OrderedSparse,
    )
    force_fn = quantity.force(energy_fn)
    force_mask = jnp.asarray(external_force_weights(species_np, config.force_mode), dtype=jnp.float64)[:, None]
    field_vector = jnp.asarray([effective_field(external_field), 0.0, 0.0], dtype=jnp.float64)

    def total_force_fn(position, neighbor):
        conservative = force_fn(position, neighbor=neighbor)
        return conservative + force_mask * field_vector

    mass_values = jnp.asarray(
        [effective_mass(config.mass_w), effective_mass(config.mass_a), effective_mass(config.mass_b)],
        dtype=jnp.float64,
    )
    mass = mass_values[species][:, None]
    return neighbor_fn, shift, total_force_fn, mass


def run_sampled_reference(
    config: BulkAATargetConfig,
    initial_positions: np.ndarray,
    external_field: float,
    n_samples: int,
    sample_interval: int,
    seed: int,
    extra_capacity: int,
    adaptive_steady: bool = False,
    min_samples: int | None = None,
    steady_check_interval: int = 100,
    drift_burn_fraction: float = 0.8,
    steady_absolute_tolerance: float = 0.002,
    steady_relative_tolerance: float = 0.05,
    steady_window_fraction: float = 0.2,
    max_all_temperature: float | None = None,
    species_rdf_radii: np.ndarray | None = None,
    species_rdf_sigma: float = 0.10,
    species_rdf_stride: int = 1,
    species_rdf_extra_capacity: int | None = None,
    species_rdf_subset_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Run a sampled explicit-reference simulation and return retained positions/velocities plus final positions."""
    stack = require_jax_stack()
    jax, jnp = stack.jax, stack.jnp
    from jax_md import simulate

    neighbor_fn, shift_fn, total_force_fn, mass = _build_reference_system(stack, config, external_field)
    position = jnp.asarray(initial_positions, dtype=jnp.float64)
    neighbors = neighbor_fn.allocate(position, extra_capacity=extra_capacity)
    key = jax.random.PRNGKey(seed)
    target_temperature = effective_temperature(config)
    dt_value = effective_dt(config)
    species_np = make_species(config)
    n_species_groups = int(np.max(species_np)) + 1
    species_ids = jnp.asarray(species_np, dtype=jnp.int32)
    keep = jnp.asarray(retained_indices(config), dtype=jnp.int32)
    solvent = jnp.asarray(np.arange(0, config.n_solvent, dtype=np.int32), dtype=jnp.int32)
    mass_column = mass.reshape((-1, 1))
    mass_vector = mass.reshape((-1,))
    mode = normalize_force_mode(config.force_mode)
    thermostat = config.thermostat.strip().lower().replace("_", "-")
    langevin_mask = jnp.asarray(langevin_thermostat_mask(species_np, config.langevin_mode), dtype=jnp.float64)
    solvent_rdf_subset = None
    solvent_rdf_positions = []
    rdf_count = 0
    species_rdf_stride = max(1, int(species_rdf_stride))
    if species_rdf_radii is not None:
        rng = np.random.default_rng(seed + 7919)
        subset_size = max(2, min(int(species_rdf_subset_size), config.n_solvent))
        solvent_subset_np = rng.choice(np.arange(config.n_solvent, dtype=np.int32), size=subset_size, replace=False)
        solvent_rdf_subset = jnp.asarray(np.sort(solvent_subset_np), dtype=jnp.int32)

    retained_positions = []
    retained_velocities = []
    trace_rows = []
    min_samples = n_samples if min_samples is None else int(min_samples)
    min_samples = max(1, min(min_samples, n_samples))
    steady_check_interval = max(1, int(steady_check_interval))

    def reached_adaptive_steady() -> bool:
        if not adaptive_steady or len(trace_rows) < min_samples:
            return False
        if len(trace_rows) < n_samples and len(trace_rows) % steady_check_interval != 0:
            return False
        trace = _stack_trace_rows(trace_rows)
        burn = int(max(0, min(len(trace["sample_index"]) - 1, round(len(trace["sample_index"]) * drift_burn_fraction))))
        summary = summarize_drift_trace(
            trace,
            burn,
            config.drift_reference,
            absolute_tolerance=steady_absolute_tolerance,
            relative_tolerance=steady_relative_tolerance,
            window_fraction=steady_window_fraction,
            max_all_temperature=max_all_temperature,
        )
        return not summary["warnings"]

    def maybe_accumulate_species_rdf(sample_index: int, current_position: Any) -> None:
        nonlocal rdf_count
        if solvent_rdf_subset is None or sample_index % species_rdf_stride != 0:
            return
        solvent_rdf_positions.append(np.asarray(current_position[solvent_rdf_subset], dtype=float))
        rdf_count += 1

    if thermostat == "nve":
        key, velocity_key = jax.random.split(key)
        velocity = jax.random.normal(velocity_key, position.shape, dtype=jnp.float64) * jnp.sqrt(target_temperature / mass_column)
        velocity = velocity - jnp.mean(velocity, axis=0, keepdims=True)
        inv_mass = 1.0 / mass_column
        dt = jnp.asarray(dt_value, dtype=jnp.float64)

        def body_fn(_, state_neighbors):
            inner_position, inner_velocity, inner_neighbors = state_neighbors
            inner_neighbors = inner_neighbors.update(inner_position)
            force = total_force_fn(inner_position, inner_neighbors)
            inner_velocity = inner_velocity + 0.5 * dt * force * inv_mass
            inner_position = shift_fn(inner_position, inner_velocity * dt)
            inner_neighbors = inner_neighbors.update(inner_position)
            force = total_force_fn(inner_position, inner_neighbors)
            inner_velocity = inner_velocity + 0.5 * dt * force * inv_mass
            return inner_position, inner_velocity, inner_neighbors

        for sample_index in range(n_samples):
            new_position, new_velocity, new_neighbors = jax.lax.fori_loop(
                0,
                sample_interval,
                body_fn,
                (position, velocity, neighbors),
            )
            if bool(new_neighbors.did_buffer_overflow):
                neighbors = neighbor_fn.allocate(position, extra_capacity=extra_capacity * 2)
                new_position, new_velocity, new_neighbors = jax.lax.fori_loop(
                    0,
                    sample_interval,
                    body_fn,
                    (position, velocity, neighbors),
                )
            position, velocity, neighbors = new_position, new_velocity, new_neighbors
            retained_positions.append(np.asarray(position[keep]))
            retained_velocities.append(np.asarray(velocity[keep]))
            trace_rows.append(_sample_trace_row(sample_index, velocity, mass_column, keep, solvent, dt_value, sample_interval, species_np))
            maybe_accumulate_species_rdf(sample_index, position)
            if reached_adaptive_steady():
                break
    elif thermostat == "nose-hoover":
        init_fn, apply_fn = simulate.nvt_nose_hoover(total_force_fn, shift_fn, dt_value, target_temperature)
        init_step = getattr(init_fn, "__wrapped__", init_fn)
        apply_step = getattr(apply_fn, "__wrapped__", apply_fn)
        # Current JAX-MD traces Nose-Hoover initialization with neighbor-list
        # metadata as static data. The legacy notebook initialized this path
        # eagerly, so we do the same and keep the inner stepping compiled.
        with jax.disable_jit():
            state = init_step(key, position, neighbor=neighbors, mass=mass_vector)

        def body_fn(_, state_neighbors):
            inner_state, inner_neighbors = state_neighbors
            inner_neighbors = inner_neighbors.update(inner_state.position)
            inner_state = apply_step(inner_state, neighbor=inner_neighbors)
            return inner_state, inner_neighbors

        for sample_index in range(n_samples):
            new_state, new_neighbors = jax.lax.fori_loop(0, sample_interval, body_fn, (state, neighbors))
            if bool(new_neighbors.did_buffer_overflow):
                neighbors = neighbor_fn.allocate(state.position, extra_capacity=extra_capacity * 2)
                new_state, new_neighbors = jax.lax.fori_loop(0, sample_interval, body_fn, (state, neighbors))
            state, neighbors = new_state, new_neighbors
            velocity = state.momentum / state.mass
            position = state.position
            retained_positions.append(np.asarray(position[keep]))
            retained_velocities.append(np.asarray(velocity[keep]))
            trace_rows.append(_sample_trace_row(sample_index, velocity, mass_column, keep, solvent, dt_value, sample_interval, species_np))
            maybe_accumulate_species_rdf(sample_index, position)
            if reached_adaptive_steady():
                break
    elif thermostat == "nose-hoover-masked":
        key, velocity_key = jax.random.split(key)
        velocity = jax.random.normal(velocity_key, position.shape, dtype=jnp.float64) * jnp.sqrt(target_temperature / mass_column)
        velocity = velocity - jnp.mean(velocity, axis=0, keepdims=True)
        inv_mass = 1.0 / mass_column
        dt = jnp.asarray(dt_value, dtype=jnp.float64)
        dof = jnp.maximum(jnp.sum(langevin_mask), 1.0)
        tau = jnp.asarray(dt_value * 100.0, dtype=jnp.float64)
        thermostat_mass = dof * target_temperature * tau**2
        xi = jnp.asarray(0.0, dtype=jnp.float64)

        def masked_kinetic_energy(inner_velocity):
            masked_velocity = inner_velocity * langevin_mask
            return 0.5 * jnp.sum(mass_column * masked_velocity**2)

        def xi_force(inner_velocity):
            return (2.0 * masked_kinetic_energy(inner_velocity) - dof * target_temperature) / thermostat_mass

        def body_fn(_, state_neighbors):
            inner_position, inner_velocity, inner_xi, inner_neighbors = state_neighbors
            inner_neighbors = inner_neighbors.update(inner_position)
            inner_xi = inner_xi + 0.5 * dt * xi_force(inner_velocity)
            inner_velocity = inner_velocity * jnp.exp(-0.5 * dt * inner_xi * langevin_mask)
            force = total_force_fn(inner_position, inner_neighbors)
            inner_velocity = inner_velocity + 0.5 * dt * force * inv_mass
            inner_position = shift_fn(inner_position, inner_velocity * dt)
            inner_neighbors = inner_neighbors.update(inner_position)
            force = total_force_fn(inner_position, inner_neighbors)
            inner_velocity = inner_velocity + 0.5 * dt * force * inv_mass
            inner_velocity = inner_velocity * jnp.exp(-0.5 * dt * inner_xi * langevin_mask)
            inner_xi = inner_xi + 0.5 * dt * xi_force(inner_velocity)
            return inner_position, inner_velocity, inner_xi, inner_neighbors

        for sample_index in range(n_samples):
            new_position, new_velocity, new_xi, new_neighbors = jax.lax.fori_loop(
                0,
                sample_interval,
                body_fn,
                (position, velocity, xi, neighbors),
            )
            if bool(new_neighbors.did_buffer_overflow):
                neighbors = neighbor_fn.allocate(position, extra_capacity=extra_capacity * 2)
                new_position, new_velocity, new_xi, new_neighbors = jax.lax.fori_loop(
                    0,
                    sample_interval,
                    body_fn,
                    (position, velocity, xi, neighbors),
                )
            position, velocity, xi, neighbors = new_position, new_velocity, new_xi, new_neighbors
            retained_positions.append(np.asarray(position[keep]))
            retained_velocities.append(np.asarray(velocity[keep]))
            trace_rows.append(_sample_trace_row(sample_index, velocity, mass_column, keep, solvent, dt_value, sample_interval, species_np))
            maybe_accumulate_species_rdf(sample_index, position)
            if reached_adaptive_steady():
                break
    elif thermostat == "langevin":
        key, velocity_key = jax.random.split(key)
        velocity = jax.random.normal(velocity_key, position.shape, dtype=jnp.float64) * jnp.sqrt(target_temperature / mass_column)
        velocity = velocity - jnp.mean(velocity, axis=0, keepdims=True)
        inv_mass = 1.0 / mass_column
        dt = jnp.asarray(dt_value, dtype=jnp.float64)
        gamma = jnp.asarray(config.gamma, dtype=jnp.float64)
        noise_scale = jnp.sqrt(2.0 * gamma * target_temperature * dt * inv_mass)

        def body_fn(_, state_neighbors):
            inner_position, inner_velocity, inner_key, inner_neighbors = state_neighbors
            inner_neighbors = inner_neighbors.update(inner_position)
            force = total_force_fn(inner_position, inner_neighbors)
            inner_key, subkey = jax.random.split(inner_key)
            noise = jax.random.normal(subkey, inner_velocity.shape, dtype=jnp.float64)
            inner_velocity = inner_velocity + dt * (force * inv_mass - gamma * langevin_mask * inner_velocity) + noise_scale * noise * langevin_mask
            inner_position = shift_fn(inner_position, inner_velocity * dt)
            return inner_position, inner_velocity, inner_key, inner_neighbors

        for sample_index in range(n_samples):
            new_position, new_velocity, new_key, new_neighbors = jax.lax.fori_loop(
                0,
                sample_interval,
                body_fn,
                (position, velocity, key, neighbors),
            )
            if bool(new_neighbors.did_buffer_overflow):
                neighbors = neighbor_fn.allocate(position, extra_capacity=extra_capacity * 2)
                new_position, new_velocity, new_key, new_neighbors = jax.lax.fori_loop(
                    0,
                    sample_interval,
                    body_fn,
                    (position, velocity, key, neighbors),
                )
            position, velocity, key, neighbors = new_position, new_velocity, new_key, new_neighbors
            retained_positions.append(np.asarray(position[keep]))
            retained_velocities.append(np.asarray(velocity[keep]))
            trace_rows.append(_sample_trace_row(sample_index, velocity, mass_column, keep, solvent, dt_value, sample_interval, species_np))
            maybe_accumulate_species_rdf(sample_index, position)
            if reached_adaptive_steady():
                break
    elif thermostat == "langevin-peculiar":
        key, velocity_key = jax.random.split(key)
        velocity = jax.random.normal(velocity_key, position.shape, dtype=jnp.float64) * jnp.sqrt(target_temperature / mass_column)
        velocity = velocity - jnp.mean(velocity, axis=0, keepdims=True)
        inv_mass = 1.0 / mass_column
        dt = jnp.asarray(dt_value, dtype=jnp.float64)
        gamma = jnp.asarray(config.gamma, dtype=jnp.float64)
        noise_scale = jnp.sqrt(2.0 * gamma * target_temperature * dt * inv_mass)

        def group_streaming_velocity(inner_velocity):
            group_mass = jnp.zeros((n_species_groups, 1), dtype=jnp.float64).at[species_ids].add(mass_column)
            group_momentum = jnp.zeros((n_species_groups, 3), dtype=jnp.float64).at[species_ids].add(mass_column * inner_velocity)
            return group_momentum[species_ids] / group_mass[species_ids]

        def zero_group_momentum_delta(delta_velocity):
            masked_mass = mass_column * langevin_mask
            group_masked_mass = jnp.zeros((n_species_groups, 3), dtype=jnp.float64).at[species_ids].add(masked_mass)
            group_delta_momentum = jnp.zeros((n_species_groups, 3), dtype=jnp.float64).at[species_ids].add(mass_column * delta_velocity)
            safe_mass = jnp.where(group_masked_mass > 0.0, group_masked_mass, 1.0)
            group_delta_velocity = jnp.where(group_masked_mass > 0.0, group_delta_momentum / safe_mass, 0.0)
            return (delta_velocity - group_delta_velocity[species_ids]) * langevin_mask

        def body_fn(_, state_neighbors):
            inner_position, inner_velocity, inner_key, inner_neighbors = state_neighbors
            inner_neighbors = inner_neighbors.update(inner_position)
            force = total_force_fn(inner_position, inner_neighbors)
            inner_velocity = inner_velocity + dt * force * inv_mass
            stream_velocity = group_streaming_velocity(inner_velocity)
            peculiar_velocity = inner_velocity - stream_velocity
            inner_key, subkey = jax.random.split(inner_key)
            noise = jax.random.normal(subkey, inner_velocity.shape, dtype=jnp.float64)
            stochastic_delta = zero_group_momentum_delta(noise_scale * noise * langevin_mask)
            thermostat_delta = -gamma * dt * peculiar_velocity * langevin_mask + stochastic_delta
            inner_velocity = inner_velocity + thermostat_delta
            inner_position = shift_fn(inner_position, inner_velocity * dt)
            return inner_position, inner_velocity, inner_key, inner_neighbors

        for sample_index in range(n_samples):
            new_position, new_velocity, new_key, new_neighbors = jax.lax.fori_loop(
                0,
                sample_interval,
                body_fn,
                (position, velocity, key, neighbors),
            )
            if bool(new_neighbors.did_buffer_overflow):
                neighbors = neighbor_fn.allocate(position, extra_capacity=extra_capacity * 2)
                new_position, new_velocity, new_key, new_neighbors = jax.lax.fori_loop(
                    0,
                    sample_interval,
                    body_fn,
                    (position, velocity, key, neighbors),
                )
            position, velocity, key, neighbors = new_position, new_velocity, new_key, new_neighbors
            retained_positions.append(np.asarray(position[keep]))
            retained_velocities.append(np.asarray(velocity[keep]))
            trace_rows.append(_sample_trace_row(sample_index, velocity, mass_column, keep, solvent, dt_value, sample_interval, species_np))
            maybe_accumulate_species_rdf(sample_index, position)
            if reached_adaptive_steady():
                break
    else:
        raise ValueError(
            f"Unknown thermostat {config.thermostat!r}; expected 'nose-hoover', "
            "'nose-hoover-masked', 'langevin', 'langevin-peculiar', or 'nve'."
        )

    trace = _stack_trace_rows(trace_rows)
    trace["force_mode"] = np.asarray([mode] * len(trace_rows), dtype=object)
    if rdf_count > 0 and solvent_rdf_positions:
        trace["solvent_rdf_positions"] = np.asarray(solvent_rdf_positions, dtype=float)
        trace["species_rdf_count"] = np.asarray(rdf_count, dtype=int)
    return np.asarray(retained_positions), np.asarray(retained_velocities), np.asarray(position), trace


def _sample_trace_row(
    sample_index: int,
    velocity: Any,
    mass: Any,
    keep: Any,
    solvent: Any,
    dt_value: float,
    sample_interval: int,
    species: Any | None = None,
) -> dict[str, float]:
    solute_velocity = velocity[keep]
    if solvent.shape[0] > 0:
        solvent_velocity = velocity[solvent]
        solvent_vx = float(np.asarray(solvent_velocity[:, 0]).mean())
    else:
        solvent_velocity = velocity[:0]
        solvent_vx = np.nan
    solute_temp = jnp_temperature(solute_velocity, mass[keep])
    all_temp = jnp_temperature(velocity, mass)
    solute_vx = float(np.asarray(solute_velocity[:, 0]).mean())
    all_vx = float(np.asarray(velocity[:, 0]).mean())
    all_com_vx = mass_weighted_vx(velocity, mass)
    velocity_array = np.asarray(velocity, dtype=float)
    mass_array = np.asarray(mass, dtype=float)
    if species is None:
        peculiar_velocity = velocity_array - np.sum(mass_array * velocity_array, axis=0, keepdims=True) / np.sum(mass_array)
    else:
        peculiar_velocity = group_peculiar_velocity_np(velocity_array, mass_array, np.asarray(species, dtype=np.int32))
    solute_peculiar_velocity = peculiar_velocity[np.asarray(keep)]
    solvent_indices = np.asarray(solvent)
    solvent_peculiar_velocity = peculiar_velocity[solvent_indices] if solvent_indices.size else peculiar_velocity[:0]
    return {
        "sample_index": float(sample_index),
        "time": float((sample_index + 1) * sample_interval * dt_value),
        "solute_vx_mean": solute_vx,
        "solvent_vx_mean": solvent_vx,
        "all_vx_mean": all_vx,
        "all_com_vx_mean": all_com_vx,
        "solute_relative_solvent_vx_mean": solute_vx - solvent_vx,
        "solute_relative_all_com_vx_mean": solute_vx - all_com_vx,
        "solute_temperature": float(np.asarray(solute_temp)),
        "all_temperature": float(np.asarray(all_temp)),
        "solvent_peculiar_temperature": kinetic_temperature_np(solvent_peculiar_velocity, mass_array[solvent_indices]) if solvent_indices.size else float("nan"),
        "solute_peculiar_temperature": kinetic_temperature_np(solute_peculiar_velocity, mass_array[np.asarray(keep)]),
        "all_peculiar_temperature": kinetic_temperature_np(peculiar_velocity, mass_array),
    }


def mass_weighted_vx(velocity: Any, mass: Any) -> float:
    mass_array = np.asarray(mass, dtype=float).reshape((-1,))
    velocity_array = np.asarray(velocity, dtype=float)
    return float(np.sum(mass_array * velocity_array[:, 0]) / np.sum(mass_array))


def jnp_temperature(velocity: Any, mass: Any) -> Any:
    if velocity.shape[0] == 0:
        return np.nan
    return np.sum(np.asarray(mass) * np.asarray(velocity) ** 2) / (3.0 * velocity.shape[0])


def _stack_trace_rows(rows: list[dict[str, float]]) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "sample_index": np.empty(0),
            "time": np.empty(0),
            "solute_vx_mean": np.empty(0),
            "solvent_vx_mean": np.empty(0),
            "all_vx_mean": np.empty(0),
            "all_com_vx_mean": np.empty(0),
            "solute_relative_solvent_vx_mean": np.empty(0),
            "solute_relative_all_com_vx_mean": np.empty(0),
            "solute_temperature": np.empty(0),
            "all_temperature": np.empty(0),
            "solvent_peculiar_temperature": np.empty(0),
            "solute_peculiar_temperature": np.empty(0),
            "all_peculiar_temperature": np.empty(0),
        }
    return {key: np.asarray([row[key] for row in rows], dtype=float) for key in rows[0]}


def run_bulk_aa_targets(
    config: BulkAATargetConfig,
    output_dir: Path,
    fields: tuple[float, ...],
    equilibration_samples: int,
    equilibrium_samples: int,
    field_samples: int,
    sample_interval: int,
    field_sample_interval: int,
    vacf_window: int,
    drift_burn_fraction: float,
    extra_capacity: int,
    save_trajectories: bool = True,
    equilibrium_thermostat: str | None = None,
    equilibrium_thermostat_mode: str | None = None,
    field_thermostat: str | None = None,
    field_thermostat_mode: str | None = None,
    adaptive_fields: bool = False,
    field_min_samples: int | None = None,
    field_max_samples: int | None = None,
    steady_check_interval: int = 100,
    steady_absolute_tolerance: float = 0.002,
    steady_relative_tolerance: float = 0.05,
    steady_window_fraction: float = 0.2,
    max_all_temperature: float | None = None,
    rdf_smoothing_sigma: float = 2.0,
    rdf_jax_sigma: float = 0.10,
    rdf_sample_stride: int = 1,
    include_solvent_rdf: bool = True,
    solvent_rdf_subset_size: int = 512,
) -> dict[str, Any]:
    """Generate equilibrium RDF/VACF targets and driven-field mobility targets."""
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial = make_initial_positions(config)

    _, _, equilibrated_positions, _ = run_sampled_reference(
        config,
        initial,
        external_field=0.0,
        n_samples=equilibration_samples,
        sample_interval=sample_interval,
        seed=config.seed,
        extra_capacity=extra_capacity,
    )
    rdf_edges = np.arange(0.0, config.rdf_cutoff + config.rdf_dr, config.rdf_dr)
    rdf_centers = 0.5 * (rdf_edges[:-1] + rdf_edges[1:])
    equilibrium_config = replace(
        config,
        thermostat=equilibrium_thermostat or config.thermostat,
        langevin_mode=equilibrium_thermostat_mode or config.langevin_mode,
    )
    retained_pos, retained_vel, _, equilibrium_trace = run_sampled_reference(
        equilibrium_config,
        equilibrated_positions,
        external_field=0.0,
        n_samples=equilibrium_samples,
        sample_interval=sample_interval,
        seed=config.seed + 1,
        extra_capacity=extra_capacity,
        species_rdf_radii=rdf_centers if include_solvent_rdf else None,
        species_rdf_sigma=rdf_jax_sigma,
        species_rdf_stride=rdf_sample_stride,
        species_rdf_subset_size=solvent_rdf_subset_size,
    )

    centers, rdfs = compute_retained_rdfs(retained_pos, config)
    rdf_path = write_rdf_csv(output_dir / "AA_retained_rdf_targets.csv", centers, rdfs)
    full_rdfs = dict(rdfs)
    species_rdf_count = int(np.asarray(equilibrium_trace.get("species_rdf_count", 0)).reshape(-1)[0])
    if include_solvent_rdf and "solvent_rdf_positions" in equilibrium_trace:
        solvent_centers, solvent_rdf = compute_same_species_rdf(
            equilibrium_trace["solvent_rdf_positions"],
            config.box_size,
            config.rdf_cutoff,
            config.rdf_dr,
        )
        if solvent_centers.shape == centers.shape and np.allclose(solvent_centers, centers):
            full_rdfs["g_r_WW"] = solvent_rdf
    rdf_full_path = write_full_rdf_csv(
        output_dir / "AA_rdf_targets.csv",
        centers,
        smooth_and_tail_normalize_rdfs(full_rdfs, sigma=0.0),
    )
    rdf_smooth_path = write_full_rdf_csv(
        output_dir / "AA_rdf_targets_smooth.csv",
        centers,
        smooth_and_tail_normalize_rdfs(full_rdfs, sigma=rdf_smoothing_sigma),
    )
    vacf_time, vacf = compute_vacf(retained_vel, dt=effective_dt(config) * sample_interval, window=vacf_window)
    vacf_path = write_vacf_csv(output_dir / "AA_retained_vacf_target.csv", vacf_time, vacf)

    trajectory_paths: dict[str, str] = {}
    if save_trajectories:
        traj_path = output_dir / "traj_cg.npy"
        vel_path = output_dir / "vel_cg.npy"
        np.save(traj_path, retained_pos)
        np.save(vel_path, retained_vel)
        trajectory_paths = {"traj_cg": str(traj_path), "vel_cg": str(vel_path)}

    mobility_rows = []
    mobility_path = output_dir / "AA_mobility_targets.csv"
    trace_paths: dict[str, str] = {}
    field_diagnostics = []
    production_config = replace(
        config,
        thermostat=field_thermostat or config.thermostat,
        langevin_mode=field_thermostat_mode or config.langevin_mode,
    )
    requested_field_samples = int(field_samples)
    effective_field_min_samples = int(field_min_samples or field_samples)
    effective_field_max_samples = int(field_max_samples or field_samples)
    if adaptive_fields:
        effective_field_min_samples = min(effective_field_min_samples, effective_field_max_samples)
    else:
        effective_field_min_samples = requested_field_samples
        effective_field_max_samples = requested_field_samples
    for field_index, field in enumerate(fields):
        print(
            (
                f"Running field {field:g} with "
                f"{'adaptive' if adaptive_fields else 'fixed'} sampling "
                f"(min={effective_field_min_samples}, max={effective_field_max_samples})"
            ),
            flush=True,
        )
        _, _field_vel, _, trace = run_sampled_reference(
            production_config,
            equilibrated_positions,
            external_field=field,
            n_samples=effective_field_max_samples,
            sample_interval=field_sample_interval,
            seed=config.seed + 100 + field_index,
            extra_capacity=extra_capacity,
            adaptive_steady=adaptive_fields,
            min_samples=effective_field_min_samples,
            steady_check_interval=steady_check_interval,
            drift_burn_fraction=drift_burn_fraction,
            steady_absolute_tolerance=steady_absolute_tolerance,
            steady_relative_tolerance=steady_relative_tolerance,
            steady_window_fraction=steady_window_fraction,
            max_all_temperature=max_all_temperature,
        )
        actual_field_samples = len(trace["sample_index"])
        burn = int(max(0, min(actual_field_samples - 1, round(actual_field_samples * drift_burn_fraction))))
        drift = float(np.mean(drift_observable(trace, config.drift_reference)[burn:]))
        mobility_rows.append((field, drift))
        trace_path = write_drift_trace_csv(output_dir / f"AA_drift_trace_{_sanitize_field_label(field)}_{normalize_force_mode(config.force_mode)}.csv", trace)
        trace_paths[f"drift_trace_{_sanitize_field_label(field)}"] = str(trace_path)
        diagnostic = summarize_drift_trace(
            trace,
            burn,
            config.drift_reference,
            absolute_tolerance=steady_absolute_tolerance,
            relative_tolerance=steady_relative_tolerance,
            window_fraction=steady_window_fraction,
            max_all_temperature=max_all_temperature,
        )
        stop_reason = "fixed_samples"
        if adaptive_fields and diagnostic["warnings"]:
            stop_reason = "max_samples_reached_before_steady"
            diagnostic["status"] = "max_samples_reached_needs_longer_run"
        elif adaptive_fields:
            stop_reason = "adaptive_steady_reached"
            diagnostic["status"] = "adaptive_steady_reached"
        field_diagnostics.append(
            {
                "field": float(field),
                "drift_velocity": drift,
                "actual_field_samples": int(actual_field_samples),
                "requested_min_field_samples": int(effective_field_min_samples),
                "requested_max_field_samples": int(effective_field_max_samples),
                "stop_reason": stop_reason,
                **diagnostic,
            }
        )
        np.savetxt(
            mobility_path,
            np.asarray(mobility_rows),
            delimiter=",",
            header="external_field,drift_velocity",
            comments="",
        )
        partial_report = {
            "status": "running",
            "elapsed_seconds": time.time() - started,
            "force_mode": normalize_force_mode(config.force_mode),
            "drift_reference": normalize_drift_reference(config.drift_reference),
            "equilibrium_thermostat": equilibrium_config.thermostat,
            "equilibrium_thermostat_mode": normalize_langevin_mode(equilibrium_config.langevin_mode),
            "production_thermostat": production_config.thermostat,
            "production_thermostat_mode": normalize_langevin_mode(production_config.langevin_mode),
            "adaptive_fields": adaptive_fields,
            "completed_fields": [float(row[0]) for row in mobility_rows],
            "field_diagnostics": field_diagnostics,
        }
        (output_dir / "AA_target_generation_report.partial.json").write_text(json.dumps(partial_report, indent=2, sort_keys=True) + "\n")
        print(
            (
                f"Field {field:g}: drift={drift:.6g}, "
                f"samples={actual_field_samples}, status={field_diagnostics[-1]['status']}"
            ),
            flush=True,
        )
    if not mobility_rows:
        np.savetxt(
            mobility_path,
            np.empty((0, 2), dtype=float),
            delimiter=",",
            header="external_field,drift_velocity",
            comments="",
        )

    report = {
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "config": config.to_dict(),
        "effective_dt": effective_dt(config),
        "effective_temperature": effective_temperature(config),
        "force_mode": normalize_force_mode(config.force_mode),
        "drift_reference": normalize_drift_reference(config.drift_reference),
        "thermostat": config.thermostat,
        "equilibrium_thermostat": equilibrium_config.thermostat,
        "production_thermostat": production_config.thermostat,
        "langevin_mode": normalize_langevin_mode(config.langevin_mode),
        "equilibrium_thermostat_mode": normalize_langevin_mode(equilibrium_config.langevin_mode),
        "production_thermostat_mode": normalize_langevin_mode(production_config.langevin_mode),
        "force_mode_counts": force_mode_counts(config),
        "fields": list(fields),
        "equilibration_samples": equilibration_samples,
        "equilibrium_samples": equilibrium_samples,
        "field_samples": field_samples,
        "adaptive_fields": adaptive_fields,
        "field_min_samples": effective_field_min_samples,
        "field_max_samples": effective_field_max_samples,
        "steady_check_interval": steady_check_interval,
        "steady_absolute_tolerance": steady_absolute_tolerance,
        "steady_relative_tolerance": steady_relative_tolerance,
        "steady_window_fraction": steady_window_fraction,
        "max_all_temperature": max_all_temperature,
        "sample_interval": sample_interval,
        "field_sample_interval": field_sample_interval,
        "rdf_smoothing_sigma": rdf_smoothing_sigma,
        "rdf_jax_sigma": rdf_jax_sigma,
        "rdf_sample_stride": rdf_sample_stride,
        "solvent_rdf_samples": species_rdf_count,
        "solvent_rdf_subset_size": solvent_rdf_subset_size if include_solvent_rdf else 0,
        "include_solvent_rdf": include_solvent_rdf,
        "drift_burn_fraction": drift_burn_fraction,
        "outputs": {
            "rdf": str(rdf_path),
            "rdf_full": str(rdf_full_path),
            "rdf_smooth": str(rdf_smooth_path),
            "vacf": str(vacf_path),
            "mobility": str(mobility_path),
            **trace_paths,
            **trajectory_paths,
        },
        "field_diagnostics": field_diagnostics,
    }
    (output_dir / "AA_target_generation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
