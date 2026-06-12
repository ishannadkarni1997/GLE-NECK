#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.bulk.config import DEFAULT_BULK_MODEL
from gleneck.bulk.io import (
    default_bulk_input_paths,
    load_cg_potential,
    load_cg_potential_npz,
    validate_bulk_inputs,
    write_cg_potential_npz,
)
from gleneck.bulk.memory import prepare_memory_terms
from gleneck.bulk.neck import initialize_kernel_params, require_jax_stack
from gleneck.bulk.gle import make_force_tables, make_species_and_mass, pairwise_tabulated_force
from gleneck.paths import ProjectPaths


def _load_or_create_potential(root: Path):
    processed = root / "data" / "processed" / "bulk" / "cg_potentials_NVE242.npz"
    if processed.exists():
        return load_cg_potential_npz(processed), processed

    legacy = root / "code" / "Bulk" / "CGpotentials_int_NVE242.pkl"
    potential = load_cg_potential(legacy)
    return potential, write_cg_potential_npz(potential, processed)


def _synthetic_state(stack, config):
    jnp = stack.jnp
    box = jnp.asarray(config.box_size, dtype=jnp.float32)
    n_axis = math.ceil(config.n_cg ** (1.0 / 3.0))
    axes = [
        jnp.linspace(0.0, float(box[dim]), n_axis, endpoint=False, dtype=jnp.float32)
        + float(box[dim]) / (2.0 * n_axis)
        for dim in range(3)
    ]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    positions = jnp.stack([axis.reshape(-1) for axis in mesh], axis=1)[: config.n_cg]
    velocities = jnp.zeros((config.n_cg, 3), dtype=jnp.float32)
    return positions, velocities


def _private_or_synthetic_state(stack, root: Path, private_root: Path | None, synthetic: bool):
    jnp = stack.jnp
    paths = default_bulk_input_paths(root, private_root)
    missing = [record for record in validate_bulk_inputs(paths) if record.required and not record.exists]
    missing_names = {record.name for record in missing}
    if not synthetic and {"trajectory", "velocity"}.intersection(missing_names):
        missing_paths = "\n".join(str(record.path) for record in missing if record.name in {"trajectory", "velocity"})
        raise FileNotFoundError(
            "Bulk trajectory inputs are missing. Stage them or rerun with --synthetic.\n"
            f"{missing_paths}"
        )
    if synthetic:
        return _synthetic_state(stack, DEFAULT_BULK_MODEL), "synthetic"

    import numpy as np

    trajectory = np.load(paths.trajectory)
    velocity = np.load(paths.velocity)
    return (jnp.asarray(trajectory[-1]), jnp.asarray(velocity[-1])), "private"


def _tabulated_bulk_energy(stack, potential, config):
    jax, jnp = stack.jax, stack.jnp
    from jax_md import energy, partition, quantity, smap, space
    from jax_md.smap import ParameterTree, ParameterTreeMapping

    box = jnp.asarray(config.box_size, dtype=jnp.float32)
    displacement, _ = space.periodic(box)
    r_bins = jnp.linspace(config.r_min, config.r_max, config.num_bins)

    species = jnp.concatenate(
        [
            jnp.zeros(config.n_cg // 2, dtype=jnp.int32),
            jnp.ones(config.n_cg // 2, dtype=jnp.int32),
        ]
    )
    u_table = jnp.asarray(
        [
            [potential[(0, 0)], potential[(0, 1)]],
            [potential[(0, 1)], potential[(1, 1)]],
        ],
        dtype=jnp.float32,
    )
    u_table_pt = ParameterTree(u_table, mapping=ParameterTreeMapping.PerSpecies)

    def tabulated_potential(dr, u_table):
        return jax.vmap(lambda radius, table: jnp.interp(radius, r_bins, table))(dr, u_table)

    energy_fn = smap.pair_neighbor_list(
        energy.multiplicative_isotropic_cutoff(
            tabulated_potential,
            r_onset=config.r_onset,
            r_cutoff=config.r_cutoff,
        ),
        space.canonicalize_displacement_or_metric(displacement),
        species=species,
        u_table=u_table_pt,
    )
    neighbor_fn = partition.neighbor_list(
        displacement,
        box,
        r_cutoff=config.r_cutoff,
        dr_threshold=config.dr_threshold,
        format=partition.OrderedSparse,
    )
    force_fn = quantity.force(energy_fn)
    return energy_fn, force_fn, neighbor_fn


def command_smoke(args: argparse.Namespace) -> int:
    root = ProjectPaths.discover(args.root).root
    config = DEFAULT_BULK_MODEL
    stack = require_jax_stack()
    jax, jnp = stack.jax, stack.jnp

    potential, potential_path = _load_or_create_potential(root)
    (positions, velocities), state_source = _private_or_synthetic_state(stack, root, args.private_root, args.synthetic)

    memory_path = root / "data" / "processed" / "bulk" / "fitted_memory_kernal.npy"
    import numpy as np

    memory_terms = prepare_memory_terms(np.load(memory_path), l_max=config.l_max, orig_interval=config.memory_orig_interval)
    transformed, params, _, _ = initialize_kernel_params(config)
    correction = transformed.apply(params, None, jnp.asarray(args.field, dtype=jnp.float32))
    total_kernel = jnp.asarray(memory_terms.memory, dtype=jnp.float32) + correction

    energy_fn, force_fn, neighbor_fn = _tabulated_bulk_energy(stack, potential, config)
    neighbors = neighbor_fn.allocate(positions, extra_capacity=60)
    conservative_energy = energy_fn(positions, neighbor=neighbors)
    conservative_force = force_fn(positions, neighbors)
    r_bins_np, potential_table_np, _ = make_force_tables(potential, config)
    species_clean, _ = make_species_and_mass(stack, config)
    clean_force = pairwise_tabulated_force(
        stack,
        positions,
        species_clean,
        jnp.asarray(r_bins_np, dtype=positions.dtype),
        jnp.asarray(potential_table_np, dtype=positions.dtype),
        config,
    )
    force_delta = clean_force - conservative_force
    force_norm = jnp.linalg.norm(conservative_force)
    force_delta_norm = jnp.linalg.norm(force_delta)
    force_relative_l2 = force_delta_norm / jnp.maximum(force_norm, jnp.asarray(1e-12, dtype=positions.dtype))
    force_cosine = jnp.vdot(clean_force.reshape(-1), conservative_force.reshape(-1)) / jnp.maximum(
        jnp.linalg.norm(clean_force) * force_norm,
        jnp.asarray(1e-12, dtype=positions.dtype),
    )

    report = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "state_source": state_source,
        "potential_path": str(potential_path),
        "memory_path": str(memory_path),
        "positions_shape": list(positions.shape),
        "velocities_shape": list(velocities.shape),
        "field": args.field,
        "conservative_energy": float(conservative_energy),
        "force_l2_norm": float(jnp.linalg.norm(conservative_force)),
        "force_shape": list(conservative_force.shape),
        "clean_force_l2_norm": float(jnp.linalg.norm(clean_force)),
        "clean_vs_jaxmd_force_relative_l2": float(force_relative_l2),
        "clean_vs_jaxmd_force_max_abs": float(jnp.max(jnp.abs(force_delta))),
        "clean_vs_jaxmd_force_cosine": float(force_cosine),
        "memory_shape": list(memory_terms.memory.shape),
        "noise_filter_shape": list(memory_terms.noise_filter.shape),
        "corrective_kernel_l2_norm": float(jnp.linalg.norm(correction)),
        "total_kernel_first_value": float(total_kernel[0]),
    }

    output_dir = args.output_dir or (root / "runs" / "bulk_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "bulk_jax_smoke.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote smoke report: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a minimal bulk JAX/JAX-MD/GLE-NECK smoke check.")
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to auto-discovery.")
    parser.add_argument("--private-root", type=Path, default=None, help="Directory containing private traj_cg.npy and vel_cg.npy.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for smoke report JSON.")
    parser.add_argument("--field", type=float, default=0.0347, help="External field used for corrective-kernel evaluation.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic positions/velocities instead of private trajectory arrays.")
    return parser


def main() -> int:
    return command_smoke(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
