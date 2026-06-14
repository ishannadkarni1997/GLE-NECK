#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.artifacts import read_numeric_table
from gleneck.bulk.config import DEFAULT_BULK_MODEL
from gleneck.bulk.io import write_cg_potential_npz
from gleneck.bulk.memory import estimate_memory_kernel_from_vacf, gaussian_smooth
from gleneck.bulk.units import internal_time_to_ps, memory_internal_to_per_ps2


JAX_MD_REAL_TEMPERATURE_UNIT = 0.0019872034899890423
DEFAULT_GLE_EFFECTIVE_DT_INTERNAL = 0.0204548282835039
PAIR_COLUMNS = (
    ("AA", "g_r_AA", (0, 0), "u_aa"),
    ("AB", "g_r_AB", (0, 1), "u_ab"),
    ("BB", "g_r_BB", (1, 1), "u_bb"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fresh bulk GLE baseline artifacts from AA equilibrium RDF and VACF targets."
    )
    parser.add_argument("--rdf", type=Path, required=True, help="Smoothed AA RDF target CSV.")
    parser.add_argument("--vacf", type=Path, required=True, help="AA retained-solute VACF target CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated baseline artifacts.")
    parser.add_argument("--temperature", type=float, default=40.0, help="AA target temperature in notebook Kelvin-like units.")
    parser.add_argument(
        "--temperature-unit",
        type=float,
        default=JAX_MD_REAL_TEMPERATURE_UNIT,
        help="JAX-MD real-unit temperature scalar used to compute kBT.",
    )
    parser.add_argument("--kbt", type=float, default=None, help="Override kBT directly in simulation energy units.")
    parser.add_argument("--rdf-floor", type=float, default=1.0e-8, help="Lower RDF clip before Boltzmann inversion.")
    parser.add_argument(
        "--core-valid-floor",
        type=float,
        default=1.0e-4,
        help="First RDF value above this threshold marks the end of the hard-core fill region.",
    )
    parser.add_argument("--core-cap", type=float, default=1.5, help="Short-range repulsive PMF value at r_min.")
    parser.add_argument("--tail-bins", type=int, default=10, help="Final bins used to shift PMFs to a zero tail.")
    parser.add_argument("--alpha", type=float, default=1.0e-10, help="Tikhonov regularization for Volterra memory solve.")
    parser.add_argument("--max-lags", type=int, default=None, help="Optional maximum VACF lags for memory fitting.")
    parser.add_argument("--memory-smooth-sigma", type=float, default=0.0, help="Gaussian smoothing sigma for memory kernel.")
    parser.add_argument("--drop-initial", type=int, default=0, help="Drop this many initial memory points after solving.")
    parser.add_argument("--pad-end", type=int, default=0, help="Append this many zero memory points after fitting.")
    parser.add_argument(
        "--gle-effective-dt-ps",
        type=float,
        default=DEFAULT_GLE_EFFECTIVE_DT_INTERNAL,
        help=(
            "Backward-compatible name: this value is the GLE integrator timestep "
            "in JAX-MD internal time, not ps."
        ),
    )
    return parser


def _require_columns(table_path: Path, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    table = read_numeric_table(table_path)
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{table_path} is missing required columns: {missing}")
    return {column: np.asarray(table.columns[column], dtype=float) for column in columns}


def _regularized_pmf(
    r_source: np.ndarray,
    g_source: np.ndarray,
    r_grid: np.ndarray,
    *,
    kbt: float,
    rdf_floor: float,
    core_valid_floor: float,
    core_cap: float,
    tail_bins: int,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    g_grid = np.interp(r_grid, r_source, g_source)
    safe_g = np.clip(g_grid, rdf_floor, None)
    potential = -kbt * np.log(safe_g)

    tail_shift = 0.0
    if tail_bins > 0:
        tail = potential[-tail_bins:]
        finite_tail = tail[np.isfinite(tail)]
        if finite_tail.size:
            tail_shift = float(np.mean(finite_tail))
            potential = potential - tail_shift

    valid = np.flatnonzero(g_grid > core_valid_floor)
    core_fill_until_index: int | None = None
    first_valid_r: float | None = None
    if valid.size:
        first = int(valid[0])
        core_fill_until_index = first
        first_valid_r = float(r_grid[first])
        if first > 0:
            first_u = float(potential[first])
            high_u = max(float(core_cap), first_u)
            span = max(float(r_grid[first] - r_grid[0]), np.finfo(float).eps)
            fraction = (r_grid[first] - r_grid[:first]) / span
            potential[:first] = first_u + (high_u - first_u) * fraction**2

    return potential, {
        "tail_shift": tail_shift,
        "core_fill_until_index": core_fill_until_index,
        "first_valid_r": first_valid_r,
        "min": float(np.nanmin(potential)),
        "max": float(np.nanmax(potential)),
    }


def _write_potential_csv(path: Path, r_grid: np.ndarray, potentials: dict[tuple[int, int], np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        path,
        np.column_stack([r_grid, potentials[(0, 0)], potentials[(0, 1)], potentials[(1, 1)]]),
        delimiter=",",
        header="r,u_aa,u_ab,u_bb",
        comments="",
    )
    return path


def _write_memory_csv(path: Path, time_lag: np.ndarray, raw: np.ndarray, fitted: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_lag = np.asarray(time_lag, dtype=float)
    np.savetxt(
        path,
        np.column_stack(
            [
                time_lag,
                time_lag,
                internal_time_to_ps(time_lag),
                raw,
                fitted,
                memory_internal_to_per_ps2(raw),
                memory_internal_to_per_ps2(fitted),
            ]
        ),
        delimiter=",",
        header=(
            "time_lag,time_lag_internal,time_lag_ps,"
            "raw_memory_internal,fitted_memory_internal,"
            "raw_memory_ps2,fitted_memory_ps2"
        ),
        comments="",
    )
    return path


def main() -> int:
    args = build_parser().parse_args()
    if args.rdf_floor <= 0 or args.core_valid_floor <= 0:
        print("--rdf-floor and --core-valid-floor must be positive.", file=sys.stderr)
        return 2
    if args.core_cap <= 0:
        print("--core-cap must be positive.", file=sys.stderr)
        return 2
    if args.tail_bins < 0 or args.drop_initial < 0 or args.pad_end < 0:
        print("--tail-bins, --drop-initial, and --pad-end must be non-negative.", file=sys.stderr)
        return 2
    if args.alpha < 0 or args.memory_smooth_sigma < 0:
        print("--alpha and --memory-smooth-sigma must be non-negative.", file=sys.stderr)
        return 2
    if args.max_lags is not None and args.max_lags < 2:
        print("--max-lags must be at least 2.", file=sys.stderr)
        return 2
    if args.gle_effective_dt_ps <= 0:
        print("--gle-effective-dt-ps must be positive.", file=sys.stderr)
        return 2

    kbt = float(args.kbt) if args.kbt is not None else float(args.temperature * args.temperature_unit)
    if kbt <= 0:
        print("kBT must be positive.", file=sys.stderr)
        return 2

    rdf_columns = _require_columns(args.rdf, ("r_distance", "g_r_AA", "g_r_BB", "g_r_AB"))
    r_source = rdf_columns["r_distance"]
    r_grid = np.linspace(DEFAULT_BULK_MODEL.r_min, DEFAULT_BULK_MODEL.r_max, DEFAULT_BULK_MODEL.num_bins)

    potentials: dict[tuple[int, int], np.ndarray] = {}
    potential_metadata: dict[str, object] = {}
    for label, column, pair, _npz_key in PAIR_COLUMNS:
        potential, metadata = _regularized_pmf(
            r_source,
            rdf_columns[column],
            r_grid,
            kbt=kbt,
            rdf_floor=args.rdf_floor,
            core_valid_floor=args.core_valid_floor,
            core_cap=args.core_cap,
            tail_bins=args.tail_bins,
        )
        potentials[pair] = potential
        potential_metadata[label] = metadata

    vacf_columns = _require_columns(args.vacf, ("time_lag", "vacf_solute_norm"))
    time_lag = vacf_columns["time_lag"]
    vacf = vacf_columns["vacf_solute_norm"]
    if args.max_lags is not None:
        time_lag = time_lag[: args.max_lags]
        vacf = vacf[: args.max_lags]

    raw_memory, source_dt = estimate_memory_kernel_from_vacf(time_lag, vacf, alpha=args.alpha)
    if args.drop_initial:
        raw_memory = raw_memory[args.drop_initial :]
    if args.pad_end:
        raw_memory = np.concatenate([raw_memory, np.zeros(args.pad_end, dtype=float)])
    memory_time = np.arange(raw_memory.size, dtype=float) * source_dt
    fitted_memory = gaussian_smooth(raw_memory, args.memory_smooth_sigma)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    potential_npz = write_cg_potential_npz(potentials, output_dir / "cg_potentials_from_aa_rdf.npz")
    potential_csv = _write_potential_csv(output_dir / "cg_potentials_from_aa_rdf.csv", r_grid, potentials)
    raw_memory_path = output_dir / "raw_memory_kernel.npy"
    fitted_memory_path = output_dir / "fitted_memory_kernel.npy"
    np.save(raw_memory_path, raw_memory)
    np.save(fitted_memory_path, fitted_memory)
    memory_csv = _write_memory_csv(output_dir / "memory_kernel.csv", memory_time, raw_memory, fitted_memory)

    metadata = {
        "rdf_input": str(args.rdf),
        "vacf_input": str(args.vacf),
        "temperature": args.temperature,
        "temperature_unit": args.temperature_unit,
        "kbt": kbt,
        "rdf_floor": args.rdf_floor,
        "core_valid_floor": args.core_valid_floor,
        "core_cap": args.core_cap,
        "tail_bins": args.tail_bins,
        "r_grid_min": float(r_grid[0]),
        "r_grid_max": float(r_grid[-1]),
        "r_grid_size": int(r_grid.size),
        "potential_metadata": potential_metadata,
        "alpha": args.alpha,
        "max_lags": args.max_lags,
        "memory_smooth_sigma": args.memory_smooth_sigma,
        "drop_initial": args.drop_initial,
        "pad_end": args.pad_end,
        "source_dt": source_dt,
        "source_dt_internal": source_dt,
        "source_dt_ps": float(internal_time_to_ps(source_dt)),
        "gle_effective_dt_ps": args.gle_effective_dt_ps,
        "gle_effective_dt_internal": args.gle_effective_dt_ps,
        "gle_effective_dt_physical_ps": float(internal_time_to_ps(args.gle_effective_dt_ps)),
        "memory_orig_interval_for_gle": source_dt / args.gle_effective_dt_ps,
        "memory_length": int(fitted_memory.size),
        "memory_kernel_unit": "internal_time^-2",
        "raw_memory_min": float(np.nanmin(raw_memory)),
        "raw_memory_max": float(np.nanmax(raw_memory)),
        "fitted_memory_min": float(np.nanmin(fitted_memory)),
        "fitted_memory_max": float(np.nanmax(fitted_memory)),
        "potential_npz": str(potential_npz),
        "potential_csv": str(potential_csv),
        "raw_memory_output": str(raw_memory_path),
        "fitted_memory_output": str(fitted_memory_path),
        "memory_csv": str(memory_csv),
    }
    metadata_path = output_dir / "baseline_target_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
