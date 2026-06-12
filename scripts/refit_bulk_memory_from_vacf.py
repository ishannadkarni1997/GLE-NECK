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
from gleneck.bulk.memory import estimate_memory_kernel_from_vacf, gaussian_smooth
from gleneck.bulk.units import effective_dt, internal_time_to_ps, memory_internal_to_per_ps2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refit a bulk equilibrium memory kernel from an AA VACF CSV.")
    parser.add_argument("--vacf", type=Path, required=True, help="CSV containing a time-lag column and normalized VACF column.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for memory-kernel artifacts.")
    parser.add_argument("--time-column", default="time_lag")
    parser.add_argument("--vacf-column", default="vacf_solute_norm")
    parser.add_argument("--alpha", type=float, default=1e-10, help="Tikhonov regularization used in the Volterra solve.")
    parser.add_argument("--max-lags", type=int, default=None, help="Optional maximum number of VACF lags to use.")
    parser.add_argument("--smooth-sigma", type=float, default=0.0, help="Gaussian smoothing sigma for fitted output. Use 0 for raw.")
    parser.add_argument("--drop-initial", type=int, default=0, help="Drop this many initial kernel points after solving.")
    parser.add_argument("--pad-end", type=int, default=0, help="Append this many zeros after optional initial-point drop.")
    parser.add_argument(
        "--gle-effective-dt",
        type=float,
        default=None,
        help=(
            "Effective GLE timestep in JAX-MD internal time used to report "
            "memory_orig_interval. Defaults to DEFAULT_BULK_MODEL internal effective_dt."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.alpha < 0:
        print("--alpha must be non-negative.", file=sys.stderr)
        return 2
    if args.max_lags is not None and args.max_lags < 2:
        print("--max-lags must be at least 2.", file=sys.stderr)
        return 2
    if args.smooth_sigma < 0 or args.drop_initial < 0 or args.pad_end < 0:
        print("--smooth-sigma, --drop-initial, and --pad-end must be non-negative.", file=sys.stderr)
        return 2

    table = read_numeric_table(args.vacf)
    if args.time_column not in table.columns:
        print(f"Missing time column {args.time_column!r} in {args.vacf}.", file=sys.stderr)
        return 2
    if args.vacf_column not in table.columns:
        print(f"Missing VACF column {args.vacf_column!r} in {args.vacf}.", file=sys.stderr)
        return 2

    time_lag = np.asarray(table.columns[args.time_column], dtype=float)
    vacf = np.asarray(table.columns[args.vacf_column], dtype=float)
    if args.max_lags is not None:
        time_lag = time_lag[: args.max_lags]
        vacf = vacf[: args.max_lags]

    raw, source_dt = estimate_memory_kernel_from_vacf(time_lag, vacf, alpha=args.alpha)
    if args.drop_initial:
        raw = raw[args.drop_initial :]
    if args.pad_end:
        raw = np.concatenate([raw, np.zeros(args.pad_end, dtype=float)])
    fitted = gaussian_smooth(raw, args.smooth_sigma)

    gle_dt = float(args.gle_effective_dt) if args.gle_effective_dt is not None else effective_dt(DEFAULT_BULK_MODEL)
    if gle_dt <= 0:
        print("--gle-effective-dt must be positive.", file=sys.stderr)
        return 2
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_memory_kernel.npy"
    fitted_path = output_dir / "fitted_memory_kernel.npy"
    csv_path = output_dir / "memory_kernel.csv"
    metadata_path = output_dir / "memory_kernel_metadata.json"

    np.save(raw_path, raw)
    np.save(fitted_path, fitted)
    time_lag_internal = np.arange(fitted.size, dtype=float) * source_dt
    np.savetxt(
        csv_path,
        np.column_stack(
            [
                time_lag_internal,
                time_lag_internal,
                internal_time_to_ps(time_lag_internal),
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
    metadata = {
        "vacf_input": str(args.vacf),
        "time_column": args.time_column,
        "vacf_column": args.vacf_column,
        "alpha": args.alpha,
        "max_lags": args.max_lags,
        "smooth_sigma": args.smooth_sigma,
        "drop_initial": args.drop_initial,
        "pad_end": args.pad_end,
        "source_dt": source_dt,
        "source_dt_internal": source_dt,
        "source_dt_ps": float(internal_time_to_ps(source_dt)),
        "gle_effective_dt": gle_dt,
        "gle_effective_dt_internal": gle_dt,
        "gle_effective_dt_ps": float(internal_time_to_ps(gle_dt)),
        "memory_orig_interval_for_gle": source_dt / gle_dt,
        "memory_kernel_unit": "internal_time^-2",
        "raw_memory_output": str(raw_path),
        "fitted_memory_output": str(fitted_path),
        "csv_output": str(csv_path),
        "kernel_length": int(fitted.size),
        "raw_memory_min": float(np.min(raw)),
        "raw_memory_max": float(np.max(raw)),
        "fitted_memory_min": float(np.min(fitted)),
        "fitted_memory_max": float(np.max(fitted)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
