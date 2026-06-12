#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.bulk.gle import load_mobility_curve, model_config_with_overrides, run_mobility_curve
from gleneck.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline bulk GLE mobility against legacy field targets.")
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to auto-discovery.")
    parser.add_argument("--private-root", type=Path, default=None, help="Directory containing retained traj_cg.npy and vel_cg.npy.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated baseline outputs.")
    parser.add_argument("--target", type=Path, default=None, help="Mobility target CSV whose field grid should be reused.")
    parser.add_argument("--fields", type=str, default=None, help="Optional comma-separated field grid.")
    parser.add_argument("--potential-path", type=Path, default=None, help="Optional CG potential NPZ. Defaults to processed legacy potential.")
    parser.add_argument("--memory-path", type=Path, default=None, help="Optional fitted memory-kernel NPY. Defaults to processed legacy kernel.")
    parser.add_argument("--steps", type=int, default=100, help="GLE steps per field.")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Zero-field GLE warmup steps before measuring mobility.")
    parser.add_argument("--drift-window", type=int, default=None, help="Number of final steps used for drift averaging.")
    parser.add_argument("--dt", type=float, default=None, help="Override the notebook-derived GLE timestep.")
    parser.add_argument("--l-max", type=int, default=None, help="Override the memory/history length.")
    parser.add_argument("--memory-orig-interval", type=float, default=None, help="Override memory-kernel source spacing.")
    parser.add_argument(
        "--memory-history-scaling",
        choices=("force-units", "legacy-training"),
        default=None,
        help="History-memory scaling convention for the GLE integrator.",
    )
    parser.add_argument("--init-velocity-scale", type=float, default=None, help="Scale generated Maxwell-Boltzmann velocities.")
    parser.add_argument("--noise-scale", type=float, default=None, help="Scale the colored-noise white-noise draw.")
    parser.add_argument(
        "--allow-unstable",
        action="store_true",
        help="Return success even if stability diagnostics flag the trajectory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--init-mode",
        choices=("generated", "legacy"),
        default="generated",
        help="Initialization source: generated thermal A/B state or legacy traj_cg.npy/vel_cg.npy.",
    )
    return parser


def _parse_fields(text: str) -> np.ndarray:
    fields = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not fields:
        raise ValueError("No fields were supplied.")
    return np.asarray(fields, dtype=float)


def main() -> int:
    args = build_parser().parse_args()
    paths = ProjectPaths.discover(args.root)
    root = paths.root
    output_dir = args.output_dir or (root / "runs" / "bulk_gle_baseline")
    if args.fields:
        fields = _parse_fields(args.fields)
    else:
        target = args.target or (root / "data" / "processed" / "bulk" / "AA_mobility_data.csv")
        fields = load_mobility_curve(target).fields
    config = model_config_with_overrides(
        dt=args.dt,
        l_max=args.l_max,
        memory_orig_interval=args.memory_orig_interval,
        memory_history_scaling=args.memory_history_scaling,
        init_velocity_scale=args.init_velocity_scale,
        noise_scale=args.noise_scale,
    )
    report = run_mobility_curve(
        root=root,
        fields=fields,
        output_dir=output_dir,
        private_root=args.private_root,
        potential_path=args.potential_path,
        memory_path=args.memory_path,
        init_mode=args.init_mode,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        drift_window=args.drift_window,
        seed=args.seed,
        config=config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "unstable" and not args.allow_unstable:
        print(
            "GLE stability diagnostics flagged this run; inspect "
            f"{report['diagnostics_output']} or rerun with --allow-unstable for debugging.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
