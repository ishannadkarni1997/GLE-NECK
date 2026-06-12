#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.bulk.gle import model_config_with_overrides, run_mobility_curve
from gleneck.bulk.validation import (
    memory_validation_report,
    potential_validation_report,
    unit_validation_report,
    validation_issues,
)
from gleneck.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal physics validation checks for the bulk GLE baseline.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--private-root", type=Path, default=None)
    parser.add_argument("--init-mode", choices=("generated", "legacy"), default="generated")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--memory-orig-interval", type=float, default=None)
    parser.add_argument("--init-velocity-scale", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--zero-field-warmup-steps", type=int, default=100)
    parser.add_argument("--zero-field-steps", type=int, default=500)
    parser.add_argument("--drift-window", type=int, default=None)
    parser.add_argument("--zero-field-drift-tol", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-on-issues", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    paths = ProjectPaths.discover(args.root)
    root = paths.root
    output_dir = args.output_dir or (root / "runs" / "bulk_physics_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model_config_with_overrides(
        dt=args.dt,
        l_max=args.l_max,
        memory_orig_interval=args.memory_orig_interval,
        init_velocity_scale=args.init_velocity_scale,
        noise_scale=args.noise_scale,
    )

    memory_path = paths.processed_bulk / "fitted_memory_kernal.npy"
    potential_path = paths.processed_bulk / "cg_potentials_NVE242.npz"
    report = {
        "status": "completed",
        "elapsed_seconds": None,
        "init_mode": args.init_mode,
        "units": unit_validation_report(config),
        "memory": memory_validation_report(np.load(memory_path), config),
        "potential": potential_validation_report(potential_path, config),
    }

    zero_field_report = run_mobility_curve(
        root=root,
        fields=np.asarray([0.0], dtype=float),
        output_dir=output_dir / "zero_field",
        private_root=args.private_root,
        init_mode=args.init_mode,
        warmup_steps=args.zero_field_warmup_steps,
        steps=args.zero_field_steps,
        drift_window=args.drift_window,
        seed=args.seed,
        config=config,
    )
    zero_drift = float(zero_field_report["drifts"][0])
    report["zero_field"] = {
        "status": zero_field_report["status"],
        "diagnostic_warnings": zero_field_report["diagnostic_warnings"],
        "drift": zero_drift,
        "abs_drift": abs(zero_drift),
        "drift_tolerance": args.zero_field_drift_tol,
        "passes_drift_tolerance": bool(abs(zero_drift) <= args.zero_field_drift_tol),
        "report": zero_field_report,
    }

    issues = validation_issues(report)
    if not report["zero_field"]["passes_drift_tolerance"]:
        issues.append("zero_field_drift_above_tolerance")
    report["issues"] = issues
    report["status"] = "needs_attention" if issues else "completed"
    report["elapsed_seconds"] = time.time() - started
    report_path = output_dir / "bulk_physics_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.fail_on_issues and issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
