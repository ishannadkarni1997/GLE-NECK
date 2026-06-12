#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from gleneck.bulk.gle import load_mobility_curve, model_config_with_overrides, train_gleneck_legacy_targets
from gleneck.paths import ProjectPaths


def _parse_fields(text: str) -> list[float]:
    fields = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not fields:
        raise ValueError("No fields were supplied.")
    return fields


def _parse_positive_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated float.")
    if any(value <= 0 for value in values):
        raise ValueError("All values must be positive.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train bulk GLE-NECK corrective kernels against legacy drift targets.")
    parser.add_argument("--root", type=Path, default=None, help="Project root. Defaults to auto-discovery.")
    parser.add_argument("--private-root", type=Path, default=None, help="Directory containing retained traj_cg.npy and vel_cg.npy.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated training outputs.")
    parser.add_argument("--mode", choices=("mpt", "spt-low", "spt-high", "spt-custom"), default="mpt")
    parser.add_argument("--training-target", type=Path, default=None, help="Fresh mobility CSV used to select training fields/drifts.")
    parser.add_argument("--train-fields", type=str, default=None, help="Optional comma-separated training fields to pull from --training-target.")
    parser.add_argument("--potential-path", type=Path, default=None, help="Optional CG potential NPZ. Defaults to processed legacy potential.")
    parser.add_argument("--memory-path", type=Path, default=None, help="Optional fitted memory-kernel NPY. Defaults to processed legacy kernel.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs. Defaults to a smoke-size run.")
    parser.add_argument(
        "--optimizer",
        choices=("adam", "adam-lbfgs"),
        default="adam",
        help="Optimization schedule. adam-lbfgs runs Adam for --epochs, then L-BFGS for --lbfgs-epochs.",
    )
    parser.add_argument("--lbfgs-epochs", type=int, default=0, help="L-BFGS continuation iterations after Adam warmup.")
    parser.add_argument("--lbfgs-memory-size", type=int, default=10, help="L-BFGS history size.")
    parser.add_argument(
        "--checkpoint-stride",
        type=int,
        default=0,
        help="Write training checkpoints every N optimizer iterations. Zero disables checkpointing.",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("absolute", "relative"),
        default="absolute",
        help="Train on raw drift MSE or relative drift error.",
    )
    parser.add_argument("--steps-per-loss", type=int, default=50, help="GLE steps per loss evaluation.")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Zero-field GLE warmup steps before training.")
    parser.add_argument("--drift-window", type=int, default=None, help="Number of final steps used for drift averaging.")
    parser.add_argument("--eval-fields", type=str, default=None, help="Optional comma-separated field grid for final trained-kernel evaluation.")
    parser.add_argument("--eval-target", type=Path, default=None, help="Mobility CSV whose field grid should be used for final evaluation.")
    parser.add_argument("--dt", type=float, default=None, help="Override the notebook-derived GLE timestep.")
    parser.add_argument("--l-max", type=int, default=None, help="Override the memory/history length.")
    parser.add_argument("--memory-orig-interval", type=float, default=None, help="Override memory-kernel source spacing.")
    parser.add_argument(
        "--kernel-model",
        choices=("asym_softmax_poly", "asym_exp_basis", "e2_direct", "e2_lag_embed", "e2_tau_mlp", "e2_film_tau"),
        default=None,
        help="Corrective-kernel architecture.",
    )
    parser.add_argument(
        "--kernel-field-scale",
        type=float,
        default=None,
        help="Scale raw external fields before neural kernel conditioning and the E^2 gate.",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=None,
        help="Polynomial degree for lag embedding in polynomial corrective-kernel models.",
    )
    parser.add_argument(
        "--kernel-tau-prior-ps",
        type=float,
        default=None,
        help="Optional exponential lag prior for corrective-kernel softmax weights, in ps. Zero disables it.",
    )
    parser.add_argument(
        "--kernel-exp-basis-taus-ps",
        type=str,
        default=None,
        help="Comma-separated exponential-basis timescales in ps for --kernel-model asym_exp_basis.",
    )
    parser.add_argument(
        "--kernel-shape-regularization-weight",
        type=float,
        default=None,
        help=(
            "Optional weight for a soft penalty that matches the normalized positive corrective-kernel "
            "shape to the normalized positive equilibrium memory kernel. Zero disables it."
        ),
    )
    parser.add_argument(
        "--kernel-l2-regularization-weight",
        type=float,
        default=None,
        help="Optional weight for ridge/L2 magnitude regularization on the corrective kernel. Zero disables it.",
    )
    parser.add_argument(
        "--kernel-time-moment-regularization-weight",
        type=float,
        default=None,
        help="Optional weight for tau^2-weighted finite-memory regularization on the corrective kernel. Zero disables it.",
    )
    parser.add_argument(
        "--kernel-derivative-regularization-weight",
        type=float,
        default=None,
        help="Optional weight for d(Delta M)/d tau smoothness regularization on the corrective kernel. Zero disables it.",
    )
    parser.add_argument(
        "--kernel-second-derivative-regularization-weight",
        type=float,
        default=None,
        help="Optional weight for d2(Delta M)/d tau2 curvature regularization on the corrective kernel. Zero disables it.",
    )
    parser.add_argument(
        "--memory-history-scaling",
        choices=("force-units", "legacy-training"),
        default=None,
        help="History-memory scaling convention for the GLE integrator.",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Override Adam learning rate.")
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


def _select_training_targets(mode: str, target_path: Path, train_fields_text: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    curve = load_mobility_curve(target_path)
    fields = np.asarray(curve.fields, dtype=float)
    drifts = np.asarray(curve.drifts, dtype=float)
    order = np.argsort(fields)
    fields = fields[order]
    drifts = drifts[order]
    if train_fields_text:
        requested = np.asarray(_parse_fields(train_fields_text), dtype=float)
        selected_fields = []
        selected_drifts = []
        for field in requested:
            index = int(np.argmin(np.abs(fields - field)))
            if abs(float(fields[index]) - float(field)) > 1.0e-8:
                raise ValueError(f"Requested training field {field} is not present in {target_path}.")
            selected_fields.append(float(fields[index]))
            selected_drifts.append(float(drifts[index]))
        label = "_".join(f"E{field:.6g}" for field in selected_fields).replace(".", "p")
        return np.asarray(selected_fields, dtype=float), np.asarray(selected_drifts, dtype=float), f"fresh_custom_{label}"

    nonzero = np.flatnonzero(np.abs(fields) > 1.0e-12)
    if nonzero.size == 0:
        raise ValueError(f"{target_path} has no nonzero fields for GLE-NECK training.")
    if mode == "spt-low":
        index = int(nonzero[0])
        name = f"fresh_spt_low_E{fields[index]:.6g}"
        return fields[index : index + 1], drifts[index : index + 1], name
    if mode == "spt-high":
        index = int(nonzero[-1])
        name = f"fresh_spt_high_E{fields[index]:.6g}"
        return fields[index : index + 1], drifts[index : index + 1], name
    if mode == "mpt":
        name = f"fresh_mpt_{nonzero.size}_fields"
        return fields[nonzero], drifts[nonzero], name
    if mode == "spt-custom":
        name = f"fresh_spt_custom_{nonzero.size}_fields"
        return fields[nonzero], drifts[nonzero], name
    raise ValueError(f"Unknown mode {mode!r}.")


def main() -> int:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.steps_per_loss < 1 or args.lbfgs_epochs < 0 or args.checkpoint_stride < 0:
        print(
            "--epochs and --steps-per-loss must be positive; --lbfgs-epochs and --checkpoint-stride must be non-negative.",
            file=sys.stderr,
        )
        return 2
    if args.optimizer == "adam-lbfgs" and args.lbfgs_epochs < 1:
        print("--optimizer adam-lbfgs requires --lbfgs-epochs > 0.", file=sys.stderr)
        return 2
    paths = ProjectPaths.discover(args.root)
    root = paths.root
    output_dir = args.output_dir or (root / "runs" / f"bulk_gleneck_{args.mode}")
    eval_fields = None
    if args.eval_fields:
        eval_fields = _parse_fields(args.eval_fields)
    elif args.eval_target:
        eval_fields = load_mobility_curve(args.eval_target).fields
    training_fields = None
    training_target_drifts = None
    training_name = None
    if args.training_target:
        try:
            training_fields, training_target_drifts, training_name = _select_training_targets(
                args.mode,
                args.training_target,
                args.train_fields,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    kernel_exp_basis_taus_ps = None
    if args.kernel_exp_basis_taus_ps:
        try:
            kernel_exp_basis_taus_ps = _parse_positive_floats(args.kernel_exp_basis_taus_ps)
        except ValueError as exc:
            print(f"--kernel-exp-basis-taus-ps: {exc}", file=sys.stderr)
            return 2
    config = model_config_with_overrides(
        dt=args.dt,
        l_max=args.l_max,
        memory_orig_interval=args.memory_orig_interval,
        kernel_model=args.kernel_model,
        poly_degree=args.poly_degree,
        kernel_field_scale=args.kernel_field_scale,
        kernel_tau_prior_ps=args.kernel_tau_prior_ps,
        kernel_exp_basis_taus_ps=kernel_exp_basis_taus_ps,
        kernel_shape_regularization_weight=args.kernel_shape_regularization_weight,
        kernel_l2_regularization_weight=args.kernel_l2_regularization_weight,
        kernel_time_moment_regularization_weight=args.kernel_time_moment_regularization_weight,
        kernel_derivative_regularization_weight=args.kernel_derivative_regularization_weight,
        kernel_second_derivative_regularization_weight=args.kernel_second_derivative_regularization_weight,
        memory_history_scaling=args.memory_history_scaling,
        learning_rate=args.learning_rate,
        init_velocity_scale=args.init_velocity_scale,
        noise_scale=args.noise_scale,
    )
    report = train_gleneck_legacy_targets(
        root=root,
        output_dir=output_dir,
        mode=args.mode,
        private_root=args.private_root,
        potential_path=args.potential_path,
        memory_path=args.memory_path,
        training_fields=training_fields,
        training_target_drifts=training_target_drifts,
        training_name=training_name,
        init_mode=args.init_mode,
        warmup_steps=args.warmup_steps,
        epochs=args.epochs,
        steps_per_loss=args.steps_per_loss,
        drift_window=args.drift_window,
        eval_fields=eval_fields,
        loss_weighting=args.loss_weighting,
        optimizer=args.optimizer,
        lbfgs_epochs=args.lbfgs_epochs,
        lbfgs_memory_size=args.lbfgs_memory_size,
        checkpoint_stride=args.checkpoint_stride,
        seed=args.seed,
        config=config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "unstable" and not args.allow_unstable:
        print(
            "GLE-NECK stability diagnostics flagged this run; inspect "
            f"{report['diagnostics_output']} or rerun with --allow-unstable for debugging.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
