#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.bulk.gle import (
    load_mobility_curve,
    model_config_with_overrides,
    run_mobility_curve,
    train_gleneck_legacy_targets,
)
from gleneck.paths import ProjectPaths


def curve_metrics(pred_fields: np.ndarray, pred_drifts: np.ndarray, reference_path: Path, label: str) -> dict[str, float | str]:
    reference = load_mobility_curve(reference_path)
    reference_drifts = np.interp(pred_fields, reference.fields, reference.drifts)
    error = pred_drifts - reference_drifts
    return {
        "label": label,
        "reference": str(reference_path),
        "n_points": int(pred_fields.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def write_metrics_csv(path: Path, metrics: list[dict[str, float | str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "label,reference,n_points,mae,rmse,max_abs_error"
    rows = [
        f"{item['label']},{item['reference']},{item['n_points']},{item['mae']},{item['rmse']},{item['max_abs_error']}"
        for item in metrics
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark clean bulk GLE/GLE-NECK runs against legacy target artifacts.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--private-root", type=Path, default=None)
    parser.add_argument("--init-mode", choices=("generated", "legacy"), default="generated")
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--l-max", type=int, default=None)
    parser.add_argument("--memory-orig-interval", type=float, default=None)
    parser.add_argument("--init-velocity-scale", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--baseline-steps", type=int, default=200)
    parser.add_argument("--baseline-warmup-steps", type=int, default=50)
    parser.add_argument("--train-mode", choices=("mpt", "spt-low", "spt-high"), default="mpt")
    parser.add_argument("--train-epochs", type=int, default=5)
    parser.add_argument("--steps-per-loss", type=int, default=20)
    parser.add_argument("--train-warmup-steps", type=int, default=20)
    parser.add_argument("--drift-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-training", action="store_true", help="Only benchmark the baseline GLE mobility curve.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.time()
    paths = ProjectPaths.discover(args.root)
    root = paths.root
    output_dir = args.output_dir or (root / "runs" / "bulk_legacy_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = model_config_with_overrides(
        dt=args.dt,
        l_max=args.l_max,
        memory_orig_interval=args.memory_orig_interval,
        init_velocity_scale=args.init_velocity_scale,
        noise_scale=args.noise_scale,
    )

    aa_target_path = paths.processed_bulk / "AA_mobility_data.csv"
    legacy_gle_path = paths.processed_bulk / "GLE_mobility_data.csv"
    legacy_neck_path = paths.processed_bulk / "asym_MPT_GLENECK_mobility_data_900Epoch_E0p0347_0p06883_0p1027.csv"
    fields = load_mobility_curve(aa_target_path).fields

    baseline_dir = output_dir / "gle_baseline"
    baseline_report = run_mobility_curve(
        root=root,
        fields=fields,
        output_dir=baseline_dir,
        private_root=args.private_root,
        init_mode=args.init_mode,
        warmup_steps=args.baseline_warmup_steps,
        steps=args.baseline_steps,
        drift_window=args.drift_window,
        seed=args.seed,
        config=config,
    )

    metrics: list[dict[str, float | str]] = []
    baseline_curve = load_mobility_curve(Path(baseline_report["mobility_output"]))
    metrics.append(curve_metrics(baseline_curve.fields, baseline_curve.drifts, aa_target_path, "clean_gle_vs_legacy_aa"))
    metrics.append(curve_metrics(baseline_curve.fields, baseline_curve.drifts, legacy_gle_path, "clean_gle_vs_legacy_gle"))

    training_report = None
    if not args.skip_training:
        training_dir = output_dir / f"gleneck_{args.train_mode}"
        training_report = train_gleneck_legacy_targets(
            root=root,
            output_dir=training_dir,
            mode=args.train_mode,
            private_root=args.private_root,
            init_mode=args.init_mode,
            warmup_steps=args.train_warmup_steps,
            epochs=args.train_epochs,
            steps_per_loss=args.steps_per_loss,
            drift_window=args.drift_window,
            eval_fields=fields,
            seed=args.seed,
            config=config,
        )
        neck_curve = load_mobility_curve(Path(training_report["eval_mobility_output"]))
        metrics.append(curve_metrics(neck_curve.fields, neck_curve.drifts, aa_target_path, "clean_gleneck_vs_legacy_aa"))
        metrics.append(curve_metrics(neck_curve.fields, neck_curve.drifts, legacy_neck_path, "clean_gleneck_vs_legacy_gleneck"))

    metrics_path = write_metrics_csv(output_dir / "bulk_legacy_benchmark_metrics.csv", metrics)
    report = {
        "status": "completed",
        "elapsed_seconds": time.time() - started,
        "init_mode": args.init_mode,
        "baseline_report": baseline_report,
        "training_report": training_report,
        "metrics": metrics,
        "metrics_output": str(metrics_path),
    }
    report_path = output_dir / "bulk_legacy_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

    unstable = baseline_report["status"] != "completed" or (training_report is not None and training_report["status"] != "completed")
    return 1 if unstable else 0


if __name__ == "__main__":
    raise SystemExit(main())
