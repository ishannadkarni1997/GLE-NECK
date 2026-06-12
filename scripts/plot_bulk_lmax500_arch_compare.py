#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for extra in (SCRIPT_DIR, REPO_ROOT / "src"):
    text = str(extra)
    if text not in sys.path:
        sys.path.insert(0, text)

import numpy as np

from plot_bulk_kernel_training_legacy import (
    _configure_matplotlib,
    _style_axis,
    plot_mpt_legacy_style,
)


def _curve_arg(text: str) -> tuple[str, Path]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("Expected LABEL:PATH")
    label, path = text.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Curve label cannot be empty")
    return label, Path(path)


def _load_report(run_dir: Path) -> dict:
    return json.loads((run_dir / "GLENECK_training_report.json").read_text())


def _read_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names or ()
    field_key = "field" if "field" in names else names[0]
    drift_key = "drift_velocity" if "drift_velocity" in names else names[1]
    x = np.asarray(data[field_key], dtype=float)
    y = np.asarray(data[drift_key], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def _resolve_existing_path(root: Path, candidate: Path | None) -> Path | None:
    if candidate is None:
        return None
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate
    return resolved if resolved.exists() else None


def _resolve_baseline_curve(root: Path, candidate: Path | None) -> Path | None:
    resolved = _resolve_existing_path(root, candidate)
    if resolved is None:
        return None
    if resolved.name == "GLE_baseline_report.json":
        report = json.loads(resolved.read_text())
        mobility = Path(report["mobility_output"])
        return mobility
    return resolved


def _load_loss(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    diagnostics_path = run_dir / "per_field_training_diagnostics.csv"
    if diagnostics_path.exists():
        data = np.genfromtxt(diagnostics_path, delimiter=",", names=True)
        names = data.dtype.names or ()
        if "epoch" in names and "total_loss" in names:
            return np.asarray(data["epoch"], dtype=float), np.asarray(data["total_loss"], dtype=float)

    loss_path = run_dir / "training_loss.csv"
    if loss_path.exists():
        data = np.genfromtxt(loss_path, delimiter=",", names=True)
        names = data.dtype.names or ()
        if "epoch" in names and "loss" in names:
            return np.asarray(data["epoch"], dtype=float), np.asarray(data["loss"], dtype=float)

    raise FileNotFoundError(f"No loss file found in {run_dir}")


def _resolve_curve(report: dict, *, prefer_best: bool) -> Path:
    key = "best_eval_mobility_output" if prefer_best else "eval_mobility_output"
    return Path(report[key])


def _slug(label: str) -> str:
    return (
        label.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot l_max=500 bulk GLE-NECK architecture comparisons.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_curve_arg,
        help="Architecture run as LABEL:RUN_DIR. Supply three or more.",
    )
    parser.add_argument(
        "--aa-target",
        type=Path,
        default=Path(
            "outputs/bulk_fresh_baseline/langevin_peculiar_solvent_yz_memory_drop2/"
            "AA_mobility_targets_peculiar_solvent_yz_with_high_fields.csv"
        ),
        help="AA mobility target CSV.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("outputs/cluster_analysis/gle_baseline_lmax250_104141/GLE_baseline_mobility.csv"),
        help="Baseline GLE mobility CSV or GLE_baseline_report.json. Skipped if the path does not exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cluster_analysis/lmax500_arch_compare"),
        help="Directory for comparison plots.",
    )
    parser.add_argument(
        "--kernel-start-epoch",
        type=float,
        default=0.0,
        help="Minimum epoch to show in the kernel-evolution figures.",
    )
    parser.add_argument(
        "--kernel-x-max-ps",
        type=float,
        default=None,
        help="Optional maximum tau value, in ps, for kernel-evolution panels.",
    )
    parser.add_argument(
        "--use-final",
        action="store_true",
        help="Use eval_mobility_output instead of best_eval_mobility_output for the comparison figure.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plt = _configure_matplotlib()

    aa_path = (root / args.aa_target).resolve() if not args.aa_target.is_absolute() else args.aa_target
    aa_field, aa_drift = _read_curve(aa_path)
    baseline_path = _resolve_baseline_curve(root, args.baseline)
    baseline_field = baseline_drift = None
    if baseline_path is not None:
        baseline_field, baseline_drift = _read_curve(baseline_path)

    runs: list[tuple[str, Path, dict, Path]] = []
    for label, run_dir in args.run:
        resolved_dir = (root / run_dir).resolve() if not run_dir.is_absolute() else run_dir
        report = _load_report(resolved_dir)
        curve_path = _resolve_curve(report, prefer_best=not args.use_final)
        runs.append((label, resolved_dir, report, curve_path))

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    ax.plot(aa_field, aa_drift, marker="o", color="black", linewidth=2.4, label="AA target")
    if baseline_field is not None and baseline_drift is not None:
        ax.plot(
            baseline_field,
            baseline_drift,
            marker="s",
            linestyle="--",
            color="0.45",
            linewidth=2.0,
            label="baseline GLE",
        )
    markers = ["^", "D", "X", "P", "v"]
    linestyles = ["-.", "-", ":", "--", "-."]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for idx, (label, _run_dir, report, curve_path) in enumerate(runs):
        field, drift = _read_curve(curve_path)
        train_fields = ", ".join(f"{float(value):g}" for value in report.get("fields", []))
        ax.plot(
            field,
            drift,
            marker=markers[idx % len(markers)],
            linestyle=linestyles[idx % len(linestyles)],
            color=colors[idx % len(colors)],
            linewidth=2.0,
            label=f"{label} (train {train_fields})",
        )
    ax.set_xlabel("External field")
    ax.set_ylabel("Drift velocity")
    ax.set_title("Bulk MPT mobility comparison, l_max=500")
    ax.legend(frameon=False, fontsize=10)
    _style_axis(ax)
    fig.savefig(output_dir / "bulk_lmax500_mobility_compare.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.2))
    for idx, (label, run_dir, _report, _curve_path) in enumerate(runs):
        epoch, loss = _load_loss(run_dir)
        ax.semilogy(
            epoch,
            loss,
            color=colors[idx % len(colors)],
            linewidth=2.0,
            label=label,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Bulk MPT loss comparison, l_max=500")
    ax.legend(frameon=False, fontsize=10)
    _style_axis(ax)
    fig.savefig(output_dir / "bulk_lmax500_loss_compare.png", bbox_inches="tight")
    plt.close(fig)

    summary_lines = []
    for label, run_dir, report, curve_path in runs:
        slug = _slug(label)
        kernel_output = output_dir / f"bulk_lmax500_{slug}_kernel_evolution.png"
        plot_mpt_legacy_style(
            run_dir,
            kernel_output,
            field_titles=[f"{float(field):g}" for field in report.get("fields", [])],
            start_epoch=args.kernel_start_epoch,
            x_max_ps=args.kernel_x_max_ps,
        )
        summary_lines.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "curve_path": str(curve_path),
                "fields": report.get("fields", []),
                "best_loss": report.get("best_loss"),
                "kernel_figure": str(kernel_output),
            }
        )

    (output_dir / "bulk_lmax500_arch_compare_summary.json").write_text(json.dumps(summary_lines, indent=2))
    print(output_dir / "bulk_lmax500_mobility_compare.png")
    print(output_dir / "bulk_lmax500_loss_compare.png")
    for entry in summary_lines:
        print(entry["kernel_figure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
