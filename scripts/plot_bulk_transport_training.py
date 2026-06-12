#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gleneck.bulk.units import internal_time_to_ps, memory_internal_to_per_ps2


def _load_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(f"No numeric curve rows found in {path}")
    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def _curve_arg(text: str) -> tuple[str, Path]:
    if ":" not in text:
        raise argparse.ArgumentTypeError("Curve arguments must be LABEL:PATH")
    label, path = text.split(":", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Curve label cannot be empty")
    return label.strip(), Path(path)


def _plot_mobility(ax, label: str, path: Path, *, marker: str, linestyle: str, linewidth: float) -> None:
    fields, drifts = _load_curve(path)
    ax.plot(fields, drifts, marker=marker, linestyle=linestyle, linewidth=linewidth, label=label)


def _field_slug(field: float) -> str:
    return f"E{float(field):.6g}".replace("-", "m").replace(".", "p")


def _load_report(training_dir: Path | None) -> dict:
    if training_dir is None:
        return {}
    report_path = training_dir / "GLENECK_training_report.json"
    if not report_path.exists():
        return {}
    return json.loads(report_path.read_text())


def _fmt_values(values: list[float] | tuple[float, ...] | np.ndarray, precision: int = 4) -> str:
    return ", ".join(f"{float(value):.{precision}g}" for value in values)


def _kernel_unit_label(unit: str | None) -> str:
    if unit == "ps^-2":
        return r"$\Delta M(\tau)$ (ps$^{-2}$)"
    if unit == "internal_time^-2":
        return r"$\Delta M(\tau)$ (internal time$^{-2}$)"
    return r"$\Delta M(\tau)$"


def _reported_tau_prior_ps(report: dict) -> float:
    value = float(report.get("kernel_tau_prior_ps", 0.0))
    if value <= 0.0:
        return 0.0
    if report.get("exported_kernel_unit") == "ps^-2" or report.get("effective_dt_ps") is not None:
        return value
    return float(internal_time_to_ps(value))


def _plot_kernel_history(fig, ax, training_dir: Path, kernel_field: float | None = None) -> float | None:
    import matplotlib.pyplot as plt

    report = _load_report(training_dir)
    if kernel_field is None and report.get("kernel_history_field") is not None:
        kernel_field = float(report["kernel_history_field"])
    table_path = training_dir / "corrective_kernel_evolution_legacy.csv"
    if kernel_field is not None:
        field_table_path = training_dir / f"corrective_kernel_evolution_{_field_slug(kernel_field)}_legacy.csv"
        if field_table_path.exists():
            table_path = field_table_path
    if table_path.exists():
        data = np.genfromtxt(table_path, delimiter=",", names=True)
        names = data.dtype.names or ()
        kernel_columns = [name for name in names if name.startswith("epoch_")]
        if kernel_columns:
            if "tau_internal" in names or "original_kernel_ps2" in names or report.get("effective_dt_ps") is not None:
                x = np.asarray(data["tau_ps"], dtype=float)
                value_converter = lambda values: np.asarray(values, dtype=float)
            else:
                x = np.asarray(internal_time_to_ps(data["tau_ps"]), dtype=float)
                value_converter = lambda values: np.asarray(memory_internal_to_per_ps2(values), dtype=float)
            kernel_unit = "ps^-2"
            mask = x <= 0.5
            if not np.any(mask):
                mask = np.ones_like(x, dtype=bool)
            n_lines = min(40, len(kernel_columns))
            indices = np.unique(np.linspace(0, len(kernel_columns) - 1, n_lines).astype(int))
            cmap = plt.get_cmap("viridis")
            for idx in indices:
                column = kernel_columns[idx]
                color = cmap(float(idx) / max(1, len(kernel_columns) - 1))
                ax.plot(x[mask], value_converter(data[column])[mask], color=color, alpha=0.75, linewidth=1.1)
            final_column = kernel_columns[-1]
            ax.plot(x[mask], value_converter(data[final_column])[mask], color="black", linewidth=2.0, label="final")
            if report.get("best_epoch") is not None:
                best_column = f"epoch_{int(report['best_epoch'])}"
                if best_column in names:
                    ax.plot(
                        x[mask],
                        value_converter(data[best_column])[mask],
                        color="tab:red",
                        linestyle="--",
                        linewidth=2.0,
                        label="best",
                    )
            ax.legend(frameon=False)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=max(0, len(kernel_columns) - 1)))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Epoch")
            ax.set_xlabel(r"$\tau$ (ps)")
            ax.set_ylabel(_kernel_unit_label(kernel_unit))
            return kernel_field

    kernel_path = training_dir / "corrective_kernel_history.npz"
    if kernel_path.exists():
        data = np.load(kernel_path)
        kernel_unit = "internal_time^-2"
        if "kernels_per_ps2" in data.files:
            kernels = np.asarray(data["kernels_per_ps2"], dtype=float)
            kernel_unit = "ps^-2"
        else:
            kernels = np.asarray(data["kernels"], dtype=float)
        if kernel_field is not None and "kernel_history_fields" in data.files:
            history_fields = np.asarray(data["kernel_history_fields"], dtype=float)
            field_index = int(np.argmin(np.abs(history_fields - kernel_field)))
            if abs(float(history_fields[field_index]) - float(kernel_field)) <= 1.0e-8:
                if "kernels_by_field_per_ps2" in data.files:
                    kernels = np.asarray(data["kernels_by_field_per_ps2"][:, field_index, :], dtype=float)
                    kernel_unit = "ps^-2"
                elif "kernels_by_field" in data.files:
                    kernels = np.asarray(data["kernels_by_field"][:, field_index, :], dtype=float)
                    kernel_unit = "internal_time^-2"
        epochs = np.asarray(data["epochs"], dtype=float)
        if "tau_internal" in data.files:
            tau_ps = np.asarray(data["tau_ps"], dtype=float)
        elif "tau_ps" in data.files and kernel_unit == "ps^-2":
            tau_ps = np.asarray(data["tau_ps"], dtype=float)
        elif "tau_ps" in data.files:
            tau_ps = np.asarray(internal_time_to_ps(data["tau_ps"]), dtype=float)
            kernels = np.asarray(memory_internal_to_per_ps2(kernels), dtype=float)
            kernel_unit = "ps^-2"
        else:
            tau_ps = np.arange(kernels.shape[1], dtype=float)
        if kernels.ndim == 2 and kernels.shape[0] > 0:
            mask = tau_ps <= 0.5 if "tau_ps" in data.files else np.ones_like(tau_ps, dtype=bool)
            n_lines = min(40, kernels.shape[0])
            indices = np.unique(np.linspace(0, kernels.shape[0] - 1, n_lines).astype(int))
            cmap = plt.get_cmap("viridis")
            for idx in indices:
                color = cmap(float(idx) / max(1, kernels.shape[0] - 1))
                ax.plot(tau_ps[mask], kernels[idx, mask], color=color, alpha=0.75, linewidth=1.1)
            ax.plot(tau_ps[mask], kernels[-1, mask], color="black", linewidth=2.0, label="final")
            if "best_epoch" in data.files:
                best_epoch = int(np.asarray(data["best_epoch"]).reshape(-1)[0])
                if 0 <= best_epoch < kernels.shape[0]:
                    ax.plot(
                        tau_ps[mask],
                        kernels[best_epoch, mask],
                        color="tab:red",
                        linestyle="--",
                        linewidth=2.0,
                        label="best",
                    )
            ax.legend(frameon=False)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=float(epochs[0]), vmax=float(epochs[-1])))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label="Epoch")
            ax.set_xlabel(r"$\tau$ (ps)" if "tau_ps" in data.files else "GLE lag index")
            ax.set_ylabel(_kernel_unit_label(kernel_unit))
            return kernel_field
    return kernel_field


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot fresh bulk GLE/GLE-NECK transport training diagnostics.")
    parser.add_argument("--aa-target", type=Path, required=True, help="AA mobility target CSV.")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline GLE mobility CSV.")
    parser.add_argument(
        "--trained",
        action="append",
        type=_curve_arg,
        default=[],
        help="Trained curve as LABEL:PATH. Can be supplied multiple times.",
    )
    parser.add_argument("--training-dir", type=Path, default=None, help="Optional GLE-NECK training output directory.")
    parser.add_argument("--kernel-field", type=float, default=None, help="Training field to show in the kernel panel.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_training = args.training_dir is not None
    report = _load_report(args.training_dir)
    if has_training:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
        mobility_ax, loss_ax, kernel_ax = axes
    else:
        fig, mobility_ax = plt.subplots(1, 1, figsize=(6.2, 4.3), constrained_layout=True)
        loss_ax = kernel_ax = None

    _plot_mobility(mobility_ax, "AA target", args.aa_target, marker="o", linestyle="-", linewidth=2.4)
    if args.baseline is not None:
        _plot_mobility(mobility_ax, "baseline GLE", args.baseline, marker="s", linestyle="--", linewidth=2.0)
    markers = ["^", "D", "X", "P"]
    for index, (label, path) in enumerate(args.trained):
        _plot_mobility(
            mobility_ax,
            label,
            path,
            marker=markers[index % len(markers)],
            linestyle="-.",
            linewidth=2.0,
        )
    training_fields = [float(field) for field in report.get("fields", [])]
    for field in training_fields:
        mobility_ax.axvline(field, color="0.75", linestyle=":", linewidth=1.1, zorder=0)
    mobility_ax.set_xlabel("External field")
    mobility_ax.set_ylabel("Drift velocity")
    if training_fields:
        mobility_ax.set_title(f"Mobility response\ntrain E={_fmt_values(training_fields)}")
    else:
        mobility_ax.set_title("Mobility response")
    mobility_ax.legend(frameon=False)

    if has_training and loss_ax is not None and kernel_ax is not None:
        loss_path = args.training_dir / "training_loss.csv"
        if loss_path.exists():
            loss = np.genfromtxt(loss_path, delimiter=",", names=True)
            loss_ax.semilogy(loss["epoch"], loss["loss"], color="tab:orange", linewidth=2.0)
            if report.get("best_epoch") is not None:
                best_epoch = int(report["best_epoch"])
                loss_ax.axvline(best_epoch, color="0.25", linestyle=":", linewidth=1.4, label="best")
                loss_ax.legend(frameon=False)
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel("Loss")
        loss_title = "Training loss"
        if report:
            loss_title += f"\n{report.get('mode', 'training').upper()}, loss={report.get('loss_weighting', 'n/a')}"
            if report.get("best_epoch_one_indexed") is not None:
                loss_title += f", best={int(report['best_epoch_one_indexed'])}"
        loss_ax.set_title(loss_title)

        shown_kernel_field = _plot_kernel_history(fig, kernel_ax, args.training_dir, args.kernel_field)
        if not kernel_ax.get_xlabel():
            kernel_ax.set_xlabel("GLE lag index")
        if not kernel_ax.get_ylabel():
            kernel_ax.set_ylabel(r"$\Delta M(\tau)$")
        kernel_title = "Corrective kernel"
        if shown_kernel_field is not None:
            kernel_title += f"\n" + r"$\Delta M(\tau; E=$" + f"{shown_kernel_field:g}" + r"$)$"
        if report:
            kernel_title += (
                f"\nmodel={report.get('kernel_model', 'n/a')}, "
                f"scale={float(report.get('kernel_field_scale', 1.0)):g}, "
                f"tau_prior={_reported_tau_prior_ps(report):g} ps"
            )
        kernel_ax.set_title(kernel_title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
