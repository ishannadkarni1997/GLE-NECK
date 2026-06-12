#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np

from gleneck.bulk.units import memory_internal_to_per_ps2


FIGURE_DPI = 600
COLORS = {
    "orange": "#D55E00",
    "black": "#000000",
}


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.titlesize": 16,
            "lines.linewidth": 2.2,
            "lines.markersize": 6,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "savefig.dpi": FIGURE_DPI,
        }
    )
    return plt


def _style_axis(ax) -> None:
    ax.tick_params(direction="out", width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def _load_report(training_dir: Path) -> dict:
    return json.loads((training_dir / "GLENECK_training_report.json").read_text())


def _effective_dt_ps(report: dict) -> float:
    if report.get("effective_dt_ps") is not None:
        return float(report["effective_dt_ps"])
    config = report.get("config", {})
    if config.get("dt") is not None:
        return float(config["dt"]) * 1.0e-3
    if report.get("dt") is not None:
        return float(report["dt"]) * 1.0e-3
    return 1.0e-3


def _load_loss(training_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    loss_path = training_dir / "training_loss.csv"
    if loss_path.exists():
        data = np.genfromtxt(loss_path, delimiter=",", names=True)
        return np.asarray(data["epoch"], dtype=float), np.asarray(data["loss"], dtype=float)

    diagnostics_path = training_dir / "per_field_training_diagnostics.csv"
    if diagnostics_path.exists():
        data = np.genfromtxt(diagnostics_path, delimiter=",", names=True)
        return np.asarray(data["epoch"], dtype=float), np.asarray(data["total_loss"], dtype=float)

    raise FileNotFoundError(f"No loss artifact found in {training_dir}")


def _sorted_epoch_columns(names: tuple[str, ...]) -> list[str]:
    epoch_cols = [name for name in names if name.startswith("epoch_")]
    return sorted(epoch_cols, key=lambda name: int(name.split("_")[-1]))


def _load_csv_kernel_evolution(path: Path, report: dict) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names or ()
    epoch_cols = _sorted_epoch_columns(names)
    if not epoch_cols:
        raise ValueError(f"No epoch columns found in {path}")

    dt_ps = _effective_dt_ps(report)
    lag_index = np.asarray(data["lag_index"], dtype=float)
    tau_ps = lag_index * dt_ps

    if "original_kernel_ps2" in names:
        equilibrium = np.asarray(data["original_kernel_ps2"], dtype=float)
        histories = np.stack([np.asarray(data[name], dtype=float) for name in epoch_cols], axis=0)
    elif "original_kernel_internal" in names:
        equilibrium = np.asarray(memory_internal_to_per_ps2(data["original_kernel_internal"]), dtype=float)
        histories = np.stack(
            [np.asarray(memory_internal_to_per_ps2(data[name]), dtype=float) for name in epoch_cols],
            axis=0,
        )
    elif "original_kernel" in names:
        equilibrium = np.asarray(memory_internal_to_per_ps2(data["original_kernel"]), dtype=float)
        histories = np.stack(
            [np.asarray(memory_internal_to_per_ps2(data[name]), dtype=float) for name in epoch_cols],
            axis=0,
        )
    else:
        raise ValueError(f"Could not find an equilibrium-kernel column in {path}")

    epochs = np.asarray([int(name.split("_")[-1]) for name in epoch_cols], dtype=float)
    return {
        "tau_ps": tau_ps,
        "equilibrium_ps2": equilibrium,
        "histories_ps2": histories,
        "epochs": epochs,
    }


def _load_npz_mpt_history(training_dir: Path) -> dict[str, np.ndarray]:
    path = training_dir / "corrective_kernel_history.npz"
    data = np.load(path)
    if "kernels_by_field_per_ps2" not in data.files or "original_kernel_per_ps2" not in data.files:
        raise ValueError(f"{path} does not contain per-ps^2 kernel histories")
    return {
        "tau_ps": np.asarray(data["tau_ps"], dtype=float),
        "equilibrium_ps2": np.asarray(data["original_kernel_per_ps2"], dtype=float),
        "fields": np.asarray(data["fields"], dtype=float),
        "histories_ps2": np.asarray(data["kernels_by_field_per_ps2"], dtype=float),
        "epochs": np.asarray(data["epochs"], dtype=float),
    }


def _plot_loss_inset(ax, loss_epoch: np.ndarray, loss_value: np.ndarray, *, loc: str = "upper right") -> None:
    bounds = {
        "upper right": [0.56, 0.43, 0.36, 0.33],
        "upper center": [0.46, 0.43, 0.36, 0.33],
        "upper left": [0.12, 0.43, 0.36, 0.33],
    }[loc]
    inset = ax.inset_axes(bounds)
    inset.plot(loss_epoch, loss_value, color=COLORS["orange"])
    inset.set_yscale("log")
    inset.set_xlabel("Epoch", fontsize=10)
    inset.set_ylabel("Loss", fontsize=10)
    inset.tick_params(axis="both", which="major", labelsize=10)
    _style_axis(inset)


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=FIGURE_DPI, bbox_inches="tight")


def plot_spt_legacy_style(
    training_dir: Path,
    output: Path,
    *,
    title_field: float,
    title_label: str,
    total_epochs: int | None = None,
) -> Path:
    plt = _configure_matplotlib()
    report = _load_report(training_dir)
    loss_epoch, loss_value = _load_loss(training_dir)
    corrective = _load_csv_kernel_evolution(training_dir / "corrective_kernel_evolution_legacy.csv", report)
    total = _load_csv_kernel_evolution(training_dir / "total_kernel_evolution_legacy.csv", report)

    tau_ps = corrective["tau_ps"]
    mask = tau_ps <= min(0.5, float(tau_ps[-1]))
    epochs = corrective["epochs"]
    total_epochs = int(total_epochs if total_epochs is not None else max(float(epochs[-1]), float(report.get("epochs", epochs[-1] + 1))))
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=total_epochs)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for idx, epoch in enumerate(epochs.astype(int)):
        color = cmap(norm(epoch))
        axes[0].plot(tau_ps[mask], corrective["histories_ps2"][idx, mask], color=color, alpha=1.0)
        axes[1].plot(tau_ps[mask], total["histories_ps2"][idx, mask], color=color, alpha=1.0)

    final_color = cmap(norm(int(epochs[-1])))
    axes[0].plot(
        tau_ps[mask],
        corrective["histories_ps2"][-1, mask],
        color=final_color,
        linewidth=3.0,
        label=r"Final Corrective Kernel ($\Delta M$)",
    )
    axes[0].plot(
        tau_ps[mask],
        corrective["equilibrium_ps2"][mask],
        color=COLORS["black"],
        linewidth=2.5,
        label=r"Equilibrium Kernel ($M_{eq}$)",
    )
    axes[0].axhline(0, color="0.5", linestyle=":", linewidth=1.5)
    axes[0].set_title(f"Evolution of Corrective Memory Kernel E = {title_label}")
    axes[0].set_xlabel(r"$\tau$ (ps)")
    axes[0].set_ylabel(r"$\Delta M(\tau)$ (ps$^{-2}$)")

    axes[1].plot(
        tau_ps[mask],
        total["equilibrium_ps2"][mask],
        color=COLORS["black"],
        linewidth=2.5,
        label=r"Equilibrium Kernel ($M_{eq}$)",
    )
    axes[1].plot(
        tau_ps[mask],
        total["histories_ps2"][-1, mask],
        color=final_color,
        linewidth=3.0,
        label=r"Final Learned Kernel ($M$)",
    )
    axes[1].set_title(f"Evolution of Total Memory Kernel E = {title_label}")
    axes[1].set_xlabel(r"$\tau$ (ps)")
    axes[1].set_ylabel(r"$M(\tau)$ (ps$^{-2}$)")

    for ax in axes:
        ax.set_xlim(float(tau_ps[mask][0]), float(tau_ps[mask][-1]))
        ax.legend(loc="upper right", frameon=True)
        _style_axis(ax)
        _plot_loss_inset(ax, loss_epoch, loss_value)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Epoch", pad=0.01)

    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.15, top=0.88, wspace=0.28)
    _save(fig, output)
    plt.close(fig)
    return output


def plot_mpt_legacy_style(
    training_dir: Path,
    output: Path,
    *,
    field_titles: list[str] | None = None,
    start_epoch: float | None = None,
    x_max_ps: float | None = None,
) -> Path:
    plt = _configure_matplotlib()
    report = _load_report(training_dir)
    loss_epoch, loss_value = _load_loss(training_dir)
    history = _load_npz_mpt_history(training_dir)

    tau_ps = history["tau_ps"]
    fields = history["fields"]
    histories = history["histories_ps2"]
    equilibrium = history["equilibrium_ps2"]
    epochs = history["epochs"]
    x_limit = float(tau_ps[-1]) if x_max_ps is None else min(float(x_max_ps), float(tau_ps[-1]))
    mask = tau_ps <= x_limit
    epoch_mask = np.ones_like(epochs, dtype=bool)
    if start_epoch is not None:
        epoch_mask = epochs >= float(start_epoch)
    if not np.any(epoch_mask):
        raise ValueError(f"No MPT epochs remain after start_epoch={start_epoch}.")
    histories = histories[epoch_mask]
    epochs = epochs[epoch_mask]

    field_titles = field_titles or [f"{field:g}" for field in fields]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(epochs[0]), vmax=float(epochs[-1]))

    fig, axes = plt.subplots(1, len(fields), figsize=(4.5 * len(fields), 4.4))
    if len(fields) == 1:
        axes = [axes]

    for ax, field, title_label, kernels in zip(axes, fields, field_titles, histories.transpose(1, 0, 2)):
        for idx, epoch in enumerate(epochs):
            ax.plot(tau_ps[mask], kernels[idx, mask], color=cmap(norm(epoch)), alpha=0.78, linewidth=1.1)
        ax.plot(
            tau_ps[mask],
            equilibrium[mask],
            color=COLORS["black"],
            linewidth=2.5,
            label=r"Equilibrium Kernel ($M_{eq}$)",
        )
        ax.plot(
            tau_ps[mask],
            kernels[-1, mask],
            color=cmap(norm(float(epochs[-1]))),
            linewidth=3.0,
            label=r"Final Corrective Kernel ($\Delta M$)",
        )
        ax.axhline(0, color="0.5", linestyle=":", linewidth=1.2)
        ax.set_xlim(float(tau_ps[mask][0]), float(tau_ps[mask][-1]))
        title = f"Corrective Kernel Evolution for E = {title_label}"
        if start_epoch is not None:
            title += f"\nshown epochs >= {start_epoch:g}"
        ax.set_title(title)
        ax.set_xlabel(r"$\tau$ (ps)")
        ax.set_ylabel(r"$\Delta M(\tau)$ (ps$^{-2}$)")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        _style_axis(ax)

    _plot_loss_inset(axes[1 if len(axes) > 1 else 0], loss_epoch, loss_value, loc="upper left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(left=0.055, right=0.93, bottom=0.18, top=0.87, wspace=0.28)
    cax = fig.add_axes([0.94, 0.18, 0.015, 0.69])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Epoch")
    _save(fig, output)
    plt.close(fig)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate legacy-style bulk GLE-NECK kernel-training figures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/cluster_analysis/current_kernel_training_figures_legacy_style"),
        help="Directory for generated PNG files.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir

    spt_mid = plot_spt_legacy_style(
        root / "outputs/cluster_analysis/gleneck_scaled_kernel_tau_prior_104286_104287_104288/spt_mid",
        output_dir / "bulk_spt_mid_legacy_style.png",
        title_field=0.5,
        title_label="0.5",
        total_epochs=100,
    )
    spt_high = plot_spt_legacy_style(
        root / "outputs/cluster_analysis/gleneck_scaled_kernel_tau_prior_104286_104287_104288/spt_high",
        output_dir / "bulk_spt_high_legacy_style.png",
        title_field=3.0,
        title_label="3.0",
        total_epochs=120,
    )
    mpt_full = plot_mpt_legacy_style(
        root / "outputs/cluster_analysis/kernel_shape_regularized_123755/bulk_gleneck_kernel_shape_reg_notau_shape0p01_123755_3",
        output_dir / "bulk_mpt_lowfield_legacy_style_full_history.png",
        field_titles=["0.25", "0.6", "1.0"],
    )
    mpt = plot_mpt_legacy_style(
        root / "outputs/cluster_analysis/kernel_shape_regularized_123755/bulk_gleneck_kernel_shape_reg_notau_shape0p01_123755_3",
        output_dir / "bulk_mpt_lowfield_legacy_style.png",
        field_titles=["0.25", "0.6", "1.0"],
        start_epoch=50,
    )

    for path in (spt_mid, spt_high, mpt_full, mpt):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
