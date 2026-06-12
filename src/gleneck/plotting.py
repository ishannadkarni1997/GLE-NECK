from __future__ import annotations

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .artifacts import BULK_MPT_KERNEL_EVOLUTION_NPZ, read_numeric_table

PS_PER_LAG = 0.01
BULK_LEGACY_VACF_STEP_PS = 0.0204548282835039
FIGURE_DPI = 600

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "gray": "#999999",
    "black": "#000000",
}


@dataclass(frozen=True)
class FigureProvenance:
    chapter_figure: str
    content: str
    status: str
    output: str
    thesis_section: str
    paper_section: str
    plotting_function: str
    input_artifacts: tuple[str, ...]
    legacy_source: str
    notes: str = ""


def _import_pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleneck_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_thesis_style(plt)
    return plt


def _import_numpy():
    import numpy as np

    return np


def _apply_thesis_style(plt) -> None:
    """Match the publication-style rcParams used in the legacy notebooks."""
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


def _style_axis(ax) -> None:
    ax.tick_params(direction="out", width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def _ensure_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save(fig, path: Path) -> None:
    _ensure_output(path)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")


def _last_snapshot_column(columns: dict[str, list[float]]) -> str:
    snapshots = [name for name in columns if name.startswith("snapshot_step")]
    if snapshots:
        return snapshots[-1]
    iterations = [name for name in columns if name.startswith("iteration_")]
    if iterations:
        return iterations[-1]
    numeric_cols = [name for name in columns if name != "lag_index"]
    return numeric_cols[-1]


def _columns_with_prefix(columns: dict[str, list[float]], prefix: str) -> list[str]:
    return [name for name in columns if name.startswith(prefix)]


def _column_suffix_value(name: str) -> float:
    try:
        return float(name.rsplit("_", 1)[-1])
    except ValueError:
        return float("nan")


def _field_value(name: str) -> float:
    try:
        return float(name.removeprefix("E_"))
    except ValueError:
        return float("nan")


def _time_axis(values, np):
    return np.asarray(values, dtype=float) * PS_PER_LAG


def _normalizer(values, np):
    array = np.asarray(values, dtype=float)
    low = float(np.nanmin(array))
    high = float(np.nanmax(array))
    scale = high - low
    if scale == 0:
        scale = 1.0

    def normalize(data):
        return (np.asarray(data, dtype=float) - low) / scale

    return normalize


def plot_bulk_equilibrium_validation(data_dir: Path, output: Path) -> None:
    plt = _import_pyplot()
    np = _import_numpy()
    from matplotlib.lines import Line2D

    aa = read_numeric_table(data_dir / "AA_rdf_plot_data.csv")
    cg = read_numeric_table(data_dir / "CG_rdf_plot_data.csv")
    vacf = read_numeric_table(data_dir / "vacf_data_aa_cg_gle.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].plot(aa.columns["r_distance"], aa.columns["g_r_AA"], color=COLORS["blue"])
    axes[0].plot(cg.columns["r"], cg.columns["g_cg_AA"], "--", color=COLORS["blue"])
    axes[0].plot(aa.columns["r_distance"], aa.columns["g_r_BB"], color=COLORS["orange"])
    axes[0].plot(cg.columns["r"], cg.columns["g_cg_BB"], "--", color=COLORS["orange"])
    axes[0].plot(aa.columns["r_distance"], aa.columns["g_r_AB"], color=COLORS["green"])
    axes[0].plot(cg.columns["r"], cg.columns["g_cg_AB"], "--", color=COLORS["green"])
    axes[0].axhline(1.0, color="0.7", linestyle=":", linewidth=1.2)
    axes[0].set_xlabel(r"$r$ ($\AA$)")
    axes[0].set_ylabel(r"$g(r)$")
    axes[0].set_title("Bulk RDF")
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=COLORS["blue"], lw=2.5, label="A-A"),
            Line2D([0], [0], color=COLORS["orange"], lw=2.5, label="B-B"),
            Line2D([0], [0], color=COLORS["green"], lw=2.5, label="A-B"),
            Line2D([0], [0], color=COLORS["black"], lw=2.5, linestyle="-", label="All-atom"),
            Line2D([0], [0], color=COLORS["black"], lw=2.5, linestyle="--", label="Coarse-grained"),
        ],
        frameon=False,
        ncol=1,
    )

    raw = np.load(data_dir / "raw_memory_kernal.npy")
    fitted = np.load(data_dir / "fitted_memory_kernal.npy")
    n_memory = min(300, len(raw), len(fitted))
    tau_memory = np.arange(n_memory) * PS_PER_LAG
    axes[1].plot(tau_memory, raw[:n_memory], color=COLORS["gray"], alpha=0.85, label="Raw estimate")
    axes[1].plot(tau_memory, fitted[:n_memory], color=COLORS["orange"], label="Fitted kernel")
    axes[1].set_xlabel(r"$\tau$ (ps)")
    axes[1].set_ylabel(r"$M(\tau)$")
    axes[1].set_title("Memory kernel")
    axes[1].legend(frameon=False)

    tau_vacf = np.asarray(vacf.columns["time_lag"], dtype=float) * BULK_LEGACY_VACF_STEP_PS
    axes[2].plot(tau_vacf, vacf.columns["vacf_aa_norm"], color=COLORS["black"], label="All-atom")
    axes[2].plot(tau_vacf, vacf.columns["vacf_ibi_norm"], color=COLORS["blue"], linestyle="--", label="IBI-CG")
    axes[2].plot(tau_vacf, vacf.columns["vacf_gle_norm"], color=COLORS["orange"], linestyle=":", label="GLE")
    axes[2].set_xlabel(r"$\tau$ (ps)")
    axes[2].set_ylabel(r"Normalized VACF, $C(\tau)$")
    axes[2].set_title("Bulk VACF")
    axes[2].legend(frameon=False)

    for ax in axes:
        _style_axis(ax)
    fig.suptitle("Bulk Equilibrium GLE Baseline")
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def plot_bulk_single_force_training(data_dir: Path, output: Path) -> None:
    plt = _import_pyplot()
    np = _import_numpy()
    kernel = read_numeric_table(data_dir / "asym_SPT_kernel_evolutionE0p1027V0p1729.csv")
    loss = read_numeric_table(data_dir / "asym_SPT_training_loss_E0p1027V0p1729.csv")
    snapshots = _columns_with_prefix(kernel.columns, "snapshot_step_")
    final = _last_snapshot_column(kernel.columns)

    lag = _time_axis(kernel.columns["lag_index"], np)
    n_lag = min(50, len(lag))
    original_kernel = np.asarray(kernel.columns["original_kernel"], dtype=float)
    total_epochs = 150
    epoch_numbers = np.linspace(0, total_epochs, len(snapshots), dtype=int)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=total_epochs)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for index, name in enumerate(snapshots):
        corrective_kernel = np.asarray(kernel.columns[name], dtype=float)
        color = cmap(norm(epoch_numbers[index]))
        axes[0].plot(lag[:n_lag], corrective_kernel[:n_lag], color=color, alpha=1.0)
        axes[1].plot(lag[:n_lag], (original_kernel + corrective_kernel)[:n_lag], color=color, alpha=1.0)

    final_corrective_kernel = np.asarray(kernel.columns[final], dtype=float)
    final_total_kernel = original_kernel + final_corrective_kernel
    final_color = cmap(norm(total_epochs))

    axes[0].plot(lag[:n_lag], final_corrective_kernel[:n_lag], color=final_color, linewidth=3.0, label=r"Final Corrective Kernel ($\Delta M$)")
    axes[0].plot(lag[:n_lag], original_kernel[:n_lag], color=COLORS["black"], linewidth=2.5, label=r"Equilibrium Kernel ($M_{eq}$)")
    axes[0].axhline(0, color="0.5", linestyle=":", linewidth=1.5)
    axes[0].set_title("Evolution of Corrective Memory Kernel E = 0.1027")
    axes[0].set_xlabel(r"$\tau$ (ps)")
    axes[0].set_ylabel(r"$\Delta M$")

    axes[1].plot(lag[:n_lag], original_kernel[:n_lag], color=COLORS["black"], linewidth=2.5, label=r"Equilibrium Kernel ($M_{eq}$)")
    axes[1].plot(lag[:n_lag], final_total_kernel[:n_lag], color=final_color, linewidth=3.0, label=r"Final Learned Kernel ($M$)")
    axes[1].set_title("Evolution of Total Memory Kernel E = 0.1027")
    axes[1].set_xlabel(r"$\tau$ (ps)")
    axes[1].set_ylabel(r"$M(\tau)$")

    for ax in axes:
        ax.set_xlim(0, 0.5)
        ax.legend(loc="upper right", frameon=True)
        _style_axis(ax)
        inset = ax.inset_axes([0.56, 0.43, 0.36, 0.33])
        inset.plot(loss.columns["training_step"], loss.columns["loss"], color=COLORS["orange"])
        inset.set_yscale("log")
        inset.set_xlabel("Epoch", fontsize=10)
        inset.set_ylabel("Loss", fontsize=10)
        inset.tick_params(axis="both", which="major", labelsize=10)
        _style_axis(inset)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Epoch", pad=0.01)

    fig.subplots_adjust(left=0.075, right=0.93, bottom=0.15, top=0.88, wspace=0.28)
    _save(fig, output)
    plt.close(fig)


def plot_bulk_mobility(data_dir: Path, output: Path) -> None:
    plt = _import_pyplot()
    aa = read_numeric_table(data_dir / "AA_mobility_data.csv")
    gle = read_numeric_table(data_dir / "GLE_mobility_data.csv")
    mpt = read_numeric_table(data_dir / "asym_MPT_GLENECK_mobility_data_900Epoch_E0p0347_0p06883_0p1027.csv")
    spt_low = read_numeric_table(data_dir / "SPT_GLENECK_mobility_dataE0p0347V0p11145.csv")
    spt_high = read_numeric_table(data_dir / "SPT_GLENECK_mobility_dataE0p1027V0p1729.csv")

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    ax.plot(aa.columns["col0"], aa.columns["col1"], "o-", color=COLORS["black"], markersize=7, label="AA")
    ax.plot(
        gle.columns["field_strength"],
        gle.columns["drift_velocity"],
        "s--",
        color=COLORS["gray"],
        markersize=7,
        label="GLE",
    )
    ax.plot(
        mpt.columns["external_field"],
        mpt.columns["drift_velocity"],
        "^-",
        color=COLORS["blue"],
        markersize=7,
        label="GLE-NECK (MPT)",
    )
    ax.plot(
        spt_low.columns["external_field"],
        spt_low.columns["drift_velocity"],
        "D",
        color=COLORS["orange"],
        linestyle="None",
        markersize=7,
        label="GLE-NECK (SPT E=0.03)",
    )
    ax.plot(
        spt_high.columns["external_field"],
        spt_high.columns["drift_velocity"],
        "X",
        color=COLORS["green"],
        linestyle="None",
        markersize=7,
        label="GLE-NECK (SPT E=0.1)",
    )
    ax.axhline(0, color="0.5", linestyle=":", linewidth=1.5)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"External Force (kcal/mol.$\AA$)")
    ax.set_ylabel("Drift Velocity")
    ax.set_title("Comparison of Model Mobility Curves")
    ax.legend(loc="upper left", frameon=True)
    _style_axis(ax)
    fig.tight_layout(pad=1.0)
    _save(fig, output)
    plt.close(fig)


def plot_bulk_multifield_kernels(data_dir: Path, output: Path) -> None:
    plt = _import_pyplot()
    np = _import_numpy()
    evolution = np.load(data_dir / BULK_MPT_KERNEL_EVOLUTION_NPZ)
    equilibrium = np.load(data_dir / "fitted_memory_kernal.npy")
    fields = ["E_0.03", "E_0.07", "E_0.10"]
    epochs = evolution["epochs"]
    total_epochs = int(np.nanmax(epochs)) + 1
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=total_epochs)
    n_lag = min(50, len(equilibrium), evolution[fields[0]].shape[1])
    lag = np.arange(n_lag) * PS_PER_LAG

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.4))
    for ax, field in zip(axes, fields):
        kernels = evolution[field]
        for index, epoch in enumerate(epochs):
            ax.plot(lag, kernels[index, :n_lag], color=cmap(norm(epoch)), alpha=0.6, linewidth=1.0)
        ax.plot(lag, equilibrium[:n_lag], color=COLORS["black"], linewidth=1.8, label=r"Equilibrium Kernel ($M_{eq}$)")
        ax.plot(lag, kernels[-1, :n_lag], color=cmap(norm(total_epochs)), linewidth=2.2, label=r"Final Corrective Kernel ($\Delta M$)")
        ax.axhline(0, color="0.5", linestyle=":", linewidth=1.0)
        ax.set_xlim(0, 0.5)
        ax.set_title(f"Corrective Kernel Evolution for {field}")
        ax.set_xlabel(r"$\tau$ (ps)")
        ax.set_ylabel(r"$\Delta M$")
        ax.legend(loc="upper right", frameon=True, fontsize=7.5, handlelength=2.2, borderpad=0.35)
        _style_axis(ax)
        ax.title.set_fontsize(10)
        ax.xaxis.label.set_size(11)
        ax.yaxis.label.set_size(11)
        ax.tick_params(labelsize=9, width=1.0, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        colorbar = fig.colorbar(sm, ax=ax, label="Epoch", pad=0.012)
        colorbar.ax.tick_params(labelsize=8, width=1.0, length=4)
        colorbar.set_label("Epoch", fontsize=9)

    fig.subplots_adjust(left=0.055, right=0.975, bottom=0.19, top=0.88, wspace=0.28)
    _save(fig, output)
    plt.close(fig)


def plot_bulk_corrective_kernels(data_dir: Path, output: Path) -> None:
    plt = _import_pyplot()
    np = _import_numpy()
    table = read_numeric_table(data_dir / "asym_MPT_900epoch_kernel_corrections_vs_field.csv")
    fields = _columns_with_prefix(table.columns, "E_")
    field_values = [_field_value(name) for name in fields]
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(field_values), max(field_values))
    lag = _time_axis(table.columns["lag_index"], np)
    n_lag = min(33, len(lag))

    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    for name in fields:
        color = cmap(norm(_field_value(name)))
        ax.plot(lag[:n_lag], table.columns[name][:n_lag], color=color)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.5)
    ax.set_xlim(0, 0.30)
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel(r"$\Delta M(\tau)$")
    ax.set_title("Learned Corrective Kernel vs. Field Strength")
    _style_axis(ax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, label=r"External Force (kcal/mol.$\AA$)", pad=0.01)
    colorbar.set_ticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
    fig.tight_layout(pad=1.0)
    _save(fig, output)
    plt.close(fig)


def plot_bulk_friction_distributions(data_dir: Path, output: Path) -> None:
    plt = _import_pyplot()
    table = read_numeric_table(data_dir / "rerun_friction_forcevsfield_dist.csv")
    fields = [name for name in _columns_with_prefix(table.columns, "E_") if name in {"E_0.00", "E_0.03", "E_0.07", "E_0.10"}]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for index, name in enumerate(fields):
        ax.plot(table.columns["bin_centers"], table.columns[name], color=plt.get_cmap("viridis")(index / max(len(fields) - 1, 1)), label=name.replace("E_", "E = "))
    ax.set_xlabel("Friction force")
    ax.set_ylabel("Probability density")
    ax.set_title("Friction-Force Distributions")
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def plot_confinement_equilibrium_validation(data_dir: Path, output: Path) -> None:
    plt = _import_pyplot()
    np = _import_numpy()
    from matplotlib.lines import Line2D

    rdf_aa = read_numeric_table(data_dir / "allatom_rdf_smooth_AA_CC_AC.csv")
    rdf_cg = read_numeric_table(data_dir / "CG_rdf_data.csv")
    density = read_numeric_table(data_dir / "density_AA_IBICG_data.csv")
    vacf = read_numeric_table(data_dir / "vacf_comparison_AA_GLE_IBI_.csv")

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.5))
    axes[0, 0].plot(rdf_aa.columns["r"], rdf_aa.columns["gAA"], color=COLORS["blue"])
    axes[0, 0].plot(rdf_cg.columns["r"], rdf_cg.columns["g_AA_cg"], "--", color=COLORS["blue"])
    axes[0, 0].plot(rdf_aa.columns["r"], rdf_aa.columns["gCC"], color=COLORS["orange"])
    axes[0, 0].plot(rdf_cg.columns["r"], rdf_cg.columns["g_CC_cg"], "--", color=COLORS["orange"])
    axes[0, 0].plot(rdf_aa.columns["r"], rdf_aa.columns["gAC"], color=COLORS["green"])
    if "g_AC_cg" in rdf_cg.columns:
        axes[0, 0].plot(rdf_cg.columns["r"], rdf_cg.columns["g_AC_cg"], "--", color=COLORS["green"])
    axes[0, 0].axhline(1.0, color="0.7", linestyle=":", linewidth=1.2)
    axes[0, 0].set_title("Confined RDF")
    axes[0, 0].set_xlabel(r"$r$ ($\AA$)")
    axes[0, 0].set_ylabel(r"$g(r)$")
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], color=COLORS["blue"], lw=2.5, label="A-A"),
            Line2D([0], [0], color=COLORS["orange"], lw=2.5, label="C-C"),
            Line2D([0], [0], color=COLORS["green"], lw=2.5, label="A-C"),
            Line2D([0], [0], color=COLORS["black"], lw=2.5, linestyle="-", label="All-atom"),
            Line2D([0], [0], color=COLORS["black"], lw=2.5, linestyle="--", label="Coarse-grained"),
        ],
        frameon=False,
        ncol=1,
    )

    axes[0, 1].plot(density.columns["y_centers"], density.columns["rho_A_target"], color=COLORS["blue"], label="A target")
    axes[0, 1].plot(density.columns["y_centers"], density.columns["rho_C_target"], color=COLORS["orange"], label="C target")
    axes[0, 1].plot(density.columns["y_centers"], density.columns["rho_A_iter_1"], "--", color=COLORS["blue"], label="A CG")
    axes[0, 1].plot(density.columns["y_centers"], density.columns["rho_C_iter_1"], "--", color=COLORS["orange"], label="C CG")
    axes[0, 1].set_title("Density profile")
    axes[0, 1].set_xlabel("Confinement coordinate")
    axes[0, 1].set_ylabel("Number density")
    axes[0, 1].legend(frameon=False)

    tau_vacf = _time_axis(vacf.columns["lag_index"], np)
    axes[1, 0].plot(tau_vacf, vacf.columns["vacf_aa"], color=COLORS["black"], label="All-atom")
    axes[1, 0].plot(tau_vacf, vacf.columns["vacf_gle"], color=COLORS["orange"], linestyle=":", label="GLE")
    axes[1, 0].plot(tau_vacf, vacf.columns["vacf_ibi"], color=COLORS["blue"], linestyle="--", label="IBI-CG")
    axes[1, 0].set_title("Confined VACF")
    axes[1, 0].set_xlabel(r"$\tau$ (ps)")
    axes[1, 0].set_ylabel(r"Normalized VACF, $C(\tau)$")
    axes[1, 0].legend(frameon=False)

    potentials = read_numeric_table(data_dir / "allatom_boltzpotentials_AA_CC_AC.csv")
    potential_cutoff = 25.0
    shown_potentials = []
    for name, color, label in (
        ("U_AA", COLORS["blue"], "A-A"),
        ("U_CC", COLORS["orange"], "C-C"),
        ("U_AC", COLORS["green"], "A-C"),
    ):
        values = np.asarray(potentials.columns[name], dtype=float)
        shown = np.where(np.isfinite(values) & (values <= potential_cutoff), values, np.nan)
        shown_potentials.append(shown)
        axes[1, 1].plot(potentials.columns["r"], shown, color=color, label=label)
    finite_shown = np.concatenate([values[np.isfinite(values)] for values in shown_potentials])
    axes[1, 1].set_ylim(float(np.nanmin(finite_shown)) - 0.2, float(np.nanmax(finite_shown)) + 0.8)
    axes[1, 1].set_title("Boltzmann potentials")
    axes[1, 1].set_xlabel(r"$r$ ($\AA$)")
    axes[1, 1].set_ylabel(r"$U(r)$")
    axes[1, 1].legend(frameon=False)

    for ax in axes.ravel():
        _style_axis(ax)
    fig.suptitle("Confined Equilibrium Baseline")
    fig.tight_layout()
    _save(fig, output)
    plt.close(fig)


def plot_confinement_profile(data_dir: Path, output: Path, profile_file: str, loss_file: str) -> None:
    plt = _import_pyplot()
    np = _import_numpy()
    table = read_numeric_table(data_dir / profile_file)
    loss = read_numeric_table(data_dir / loss_file)
    iterations = _columns_with_prefix(table.columns, "iteration_")
    normalize = _normalizer(table.columns["v_target"], np)

    fig, (profile_ax, loss_ax) = plt.subplots(1, 2, figsize=(12.2, 3.9), gridspec_kw={"width_ratios": [1.0, 1.15]})
    cmap = plt.get_cmap("viridis", max(len(iterations), 1))
    norm = plt.Normalize(vmin=0, vmax=max(len(iterations) - 1, 1))
    for index, name in enumerate(iterations):
        profile_ax.plot(table.columns["y_grid"], normalize(table.columns[name]), color=cmap(norm(index)), alpha=0.7, linewidth=1.0)
    profile_ax.plot(
        table.columns["y_grid"],
        normalize(table.columns["v_target"]),
        color=COLORS["black"],
        linewidth=3.0,
        label="Target Profile",
    )
    profile_ax.set_xlim(min(table.columns["y_grid"]), max(table.columns["y_grid"]))
    profile_ax.set_xlabel("Normalized Position")
    profile_ax.set_ylabel("Normalized Velocity")
    profile_ax.set_title("Velocity Profile Evolution During Training")
    profile_ax.legend(loc="upper right", frameon=True, fontsize=8)
    if iterations:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("viridis"))
        sm.set_array([])
        colorbar = fig.colorbar(sm, ax=profile_ax, label="Epochs", pad=0.02)
        colorbar.ax.tick_params(labelsize=9, width=1.0, length=4)
        colorbar.set_label("Epochs", fontsize=10)

    loss_ax.plot(loss.columns["iteration"], loss.columns["loss"], color=COLORS["orange"], linewidth=2.5)
    loss_ax.set_yscale("log")
    loss_ax.set_xlabel("Epochs")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training Loss Curve")

    for ax in (profile_ax, loss_ax):
        _style_axis(ax)
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
        ax.tick_params(labelsize=10, width=1.2, length=5)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.88, wspace=0.26)
    _save(fig, output)
    plt.close(fig)


def _load_kernel_matrix(path: Path, np):
    return np.loadtxt(path, delimiter=",")


def _draw_confinement_kernel_panel(fig, ax, matrix, n_lag: int):
    shown = matrix[:, :n_lag]
    im = ax.imshow(
        shown,
        aspect="auto",
        origin="lower",
        cmap="Spectral",
        vmin=0.0,
        vmax=50000.0,
        extent=[0.0, n_lag * PS_PER_LAG, 0.0, 1.0],
    )
    ax.set_xlim(0.0, n_lag * PS_PER_LAG)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel("Normalized Position")
    ax.set_title(r"Learned Corrective Kernel ($\Delta M$)")
    _style_axis(ax)
    ax.title.set_fontsize(13)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.tick_params(labelsize=10, width=1.2, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    colorbar = fig.colorbar(im, ax=ax, label=r"$\Delta M$ Magnitude", pad=0.035)
    colorbar.set_ticks([0, 10000, 20000, 30000, 40000, 50000])
    colorbar.ax.tick_params(labelsize=10, width=1.2, length=5)
    colorbar.set_label(r"$\Delta M$ Magnitude", fontsize=11)
    return im


def plot_confinement_kernel(data_dir: Path, output: Path, source_file: str, n_lag: int = 60) -> None:
    plt = _import_pyplot()
    np = _import_numpy()
    matrix = _load_kernel_matrix(data_dir / source_file, np)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    _draw_confinement_kernel_panel(fig, ax, matrix, min(n_lag, matrix.shape[1]))
    fig.tight_layout(pad=1.0)
    _save(fig, output)
    plt.close(fig)


def plot_confinement_kernels(data_dir: Path, output: Path) -> None:
    plt = _import_pyplot()
    np = _import_numpy()
    semiplug = _load_kernel_matrix(data_dir / "E_150_semiplug_epoch0_160_learnt_kernel.csv", np)
    parabolic = _load_kernel_matrix(data_dir / "E_150_parabolic_epoch0_200_learnt_kernel.csv", np)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    _draw_confinement_kernel_panel(fig, axes[0], semiplug, min(50, semiplug.shape[1]))
    _draw_confinement_kernel_panel(fig, axes[1], parabolic, min(60, parabolic.shape[1]))
    fig.subplots_adjust(left=0.065, right=0.97, bottom=0.18, top=0.88, wspace=0.34)
    _save(fig, output)
    plt.close(fig)


def figure_provenance(output_dir: Path) -> list[FigureProvenance]:
    return [
        FigureProvenance(
            "5.1",
            "GLE-NECK delta-learning framework",
            "missing schematic asset",
            "",
            "5.1 Motivation and Chapter Overview",
            "Introduction / Method overview",
            "",
            (),
            "PPT/chapter draft only",
            "Standalone method schematic still needs to be drawn.",
        ),
        FigureProvenance(
            "5.2",
            "Neural corrective-kernel architecture with asymptotic gating",
            "missing schematic asset",
            "",
            "5.6 Asymptotic Constraints and Neural Corrective Kernel",
            "Methods",
            "",
            (),
            "PPT/chapter draft only",
            "Standalone architecture schematic still needs to be drawn.",
        ),
        FigureProvenance(
            "5.3",
            "Bulk differentiable transport-targeted training workflow",
            "missing schematic asset",
            "",
            "5.7 Differentiable GLE Integrator and 5.8 Bulk Transport",
            "Methods / Bulk results",
            "",
            (),
            "PPT/chapter draft only",
            "Workflow schematic still needs a standalone thesis-quality asset.",
        ),
        FigureProvenance(
            "5.4",
            "Bulk single-force corrective-kernel training",
            "artifact-backed",
            str(output_dir / "ch5_fig04_bulk_single_force_training.png"),
            "5.8.1 Single-force training",
            "Bulk results",
            "plot_bulk_single_force_training",
            (
                "data/processed/bulk/asym_SPT_kernel_evolutionE0p1027V0p1729.csv",
                "data/processed/bulk/asym_SPT_training_loss_E0p1027V0p1729.csv",
            ),
            "code/Bulk/clean_bulk_system.ipynb exports E0p1027V0p1729 artifacts; code/Bulk/figures.ipynb cell 4 plotting pattern",
        ),
        FigureProvenance(
            "5.5",
            "Bulk multi-force learned kernels",
            "artifact-backed",
            str(output_dir / "ch5_fig05_bulk_multifield_kernels.png"),
            "5.8.2 Multi-force training",
            "Bulk results",
            "plot_bulk_multifield_kernels",
            (
                f"data/processed/bulk/{BULK_MPT_KERNEL_EVOLUTION_NPZ}",
                "data/processed/bulk/fitted_memory_kernal.npy",
            ),
            "code/Bulk/clean_bulk_system.ipynb cells 173-178",
            "Derived NPZ is normalized from the legacy 0-700 and 700-900 epoch JAX-array pickle snapshots.",
        ),
        FigureProvenance(
            "5.6",
            "Bulk mobility response",
            "artifact-backed",
            str(output_dir / "ch5_fig06_bulk_mobility_response.png"),
            "5.8.3 Mobility response",
            "Bulk results",
            "plot_bulk_mobility",
            (
                "data/processed/bulk/AA_mobility_data.csv",
                "data/processed/bulk/GLE_mobility_data.csv",
                "data/processed/bulk/asym_MPT_GLENECK_mobility_data_900Epoch_E0p0347_0p06883_0p1027.csv",
                "data/processed/bulk/SPT_GLENECK_mobility_dataE0p0347V0p11145.csv",
                "data/processed/bulk/SPT_GLENECK_mobility_dataE0p1027V0p1729.csv",
            ),
            "code/Bulk/clean_bulk_system.ipynb cells 180-181; code/Bulk/figures.ipynb cell 6",
        ),
        FigureProvenance(
            "5.7",
            "Bulk zero-field corrective-kernel behavior",
            "artifact-backed",
            str(output_dir / "ch5_fig07_bulk_corrective_kernels.png"),
            "5.8.4 Approach to equilibrium limit",
            "Bulk results",
            "plot_bulk_corrective_kernels",
            ("data/processed/bulk/asym_MPT_900epoch_kernel_corrections_vs_field.csv",),
            "code/Bulk/clean_bulk_system.ipynb cell 177; code/Bulk/figures.ipynb cell 7",
        ),
        FigureProvenance(
            "5.8",
            "Confined differentiable transport-targeted training workflow",
            "missing schematic asset",
            "",
            "5.9 Confined Transport",
            "Methods / Confinement results",
            "",
            (),
            "PPT/chapter draft only",
            "Workflow schematic still needs a standalone thesis-quality asset.",
        ),
        FigureProvenance(
            "5.9",
            "Confined plug-like velocity profile",
            "artifact-backed",
            str(output_dir / "ch5_fig09_confinement_semiplug_profile.png"),
            "5.9.3 Plug-like and parabolic flow profiles",
            "Confinement results",
            "plot_confinement_profile",
            (
                "data/processed/confinement/epoch0_160_E150_parabolic_semiplug_profile_evolution.csv",
                "data/processed/confinement/epoch0_160_E150_semiplug_training_loss.csv",
            ),
            "code/Confinement/clean_confined_system.ipynb cells 139-143; figures/Bulk_confinement figure3 pattern",
        ),
        FigureProvenance(
            "5.10",
            "Confined parabolic velocity profile",
            "artifact-backed",
            str(output_dir / "ch5_fig10_confinement_parabolic_profile.png"),
            "5.9.3 Plug-like and parabolic flow profiles",
            "Confinement results",
            "plot_confinement_profile",
            (
                "data/processed/confinement/epoch0_200_E150_parabolic_velocity_profile_evolution.csv",
                "data/processed/confinement/epoch0_200_E150_parabolic_training_loss.csv",
            ),
            "code/Confinement/clean_confined_system.ipynb cells 139-143; code/Confinement/conf_figures.ipynb cell 3",
        ),
        FigureProvenance(
            "5.11",
            "Position-dependent corrective kernels in confinement",
            "artifact-backed",
            str(output_dir / "ch5_fig11_confinement_kernels.png"),
            "5.9.4 Position dependence of learned kernel",
            "Confinement results",
            "plot_confinement_kernels",
            (
                "data/processed/confinement/E_150_semiplug_epoch0_160_learnt_kernel.csv",
                "data/processed/confinement/E_150_parabolic_epoch0_200_learnt_kernel.csv",
            ),
            "code/Confinement/clean_confined_system.ipynb cells 147-148; code/Confinement/conf_figures.ipynb cell 4",
        ),
        FigureProvenance(
            "support-bulk",
            "Bulk RDF, memory, and VACF equilibrium validation",
            "artifact-backed support figure",
            str(output_dir / "ch5_support_bulk_equilibrium_validation.png"),
            "5.3 Constructing the Equilibrium GLE Baseline",
            "Methods / Validation",
            "plot_bulk_equilibrium_validation",
            (
                "data/processed/bulk/AA_rdf_plot_data.csv",
                "data/processed/bulk/CG_rdf_plot_data.csv",
                "data/processed/bulk/raw_memory_kernal.npy",
                "data/processed/bulk/fitted_memory_kernal.npy",
                "data/processed/bulk/vacf_data_aa_cg_gle.csv",
            ),
            "code/Bulk/figures.ipynb cells 1-3",
        ),
        FigureProvenance(
            "support-bulk-friction",
            "Bulk friction-force distributions across field",
            "artifact-backed support figure",
            str(output_dir / "ch5_support_bulk_friction_distributions.png"),
            "5.8.5 Discussion",
            "Supplementary / Discussion",
            "plot_bulk_friction_distributions",
            ("data/processed/bulk/rerun_friction_forcevsfield_dist.csv",),
            "code/Bulk/figures.ipynb later analysis cells",
        ),
        FigureProvenance(
            "support-confinement",
            "Confined RDF, density, potential, and VACF validation",
            "artifact-backed support figure",
            str(output_dir / "ch5_support_confinement_equilibrium_validation.png"),
            "5.9.1 Confined equilibrium baseline",
            "Methods / Validation",
            "plot_confinement_equilibrium_validation",
            (
                "data/processed/confinement/allatom_rdf_smooth_AA_CC_AC.csv",
                "data/processed/confinement/CG_rdf_data.csv",
                "data/processed/confinement/density_AA_IBICG_data.csv",
                "data/processed/confinement/allatom_boltzpotentials_AA_CC_AC.csv",
                "data/processed/confinement/vacf_comparison_AA_GLE_IBI_.csv",
            ),
            "code/Confinement/conf_figures.ipynb validation cells",
            "CG RDF artifact contains A-A and C-C columns; the all-atom A-C curve is shown without a CG A-C counterpart.",
        ),
    ]


def write_figure_provenance(output_dir: Path, filename: str = "figure_provenance.csv") -> Path:
    path = output_dir / filename
    _ensure_output(path)
    fieldnames = [
        "chapter_figure",
        "content",
        "status",
        "output",
        "thesis_section",
        "paper_section",
        "plotting_function",
        "input_artifacts",
        "legacy_source",
        "notes",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in figure_provenance(output_dir):
            writer.writerow(
                {
                    "chapter_figure": record.chapter_figure,
                    "content": record.content,
                    "status": record.status,
                    "output": record.output,
                    "thesis_section": record.thesis_section,
                    "paper_section": record.paper_section,
                    "plotting_function": record.plotting_function,
                    "input_artifacts": "; ".join(record.input_artifacts),
                    "legacy_source": record.legacy_source,
                    "notes": record.notes,
                }
            )
    return path


def reproduce_all_figures(bulk_dir: Path, confinement_dir: Path, output_dir: Path) -> list[Path]:
    outputs = [
        output_dir / "ch5_support_bulk_equilibrium_validation.png",
        output_dir / "ch5_fig04_bulk_single_force_training.png",
        output_dir / "ch5_fig05_bulk_multifield_kernels.png",
        output_dir / "ch5_fig06_bulk_mobility_response.png",
        output_dir / "ch5_fig07_bulk_corrective_kernels.png",
        output_dir / "ch5_support_bulk_friction_distributions.png",
        output_dir / "ch5_support_confinement_equilibrium_validation.png",
        output_dir / "ch5_fig09_confinement_semiplug_profile.png",
        output_dir / "ch5_fig10_confinement_parabolic_profile.png",
        output_dir / "ch5_fig11_confinement_kernels.png",
    ]

    plot_bulk_equilibrium_validation(bulk_dir, outputs[0])
    plot_bulk_single_force_training(bulk_dir, outputs[1])
    plot_bulk_multifield_kernels(bulk_dir, outputs[2])
    plot_bulk_mobility(bulk_dir, outputs[3])
    plot_bulk_corrective_kernels(bulk_dir, outputs[4])
    plot_bulk_friction_distributions(bulk_dir, outputs[5])
    plot_confinement_equilibrium_validation(confinement_dir, outputs[6])
    plot_confinement_profile(
        confinement_dir,
        outputs[7],
        "epoch0_160_E150_parabolic_semiplug_profile_evolution.csv",
        "epoch0_160_E150_semiplug_training_loss.csv",
    )
    plot_confinement_profile(
        confinement_dir,
        outputs[8],
        "epoch0_200_E150_parabolic_velocity_profile_evolution.csv",
        "epoch0_200_E150_parabolic_training_loss.csv",
    )
    plot_confinement_kernels(confinement_dir, outputs[9])
    return outputs
