from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

from gleneck.artifacts import read_numeric_table
from gleneck.paths import ProjectPaths

FIGURE_DPI = 450
PAIR_COLORS = {
    "AA": "#0072B2",
    "BB": "#D55E00",
    "AB": "#009E73",
    "WW": "#8E8E8E",
}
MODEL_COLORS = {
    "aa": "#000000",
    "baseline": "#D55E00",
    "mpt": "#009E73",
    "spt_0.5": "#0072B2",
    "spt_1": "#CC79A7",
    "spt_2": "#E69F00",
}
REQUIRED_FIGURES = (
    "01_aa_equilibrium_targets",
    "02_gle_baseline_benchmark",
    "03_baseline_mobility",
    "04_spt_vs_mpt_loss",
    "05_spt_vs_mpt_mobility",
    "06_mpt_kernel_evolution_logtau",
    "07_field_conditioned_kernel",
)


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "gleneck_matplotlib"))
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
            "figure.titlesize": 17,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.4,
            "lines.linewidth": 2.4,
            "lines.markersize": 6,
            "savefig.dpi": FIGURE_DPI,
        }
    )
    return plt


def _array(table, column: str) -> np.ndarray:
    return np.asarray(table.columns[column], dtype=float)


def _style(ax) -> None:
    ax.tick_params(direction="out", width=1.4, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


def _read_csv(path: Path):
    return read_numeric_table(path)


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=FIGURE_DPI)
    return paths


def _load_optional_table(path: Path):
    return read_numeric_table(path) if path.exists() else None


def _plot_mobility_curve(
    ax,
    table,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str = "o",
    max_field: float | None = None,
) -> None:
    fields = _array(table, "field")
    drifts = _array(table, "drift_velocity")
    if max_field is not None:
        mask = fields <= max_field + 1.0e-12
        fields = fields[mask]
        drifts = drifts[mask]
    order = np.argsort(fields)
    ax.plot(fields[order], drifts[order], marker=marker, linestyle=linestyle, color=color, label=label)


def _plot_spt_mobility(ax, table, *, max_field: float | None = None) -> None:
    fields = _array(table, "field")
    drifts = _array(table, "drift_velocity")
    training = _array(table, "training_field")
    if max_field is not None:
        mask = fields <= max_field + 1.0e-12
        fields = fields[mask]
        drifts = drifts[mask]
        training = training[mask]
    for train_field in sorted(np.unique(training)):
        mask = np.isclose(training, train_field)
        order = np.argsort(fields[mask])
        key = f"spt_{train_field:g}"
        ax.plot(
            fields[mask][order],
            drifts[mask][order],
            linestyle="--",
            marker="^",
            color=MODEL_COLORS.get(key, None),
            label=fr"SPT $E={train_field:g}$",
        )


def _interpolate_kernels_by_field(fields: np.ndarray, kernels: np.ndarray, target_fields: np.ndarray) -> np.ndarray:
    """Linear field interpolation with the exact zero-field asymptotic kernel."""
    order = np.argsort(fields)
    source_fields = np.concatenate(([0.0], fields[order]))
    source_kernels = np.vstack([np.zeros_like(kernels[0]), kernels[order]])
    interpolated = []
    for target in target_fields:
        interpolated.append(np.array([np.interp(target, source_fields, source_kernels[:, lag]) for lag in range(source_kernels.shape[1])]))
    return np.asarray(interpolated)


def plot_aa_equilibrium_targets(data_dir: Path, output_dir: Path) -> list[Path]:
    plt = _pyplot()
    rdf = _read_csv(data_dir / "aa_equilibrium_rdf.csv")
    potentials = _read_csv(data_dir / "ibi_potentials.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    r = _array(rdf, "r")
    ax.plot(r, _array(rdf, "g_AA"), color=PAIR_COLORS["AA"], label="A-A")
    ax.plot(r, _array(rdf, "g_BB"), color=PAIR_COLORS["BB"], label="B-B")
    ax.plot(r, _array(rdf, "g_AB"), color=PAIR_COLORS["AB"], label="A-B")
    ax.plot(r, _array(rdf, "g_WW"), color=PAIR_COLORS["WW"], label="W-W")
    ax.axhline(1.0, color="0.65", linestyle=":", linewidth=1.5)
    ax.set_xlabel(r"$r$ ($\AA$)")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title("All-atom equilibrium RDFs")
    ax.set_xlim(0.0, 10.0)
    ax.legend(frameon=False, ncol=2)
    _style(ax)

    ax = axes[1]
    rp = _array(potentials, "r")
    ax.plot(rp, _array(potentials, "u_aa"), color=PAIR_COLORS["AA"], label="A-A")
    ax.plot(rp, _array(potentials, "u_bb"), color=PAIR_COLORS["BB"], label="B-B")
    ax.plot(rp, _array(potentials, "u_ab"), color=PAIR_COLORS["AB"], label="A-B")
    ax.axhline(0.0, color="0.65", linestyle=":", linewidth=1.5)
    ax.set_xlabel(r"$r$ ($\AA$)")
    ax.set_ylabel(r"$U(r)$")
    ax.set_title("IBI/PMF solute potentials")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-0.08, 0.22)
    ax.legend(frameon=False)
    _style(ax)

    fig.suptitle("Bulk Equilibrium Targets")
    fig.tight_layout()
    paths = _save(fig, output_dir, "01_aa_equilibrium_targets")
    plt.close(fig)
    return paths


def plot_gle_baseline_benchmark(data_dir: Path, output_dir: Path) -> list[Path]:
    plt = _pyplot()
    aa_rdf = _read_csv(data_dir / "aa_equilibrium_rdf.csv")
    gle_rdf = _read_csv(data_dir / "gle_baseline_rdf.csv")
    memory = _read_csv(data_dir / "memory_kernel.csv")
    aa_vacf = _read_csv(data_dir / "aa_vacf.csv")
    gle_vacf = _read_csv(data_dir / "gle_baseline_vacf.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    ax = axes[0]
    r = _array(aa_rdf, "r")
    rg = _array(gle_rdf, "r")
    for pair, aa_col, gle_col in (("A-A", "g_AA", "g_AA"), ("B-B", "g_BB", "g_BB"), ("A-B", "g_AB", "g_AB")):
        color = PAIR_COLORS[pair.replace("-", "")]
        ax.plot(r, _array(aa_rdf, aa_col), color=color, label=pair)
        ax.plot(rg, _array(gle_rdf, gle_col), color=color, linestyle=":", alpha=0.95)
    ax.plot(r, _array(aa_rdf, "g_WW"), color=PAIR_COLORS["WW"], alpha=0.8, label="W-W AA")
    ax.axhline(1.0, color="0.65", linestyle=":", linewidth=1.4)
    ax.set_xlim(0.0, 10.0)
    ax.set_xlabel(r"$r$ ($\AA$)")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title("Bulk RDF")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)

    ax = axes[1]
    tau = _array(memory, "time_ps")
    keep = tau <= 1.2
    ax.plot(tau[keep], _array(memory, "raw_memory_ps2")[keep], color="0.65", label="Raw estimate")
    ax.plot(tau[keep], _array(memory, "fitted_memory_ps2")[keep], color=MODEL_COLORS["baseline"], label="Fitted kernel")
    ax.axhline(0.0, color="0.65", linestyle=":", linewidth=1.4)
    ax.set_xlim(0.0, 1.2)
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel(r"$M(\tau)$ (ps$^{-2}$)")
    ax.set_title("Memory kernel")
    ax.legend(frameon=False)
    _style(ax)

    ax = axes[2]
    aa_tau = _array(aa_vacf, "time_ps")
    gle_tau = _array(gle_vacf, "time_ps")
    aa_keep = aa_tau <= 200.0
    gle_keep = gle_tau <= 200.0
    ax.plot(aa_tau[aa_keep], _array(aa_vacf, "vacf")[aa_keep], color=MODEL_COLORS["aa"], label="All-atom")
    ax.plot(
        gle_tau[gle_keep],
        _array(gle_vacf, "vacf")[gle_keep],
        color=PAIR_COLORS["AB"],
        linestyle="-.",
        label="GLE",
    )
    ax.set_xlim(0.0, 200.0)
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel(r"Normalized VACF, $C(\tau)$")
    ax.set_title("Bulk VACF")
    ax.legend(frameon=False)
    _style(ax)

    fig.suptitle("Bulk Equilibrium GLE Baseline")
    fig.tight_layout()
    paths = _save(fig, output_dir, "02_gle_baseline_benchmark")
    plt.close(fig)
    return paths


def plot_baseline_mobility(data_dir: Path, output_dir: Path) -> list[Path]:
    plt = _pyplot()
    aa = _read_csv(data_dir / "aa_mobility.csv")
    baseline = _read_csv(data_dir / "gle_baseline_mobility.csv")

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    _plot_mobility_curve(ax, aa, label="AA target", color=MODEL_COLORS["aa"], marker="o")
    _plot_mobility_curve(ax, baseline, label="baseline GLE", color=MODEL_COLORS["baseline"], linestyle="--", marker="s")
    ax.set_xlabel("External field")
    ax.set_ylabel("Drift velocity")
    ax.set_title("Bulk Mobility Response")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    paths = _save(fig, output_dir, "03_baseline_mobility")
    plt.close(fig)
    return paths


def plot_spt_vs_mpt_loss(data_dir: Path, output_dir: Path) -> list[Path]:
    plt = _pyplot()
    mpt = _read_csv(data_dir / "gleneck_mpt_training_loss.csv")
    spt = _load_optional_table(data_dir / "gleneck_spt_training_loss.csv")

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if spt is not None:
        epoch = _array(spt, "epoch")
        loss = _array(spt, "loss")
        training = _array(spt, "training_field")
        for train_field in sorted(np.unique(training)):
            mask = np.isclose(training, train_field)
            ax.plot(
                epoch[mask],
                np.clip(loss[mask], 1.0e-18, None),
                color=MODEL_COLORS.get(f"spt_{train_field:g}", None),
                label=fr"SPT $E={train_field:g}$",
            )
    ax.plot(_array(mpt, "epoch"), np.clip(_array(mpt, "loss"), 1.0e-18, None), color=MODEL_COLORS["mpt"], label=r"MPT $E=0.5,1.0,2.0$")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE drift loss")
    ax.set_title("Corrective-Kernel Training Loss")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    paths = _save(fig, output_dir, "04_spt_vs_mpt_loss")
    plt.close(fig)
    return paths


def plot_spt_vs_mpt_mobility(data_dir: Path, output_dir: Path) -> list[Path]:
    plt = _pyplot()
    aa = _read_csv(data_dir / "aa_mobility.csv")
    baseline = _read_csv(data_dir / "gle_baseline_mobility.csv")
    mpt = _read_csv(data_dir / "gleneck_mpt_mobility.csv")
    spt = _load_optional_table(data_dir / "gleneck_spt_mobility.csv")

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    max_field = 2.0
    _plot_mobility_curve(ax, aa, label="AA target", color=MODEL_COLORS["aa"], marker="o", max_field=max_field)
    _plot_mobility_curve(
        ax,
        baseline,
        label="baseline GLE",
        color=MODEL_COLORS["baseline"],
        linestyle="--",
        marker="s",
        max_field=max_field,
    )
    if spt is not None:
        _plot_spt_mobility(ax, spt, max_field=max_field)
    _plot_mobility_curve(
        ax,
        mpt,
        label=r"MPT $E=0.5,1.0,2.0$",
        color=MODEL_COLORS["mpt"],
        linestyle="-",
        marker="D",
        max_field=max_field,
    )
    ax.set_xlim(-0.03, 2.05)
    ax.set_xlabel("External field")
    ax.set_ylabel("Drift velocity")
    ax.set_title(r"GLE-NECK Mobility Response ($E \leq 2$)")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    paths = _save(fig, output_dir, "05_spt_vs_mpt_mobility")
    plt.close(fig)
    return paths


def plot_mpt_kernel_evolution_logtau(data_dir: Path, output_dir: Path) -> list[Path]:
    plt = _pyplot()
    data = np.load(data_dir / "gleneck_kernel_evolution.npz")
    fields = np.asarray(data["fields"], dtype=float)
    epochs = np.asarray(data["epochs"], dtype=float)
    tau = np.asarray(data["tau_ps"], dtype=float)
    kernels = np.asarray(data["kernels_by_field_ps2"], dtype=float)
    final = np.asarray(data["best_kernels_by_field_ps2"], dtype=float)
    eq = np.asarray(data["equilibrium_memory_ps2"], dtype=float)

    keep = (tau >= 1.0e-3) & (tau <= 0.3)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=float(np.nanmin(epochs)), vmax=float(np.nanmax(epochs)))
    fig, axes = plt.subplots(1, len(fields), figsize=(14.5, 4.2), sharex=True, constrained_layout=True)
    if len(fields) == 1:
        axes = [axes]
    for index, (ax, field) in enumerate(zip(axes, fields)):
        for epoch_index, epoch in enumerate(epochs):
            ax.plot(tau[keep], kernels[epoch_index, index, keep], color=cmap(norm(epoch)), alpha=0.58, linewidth=0.95, zorder=1)
        final_line = ax.plot(tau[keep], final[index, keep], color="#F0E442", linewidth=3.0, zorder=3)[0]
        eq_line = ax.plot(tau[keep], eq[keep], color="black", linewidth=3.0, zorder=5)[0]
        ax.axhline(0.0, color="0.65", linestyle=":", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_title(fr"$E={field:g}$")
        ax.set_xlabel(r"$\tau$ (ps)")
        if index == 0:
            ax.set_ylabel(r"$\Delta M(\tau)$ (ps$^{-2}$)")
        if index == len(fields) - 1:
            ax.legend([eq_line, final_line], [r"$M_{eq}$", r"final $\Delta M$"], frameon=False)
        _style(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes), label="Epoch", location="right", fraction=0.025, pad=0.02)
    fig.suptitle("MPT Corrective Kernel Evolution")
    paths = _save(fig, output_dir, "06_mpt_kernel_evolution_logtau")
    plt.close(fig)
    return paths


def plot_field_conditioned_kernel(data_dir: Path, output_dir: Path) -> list[Path]:
    plt = _pyplot()
    data = np.load(data_dir / "gleneck_kernel_evolution.npz")
    fields = np.asarray(data["fields"], dtype=float)
    tau = np.asarray(data["tau_ps"], dtype=float)
    final = np.asarray(data["best_kernels_by_field_ps2"], dtype=float)
    keep = (tau >= 1.0e-3) & (tau <= 0.3)
    target_fields = np.arange(0.0, 2.0 + 1.0e-12, 0.25)
    field_kernels = _interpolate_kernels_by_field(fields, final, target_fields)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.0, vmax=2.0)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for field, kernel in zip(target_fields, field_kernels):
        linewidth = 2.8 if np.any(np.isclose(field, [0.0, 0.5, 1.0, 2.0])) else 2.0
        ax.plot(tau[keep], kernel[keep], color=cmap(norm(field)), linewidth=linewidth)
    ax.axhline(0.0, color="0.65", linestyle=":", linewidth=1.3)
    ax.set_xscale("log")
    ax.set_xlim(1.0e-3, 0.3)
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel(r"$\Delta M(\tau)$ (ps$^{-2}$)")
    ax.set_title("Field-Conditioned Corrective Kernel")
    _style(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="External field", pad=0.02)
    fig.tight_layout()
    paths = _save(fig, output_dir, "07_field_conditioned_kernel")
    plt.close(fig)
    return paths


FIGURE_BUILDERS = (
    plot_aa_equilibrium_targets,
    plot_gle_baseline_benchmark,
    plot_baseline_mobility,
    plot_spt_vs_mpt_loss,
    plot_spt_vs_mpt_mobility,
    plot_mpt_kernel_evolution_logtau,
    plot_field_conditioned_kernel,
)


def make_all_bulk_figures(root: Path, output_dir: Path | None = None) -> list[Path]:
    paths = ProjectPaths.discover(root)
    data_dir = paths.processed_bulk
    out = output_dir or paths.bulk_figures
    generated: list[Path] = []
    for builder in FIGURE_BUILDERS:
        generated.extend(builder(data_dir, out))
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate final bulk GLE-NECK figures from processed artifacts.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root or any path inside it.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure output directory. Defaults to figures/bulk.")
    parser.add_argument("--all", action="store_true", help="Generate all public bulk figures.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.all:
        raise SystemExit("Use --all to generate the public bulk figure set.")
    generated = make_all_bulk_figures(args.root, args.output_dir)
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
