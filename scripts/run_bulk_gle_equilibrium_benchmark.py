#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d as _scipy_gaussian_filter1d
except Exception:  # pragma: no cover - scipy is optional for the clean benchmark path.
    _scipy_gaussian_filter1d = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.artifacts import read_numeric_table
from gleneck.bulk.aa_targets import (
    compute_retained_rdfs,
    compute_vacf,
    smooth_and_tail_normalize_rdfs,
    write_rdf_csv,
    write_vacf_csv,
)
from gleneck.bulk.config import BulkAATargetConfig
from gleneck.bulk.gle import (
    diagnostic_status,
    diagnostic_warnings,
    load_or_create_potential,
    load_retained_state,
    model_config_with_overrides,
    run_gle,
    stability_diagnostics,
    warmup_retained_state,
)
from gleneck.bulk.memory import prepare_memory_terms
from gleneck.bulk.neck import require_jax_stack
from gleneck.bulk.units import effective_dt, effective_dt_ps, internal_time_to_ps, memory_internal_to_per_ps2
from gleneck.paths import ProjectPaths
from gleneck.plotting import COLORS, FIGURE_DPI, _import_pyplot, _style_axis


LEGACY_BULK_VACF_STEP_PS = 0.0204548282835039


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark zero-field bulk GLE RDF and VACF against AA/CG targets.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--private-root", type=Path, default=None, help="Directory containing retained traj_cg.npy and vel_cg.npy.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--init-mode", choices=("generated", "legacy"), default="legacy")
    parser.add_argument("--potential-path", type=Path, default=None, help="Optional CG potential NPZ. Defaults to processed legacy potential.")
    parser.add_argument("--memory-path", type=Path, default=None, help="Fitted memory-kernel NPY. Defaults to processed legacy kernel.")
    parser.add_argument("--raw-memory-path", type=Path, default=None, help="Optional raw memory-kernel NPY for plotting.")
    parser.add_argument("--target-rdf", type=Path, default=None, help="Optional AA RDF target CSV with g_r_AA/g_r_BB/g_r_AB columns.")
    parser.add_argument("--target-vacf", type=Path, default=None, help="Optional AA VACF target CSV with time_lag/vacf_solute_norm columns.")
    parser.add_argument(
        "--legacy-vacf-time-scale",
        type=float,
        default=LEGACY_BULK_VACF_STEP_PS,
        help=(
            "Scale factor converting the legacy processed bulk VACF time_lag column "
            "from notebook MD-step units to ps. Ignored when --target-vacf is supplied."
        ),
    )
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--l-max", type=int, default=3000)
    parser.add_argument("--memory-orig-interval", type=float, default=10.0)
    parser.add_argument(
        "--memory-history-scaling",
        choices=("force-units", "legacy-training"),
        default=None,
        help="History-memory scaling convention for the GLE integrator.",
    )
    parser.add_argument("--init-velocity-scale", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--vacf-window", type=int, default=1000)
    parser.add_argument(
        "--rdf-smoothing-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma, in RDF bins, used for smoothed GLE RDF plot/metrics. Use 0 for raw RDF.",
    )
    parser.add_argument(
        "--vacf-tail-samples",
        type=int,
        default=None,
        help="Use only this many sampled frames from the end of the trajectory for VACF.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _gaussian_smooth_1d(values: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    """Gaussian smoothing using the legacy SciPy path when available."""
    data = np.asarray(values, dtype=float)
    if sigma <= 0:
        return data.copy()
    if _scipy_gaussian_filter1d is not None:
        return np.asarray(_scipy_gaussian_filter1d(data, sigma=sigma), dtype=float)
    radius = int(truncate * sigma + 0.5)
    if radius < 1:
        return data.copy()
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(data, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _smooth_rdfs(rdfs: dict[str, np.ndarray], sigma: float) -> dict[str, np.ndarray]:
    return {name: _gaussian_smooth_1d(values, sigma) for name, values in rdfs.items()}


def _rmse(actual: np.ndarray, target: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(target)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((actual[mask] - target[mask]) ** 2)))


def _interp(x_source: np.ndarray, y_source: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(x_target, dtype=float), np.asarray(x_source, dtype=float), np.asarray(y_source, dtype=float))


def _load_csv_columns(path: Path) -> dict[str, np.ndarray]:
    table = read_numeric_table(path)
    return {name: np.asarray(values, dtype=float) for name, values in table.columns.items()}


def _comparison_metrics(
    root: Path,
    rdf_path: Path,
    vacf_path: Path,
    target_rdf_path: Path | None = None,
    target_vacf_path: Path | None = None,
    legacy_vacf_time_scale: float = LEGACY_BULK_VACF_STEP_PS,
) -> dict[str, object]:
    bulk_dir = root / "data" / "processed" / "bulk"
    gle_rdf = _load_csv_columns(rdf_path)
    gle_vacf = _load_csv_columns(vacf_path)

    r_gle = gle_rdf["r_distance"]
    rdf_metrics: dict[str, dict[str, float]] = {}
    if target_rdf_path is not None:
        aa_rdf = _load_csv_columns(target_rdf_path)
        r_aa = aa_rdf["r_distance"]
        for label, aa_col, gle_col in (
            ("AA", "g_r_AA", "g_r_AA"),
            ("BB", "g_r_BB", "g_r_BB"),
            ("AB", "g_r_AB", "g_r_AB"),
        ):
            gle_on_aa = _interp(r_gle, gle_rdf[gle_col], r_aa)
            rdf_metrics[label] = {"gle_vs_aa_rmse": _rmse(gle_on_aa, aa_rdf[aa_col])}
    else:
        aa_rdf = _load_csv_columns(bulk_dir / "AA_rdf_plot_data.csv")
        cg_rdf = _load_csv_columns(bulk_dir / "CG_rdf_plot_data.csv")
        r_aa = aa_rdf["r_distance"]
        r_cg = cg_rdf["r"]
        for label, aa_col, cg_col, gle_col in (
            ("AA", "g_r_AA", "g_cg_AA", "g_r_AA"),
            ("BB", "g_r_BB", "g_cg_BB", "g_r_BB"),
            ("AB", "g_r_AB", "g_cg_AB", "g_r_AB"),
        ):
            gle_on_aa = _interp(r_gle, gle_rdf[gle_col], r_aa)
            cg_on_aa = _interp(r_cg, cg_rdf[cg_col], r_aa)
            rdf_metrics[label] = {
                "gle_vs_aa_rmse": _rmse(gle_on_aa, aa_rdf[aa_col]),
                "gle_vs_cg_rmse": _rmse(_interp(r_gle, gle_rdf[gle_col], r_cg), cg_rdf[cg_col]),
                "cg_vs_aa_rmse": _rmse(cg_on_aa, aa_rdf[aa_col]),
            }

    if target_vacf_path is not None:
        target_vacf = _load_csv_columns(target_vacf_path)
        target_time = target_vacf["time_lag"]
        target_values = target_vacf["vacf_solute_norm"]
        gle_vacf_on_target = _interp(gle_vacf["time_lag"], gle_vacf["vacf_solute_norm"], target_time)
        vacf_metrics = {"fresh_gle_vs_aa_rmse": _rmse(gle_vacf_on_target, target_values)}
    else:
        target_vacf = _load_csv_columns(bulk_dir / "vacf_data_aa_cg_gle.csv")
        target_time = target_vacf["time_lag"] * legacy_vacf_time_scale
        gle_vacf_on_target = _interp(gle_vacf["time_lag"], gle_vacf["vacf_solute_norm"], target_time)
        vacf_metrics = {
            "fresh_gle_vs_aa_rmse": _rmse(gle_vacf_on_target, target_vacf["vacf_aa_norm"]),
            "fresh_gle_vs_legacy_gle_rmse": _rmse(gle_vacf_on_target, target_vacf["vacf_gle_norm"]),
            "legacy_gle_vs_aa_rmse": _rmse(target_vacf["vacf_gle_norm"], target_vacf["vacf_aa_norm"]),
            "ibi_cg_vs_aa_rmse": _rmse(target_vacf["vacf_ibi_norm"], target_vacf["vacf_aa_norm"]),
        }
    return {"rdf": rdf_metrics, "vacf": vacf_metrics}


def _plot_benchmark(
    root: Path,
    rdf_path: Path,
    vacf_path: Path,
    output_path: Path,
    memory_path: Path,
    raw_memory_path: Path | None = None,
    target_rdf_path: Path | None = None,
    target_vacf_path: Path | None = None,
    legacy_vacf_time_scale: float = LEGACY_BULK_VACF_STEP_PS,
) -> None:
    plt = _import_pyplot()
    from matplotlib.lines import Line2D

    bulk_dir = root / "data" / "processed" / "bulk"
    gle_rdf = _load_csv_columns(rdf_path)
    gle_vacf = _load_csv_columns(vacf_path)
    aa = _load_csv_columns(target_rdf_path) if target_rdf_path is not None else _load_csv_columns(bulk_dir / "AA_rdf_plot_data.csv")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    axes[0].plot(aa["r_distance"], aa["g_r_AA"], color=COLORS["blue"])
    axes[0].plot(gle_rdf["r_distance"], gle_rdf["g_r_AA"], ":", color=COLORS["blue"])
    axes[0].plot(aa["r_distance"], aa["g_r_BB"], color=COLORS["orange"])
    axes[0].plot(gle_rdf["r_distance"], gle_rdf["g_r_BB"], ":", color=COLORS["orange"])
    axes[0].plot(aa["r_distance"], aa["g_r_AB"], color=COLORS["green"])
    axes[0].plot(gle_rdf["r_distance"], gle_rdf["g_r_AB"], ":", color=COLORS["green"])
    if target_rdf_path is None:
        cg = _load_csv_columns(bulk_dir / "CG_rdf_plot_data.csv")
        axes[0].plot(cg["r"], cg["g_cg_AA"], "--", color=COLORS["blue"])
        axes[0].plot(cg["r"], cg["g_cg_BB"], "--", color=COLORS["orange"])
        axes[0].plot(cg["r"], cg["g_cg_AB"], "--", color=COLORS["green"])
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
            Line2D([0], [0], color=COLORS["black"], lw=2.5, linestyle="--", label="IBI-CG"),
            Line2D([0], [0], color=COLORS["black"], lw=2.5, linestyle=":", label="Fresh GLE"),
        ],
        frameon=False,
        ncol=1,
    )

    fitted = np.load(memory_path)
    raw = np.load(raw_memory_path) if raw_memory_path is not None and raw_memory_path.exists() else fitted
    n_memory = min(300, len(raw), len(fitted))
    memory_csv_path = memory_path.with_name("memory_kernel.csv")
    if memory_csv_path.exists():
        memory_csv = _load_csv_columns(memory_csv_path)
        if "time_lag_ps" in memory_csv:
            tau_memory = memory_csv["time_lag_ps"][:n_memory]
        elif "time_lag_internal" in memory_csv:
            tau_memory = internal_time_to_ps(memory_csv["time_lag_internal"][:n_memory])
        else:
            tau_memory = internal_time_to_ps(memory_csv["time_lag"][:n_memory])
    else:
        tau_memory = np.arange(n_memory) * effective_dt_ps(config) * config.memory_orig_interval
    axes[1].plot(tau_memory, memory_internal_to_per_ps2(raw[:n_memory]), color=COLORS["gray"], alpha=0.85, label="Raw estimate")
    axes[1].plot(tau_memory, memory_internal_to_per_ps2(fitted[:n_memory]), color=COLORS["orange"], label="Fitted kernel")
    axes[1].set_xlabel(r"$\tau$ (ps)")
    axes[1].set_ylabel(r"$M(\tau)$ (ps$^{-2}$)")
    axes[1].set_title("Memory kernel")
    axes[1].legend(frameon=False)

    if target_vacf_path is not None:
        target_vacf = _load_csv_columns(target_vacf_path)
        axes[2].plot(target_vacf["time_lag"], target_vacf["vacf_solute_norm"], color=COLORS["black"], label="All-atom target")
    else:
        target_vacf = _load_csv_columns(bulk_dir / "vacf_data_aa_cg_gle.csv")
        target_time = target_vacf["time_lag"] * legacy_vacf_time_scale
        axes[2].plot(target_time, target_vacf["vacf_aa_norm"], color=COLORS["black"], label="All-atom")
        axes[2].plot(target_time, target_vacf["vacf_ibi_norm"], color=COLORS["blue"], linestyle="--", label="IBI-CG")
        axes[2].plot(target_time, target_vacf["vacf_gle_norm"], color=COLORS["orange"], linestyle=":", alpha=0.6, label="Legacy GLE")
    axes[2].plot(gle_vacf["time_lag"], gle_vacf["vacf_solute_norm"], color=COLORS["green"], linestyle="-.", label="Fresh GLE")
    axes[2].set_xlabel(r"$\tau$ (ps)")
    axes[2].set_ylabel(r"Normalized VACF, $C(\tau)$")
    axes[2].set_title("Bulk VACF")
    axes[2].legend(frameon=False)

    for ax in axes:
        _style_axis(ax)
    fig.suptitle("Bulk Equilibrium GLE Baseline Benchmark")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    if args.steps < 1 or args.warmup_steps < 0:
        print("--steps must be positive and --warmup-steps must be non-negative.", file=sys.stderr)
        return 2
    if args.sample_stride < 1:
        print("--sample-stride must be positive.", file=sys.stderr)
        return 2
    if args.rdf_smoothing_sigma < 0:
        print("--rdf-smoothing-sigma must be non-negative.", file=sys.stderr)
        return 2

    started = time.time()
    paths = ProjectPaths.discover(args.root)
    root = paths.root
    output_dir = args.output_dir or (root / "runs" / "bulk_gle_equilibrium_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = model_config_with_overrides(
        dt=args.dt,
        l_max=args.l_max,
        memory_orig_interval=args.memory_orig_interval,
        memory_history_scaling=args.memory_history_scaling,
        init_velocity_scale=args.init_velocity_scale,
        noise_scale=args.noise_scale,
    )

    stack = require_jax_stack()
    potential, potential_path = load_or_create_potential(root, args.potential_path)
    memory_path = args.memory_path or (root / "data" / "processed" / "bulk" / "fitted_memory_kernal.npy")
    terms = prepare_memory_terms(np.load(memory_path), l_max=config.l_max, orig_interval=config.memory_orig_interval)
    (positions, velocities, history_velocities), state_source = load_retained_state(
        stack,
        root,
        args.private_root,
        args.init_mode,
        args.seed,
        config,
    )

    diagnostics = [stability_diagnostics(positions, velocities, "initial_state", config)]
    positions, velocities, history_velocities = warmup_retained_state(
        stack,
        positions,
        velocities,
        potential,
        terms.memory,
        terms.noise_filter,
        warmup_steps=args.warmup_steps,
        history_velocities=history_velocities,
        seed=args.seed + 10_000,
        config=config,
    )
    diagnostics.append(stability_diagnostics(positions, velocities, "post_warmup_state", config))

    traj, vels = run_gle(
        stack,
        positions,
        velocities,
        potential,
        terms.memory,
        terms.noise_filter,
        field=0.0,
        steps=args.steps,
        seed=args.seed + 20_000,
        history_velocities=history_velocities,
        config=config,
    )
    traj_np = np.asarray(traj, dtype=float)[:: args.sample_stride]
    vels_np = np.asarray(vels, dtype=float)[:: args.sample_stride]
    diagnostics.append(stability_diagnostics(traj_np, vels_np, "zero_field_equilibrium", config))
    vacf_vels_np = vels_np
    if args.vacf_tail_samples is not None:
        if args.vacf_tail_samples < 1:
            print("--vacf-tail-samples must be positive when provided.", file=sys.stderr)
            return 2
        vacf_vels_np = vels_np[-args.vacf_tail_samples :]

    rdf_config = BulkAATargetConfig(
        n_solvent=0,
        n_a=config.n_a,
        n_b=config.n_b,
        box_size=config.box_size,
        rdf_cutoff=10.0,
        rdf_dr=0.1,
    )
    centers, rdfs = compute_retained_rdfs(traj_np, rdf_config)
    rdf_path = write_rdf_csv(output_dir / "GLE_retained_rdf.csv", centers, rdfs)
    rdf_comparison_path = rdf_path
    smoothed_rdf_path = None
    if args.rdf_smoothing_sigma > 0:
        smoothed_rdfs = smooth_and_tail_normalize_rdfs(rdfs, sigma=args.rdf_smoothing_sigma)
        smoothed_rdf_path = write_rdf_csv(output_dir / "GLE_retained_rdf_smooth.csv", centers, smoothed_rdfs)
        rdf_comparison_path = smoothed_rdf_path
    vacf_time, vacf = compute_vacf(
        vacf_vels_np,
        dt=effective_dt(config) * args.sample_stride,
        window=args.vacf_window,
    )
    vacf_path = write_vacf_csv(output_dir / "GLE_retained_vacf.csv", vacf_time, vacf)
    metrics = _comparison_metrics(
        root,
        rdf_comparison_path,
        vacf_path,
        args.target_rdf,
        args.target_vacf,
        args.legacy_vacf_time_scale,
    )
    plot_path = output_dir / "bulk_gle_equilibrium_benchmark.png"
    _plot_benchmark(
        root,
        rdf_comparison_path,
        vacf_path,
        plot_path,
        memory_path,
        args.raw_memory_path,
        args.target_rdf,
        args.target_vacf,
        args.legacy_vacf_time_scale,
    )

    warnings = diagnostic_warnings(diagnostics)
    report = {
        "status": diagnostic_status(diagnostics),
        "elapsed_seconds": time.time() - started,
        "state_source": state_source,
        "init_mode": args.init_mode,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "sample_stride": args.sample_stride,
        "sampled_frames": int(traj_np.shape[0]),
        "vacf_window": args.vacf_window,
        "vacf_tail_samples": args.vacf_tail_samples,
        "rdf_smoothing_sigma": args.rdf_smoothing_sigma,
        "dt": config.dt,
        "effective_dt": effective_dt(config),
        "effective_dt_internal": effective_dt(config),
        "effective_dt_ps": effective_dt_ps(config),
        "vacf_lag_dt": effective_dt(config) * args.sample_stride,
        "vacf_lag_dt_ps": effective_dt_ps(config) * args.sample_stride,
        "legacy_vacf_time_scale": None if args.target_vacf is not None else args.legacy_vacf_time_scale,
        "l_max": config.l_max,
        "memory_orig_interval": config.memory_orig_interval,
        "noise_scale": config.noise_scale,
        "potential_path": str(potential_path),
        "memory_path": str(memory_path),
        "raw_memory_path": str(args.raw_memory_path) if args.raw_memory_path is not None else None,
        "target_rdf": str(args.target_rdf) if args.target_rdf is not None else None,
        "target_vacf": str(args.target_vacf) if args.target_vacf is not None else None,
        "rdf_output": str(rdf_path),
        "rdf_smoothed_output": str(smoothed_rdf_path) if smoothed_rdf_path is not None else None,
        "rdf_comparison_output": str(rdf_comparison_path),
        "vacf_output": str(vacf_path),
        "plot_output": str(plot_path),
        "metrics": metrics,
        "diagnostic_warnings": warnings,
        "diagnostics": diagnostics,
    }
    report_path = output_dir / "bulk_gle_equilibrium_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
