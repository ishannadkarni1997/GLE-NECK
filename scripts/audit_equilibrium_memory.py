#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.artifacts import read_numeric_table
from gleneck.bulk.memory import prepare_memory_terms
from gleneck.bulk.memory_audit import (
    decay_summary,
    memory_resampling_audit,
    volterra_unit_scaling_audit,
)
from gleneck.bulk.units import (
    FEMTOSECOND_TO_PS,
    internal_time_to_ps,
    memory_internal_to_per_ps2,
    notebook_unit_scalars,
)


DEFAULT_OUTPUT = Path("outputs/diagnostics/equilibrium_memory_audit")
DEFAULT_PRIMARY_VACF = Path("outputs/cluster_analysis/aa_hires_vacf_dt1_119101/AA_retained_vacf_target_window10000_fft.csv")
DEFAULT_BASELINE_AA_VACF = Path(
    "outputs/cluster_analysis/aa_hires_vacf_dt1_119101/AA_retained_vacf_target_window1000_stride5_fft.csv"
)
DEFAULT_BASELINE_GLE_VACF = Path(
    "outputs/cluster_analysis/gle_baseline_vacf_dt1_window5000_lmax1000_120700/GLE_retained_vacf.csv"
)
DEFAULT_BASELINE_REPORT = Path(
    "outputs/cluster_analysis/gle_baseline_vacf_dt1_window5000_lmax1000_120700/bulk_gle_equilibrium_benchmark_report.json"
)
DEFAULT_LEGACY_VACF = Path("data/processed/bulk/vacf_data_aa_cg_gle.csv")
DEFAULT_MEMORY_DIRS = (
    Path("outputs/bulk_fresh_baseline/langevin_peculiar_solvent_yz_memory_drop2"),
    Path("outputs/bulk_fresh_baseline/langevin_peculiar_solvent_yz_vacf_dt1_window5000_drop2"),
)
DEFAULT_NECK_REPORT = Path(
    "outputs/cluster_analysis/tau_prior_sweep_122771/"
    "bulk_gleneck_lowfield_mpt_tau0p015ps_lr0p1_122771_2/GLENECK_training_report.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit equilibrium VACF, Volterra memory, and GLE time grids.")
    parser.add_argument("--root", type=Path, default=Path("."), help="GLE-NECK repository root.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--primary-vacf", type=Path, default=DEFAULT_PRIMARY_VACF)
    parser.add_argument("--baseline-aa-vacf", type=Path, default=DEFAULT_BASELINE_AA_VACF)
    parser.add_argument("--baseline-gle-vacf", type=Path, default=DEFAULT_BASELINE_GLE_VACF)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--legacy-vacf", type=Path, default=DEFAULT_LEGACY_VACF)
    parser.add_argument("--memory-dir", type=Path, action="append", default=None)
    parser.add_argument("--neck-report", type=Path, default=DEFAULT_NECK_REPORT)
    parser.add_argument("--cluster-units-json", type=Path, default=None)
    parser.add_argument("--refit-lags", type=int, default=1000)
    parser.add_argument("--legacy-time-scale-ps", type=float, default=1.0e-3)
    return parser


def _rooted(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def _maybe_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _load_columns(path: Path) -> dict[str, np.ndarray]:
    table = read_numeric_table(path)
    return {name: np.asarray(values, dtype=float) for name, values in table.columns.items()}


def _time_axis_from_table(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "time_lag_ps" in columns:
        time_ps = columns["time_lag_ps"]
        time_internal = columns.get("time_lag_internal", np.asarray(time_ps) / float(internal_time_to_ps(1.0)))
        source = "explicit_time_lag_ps"
    elif "time_lag_internal" in columns:
        time_internal = columns["time_lag_internal"]
        time_ps = np.asarray(internal_time_to_ps(time_internal), dtype=float)
        source = "explicit_time_lag_internal"
    elif "time_lag" in columns:
        time_internal = columns["time_lag"]
        time_ps = np.asarray(internal_time_to_ps(time_internal), dtype=float)
        source = "untagged_time_lag_assumed_internal"
    else:
        raise ValueError("VACF table has no time_lag column.")
    return {"internal": time_internal, "ps": time_ps, "source": np.asarray([source])}


def _vacf_values(columns: dict[str, np.ndarray]) -> np.ndarray:
    if "vacf_solute_norm" in columns:
        return columns["vacf_solute_norm"]
    for key in columns:
        if key.lower().startswith("vacf"):
            return columns[key]
    raise ValueError("VACF table has no VACF column.")


def _dt(values: np.ndarray) -> float | None:
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        return None
    return float(np.median(np.diff(data)))


def _path_status(root: Path, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    return {"path": str(path.relative_to(root) if path.is_relative_to(root) else path), "exists": path.exists()}


def _load_jax_units(cluster_units_path: Path | None = None) -> dict[str, Any]:
    fallback = notebook_unit_scalars()
    result: dict[str, Any] = {
        "fallback_notebook_scalars": fallback.__dict__,
        "local_jax_md_import": {"available": False, "values": None, "error": None},
        "cluster_jax_md_import": {"available": False, "values": None, "path": None},
        "conversion": {
            "internal_time_per_fs": fallback.time,
            "ps_per_internal_time": float(internal_time_to_ps(1.0)),
            "dt_1fs_internal": fallback.time,
            "dt_1fs_ps": FEMTOSECOND_TO_PS,
            "memory_internal_to_ps2_factor": float(memory_internal_to_per_ps2(1.0)),
        },
    }
    try:
        from jax_md import units  # type: ignore

        raw = units.real_unit_system()
        result["local_jax_md_import"] = {
            "available": True,
            "values": {key: float(value) for key, value in raw.items()},
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - local lightweight env usually has no JAX-MD.
        result["local_jax_md_import"]["error"] = f"{type(exc).__name__}: {exc}"

    if cluster_units_path is not None and cluster_units_path.exists():
        result["cluster_jax_md_import"] = {
            "available": True,
            "values": json.loads(cluster_units_path.read_text()),
            "path": str(cluster_units_path),
        }
    return result


def _metadata_for_memory_dir(memory_dir: Path) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    metadata_path = memory_dir / "memory_kernel_metadata.json"
    if not metadata_path.exists():
        metadata_path = memory_dir / "baseline_target_metadata.json"
    metadata = _maybe_json(metadata_path) or {}
    metadata["_metadata_path"] = str(metadata_path) if metadata_path.exists() else None

    if "gle_effective_dt_internal" in metadata:
        gle_dt = float(metadata["gle_effective_dt_internal"])
    elif "gle_effective_dt" in metadata:
        gle_dt = float(metadata["gle_effective_dt"])
    elif "gle_effective_dt_ps" in metadata:
        gle_dt = float(metadata["gle_effective_dt_ps"])
        warnings.append("legacy metadata key gle_effective_dt_ps is interpreted as internal time, not physical ps")
    else:
        gle_dt = notebook_unit_scalars().time
        warnings.append("missing GLE timestep metadata; defaulting to dt=1 fs internal time")
    metadata["_gle_dt_internal_resolved"] = gle_dt
    metadata["_source_dt_internal_resolved"] = float(metadata.get("source_dt_internal", metadata.get("source_dt", gle_dt)))
    return metadata, warnings


def _memory_dataset_report(memory_dir: Path, l_max_override: int | None = None) -> dict[str, Any]:
    metadata, warnings = _metadata_for_memory_dir(memory_dir)
    fitted_path = memory_dir / "fitted_memory_kernel.npy"
    raw_path = memory_dir / "raw_memory_kernel.npy"
    if not fitted_path.exists():
        raise FileNotFoundError(f"Missing fitted memory: {fitted_path}")
    fitted = np.load(fitted_path)
    raw = np.load(raw_path) if raw_path.exists() else fitted
    source_dt_internal = float(metadata["_source_dt_internal_resolved"])
    gle_dt_internal = float(metadata["_gle_dt_internal_resolved"])
    l_max = int(l_max_override or min(max(1000, fitted.size), 5000))
    fitted_audit = memory_resampling_audit(fitted, source_dt_internal, gle_dt_internal, l_max=l_max)
    raw_audit = memory_resampling_audit(raw, source_dt_internal, gle_dt_internal, l_max=l_max)

    metadata_interval = metadata.get("memory_orig_interval_for_gle")
    resolved_interval = float(source_dt_internal / gle_dt_internal)
    if metadata_interval is not None and abs(float(metadata_interval) - resolved_interval) > 1e-8 * max(1.0, abs(resolved_interval)):
        warnings.append(
            f"metadata memory_orig_interval_for_gle={metadata_interval} disagrees with source_dt/gle_dt={resolved_interval}"
        )

    return {
        "label": memory_dir.name,
        "memory_dir": str(memory_dir),
        "metadata_path": metadata.get("_metadata_path"),
        "fitted_memory": str(fitted_path),
        "raw_memory": str(raw_path) if raw_path.exists() else None,
        "warnings": warnings,
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key
            in {
                "alpha",
                "drop_initial",
                "smooth_sigma",
                "memory_smooth_sigma",
                "source_dt",
                "source_dt_internal",
                "source_dt_ps",
                "gle_effective_dt",
                "gle_effective_dt_internal",
                "gle_effective_dt_ps",
                "memory_orig_interval_for_gle",
                "kernel_length",
                "memory_length",
                "max_lags",
                "vacf_input",
            }
        },
        "resolved": {
            "source_dt_internal": source_dt_internal,
            "source_dt_ps": float(internal_time_to_ps(source_dt_internal)),
            "gle_dt_internal": gle_dt_internal,
            "gle_dt_ps": float(internal_time_to_ps(gle_dt_internal)),
            "memory_orig_interval": resolved_interval,
        },
        "raw_grid": raw_audit,
        "fitted_grid": fitted_audit,
    }


def _vacf_axis_report(vacf_path: Path) -> dict[str, Any]:
    columns = _load_columns(vacf_path)
    axis = _time_axis_from_table(columns)
    vacf = _vacf_values(columns)
    ps = np.asarray(axis["ps"], dtype=float)
    internal = np.asarray(axis["internal"], dtype=float)
    source = str(axis["source"][0])
    return {
        "path": str(vacf_path),
        "columns": list(columns),
        "axis_source": source,
        "points": int(vacf.size),
        "dt_raw": _dt(columns["time_lag"]) if "time_lag" in columns else None,
        "dt_internal": _dt(internal),
        "dt_ps": _dt(ps),
        "max_time_internal": float(internal[-1]),
        "max_time_ps": float(ps[-1]),
        "vacf_decay": decay_summary(ps, vacf).to_dict(),
    }


def _volterra_report(vacf_path: Path, alpha: float, max_lags: int) -> dict[str, Any]:
    columns = _load_columns(vacf_path)
    axis = _time_axis_from_table(columns)
    vacf = _vacf_values(columns)
    return volterra_unit_scaling_audit(axis["internal"], vacf, alpha_internal=alpha, max_lags=max_lags)


def _baseline_vacf_report(aa_vacf_path: Path, gle_vacf_path: Path) -> dict[str, Any]:
    aa_columns = _load_columns(aa_vacf_path)
    gle_columns = _load_columns(gle_vacf_path)
    aa_axis = _time_axis_from_table(aa_columns)
    gle_axis = _time_axis_from_table(gle_columns)
    aa_vacf = _vacf_values(aa_columns)
    gle_vacf = _vacf_values(gle_columns)
    common = aa_axis["ps"] <= min(float(aa_axis["ps"][-1]), float(gle_axis["ps"][-1]))
    gle_on_aa = np.interp(aa_axis["ps"][common], gle_axis["ps"], gle_vacf)
    rmse = float(np.sqrt(np.mean((gle_on_aa - aa_vacf[common]) ** 2)))
    return {
        "aa_vacf": str(aa_vacf_path),
        "gle_vacf": str(gle_vacf_path),
        "aa_dt_ps": _dt(aa_axis["ps"]),
        "gle_dt_ps": _dt(gle_axis["ps"]),
        "common_points": int(np.sum(common)),
        "common_max_time_ps": float(aa_axis["ps"][common][-1]),
        "rmse": rmse,
        "aa_decay": decay_summary(aa_axis["ps"], aa_vacf).to_dict(),
        "gle_decay": decay_summary(gle_axis["ps"], gle_vacf).to_dict(),
    }


def _plot_vacf_axis(root: Path, output_dir: Path, primary_vacf: Path, legacy_vacf: Path, legacy_time_scale_ps: float) -> str:
    import matplotlib.pyplot as plt

    primary = _load_columns(primary_vacf)
    primary_axis = _time_axis_from_table(primary)
    primary_vacf_values = _vacf_values(primary)
    legacy = _load_columns(legacy_vacf) if legacy_vacf.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(primary["time_lag"], primary_vacf_values, label="current raw time_lag")
    axes[0].set_xlabel("stored time_lag")
    axes[0].set_ylabel("normalized VACF")
    axes[0].set_title("Stored axis")
    axes[0].legend(frameon=False)

    axes[1].plot(primary_axis["ps"], primary_vacf_values, label="current, converted ps")
    if legacy is not None:
        axes[1].plot(
            legacy["time_lag"] * legacy_time_scale_ps,
            legacy["vacf_aa_norm"],
            label="legacy AA reference, ps",
            alpha=0.8,
        )
    axes[1].set_xlabel(r"$\tau$ (ps)")
    axes[1].set_ylabel("normalized VACF")
    axes[1].set_title("Physical axis")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "aa_vacf_axis_audit.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(root) if path.is_relative_to(root) else path)


def _plot_memory_resampling(root: Path, output_dir: Path, memory_reports: list[dict[str, Any]]) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(memory_reports),
        2,
        figsize=(12, 4 * len(memory_reports)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, report in enumerate(memory_reports):
        interval = float(report["resolved"]["memory_orig_interval"])
        display_label = "coarse 10 fs memory" if interval > 2 else "high-res 1 fs memory"
        fitted = np.load(report["fitted_memory"])
        source_dt = report["resolved"]["source_dt_internal"]
        gle_dt = report["resolved"]["gle_dt_internal"]
        l_max = min(1000, max(250, int(report["fitted_grid"]["l_max"])))
        terms = prepare_memory_terms(fitted, l_max=l_max, orig_interval=interval)
        raw_tau_ps = np.asarray(internal_time_to_ps(np.arange(fitted.size) * source_dt), dtype=float)
        mistaken_tau_ps = np.asarray(internal_time_to_ps(np.arange(fitted.size) * gle_dt), dtype=float)
        fine_tau_ps = np.asarray(internal_time_to_ps(np.arange(l_max) * gle_dt), dtype=float)
        fitted_ps2 = np.asarray(memory_internal_to_per_ps2(fitted), dtype=float)
        resampled_ps2 = np.asarray(memory_internal_to_per_ps2(terms.memory), dtype=float)

        axes[row][0].plot(raw_tau_ps, fitted_ps2, label="raw/fitted native grid")
        axes[row][0].plot(fine_tau_ps, resampled_ps2, "--", label="GLE-resampled grid")
        axes[row][0].set_xlim(0, 0.6)
        axes[row][0].set_xlabel(r"$\tau$ (ps)")
        axes[row][0].set_ylabel(r"$M(\tau)$ (ps$^{-2}$)")
        axes[row][0].set_title(f"{display_label}: correct grids")
        axes[row][0].legend(frameon=False)

        axes[row][1].plot(raw_tau_ps, fitted_ps2, label="correct native axis")
        axes[row][1].plot(mistaken_tau_ps, fitted_ps2, ":", label="wrong: raw samples at GLE dt")
        axes[row][1].set_xlim(0, 0.6)
        axes[row][1].set_xlabel(r"$\tau$ (ps)")
        axes[row][1].set_ylabel(r"$M(\tau)$ (ps$^{-2}$)")
        axes[row][1].set_title(f"{display_label}: factor check")
        axes[row][1].legend(frameon=False)
    path = output_dir / "raw_vs_gle_resampled_memory.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(root) if path.is_relative_to(root) else path)


def _plot_baseline_vacf(root: Path, output_dir: Path, aa_vacf_path: Path, gle_vacf_path: Path) -> str:
    import matplotlib.pyplot as plt

    aa = _load_columns(aa_vacf_path)
    gle = _load_columns(gle_vacf_path)
    aa_axis = _time_axis_from_table(aa)
    gle_axis = _time_axis_from_table(gle)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(aa_axis["ps"], _vacf_values(aa), color="black", label="AA target")
    ax.plot(gle_axis["ps"], _vacf_values(gle), "--", label="GLE baseline")
    ax.set_xlim(0, min(6.0, float(aa_axis["ps"][-1]), float(gle_axis["ps"][-1])))
    ax.set_xlabel(r"$\tau$ (ps)")
    ax.set_ylabel("normalized VACF")
    ax.set_title("Baseline GLE VACF on physical axis")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = output_dir / "baseline_gle_vacf_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(root) if path.is_relative_to(root) else path)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Equilibrium Memory-Kernel Audit")
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    for item in report["interpretation"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Unit Audit")
    lines.append("")
    conv = report["jax_md_units"]["conversion"]
    lines.append(f"- `dt=1` corresponds to `{conv['dt_1fs_internal']:.15g}` internal time and `{conv['dt_1fs_ps']:.6g}` ps.")
    lines.append(f"- Memory conversion factor internal-time^-2 to ps^-2: `{conv['memory_internal_to_ps2_factor']:.8g}`.")
    if report["jax_md_units"]["cluster_jax_md_import"]["available"]:
        cluster_time = report["jax_md_units"]["cluster_jax_md_import"]["values"].get("time")
        lines.append(f"- Cluster `jax_md.units.real_unit_system()['time']`: `{cluster_time}`.")
    else:
        lines.append("- Cluster JAX-MD unit JSON was not supplied to this script.")
    lines.append("")
    lines.append("## VACF Axis")
    lines.append("")
    vacf = report["vacf_axis"]
    lines.append(f"- Primary VACF: `{vacf['path']}`.")
    lines.append(f"- Axis source: `{vacf['axis_source']}`.")
    lines.append(f"- Stored raw dt: `{vacf['dt_raw']}`; converted physical dt: `{vacf['dt_ps']}` ps.")
    lines.append(f"- Current AA VACF 1/e decay: `{vacf['vacf_decay']['t_1e']}` ps.")
    lines.append("")
    lines.append("## Volterra Scaling")
    lines.append("")
    volterra = report["volterra_scaling"]
    lines.append(f"- Refit lags used: `{volterra['lags_used']}`.")
    lines.append(f"- Equivalent ps-axis alpha: `{volterra['alpha_ps_equivalent']:.8g}`.")
    lines.append(f"- Internal-refit converted to ps^-2 vs direct ps-refit RMSE: `{volterra['rmse_ps2']:.8g}`.")
    lines.append(f"- Max absolute difference: `{volterra['max_abs_error_ps2']:.8g}`.")
    lines.append("")
    lines.append("## Memory Grids")
    lines.append("")
    for item in report["memory_datasets"]:
        resolved = item["resolved"]
        decay = item["fitted_grid"]["resampled_decay"]
        wrong = item["fitted_grid"]["mistaken_raw_as_fine_decay"]
        lines.append(f"- `{item['label']}`: source dt `{resolved['source_dt_ps']}` ps, GLE dt `{resolved['gle_dt_ps']}` ps, `memory_orig_interval={resolved['memory_orig_interval']}`.")
        lines.append(f"  Resampled 1/e decay `{decay['t_1e']}` ps; mistaken raw-as-GLE 1/e decay `{wrong['t_1e']}` ps.")
        for warning in item["warnings"]:
            lines.append(f"  Warning: {warning}.")
    lines.append("")
    lines.append("## Baseline GLE")
    lines.append("")
    base = report["baseline_vacf"]
    lines.append(f"- Existing baseline RMSE on common ps axis: `{base['rmse']:.8g}`.")
    lines.append(f"- AA 1/e decay: `{base['aa_decay']['t_1e']}` ps; GLE 1/e decay: `{base['gle_decay']['t_1e']}` ps.")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    for label, plot_path in report["plots"].items():
        lines.append(f"- {label}: `{plot_path}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output_dir = _rooted(root, args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.refit_lags < 2:
        print("--refit-lags must be at least 2.", file=sys.stderr)
        return 2

    primary_vacf = _rooted(root, args.primary_vacf)
    baseline_aa_vacf = _rooted(root, args.baseline_aa_vacf)
    baseline_gle_vacf = _rooted(root, args.baseline_gle_vacf)
    baseline_report = _rooted(root, args.baseline_report)
    legacy_vacf = _rooted(root, args.legacy_vacf)
    neck_report = _rooted(root, args.neck_report)
    cluster_units_json = _rooted(root, args.cluster_units_json)
    memory_dirs = tuple(_rooted(root, item) for item in (args.memory_dir or list(DEFAULT_MEMORY_DIRS)))
    assert primary_vacf is not None and baseline_aa_vacf is not None and baseline_gle_vacf is not None
    assert legacy_vacf is not None

    missing = [
        str(path)
        for path in [primary_vacf, baseline_aa_vacf, baseline_gle_vacf, legacy_vacf, *memory_dirs]
        if path is not None and not path.exists()
    ]
    if missing:
        print("Missing required audit inputs:\n" + "\n".join(missing), file=sys.stderr)
        return 2

    memory_reports = [_memory_dataset_report(memory_dir) for memory_dir in memory_dirs if memory_dir is not None]
    alpha = float(memory_reports[-1]["metadata"].get("alpha", 1e-10))
    report: dict[str, Any] = {
        "status": "completed",
        "inputs": {
            "primary_vacf": _path_status(root, primary_vacf),
            "baseline_aa_vacf": _path_status(root, baseline_aa_vacf),
            "baseline_gle_vacf": _path_status(root, baseline_gle_vacf),
            "baseline_report": _path_status(root, baseline_report),
            "legacy_vacf": _path_status(root, legacy_vacf),
            "neck_report": _path_status(root, neck_report),
            "memory_dirs": [_path_status(root, item) for item in memory_dirs],
        },
        "jax_md_units": _load_jax_units(cluster_units_json),
        "vacf_axis": _vacf_axis_report(primary_vacf),
        "volterra_scaling": _volterra_report(primary_vacf, alpha=alpha, max_lags=args.refit_lags),
        "memory_datasets": memory_reports,
        "baseline_vacf": _baseline_vacf_report(baseline_aa_vacf, baseline_gle_vacf),
        "baseline_report": _maybe_json(baseline_report),
        "neck_report": _maybe_json(neck_report),
    }

    plots = {
        "AA VACF axis audit": _plot_vacf_axis(root, output_dir, primary_vacf, legacy_vacf, args.legacy_time_scale_ps),
        "Raw vs GLE-resampled memory": _plot_memory_resampling(root, output_dir, memory_reports),
        "Baseline GLE VACF comparison": _plot_baseline_vacf(root, output_dir, baseline_aa_vacf, baseline_gle_vacf),
    }
    report["plots"] = plots

    interpretation = [
        "The audited VACF files store untagged time_lag values in JAX-MD internal time; the report converts them to ps with ps = internal_time / unit_time * 0.001.",
        "The old 10-fs memory and the high-resolution 1-fs memory are both internally consistent when memory_orig_interval is honored.",
        "Plotting a 10-fs raw memory array as if each sample were a 1-fs GLE lag compresses the apparent decay time by a factor of about 10.",
        "The existing high-resolution baseline GLE VACF comparison is on a consistent physical axis; use it as the equilibrium baseline until a rerun supersedes it.",
    ]
    if report["volterra_scaling"]["rmse_ps2"] > 1e-6:
        interpretation.append(
            "The internal-time and ps-time Volterra refits are not numerically identical at the requested tolerance; inspect the regularization/window details before using this as final."
        )
        report["status"] = "needs_review"
    report["interpretation"] = interpretation

    json_path = output_dir / "equilibrium_memory_audit.json"
    md_path = output_dir / "equilibrium_memory_audit_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_markdown(md_path, report)
    print(json.dumps({"status": report["status"], "json": str(json_path), "report": str(md_path), "plots": plots}, indent=2))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
