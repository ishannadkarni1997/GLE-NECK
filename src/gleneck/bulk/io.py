from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gleneck.artifacts import _JaxArrayShimUnpickler

from .config import DEFAULT_BULK_MODEL, DEFAULT_BULK_SPECIES, DEFAULT_TRAINING_CONFIGS, BulkModelConfig
from .memory import prepare_memory_terms


@dataclass(frozen=True)
class BulkInputPaths:
    """Paths needed for an end-to-end bulk scientific rerun."""

    cg_potential: Path
    trajectory: Path
    velocity: Path
    memory_kernel: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class InputStatus:
    name: str
    path: Path
    required: bool
    exists: bool
    size_bytes: int | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def default_bulk_input_paths(root: Path, private_root: Path | None = None) -> BulkInputPaths:
    """Resolve the default local data contract for bulk reruns."""
    root = root.resolve()
    private = (private_root or root / "data" / "private" / "bulk").resolve()
    return BulkInputPaths(
        cg_potential=root / "data" / "processed" / "bulk" / "cg_potentials_NVE242.npz",
        trajectory=private / "traj_cg.npy",
        velocity=private / "vel_cg.npy",
        memory_kernel=root / "data" / "processed" / "bulk" / "fitted_memory_kernal.npy",
    )


def validate_bulk_inputs(paths: BulkInputPaths) -> list[InputStatus]:
    """Return status records for every bulk rerun input."""
    specs = (
        ("cg_potential", paths.cg_potential, True, "Portable tabulated A/B retained-solute CG potential."),
        ("trajectory", paths.trajectory, True, "Retained-solute position trajectory, expected shape (T, 242, 3)."),
        ("velocity", paths.velocity, True, "Retained-solute velocity trajectory, expected shape (T, 242, 3)."),
        ("memory_kernel", paths.memory_kernel, True, "Fitted equilibrium GLE memory kernel."),
    )
    records: list[InputStatus] = []
    for name, path, required, note in specs:
        exists = path.exists()
        records.append(
            InputStatus(
                name=name,
                path=path,
                required=required,
                exists=exists,
                size_bytes=path.stat().st_size if exists else None,
                note=note,
            )
        )
    return records


def load_jax_pickle(path: Path) -> Any:
    """Load a legacy pickle that may contain serialized JAX arrays."""
    with path.open("rb") as handle:
        return _JaxArrayShimUnpickler(handle).load()


def load_cg_potential(path: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load the two-species tabulated CG potential dictionary."""
    loaded = load_jax_pickle(path)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} should contain a dictionary keyed by species pairs.")

    required = {(0, 0), (0, 1), (1, 1)}
    missing = sorted(required.difference(loaded))
    if missing:
        raise ValueError(f"{path} is missing species-pair potentials: {missing}")

    return {key: np.asarray(loaded[key], dtype=float) for key in required}


def write_cg_potential_npz(potential: dict[tuple[int, int], np.ndarray], output: Path) -> Path:
    """Write the legacy potential dictionary as a portable NumPy archive."""
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        species_pairs=np.asarray([[0, 0], [0, 1], [1, 1]], dtype=int),
        u_aa=np.asarray(potential[(0, 0)], dtype=float),
        u_ab=np.asarray(potential[(0, 1)], dtype=float),
        u_bb=np.asarray(potential[(1, 1)], dtype=float),
    )
    return output


def load_cg_potential_npz(path: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load a portable CG potential archive written by ``write_cg_potential_npz``."""
    data = np.load(path)
    return {
        (0, 0): np.asarray(data["u_aa"], dtype=float),
        (0, 1): np.asarray(data["u_ab"], dtype=float),
        (1, 1): np.asarray(data["u_bb"], dtype=float),
    }


def summarize_bulk_inputs(paths: BulkInputPaths, model: BulkModelConfig = DEFAULT_BULK_MODEL) -> dict[str, Any]:
    """Build a machine-readable audit of available bulk rerun inputs."""
    statuses = validate_bulk_inputs(paths)
    summary: dict[str, Any] = {
        "input_status": [record.to_dict() for record in statuses],
        "species_contract": DEFAULT_BULK_SPECIES.to_dict(),
        "model_config": model.to_dict(),
        "training_configs": [config.to_dict() for config in DEFAULT_TRAINING_CONFIGS],
    }

    if paths.memory_kernel.exists():
        memory = np.load(paths.memory_kernel)
        terms = prepare_memory_terms(memory, l_max=model.l_max, orig_interval=model.memory_orig_interval)
        summary["memory_summary"] = {
            "source_shape": list(memory.shape),
            "resampled_shape": list(terms.memory.shape),
            "noise_filter_shape": list(terms.noise_filter.shape),
            "memory_min": float(np.nanmin(terms.memory)),
            "memory_max": float(np.nanmax(terms.memory)),
            "noise_filter_min": float(np.nanmin(terms.noise_filter)),
            "noise_filter_max": float(np.nanmax(terms.noise_filter)),
        }

    if paths.cg_potential.exists():
        if paths.cg_potential.suffix == ".npz":
            potential = load_cg_potential_npz(paths.cg_potential)
        else:
            potential = load_cg_potential(paths.cg_potential)
        summary["cg_potential_summary"] = {
            f"{pair[0]}-{pair[1]}": {
                "shape": list(values.shape),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
            for pair, values in sorted(potential.items())
        }

    return summary


def write_json_report(report: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return output
