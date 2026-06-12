from __future__ import annotations

import csv
import math
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactSpec:
    system: str
    filename: str
    source_subdir: str
    required: bool = True
    note: str = ""


BULK_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("bulk", "AA_rdf_plot_data.csv", "Bulk"),
    ArtifactSpec("bulk", "CG_rdf_plot_data.csv", "Bulk"),
    ArtifactSpec("bulk", "vacf_data_aa_cg_gle.csv", "Bulk"),
    ArtifactSpec("bulk", "raw_memory_kernal.npy", "Bulk"),
    ArtifactSpec("bulk", "fitted_memory_kernal.npy", "Bulk"),
    ArtifactSpec("bulk", "temperature_data.csv", "Bulk"),
    ArtifactSpec("bulk", "AA_mobility_data.csv", "Bulk"),
    ArtifactSpec("bulk", "GLE_mobility_data.csv", "Bulk"),
    ArtifactSpec("bulk", "asym_MPT_GLENECK_mobility_data_900Epoch_E0p0347_0p06883_0p1027.csv", "Bulk"),
    ArtifactSpec("bulk", "SPT_GLENECK_mobility_dataE0p0347V0p11145.csv", "Bulk"),
    ArtifactSpec("bulk", "SPT_GLENECK_mobility_dataE0p1027V0p1729.csv", "Bulk"),
    ArtifactSpec("bulk", "asym_SPT_kernel_evolutionE0p0347V0p11145.csv", "Bulk"),
    ArtifactSpec("bulk", "asym_SPT_training_loss_E0p0347V0p11145.csv", "Bulk"),
    ArtifactSpec("bulk", "asym_SPT_kernel_evolutionE0p1027V0p1729.csv", "Bulk"),
    ArtifactSpec("bulk", "asym_SPT_training_loss_E0p1027V0p1729.csv", "Bulk"),
    ArtifactSpec("bulk", "asym_MPT_900epoch_kernel_corrections_vs_field.csv", "Bulk"),
    ArtifactSpec("bulk", "rerun_friction_forcevsfield_dist.csv", "Bulk"),
)


CONFINEMENT_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("confinement", "allatom_rdf_smooth_AA_CC_AC.csv", "Confinement"),
    ArtifactSpec("confinement", "CG_rdf_data.csv", "Confinement"),
    ArtifactSpec("confinement", "density_AA_IBICG_data.csv", "Confinement"),
    ArtifactSpec("confinement", "allatom_boltzpotentials_AA_CC_AC.csv", "Confinement"),
    ArtifactSpec("confinement", "vacf_comparison_AA_GLE_IBI_.csv", "Confinement"),
    ArtifactSpec("confinement", "M_fitted.npy", "Confinement"),
    ArtifactSpec("confinement", "temp_GLE.csv", "Confinement"),
    ArtifactSpec("confinement", "epoch0_160_E150_parabolic_semiplug_profile_evolution.csv", "Confinement"),
    ArtifactSpec("confinement", "epoch0_160_E150_semiplug_training_loss.csv", "Confinement"),
    ArtifactSpec("confinement", "semi_plug_velocity_profile_evolution.csv", "Confinement"),
    ArtifactSpec("confinement", "semi_plug_training_loss.csv", "Confinement"),
    ArtifactSpec("confinement", "semi_plug_learnt_kernel.csv", "Confinement"),
    ArtifactSpec("confinement", "epoch0_200_E150_parabolic_velocity_profile_evolution.csv", "Confinement"),
    ArtifactSpec("confinement", "epoch0_200_E150_parabolic_training_loss.csv", "Confinement"),
    ArtifactSpec("confinement", "E_150_parabolic_epoch0_200_learnt_kernel.csv", "Confinement"),
    ArtifactSpec("confinement", "E_150_semiplug_epoch0_160_learnt_kernel.csv", "Confinement"),
)


ALL_ARTIFACTS: tuple[ArtifactSpec, ...] = BULK_ARTIFACTS + CONFINEMENT_ARTIFACTS


BULK_MPT_KERNEL_EVOLUTION_NPZ = "asym_MPT_kernel_evolution_900epochs.npz"
BULK_MPT_KERNEL_EVOLUTION_SOURCES: tuple[str, ...] = (
    "asym_MPT_epoch0_700multi_kernel_evolution_E_0p0347_0p06883_0p1027.pkl",
    "asym_MPT_epoch700_900multi_kernel_evolution_E_0p0347_0p06883_0p1027.pkl",
)


@dataclass
class NumericTable:
    path: Path
    columns: dict[str, list[float]]

    def require(self, *names: str) -> None:
        missing = [name for name in names if name not in self.columns]
        if missing:
            raise ValueError(f"{self.path} is missing required columns: {', '.join(missing)}")

    def first_matching(self, prefix: str) -> str:
        for name in self.columns:
            if name.startswith(prefix):
                return name
        raise ValueError(f"{self.path} has no column starting with {prefix!r}.")


def _as_float(value: str) -> float:
    try:
        return float(value.strip())
    except ValueError:
        return math.nan


def _is_float(value: str) -> bool:
    try:
        float(value.strip())
        return True
    except ValueError:
        return False


def _unique_headers(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for index, header in enumerate(headers):
        name = header.strip() or f"col{index}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}.{count}")
    return unique


def read_numeric_table(path: Path) -> NumericTable:
    """Read a numeric CSV with or without a header row.

    Some legacy artifacts have proper column names; a few files have numeric first
    rows and no header. This reader handles both cases and preserves duplicate
    headers by adding suffixes such as ``.1``.
    """
    rows: list[list[str]] = []
    with path.open(newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if row and any(cell.strip() for cell in row):
                rows.append(row)

    if not rows:
        raise ValueError(f"{path} is empty.")

    first = rows[0]
    has_header = not all(_is_float(cell) for cell in first)
    if has_header:
        headers = _unique_headers(first)
        data_rows = rows[1:]
    else:
        headers = [f"col{i}" for i in range(len(first))]
        data_rows = rows

    columns: dict[str, list[float]] = {name: [] for name in headers}
    for row in data_rows:
        padded = row + [""] * (len(headers) - len(row))
        for name, value in zip(headers, padded):
            columns[name].append(_as_float(value))
    return NumericTable(path=path, columns=columns)


def _reconstruct_jax_array(func, args, state, metadata):
    array = func(*args)
    array.__setstate__(state)
    return array


class _JaxArrayShimUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "jax._src.array" and name == "_reconstruct_array":
            return _reconstruct_jax_array
        return super().find_class(module, name)


def _load_jax_array_pickle(path: Path):
    with path.open("rb") as handle:
        return _JaxArrayShimUnpickler(handle).load()


def normalize_mpt_kernel_evolution(repo_root: Path, dry_run: bool = False) -> tuple[Path, tuple[Path, ...], bool]:
    """Convert legacy JAX-array MPT kernel snapshots into a portable NPZ.

    The legacy pickle stores JAX arrays through a reconstruction wrapper around
    NumPy buffers. The shim above reconstructs those buffers directly as NumPy
    arrays, avoiding a JAX dependency for figure reproduction.
    """
    sources = tuple(repo_root / "code" / "Bulk" / filename for filename in BULK_MPT_KERNEL_EVOLUTION_SOURCES)
    destination = repo_root / "data" / "processed" / "bulk" / BULK_MPT_KERNEL_EVOLUTION_NPZ
    exists = all(source.exists() for source in sources)
    if not exists or dry_run:
        return destination, sources, exists

    import numpy as np

    snapshots: list[dict[str, object]] = []
    for source in sources:
        loaded = _load_jax_array_pickle(source)
        if not isinstance(loaded, list):
            raise ValueError(f"{source} should contain a list of kernel snapshots.")
        snapshots.extend(loaded)

    if not snapshots:
        raise ValueError("MPT kernel evolution snapshots are empty.")

    fields = list(snapshots[0].keys())
    arrays = {}
    for field in fields:
        arrays[field] = np.stack([np.asarray(snapshot[field], dtype=float) for snapshot in snapshots])

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        fields=np.asarray(fields),
        epochs=np.arange(len(snapshots), dtype=int),
        **arrays,
    )
    return destination, sources, exists


def copy_artifacts(repo_root: Path, dry_run: bool = False) -> list[tuple[Path, Path, bool]]:
    """Copy compact legacy artifacts into ``data/processed``."""
    copied: list[tuple[Path, Path, bool]] = []
    for spec in ALL_ARTIFACTS:
        source = repo_root / "code" / spec.source_subdir / spec.filename
        destination = repo_root / "data" / "processed" / spec.system / spec.filename
        exists = source.exists()
        copied.append((source, destination, exists))
        if exists and not dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return copied


def validate_artifacts(processed_root: Path) -> list[str]:
    """Return human-readable validation issues for processed artifacts."""
    issues: list[str] = []
    for spec in ALL_ARTIFACTS:
        path = processed_root / spec.system / spec.filename
        if spec.required and not path.exists():
            issues.append(f"missing required artifact: {path}")
        elif path.exists() and path.stat().st_size == 0:
            issues.append(f"empty artifact: {path}")
    derived = processed_root / "bulk" / BULK_MPT_KERNEL_EVOLUTION_NPZ
    if not derived.exists():
        issues.append(f"missing required derived artifact: {derived}")
    elif derived.stat().st_size == 0:
        issues.append(f"empty derived artifact: {derived}")
    return issues
