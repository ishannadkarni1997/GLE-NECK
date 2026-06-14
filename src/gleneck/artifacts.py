from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactSpec:
    filename: str
    required_columns: tuple[str, ...] = ()
    required: bool = True
    note: str = ""


BULK_ARTIFACTS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec("aa_equilibrium_rdf.csv", ("r", "g_AA", "g_BB", "g_AB", "g_WW")),
    ArtifactSpec("aa_vacf.csv", ("time_ps", "vacf")),
    ArtifactSpec("aa_mobility.csv", ("field", "drift_velocity")),
    ArtifactSpec("ibi_potentials.csv", ("r", "u_aa", "u_ab", "u_bb")),
    ArtifactSpec("ibi_potentials.npz"),
    ArtifactSpec("memory_kernel.csv", ("time_ps", "raw_memory_ps2", "fitted_memory_ps2")),
    ArtifactSpec("gle_baseline_rdf.csv", ("r", "g_AA", "g_BB", "g_AB")),
    ArtifactSpec("gle_baseline_vacf.csv", ("time_ps", "vacf")),
    ArtifactSpec("gle_baseline_mobility.csv", ("field", "drift_velocity")),
    ArtifactSpec("gleneck_mpt_mobility.csv", ("field", "drift_velocity")),
    ArtifactSpec("gleneck_mpt_training_loss.csv", ("epoch", "loss")),
    ArtifactSpec("gleneck_validation_loss.csv", ("model", "training_fields", "eval_max_field", "train_mse", "validation_mse", "full_mse")),
    ArtifactSpec("gleneck_kernel_evolution.npz"),
    ArtifactSpec("bulk_result_summary.json"),
    ArtifactSpec("gleneck_spt_mobility.csv", ("field", "drift_velocity", "training_field"), required=False),
    ArtifactSpec("gleneck_spt_training_loss.csv", ("epoch", "loss", "training_field"), required=False),
)


BULK_MPT_KERNEL_EVOLUTION_NPZ = "gleneck_kernel_evolution.npz"


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

    Some processed artifacts have proper column names; a few files have numeric
    first rows and no header. This reader handles both cases and preserves duplicate
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


def validate_artifacts(processed_root: Path) -> list[str]:
    """Return human-readable validation issues for processed artifacts."""
    issues: list[str] = []
    bulk_root = processed_root / "bulk"
    for spec in BULK_ARTIFACTS:
        path = bulk_root / spec.filename
        if spec.required and not path.exists():
            issues.append(f"missing required artifact: {path}")
            continue
        if not path.exists():
            continue
        if path.stat().st_size == 0:
            issues.append(f"empty artifact: {path}")
            continue
        if spec.required_columns and path.suffix == ".csv":
            try:
                table = read_numeric_table(path)
                table.require(*spec.required_columns)
            except ValueError as exc:
                issues.append(str(exc))
    return issues
