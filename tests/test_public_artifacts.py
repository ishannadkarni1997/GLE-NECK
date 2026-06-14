from __future__ import annotations

from pathlib import Path

import numpy as np

from gleneck.artifacts import BULK_ARTIFACTS, read_numeric_table, validate_artifacts
from gleneck.paths import ProjectPaths


def test_public_bulk_artifacts_have_required_columns():
    paths = ProjectPaths.discover(Path(__file__))
    issues = validate_artifacts(paths.root / "data" / "processed")
    assert issues == []

    names = {spec.filename for spec in BULK_ARTIFACTS}
    assert "aa_equilibrium_rdf.csv" in names
    assert "gleneck_kernel_evolution.npz" in names


def test_aa_rdf_includes_solvent_diagnostic():
    paths = ProjectPaths.discover(Path(__file__))
    table = read_numeric_table(paths.processed_bulk / "aa_equilibrium_rdf.csv")
    table.require("r", "g_AA", "g_BB", "g_AB", "g_WW")
    assert len(table.columns["r"]) >= 50
    assert max(table.columns["g_WW"]) > 1.0


def test_corrective_kernel_npz_schema():
    paths = ProjectPaths.discover(Path(__file__))
    data = np.load(paths.processed_bulk / "gleneck_kernel_evolution.npz")
    assert list(np.asarray(data["fields"], dtype=float)) == [0.5, 1.0, 2.0]
    assert data["kernels_by_field_ps2"].ndim == 3
    assert data["best_kernels_by_field_ps2"].shape == (3, data["tau_ps"].shape[0])
    assert data["equilibrium_memory_ps2"].shape == data["tau_ps"].shape
