from pathlib import Path

import numpy as np

from gleneck.artifacts import BULK_MPT_KERNEL_EVOLUTION_NPZ
from gleneck.artifacts import read_numeric_table, validate_artifacts
from gleneck.paths import ProjectPaths


def test_processed_artifacts_present_after_normalization():
    paths = ProjectPaths.discover(Path(__file__))
    issues = validate_artifacts(paths.root / "data" / "processed")
    assert issues == []


def test_bulk_vacf_columns_load():
    paths = ProjectPaths.discover(Path(__file__))
    table = read_numeric_table(paths.processed_bulk / "vacf_data_aa_cg_gle.csv")
    table.require("time_lag", "vacf_aa_norm", "vacf_ibi_norm", "vacf_gle_norm")
    assert len(table.columns["time_lag"]) > 10


def test_headerless_mobility_file_loads_as_numeric_columns():
    paths = ProjectPaths.discover(Path(__file__))
    table = read_numeric_table(paths.processed_bulk / "AA_mobility_data.csv")
    table.require("col0", "col1")
    assert len(table.columns["col0"]) >= 5


def test_bulk_mpt_kernel_evolution_npz_loads():
    paths = ProjectPaths.discover(Path(__file__))
    data = np.load(paths.processed_bulk / BULK_MPT_KERNEL_EVOLUTION_NPZ)
    assert list(data["fields"]) == ["E_0.03", "E_0.07", "E_0.10"]
    assert data["epochs"].shape == (900,)
    assert data["E_0.03"].shape[0] == 900
