from pathlib import Path

import numpy as np

from gleneck.bulk.config import DEFAULT_BULK_MODEL
from gleneck.bulk.validation import memory_validation_report, potential_validation_report, unit_validation_report, validation_issues
from gleneck.paths import ProjectPaths


def test_static_bulk_validation_reports_are_finite():
    paths = ProjectPaths.discover(Path(__file__))
    memory = np.load(paths.processed_bulk / "fitted_memory_kernal.npy")

    units = unit_validation_report(DEFAULT_BULK_MODEL)
    memory_report = memory_validation_report(memory, DEFAULT_BULK_MODEL)
    potential_report = potential_validation_report(paths.processed_bulk / "cg_potentials_NVE242.npz", DEFAULT_BULK_MODEL)

    assert units["notebook_dt"] == DEFAULT_BULK_MODEL.dt
    assert memory_report["finite_memory"] is True
    assert memory_report["finite_noise_filter"] is True
    assert potential_report["finite_potential"] is True
    assert potential_report["finite_force"] is True


def test_validation_issues_collects_high_signal_problems():
    report = {
        "memory": {"finite_memory": False, "finite_noise_filter": True, "negative_spectrum_count": 1},
        "potential": {"finite_potential": True, "finite_force": False},
        "zero_field": {"diagnostic_warnings": ["temperature_above_10x_target"]},
    }
    issues = validation_issues(report)
    assert "memory_kernel_has_nonfinite_values" in issues
    assert "memory_kernel_requires_psd_clipping_for_noise" in issues
    assert "cg_force_table_has_nonfinite_values" in issues
    assert "zero_field_temperature_above_10x_target" in issues
