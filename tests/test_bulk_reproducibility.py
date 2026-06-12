from pathlib import Path

import numpy as np

from gleneck.bulk.config import DEFAULT_BULK_MODEL, DEFAULT_MPT_TRAINING
from gleneck.bulk.io import (
    default_bulk_input_paths,
    load_cg_potential,
    load_cg_potential_npz,
    summarize_bulk_inputs,
    write_cg_potential_npz,
)
from gleneck.bulk.memory import (
    colored_noise_filter,
    estimate_memory_kernel_from_vacf,
    gaussian_smooth,
    prepare_memory_terms,
    resample_memory_kernel,
)
from gleneck.bulk.memory_audit import (
    alpha_for_time_unit_conversion,
    memory_resampling_audit,
    volterra_unit_scaling_audit,
)
from gleneck.bulk.units import internal_time_to_ps
from gleneck.paths import ProjectPaths


def test_bulk_training_defaults_match_notebook_targets():
    assert DEFAULT_MPT_TRAINING.fields == (0.0347, 0.06883, 0.1027)
    assert DEFAULT_MPT_TRAINING.target_drifts == (0.11145, 0.15468, 0.1729)
    assert DEFAULT_BULK_MODEL.l_max == 300
    assert DEFAULT_BULK_MODEL.n_cg == 242


def test_memory_terms_prepare_on_processed_kernel():
    paths = ProjectPaths.discover(Path(__file__))
    memory = np.load(paths.processed_bulk / "fitted_memory_kernal.npy")

    resampled = resample_memory_kernel(memory, l_max=DEFAULT_BULK_MODEL.l_max)
    noise = colored_noise_filter(resampled)
    terms = prepare_memory_terms(memory, l_max=DEFAULT_BULK_MODEL.l_max)

    assert resampled.shape == (DEFAULT_BULK_MODEL.l_max,)
    assert noise.shape == (DEFAULT_BULK_MODEL.l_max,)
    assert terms.memory.shape == (DEFAULT_BULK_MODEL.l_max,)
    assert terms.noise_filter.shape == (DEFAULT_BULK_MODEL.l_max,)
    assert np.all(np.isfinite(terms.memory))
    assert np.all(np.isfinite(terms.noise_filter))


def test_memory_kernel_can_be_estimated_from_smooth_vacf():
    time_lag = np.arange(50, dtype=float) * 0.2
    vacf = np.exp(-time_lag)

    memory, dt = estimate_memory_kernel_from_vacf(time_lag, vacf, alpha=1e-8)
    smoothed = gaussian_smooth(memory, sigma=1.0)

    assert np.isclose(dt, 0.2)
    assert memory.shape == vacf.shape
    assert smoothed.shape == memory.shape
    assert np.all(np.isfinite(memory))
    assert np.all(np.isfinite(smoothed))


def test_volterra_memory_scaling_is_invariant_under_physical_time_conversion():
    time_internal = np.arange(80, dtype=float) * 0.0204548282835039
    time_ps = internal_time_to_ps(time_internal)
    vacf = np.exp(-time_ps / 0.2)

    audit = volterra_unit_scaling_audit(time_internal, vacf, alpha_internal=1e-10)

    assert np.isclose(audit["dt_ps"], 0.001)
    assert audit["max_abs_error_ps2"] < 1e-8
    assert audit["rmse_ps2"] < 1e-9


def test_alpha_scales_with_time_unit_in_volterra_refit():
    ps_per_internal = float(internal_time_to_ps(1.0))
    assert np.isclose(
        alpha_for_time_unit_conversion(2.0, ps_per_internal),
        2.0 * ps_per_internal**2,
    )


def test_memory_resampling_audit_catches_raw_as_fine_timescale_error():
    memory = np.exp(-np.arange(100, dtype=float) * 0.01 / 0.05)
    source_dt_internal = 10.0 * 0.0204548282835039
    gle_dt_internal = 0.0204548282835039

    audit = memory_resampling_audit(memory, source_dt_internal, gle_dt_internal, l_max=1000)

    assert np.isclose(audit["memory_orig_interval"], 10.0)
    assert np.isclose(audit["source_dt_ps"], 0.01)
    assert np.isclose(audit["gle_dt_ps"], 0.001)
    assert audit["raw_decay"]["t_1e"] > 0.04
    assert audit["mistaken_raw_as_fine_decay"]["t_1e"] < 0.01


def test_legacy_cg_potential_can_be_portabilized(tmp_path):
    paths = ProjectPaths.discover(Path(__file__))
    potential = load_cg_potential_npz(paths.processed_bulk / "cg_potentials_NVE242.npz")
    output = write_cg_potential_npz(potential, tmp_path / "cg_potential.npz")
    reloaded = load_cg_potential_npz(output)

    assert sorted(reloaded) == [(0, 0), (0, 1), (1, 1)]
    assert reloaded[(0, 0)].shape == (100,)
    np.testing.assert_allclose(reloaded[(0, 1)], potential[(0, 1)])


def test_bulk_input_summary_reports_default_private_array_paths():
    paths = ProjectPaths.discover(Path(__file__))
    input_paths = default_bulk_input_paths(paths.root)
    report = summarize_bulk_inputs(input_paths)
    status = {item["name"]: item for item in report["input_status"]}

    assert status["cg_potential"]["exists"] is True
    assert status["memory_kernel"]["exists"] is True
    assert status["cg_potential"]["path"].endswith("data/processed/bulk/cg_potentials_NVE242.npz")
    assert status["trajectory"]["path"].endswith("data/private/bulk/traj_cg.npy")
    assert status["velocity"]["path"].endswith("data/private/bulk/vel_cg.npy")
    assert report["memory_summary"]["resampled_shape"] == [DEFAULT_BULK_MODEL.l_max]
