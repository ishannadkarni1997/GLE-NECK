from __future__ import annotations

from pathlib import Path

import numpy as np

from gleneck.bulk.config import DEFAULT_BULK_MODEL
from gleneck.bulk.gle import (
    diagnostic_status,
    diagnostic_warnings,
    lag_order_velocity_history,
    load_mobility_curve,
    make_force_tables,
    model_config_with_overrides,
    stability_diagnostics,
)
from gleneck.bulk.io import load_cg_potential_npz
from gleneck.bulk.units import effective_mass
from gleneck.paths import ProjectPaths


def test_load_legacy_aa_mobility_without_header():
    paths = ProjectPaths.discover(Path(__file__))
    curve = load_mobility_curve(paths.processed_bulk / "AA_mobility_data.csv")
    assert curve.fields.shape == (10,)
    assert curve.drifts.shape == (10,)
    assert np.isclose(curve.fields[-1], 0.1)
    assert np.isclose(curve.drifts[-1], 0.17291666666666664)


def test_load_legacy_gle_mobility_with_header():
    paths = ProjectPaths.discover(Path(__file__))
    curve = load_mobility_curve(paths.processed_bulk / "GLE_mobility_data.csv")
    assert curve.fields.shape == (10,)
    assert curve.drifts.shape == (10,)
    assert curve.drifts[0] > 0


def test_force_tables_match_bulk_config_shape():
    paths = ProjectPaths.discover(Path(__file__))
    potential = load_cg_potential_npz(paths.processed_bulk / "cg_potentials_NVE242.npz")
    r_bins, u_table, force_table = make_force_tables(potential, DEFAULT_BULK_MODEL)
    assert r_bins.shape == (DEFAULT_BULK_MODEL.num_bins,)
    assert u_table.shape == (2, 2, DEFAULT_BULK_MODEL.num_bins)
    assert force_table.shape == (2, 2, DEFAULT_BULK_MODEL.num_bins)
    np.testing.assert_allclose(u_table[0, 1], u_table[1, 0])


def test_stability_diagnostics_flags_overlap():
    positions = np.zeros((1, DEFAULT_BULK_MODEL.n_cg, 3), dtype=float)
    velocities = np.zeros_like(positions)
    report = stability_diagnostics(positions, velocities, "overlap_test")
    assert report["finite"] is True
    assert "minimum_pair_distance_below_1" in report["warnings"]


def test_stability_diagnostics_uses_peculiar_temperature_for_drift():
    grid = np.array(np.meshgrid(np.arange(7), np.arange(7), np.arange(5), indexing="ij"), dtype=float)
    positions = (2.0 * grid.reshape(3, -1).T[: DEFAULT_BULK_MODEL.n_cg])[None, :, :]

    rng = np.random.default_rng(123)
    peculiar = rng.normal(size=(DEFAULT_BULK_MODEL.n_cg, 3))
    peculiar -= peculiar.mean(axis=0, keepdims=True)
    masses = np.concatenate(
        [
            np.full(DEFAULT_BULK_MODEL.n_a, effective_mass(DEFAULT_BULK_MODEL.mass_a)),
            np.full(DEFAULT_BULK_MODEL.n_b, effective_mass(DEFAULT_BULK_MODEL.mass_b)),
        ]
    )
    kinetic = np.sum(masses[:, None] * peculiar * peculiar) / (3.0 * DEFAULT_BULK_MODEL.n_cg)
    target = DEFAULT_BULK_MODEL.temperature * 0.0019872035997726056
    peculiar *= np.sqrt(target / kinetic)

    velocities = peculiar[None, :, :]
    velocities[:, :, 0] += 5.0
    report = stability_diagnostics(positions, velocities, "driven_test")

    assert report["temperature_mean"] > 10.0 * report["target_temperature"]
    assert np.isclose(report["peculiar_temperature_mean"], report["target_temperature"])
    assert "temperature_above_10x_target" not in report["warnings"]


def test_config_overrides_and_warning_status():
    config = model_config_with_overrides(DEFAULT_BULK_MODEL, dt=1.0, l_max=32, noise_scale=0.5)
    assert config.dt == 1.0
    assert config.l_max == 32
    assert config.noise_scale == 0.5

    diagnostics = [{"warnings": []}, {"warnings": ["temperature_above_10x_target"]}]
    assert diagnostic_warnings(diagnostics) == ["temperature_above_10x_target"]
    assert diagnostic_status(diagnostics) == "unstable"


def test_config_accepts_exponential_basis_kernel():
    config = model_config_with_overrides(
        DEFAULT_BULK_MODEL,
        kernel_model="asym_exp_basis",
        kernel_exp_basis_taus_ps=(0.06, 0.14, 0.25),
    )
    assert config.kernel_model == "asym_exp_basis"
    assert config.kernel_exp_basis_taus_ps == (0.06, 0.14, 0.25)


def test_config_rejects_invalid_exponential_basis_timescale():
    try:
        model_config_with_overrides(DEFAULT_BULK_MODEL, kernel_exp_basis_taus_ps=(0.1, 0.0))
    except ValueError as exc:
        assert "kernel-exp-basis-taus-ps" in str(exc)
    else:
        raise AssertionError("Expected invalid basis timescale to raise ValueError.")


def test_config_accepts_kernel_shape_regularization_weight():
    config = model_config_with_overrides(DEFAULT_BULK_MODEL, kernel_shape_regularization_weight=0.01)
    assert config.kernel_shape_regularization_weight == 0.01


def test_config_rejects_negative_kernel_shape_regularization_weight():
    try:
        model_config_with_overrides(DEFAULT_BULK_MODEL, kernel_shape_regularization_weight=-0.1)
    except ValueError as exc:
        assert "kernel-shape-regularization-weight" in str(exc)
    else:
        raise AssertionError("Expected negative shape regularization weight to raise ValueError.")


def test_lag_order_velocity_history_reverses_recent_frames():
    velocity = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    history = lag_order_velocity_history(velocity, l_max=3)
    assert history.shape == (3, 2, 3)
    np.testing.assert_allclose(history[0], velocity[-1])
    np.testing.assert_allclose(history[1], velocity[-2])
    np.testing.assert_allclose(history[2], velocity[-3])
