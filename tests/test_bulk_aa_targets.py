from __future__ import annotations

import numpy as np

from gleneck.bulk.aa_targets import (
    compute_retained_rdfs,
    drift_observable,
    force_mode_counts,
    group_peculiar_velocity_np,
    langevin_thermostat_mask,
    make_initial_positions,
    parse_fields,
    summarize_drift_trace,
)
from gleneck.bulk.config import BulkAATargetConfig


def test_parse_fields_defaults_and_values():
    assert parse_fields("", (0.0, 0.1)) == (0.0, 0.1)
    assert parse_fields("0.0, 0.0347,0.1") == (0.0, 0.0347, 0.1)


def test_initial_positions_have_expected_shape_and_bounds():
    config = BulkAATargetConfig(n_solvent=8, n_a=2, n_b=2, box_size=(10.0, 8.0, 6.0))
    positions = make_initial_positions(config)
    assert positions.shape == (12, 3)
    assert np.all(positions >= 0.0)
    assert np.all(positions < np.asarray(config.box_size))


def test_retained_rdf_smoke_shape():
    config = BulkAATargetConfig(n_solvent=0, n_a=2, n_b=2, box_size=(10.0, 10.0, 10.0), rdf_cutoff=5.0, rdf_dr=1.0)
    retained_positions = np.asarray(
        [
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [7.0, 7.0, 7.0],
                [8.0, 7.0, 7.0],
            ]
        ]
    )
    centers, rdfs = compute_retained_rdfs(retained_positions, config)
    assert centers.shape == (5,)
    assert set(rdfs) == {"g_r_AA", "g_r_BB", "g_r_AB"}
    assert all(values.shape == centers.shape for values in rdfs.values())


def test_force_mode_counts_solute_and_all_mobile():
    config = BulkAATargetConfig(n_solvent=8, n_a=2, n_b=2)
    solute = force_mode_counts(config, "solute")
    assert solute["forced_particles"] == 4
    assert solute["forced_solvent_particles"] == 0
    assert solute["forced_solute_particles"] == 4
    assert solute["net_external_force_weight"] == 4

    all_mobile = force_mode_counts(config, "all-mobile")
    assert all_mobile["forced_particles"] == 12
    assert all_mobile["forced_solvent_particles"] == 8
    assert all_mobile["forced_solute_particles"] == 4
    assert all_mobile["net_external_force_weight"] == 12

    counter = force_mode_counts(config, "solute-counter-solvent")
    assert counter["forced_particles"] == 12
    assert counter["forced_solvent_particles"] == 8
    assert counter["forced_solute_particles"] == 4
    assert np.isclose(counter["net_external_force_weight"], 0.0)


def test_relative_drift_observable_uses_solvent_reference():
    trace = {
        "solute_vx_mean": np.asarray([0.10, 0.12, 0.14]),
        "solvent_vx_mean": np.asarray([0.02, 0.03, 0.04]),
        "all_vx_mean": np.asarray([0.03, 0.04, 0.05]),
    }
    np.testing.assert_allclose(drift_observable(trace, "lab"), [0.10, 0.12, 0.14])
    np.testing.assert_allclose(drift_observable(trace, "solvent"), [0.08, 0.09, 0.10])
    np.testing.assert_allclose(drift_observable(trace, "all-com"), [0.07, 0.08, 0.09])


def test_drift_summary_reports_reference_frame():
    trace = {
        "solute_vx_mean": np.asarray([0.10, 0.11, 0.12, 0.13, 0.14]),
        "solvent_vx_mean": np.asarray([0.02, 0.02, 0.02, 0.02, 0.02]),
        "all_vx_mean": np.asarray([0.03, 0.03, 0.03, 0.03, 0.03]),
    }
    summary = summarize_drift_trace(trace, burn=2, reference="solvent")
    assert summary["drift_reference"] == "solvent"
    assert np.isclose(summary["steady_mean"], np.mean([0.10, 0.11, 0.12]))


def test_drift_summary_flags_tail_shift_and_temperature_guard():
    trace = {
        "solute_vx_mean": np.asarray([0.1, 0.1, 0.1, 0.3, 0.3, 0.3]),
        "solvent_vx_mean": np.zeros(6),
        "all_vx_mean": np.zeros(6),
        "all_temperature": np.asarray([0.08, 0.08, 0.08, 0.08, 0.5, 0.5]),
    }

    summary = summarize_drift_trace(
        trace,
        burn=2,
        reference="solvent",
        absolute_tolerance=0.01,
        relative_tolerance=0.01,
        window_fraction=1 / 3,
        max_all_temperature=0.2,
    )

    assert summary["status"] == "needs_longer_run"
    assert "tail_mean_shift_exceeds_tolerance" in summary["warnings"]
    assert "all_temperature_guard_exceeded" in summary["warnings"]


def test_solvent_yz_langevin_mask_does_not_thermostat_driven_direction():
    species = np.asarray([0, 0, 0, 1, 2])
    mask = langevin_thermostat_mask(species, "solvent-yz")
    assert mask.shape == (5, 3)
    np.testing.assert_allclose(mask[:, 0], 0.0)
    np.testing.assert_allclose(mask[:3, 1:], 1.0)
    np.testing.assert_allclose(mask[3:, 1:], 0.0)


def test_all_yz_langevin_mask_thermostats_only_transverse_components():
    species = np.asarray([0, 0, 1, 2])
    mask = langevin_thermostat_mask(species, "all-yz")
    assert mask.shape == (4, 3)
    np.testing.assert_allclose(mask[:, 0], 0.0)
    np.testing.assert_allclose(mask[:, 1:], 1.0)


def test_group_peculiar_velocity_removes_species_streaming():
    species = np.asarray([0, 0, 1, 1])
    mass = np.asarray([[1.0], [3.0], [2.0], [4.0]])
    velocity = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 1.0, 0.0],
            [14.0, 1.0, 0.0],
        ]
    )

    peculiar = group_peculiar_velocity_np(velocity, mass, species)

    for group in np.unique(species):
        mask = species == group
        group_momentum = np.sum(mass[mask] * peculiar[mask], axis=0)
        np.testing.assert_allclose(group_momentum, 0.0, atol=1e-12)
