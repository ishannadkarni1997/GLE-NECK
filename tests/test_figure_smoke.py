from pathlib import Path

import pytest

from gleneck.paths import ProjectPaths


def test_bulk_mobility_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from gleneck.plotting import plot_bulk_mobility

    paths = ProjectPaths.discover(Path(__file__))
    output = tmp_path / "bulk_mobility.png"
    plot_bulk_mobility(paths.processed_bulk, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_confinement_profile_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from gleneck.plotting import plot_confinement_profile

    paths = ProjectPaths.discover(Path(__file__))
    output = tmp_path / "confinement_profile.png"
    plot_confinement_profile(
        paths.processed_confinement,
        output,
        "epoch0_200_E150_parabolic_velocity_profile_evolution.csv",
        "epoch0_200_E150_parabolic_training_loss.csv",
    )
    assert output.exists()
    assert output.stat().st_size > 0


def test_confinement_kernels_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    from gleneck.plotting import plot_confinement_kernels

    paths = ProjectPaths.discover(Path(__file__))
    output = tmp_path / "confinement_kernels.png"
    plot_confinement_kernels(paths.processed_confinement, output)
    assert output.exists()
    assert output.stat().st_size > 0
