from __future__ import annotations

from pathlib import Path

from gleneck.bulk.checks import check_bulk_reproducibility
from gleneck.paths import ProjectPaths


def test_repo_root_discovery_from_tests():
    paths = ProjectPaths.discover(Path(__file__))
    assert (paths.root / "pyproject.toml").exists()
    assert paths.processed_bulk.exists()


def test_reproducibility_check_without_figure_requirement():
    paths = ProjectPaths.discover(Path(__file__))
    assert check_bulk_reproducibility(paths.root, require_figures=False) == []
