from __future__ import annotations

from pathlib import Path

import pytest

from gleneck.bulk.figures import REQUIRED_FIGURES, make_all_bulk_figures
from gleneck.paths import ProjectPaths


def test_all_public_bulk_figures_generate(tmp_path):
    pytest.importorskip("matplotlib")
    paths = ProjectPaths.discover(Path(__file__))
    generated = make_all_bulk_figures(paths.root, tmp_path)

    expected = {f"{stem}{suffix}" for stem in REQUIRED_FIGURES for suffix in (".png", ".pdf")}
    actual = {path.name for path in generated}
    assert expected == actual
    assert all(path.exists() and path.stat().st_size > 0 for path in generated)
