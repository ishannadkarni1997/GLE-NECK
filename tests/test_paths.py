from pathlib import Path

from gleneck.paths import ProjectPaths


def test_compact_figure_bundle_root_discovery(tmp_path):
    bundle = tmp_path / "figure_reproduction"
    (bundle / "data" / "processed").mkdir(parents=True)
    (bundle / "scripts").mkdir()
    (bundle / "scripts" / "reproduce_figures.py").write_text("# compact bundle entrypoint\n")

    paths = ProjectPaths.discover(bundle / "data" / "processed")

    assert paths.root == bundle

