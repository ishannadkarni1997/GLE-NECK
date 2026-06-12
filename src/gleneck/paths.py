from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find either the full project root or a compact figure-bundle root."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        processed = candidate / "data" / "processed"
        reproduce_script = candidate / "scripts" / "reproduce_figures.py"
        is_full_project = (candidate / "code").is_dir() and (candidate / "chap_5GLE_NECK_pdf.pdf").exists()
        is_figure_bundle = processed.is_dir() and reproduce_script.exists()
        if is_full_project or is_figure_bundle:
            return candidate
    raise FileNotFoundError("Could not locate the GLE-NECK repository root.")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        return cls(find_repo_root(start))

    @property
    def legacy_bulk(self) -> Path:
        return self.root / "code" / "Bulk"

    @property
    def legacy_confinement(self) -> Path:
        return self.root / "code" / "Confinement"

    @property
    def processed_bulk(self) -> Path:
        return self.root / "data" / "processed" / "bulk"

    @property
    def processed_confinement(self) -> Path:
        return self.root / "data" / "processed" / "confinement"

    @property
    def figure_output(self) -> Path:
        return self.root / "outputs" / "figures"
