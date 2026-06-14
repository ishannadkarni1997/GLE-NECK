from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the public GLE-NECK repository root."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        processed = candidate / "data" / "processed"
        is_python_project = (candidate / "pyproject.toml").exists() and (candidate / "src" / "gleneck").is_dir()
        is_bulk_bundle = (processed / "bulk").is_dir() and (candidate / "scripts" / "make_bulk_figures.py").exists()
        if is_python_project or is_bulk_bundle:
            return candidate
    raise FileNotFoundError("Could not locate the GLE-NECK repository root.")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        return cls(find_repo_root(start))

    @property
    def processed_bulk(self) -> Path:
        return self.root / "data" / "processed" / "bulk"

    @property
    def bulk_figures(self) -> Path:
        return self.root / "figures" / "bulk"
