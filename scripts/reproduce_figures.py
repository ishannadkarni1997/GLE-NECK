#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.artifacts import validate_artifacts
from gleneck.paths import ProjectPaths
from gleneck.plotting import reproduce_all_figures, write_figure_provenance


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate GLE-NECK chapter figures from processed artifacts.")
    parser.add_argument("--all", action="store_true", help="Regenerate all artifact-backed figures.")
    parser.add_argument("--root", type=Path, default=None, help="Repository root. Defaults to auto-discovery.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Figure output directory.")
    args = parser.parse_args()

    if not args.all:
        parser.error("Only --all is implemented for the first reproducibility milestone.")

    paths = ProjectPaths.discover(args.root)
    output_dir = args.output_dir or paths.figure_output

    issues = validate_artifacts(paths.root / "data" / "processed")
    if issues:
        print("Processed artifacts are incomplete. Run scripts/normalize_artifacts.py first.")
        for issue in issues:
            print(f"- {issue}")
        return 1

    try:
        outputs = reproduce_all_figures(paths.processed_bulk, paths.processed_confinement, output_dir)
        provenance = write_figure_provenance(output_dir)
    except ModuleNotFoundError as exc:
        if exc.name in {"matplotlib", "numpy"}:
            print(
                f"Missing plotting dependency {exc.name!r}. "
                "Create the environment with `conda env create -f environment.yml` "
                "or install the project dependencies, then rerun this command."
            )
            return 2
        raise
    print("Generated figures:")
    for path in outputs:
        print(f"- {path}")
    print(f"Generated provenance: {provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
