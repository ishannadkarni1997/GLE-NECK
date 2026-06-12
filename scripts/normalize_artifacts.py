#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.artifacts import copy_artifacts, normalize_mpt_kernel_evolution, validate_artifacts
from gleneck.paths import ProjectPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate compact legacy artifacts into data/processed.")
    parser.add_argument("--dry-run", action="store_true", help="Show copy actions without writing files.")
    parser.add_argument("--root", type=Path, default=None, help="Repository root. Defaults to auto-discovery.")
    args = parser.parse_args()

    paths = ProjectPaths.discover(args.root)
    copied = copy_artifacts(paths.root, dry_run=args.dry_run)

    for source, destination, exists in copied:
        status = "copy" if exists else "missing"
        suffix = " (dry-run)" if args.dry_run and exists else ""
        print(f"{status}: {source} -> {destination}{suffix}")

    derived_destination, derived_sources, derived_exists = normalize_mpt_kernel_evolution(paths.root, dry_run=args.dry_run)
    derived_status = "create" if derived_exists else "missing"
    derived_suffix = " (dry-run)" if args.dry_run and derived_exists else ""
    source_list = ", ".join(str(source) for source in derived_sources)
    print(f"{derived_status}: {source_list} -> {derived_destination}{derived_suffix}")

    if not args.dry_run:
        issues = validate_artifacts(paths.root / "data" / "processed")
        if issues:
            print("\nValidation issues:")
            for issue in issues:
                print(f"- {issue}")
            return 1
        print("\nProcessed artifact validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
