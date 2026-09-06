from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

from gleneck.artifacts import BULK_MPT_KERNEL_EVOLUTION_NPZ, validate_artifacts
from gleneck.paths import ProjectPaths

REQUIRED_NPZ_KEYS = {
    "fields",
    "epochs",
    "tau_ps",
    "kernels_by_field_ps2",
    "best_kernels_by_field_ps2",
    "equilibrium_memory_ps2",
    "target_drifts",
    "best_training_predictions",
}
REQUIRED_FIGURES = (
    "01_aa_equilibrium_targets",
    "02_gle_baseline_benchmark",
    "03_baseline_mobility",
    "04_spt_vs_mpt_loss",
    "05_spt_vs_mpt_mobility",
    "06_mpt_kernel_evolution_logtau",
    "07_field_conditioned_kernel",
)
BAD_FILENAME_TOKENS = (
    ".DS_Store",
    "__pycache__",
    ".pytest_cache",
    "chapter5",
    "legacy",
    "midlong",
    "kernal",
)
PRIVATE_PATH_PATTERNS = (
    "/Users/",
    "/home/",
    "/research/",
    "St.Jude_Intern",
    "St_Judes",
)
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".gitignore",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _git_files(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _is_text_file(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", "LICENSE"}


def check_bulk_reproducibility(root: Path, require_figures: bool = True) -> list[str]:
    paths = ProjectPaths.discover(root)
    issues = validate_artifacts(paths.root / "data" / "processed")

    kernel_path = paths.processed_bulk / BULK_MPT_KERNEL_EVOLUTION_NPZ
    if kernel_path.exists():
        data = np.load(kernel_path)
        missing = sorted(REQUIRED_NPZ_KEYS.difference(data.files))
        if missing:
            issues.append(f"{kernel_path} is missing NPZ keys: {missing}")
        elif data["kernels_by_field_ps2"].ndim != 3:
            issues.append(f"{kernel_path} should store kernel history as (epoch, field, lag).")

    if require_figures:
        for stem in REQUIRED_FIGURES:
            for suffix in (".png", ".pdf"):
                path = paths.bulk_figures / f"{stem}{suffix}"
                if not path.exists():
                    issues.append(f"missing generated figure: {path}")
                elif path.stat().st_size == 0:
                    issues.append(f"empty generated figure: {path}")

    tracked = _git_files(paths.root)
    for name in tracked:
        lowered = name.lower()
        for token in BAD_FILENAME_TOKENS:
            if token.lower() in lowered:
                issues.append(f"public filename contains disallowed token {token!r}: {name}")
        if re.search(r"(?:^|[_-])\\d{5,}(?:[_./-]|$)", name):
            issues.append(f"public filename appears to contain a cluster job id: {name}")

    for name in tracked:
        if name == "src/gleneck/bulk/checks.py":
            continue
        path = paths.root / name
        if not path.exists() or not _is_text_file(path):
            continue
        text = path.read_text(errors="ignore")
        for pattern in PRIVATE_PATH_PATTERNS:
            if pattern in text:
                issues.append(f"tracked text file contains private path pattern {pattern!r}: {name}")
    return sorted(set(issues))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check bulk GLE-NECK data, generated figures, and tracked repository files.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root or any path inside it.")
    parser.add_argument("--skip-figures", action="store_true", help="Do not require generated figures to be present.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    issues = check_bulk_reproducibility(args.root, require_figures=not args.skip_figures)
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1
    print("Bulk reproducibility check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
