#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.bulk.config import DEFAULT_BULK_MODEL
from gleneck.bulk.io import (
    default_bulk_input_paths,
    load_cg_potential,
    summarize_bulk_inputs,
    validate_bulk_inputs,
    write_cg_potential_npz,
    write_json_report,
)
from gleneck.bulk.neck import jax_stack_available
from gleneck.paths import ProjectPaths


def _root_from_args(root: Path | None) -> Path:
    return ProjectPaths.discover(root).root


def _status_text(report: dict) -> str:
    lines = ["Bulk GLE-NECK reproducibility audit", ""]
    species = report.get("species_contract", {})
    if species:
        lines.append(
            "Physical model: retained species "
            f"{', '.join(species.get('retained_species', []))}; eliminated solvent "
            f"{', '.join(species.get('eliminated_species', []))}."
        )
        lines.append(f"Retained pair potentials: {', '.join(species.get('retained_pair_labels', []))}")
        lines.append("")
    for item in report["input_status"]:
        marker = "OK" if item["exists"] else "MISSING"
        size = "" if item["size_bytes"] is None else f" ({item['size_bytes']} bytes)"
        lines.append(f"- {marker}: {item['name']} -> {item['path']}{size}")
        lines.append(f"  {item['note']}")
    lines.append("")
    lines.append(f"JAX/Haiku/Optax available: {report['jax_stack_available']}")
    if "memory_summary" in report:
        memory = report["memory_summary"]
        lines.append(
            "Memory kernel: "
            f"source {memory['source_shape']} -> resampled {memory['resampled_shape']}, "
            f"noise {memory['noise_filter_shape']}"
        )
    if "cg_potential_summary" in report:
        lines.append("CG potential tables:")
        for pair, summary in report["cg_potential_summary"].items():
            lines.append(f"- species {pair}: shape {summary['shape']}, min {summary['min']:.6g}, max {summary['max']:.6g}")
    return "\n".join(lines)


def command_audit(args: argparse.Namespace) -> int:
    root = _root_from_args(args.root)
    paths = default_bulk_input_paths(root, args.private_root)
    report = summarize_bulk_inputs(paths, DEFAULT_BULK_MODEL)
    report["jax_stack_available"] = jax_stack_available()

    if args.output:
        write_json_report(report, args.output)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_status_text(report))

    missing = [item for item in report["input_status"] if item["required"] and not item["exists"]]
    return 1 if args.strict and missing else 0


def command_prepare_inputs(args: argparse.Namespace) -> int:
    root = _root_from_args(args.root)
    paths = default_bulk_input_paths(root, args.private_root)
    statuses = validate_bulk_inputs(paths)
    missing_core = [item for item in statuses if item.name in {"cg_potential", "memory_kernel"} and not item.exists]
    if missing_core:
        for item in missing_core:
            print(f"Missing required preparation input: {item.path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (root / "data" / "processed" / "bulk")
    potential = load_cg_potential(paths.cg_potential)
    potential_output = write_cg_potential_npz(potential, output_dir / "cg_potentials_NVE242.npz")

    report = summarize_bulk_inputs(paths, DEFAULT_BULK_MODEL)
    report["jax_stack_available"] = jax_stack_available()
    report["portable_outputs"] = {"cg_potential_npz": str(potential_output)}
    report_output = write_json_report(report, output_dir / "bulk_reproducibility_audit.json")

    print(f"Wrote portable CG potential: {potential_output}")
    print(f"Wrote bulk reproducibility audit: {report_output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bulk GLE-NECK scientific reproducibility utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate the bulk rerun data contract.")
    audit.add_argument("--root", type=Path, default=None, help="Project root. Defaults to auto-discovery.")
    audit.add_argument("--private-root", type=Path, default=None, help="Directory containing private traj_cg.npy and vel_cg.npy.")
    audit.add_argument("--output", type=Path, default=None, help="Optional JSON report path.")
    audit.add_argument("--json", action="store_true", help="Print the audit as JSON.")
    audit.add_argument("--strict", action="store_true", help="Return nonzero when required rerun inputs are missing.")
    audit.set_defaults(func=command_audit)

    prepare = subparsers.add_parser("prepare-inputs", help="Create portable compact inputs needed by future bulk reruns.")
    prepare.add_argument("--root", type=Path, default=None, help="Project root. Defaults to auto-discovery.")
    prepare.add_argument("--private-root", type=Path, default=None, help="Directory containing private traj_cg.npy and vel_cg.npy.")
    prepare.add_argument("--output-dir", type=Path, default=None, help="Output directory for portable inputs and audit JSON.")
    prepare.set_defaults(func=command_prepare_inputs)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
