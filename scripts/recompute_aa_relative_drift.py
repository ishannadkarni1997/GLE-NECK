#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.bulk.aa_targets import drift_observable, normalize_drift_reference, summarize_drift_trace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute AA mobility targets from saved driven drift traces.")
    parser.add_argument("--target-dir", type=Path, required=True, help="AA target directory containing drift traces and report JSON.")
    parser.add_argument(
        "--drift-reference",
        choices=("lab", "solvent", "all-com"),
        default="solvent",
        help="Reference frame for the recomputed drift target.",
    )
    parser.add_argument(
        "--burn-fraction",
        type=float,
        default=None,
        help="Fallback burn fraction if per-field burn samples are absent from the report.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path. Defaults inside --target-dir.")
    return parser


def _read_trace(path: Path) -> dict[str, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True)
    if table.shape == ():
        table = np.asarray([table], dtype=table.dtype)
    return {name: np.asarray(table[name], dtype=float) for name in table.dtype.names or ()}


def _load_report(target_dir: Path) -> dict:
    report_path = target_dir / "AA_target_generation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}")
    return json.loads(report_path.read_text())


def _trace_path_for_field(target_dir: Path, report: dict, field_index: int) -> Path:
    diagnostics = report.get("field_diagnostics", [])
    outputs = report.get("outputs", {})
    if field_index < len(diagnostics):
        field = diagnostics[field_index].get("field")
        if field is not None:
            label = f"E_{float(field):.6g}".replace("-", "m").replace(".", "p")
            output_value = outputs.get(f"drift_trace_{label}")
            if output_value:
                path = Path(output_value)
                if path.exists():
                    return path
                candidate = target_dir / path.name
                if candidate.exists():
                    return candidate
            matches = sorted(target_dir.glob(f"AA_drift_trace_{label}_*.csv"))
            if matches:
                return matches[0]
    traces = sorted(target_dir.glob("AA_drift_trace_E_*.csv"))
    if field_index < len(traces):
        return traces[field_index]
    raise FileNotFoundError(f"Could not find drift trace for field index {field_index} in {target_dir}")


def main() -> int:
    args = build_parser().parse_args()
    if args.burn_fraction is not None and not (0.0 <= args.burn_fraction < 1.0):
        print("--burn-fraction must be in [0, 1).", file=sys.stderr)
        return 2

    target_dir = args.target_dir.resolve()
    report = _load_report(target_dir)
    diagnostics = report.get("field_diagnostics", [])
    fields = [float(item["field"]) for item in diagnostics]
    if not fields:
        fields = [float(item) for item in report.get("fields", [])]
    if not fields:
        print("No fields found in AA_target_generation_report.json.", file=sys.stderr)
        return 2

    mode = normalize_drift_reference(args.drift_reference)
    rows = []
    new_diagnostics = []
    for index, field in enumerate(fields):
        trace_path = _trace_path_for_field(target_dir, report, index)
        trace = _read_trace(trace_path)
        if index < len(diagnostics) and "burn_samples" in diagnostics[index]:
            burn = int(diagnostics[index]["burn_samples"])
        else:
            burn_fraction = args.burn_fraction
            if burn_fraction is None:
                burn_fraction = float(report.get("drift_burn_fraction", 0.8))
            n_samples = len(next(iter(trace.values())))
            burn = int(max(0, min(n_samples - 1, round(n_samples * burn_fraction))))
        series = drift_observable(trace, mode)
        drift = float(np.mean(series[burn:]))
        rows.append((field, drift))
        new_diagnostics.append(
            {
                "field": field,
                "drift_velocity": drift,
                "trace_path": str(trace_path),
                **summarize_drift_trace(trace, burn, mode),
            }
        )

    output = args.output or (target_dir / f"AA_mobility_targets_relative_{mode.replace('-', '_')}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, np.asarray(rows), delimiter=",", header="external_field,drift_velocity", comments="")

    reprocess_report = {
        "source_report": str(target_dir / "AA_target_generation_report.json"),
        "source_mobility": report.get("outputs", {}).get("mobility"),
        "drift_reference": mode,
        "mobility_output": str(output),
        "field_diagnostics": new_diagnostics,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(reprocess_report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(reprocess_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
