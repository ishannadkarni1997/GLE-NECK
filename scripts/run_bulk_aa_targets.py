#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gleneck.bulk.aa_targets import config_from_args, parse_fields, run_bulk_aa_targets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate explicit-solvent bulk reference targets for GLE-NECK: "
            "solute RDF/VACF and drift velocity versus external field."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generated target artifacts.")
    parser.add_argument("--n-solvent", type=int, default=6508, help="Number of explicit solvent W/C particles.")
    parser.add_argument("--n-a", type=int, default=121, help="Number of retained A solutes.")
    parser.add_argument("--n-b", type=int, default=121, help="Number of retained B solutes.")
    parser.add_argument("--box-size", type=float, nargs=3, default=(100.0, 50.0, 50.0), metavar=("LX", "LY", "LZ"))
    parser.add_argument("--temperature", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.02, help="Langevin friction coefficient when --thermostat langevin is used.")
    parser.add_argument(
        "--thermostat",
        choices=("nve", "nose-hoover", "nose-hoover-masked", "langevin", "langevin-peculiar"),
        default="nose-hoover",
        help="Explicit-reference thermostat used for zero-field equilibration and equilibrium sampling.",
    )
    parser.add_argument(
        "--langevin-mode",
        choices=("all", "solvent", "solvent-yz", "all-yz", "none"),
        default="solvent-yz",
        help="Component mask for custom Langevin/masked Nose-Hoover modes. 'solvent-yz' leaves the driven x direction unthermostatted.",
    )
    parser.add_argument(
        "--equilibrium-thermostat",
        choices=("same", "nve", "nose-hoover", "nose-hoover-masked", "langevin", "langevin-peculiar"),
        default="same",
        help="Optional thermostat override for zero-field equilibrium RDF/VACF production after equilibration.",
    )
    parser.add_argument(
        "--equilibrium-thermostat-mode",
        choices=("same", "all", "solvent", "solvent-yz", "all-yz", "none"),
        default="same",
        help="Optional component mask override for zero-field equilibrium RDF/VACF production.",
    )
    parser.add_argument(
        "--field-thermostat",
        choices=("same", "nve", "nose-hoover", "nose-hoover-masked", "langevin", "langevin-peculiar"),
        default="same",
        help="Optional thermostat override for driven-field production runs.",
    )
    parser.add_argument(
        "--field-thermostat-mode",
        choices=("same", "all", "solvent", "solvent-yz", "all-yz", "none"),
        default="same",
        help="Optional component mask override for driven-field production runs.",
    )
    parser.add_argument(
        "--force-mode",
        choices=("solute", "all-mobile", "solute-counter-solvent"),
        default="solute",
        help=(
            "External-force protocol: retained solutes only, all mobile explicit particles, "
            "or solutes with a distributed solvent counterforce for zero net forcing."
        ),
    )
    parser.add_argument(
        "--drift-reference",
        choices=("lab", "solvent", "all-com"),
        default="solvent",
        help="Reference frame for driven drift targets. 'solvent' reports solute drift relative to the explicit bath.",
    )
    parser.add_argument("--r-cutoff", type=float, default=4.0)
    parser.add_argument("--dr-threshold", type=float, default=1.0)
    parser.add_argument("--rdf-cutoff", type=float, default=10.0)
    parser.add_argument("--rdf-dr", type=float, default=0.1)
    parser.add_argument("--fields", type=str, default=None, help="Comma-separated external fields. Defaults to 0.0..0.1.")
    parser.add_argument("--skip-fields", action="store_true", help="Generate only equilibrium RDF/VACF targets.")
    parser.add_argument("--equilibration-samples", type=int, default=100, help="Number of saved chunks for zero-field equilibration.")
    parser.add_argument("--equilibrium-samples", type=int, default=500, help="Number of saved chunks for equilibrium target observables.")
    parser.add_argument("--field-samples", type=int, default=1000, help="Number of saved chunks per driven field.")
    parser.add_argument(
        "--adaptive-fields",
        action="store_true",
        help="Keep each driven field running past the minimum length until tail windows satisfy the steady-state criterion.",
    )
    parser.add_argument(
        "--field-min-samples",
        type=int,
        default=None,
        help="Minimum saved chunks per driven field when --adaptive-fields is used. Defaults to --field-samples.",
    )
    parser.add_argument(
        "--field-max-samples",
        type=int,
        default=None,
        help="Maximum saved chunks per driven field when --adaptive-fields is used. Defaults to --field-samples.",
    )
    parser.add_argument(
        "--steady-check-interval",
        type=int,
        default=100,
        help="Check the adaptive steady-state criterion every N saved chunks.",
    )
    parser.add_argument(
        "--steady-window-fraction",
        type=float,
        default=0.2,
        help="Fraction of a drift trace used for current/previous tail-window comparison.",
    )
    parser.add_argument(
        "--steady-absolute-tolerance",
        type=float,
        default=0.002,
        help="Absolute tolerance for adaptive drift tail-window changes.",
    )
    parser.add_argument(
        "--steady-relative-tolerance",
        type=float,
        default=0.05,
        help="Relative tolerance for adaptive drift tail-window changes.",
    )
    parser.add_argument(
        "--max-all-temperature",
        type=float,
        default=None,
        help="Optional guard: mark field as nonsteady if all-particle raw temperature exceeds this value in the tail.",
    )
    parser.add_argument("--sample-interval", type=int, default=100, help="Integrator steps between equilibrium samples.")
    parser.add_argument("--field-sample-interval", type=int, default=100, help="Integrator steps between driven-field samples.")
    parser.add_argument("--vacf-window", type=int, default=500, help="Maximum VACF lag in saved samples.")
    parser.add_argument("--rdf-smoothing-sigma", type=float, default=2.0, help="Gaussian sigma, in RDF bins, for the smoothed RDF target file.")
    parser.add_argument("--rdf-jax-sigma", type=float, default=0.10, help="Reserved for older JAX-MD RDF diagnostics; not used by the subset solvent RDF path.")
    parser.add_argument("--rdf-sample-stride", type=int, default=1, help="Accumulate solvent RDF every N saved equilibrium samples.")
    parser.add_argument("--solvent-rdf-subset-size", type=int, default=512, help="Number of solvent particles sampled for the solvent-solvent RDF diagnostic.")
    parser.add_argument("--skip-solvent-rdf", action="store_true", help="Do not compute the online solvent-solvent RDF diagnostic.")
    parser.add_argument("--drift-burn-fraction", type=float, default=0.8, help="Fraction of each driven run discarded before drift averaging.")
    parser.add_argument("--extra-capacity", type=int, default=64, help="Neighbor-list extra capacity.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-save-trajectories", action="store_true", help="Do not write retained traj_cg.npy/vel_cg.npy.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.n_a != args.n_b:
        print("The reference simulation requires equal A and B particle counts.", file=sys.stderr)
        return 2
    if args.equilibration_samples < 1 or args.equilibrium_samples < 1 or args.field_samples < 1:
        print("Sample counts must be positive.", file=sys.stderr)
        return 2
    if args.sample_interval < 1 or args.field_sample_interval < 1:
        print("Sample intervals must be positive.", file=sys.stderr)
        return 2
    if args.rdf_smoothing_sigma < 0.0:
        print("--rdf-smoothing-sigma must be non-negative.", file=sys.stderr)
        return 2
    if args.rdf_jax_sigma <= 0.0:
        print("--rdf-jax-sigma must be positive.", file=sys.stderr)
        return 2
    if args.rdf_sample_stride < 1:
        print("--rdf-sample-stride must be positive.", file=sys.stderr)
        return 2
    if args.solvent_rdf_subset_size < 2:
        print("--solvent-rdf-subset-size must be at least 2.", file=sys.stderr)
        return 2
    if args.field_min_samples is not None and args.field_min_samples < 1:
        print("--field-min-samples must be positive.", file=sys.stderr)
        return 2
    if args.field_max_samples is not None and args.field_max_samples < 1:
        print("--field-max-samples must be positive.", file=sys.stderr)
        return 2
    if args.adaptive_fields and args.field_min_samples and args.field_max_samples and args.field_min_samples > args.field_max_samples:
        print("--field-min-samples cannot exceed --field-max-samples.", file=sys.stderr)
        return 2
    if args.steady_check_interval < 1:
        print("--steady-check-interval must be positive.", file=sys.stderr)
        return 2
    if not (0.0 < args.steady_window_fraction <= 0.5):
        print("--steady-window-fraction must be in (0, 0.5].", file=sys.stderr)
        return 2
    if args.steady_absolute_tolerance < 0.0 or args.steady_relative_tolerance < 0.0:
        print("Steady-state tolerances must be non-negative.", file=sys.stderr)
        return 2
    if not (0.0 <= args.drift_burn_fraction < 1.0):
        print("--drift-burn-fraction must be in [0, 1).", file=sys.stderr)
        return 2

    config = config_from_args(args)
    fields = () if args.skip_fields else parse_fields(args.fields, config.default_fields)
    report = run_bulk_aa_targets(
        config=config,
        output_dir=args.output_dir,
        fields=fields,
        equilibration_samples=args.equilibration_samples,
        equilibrium_samples=args.equilibrium_samples,
        field_samples=args.field_samples,
        sample_interval=args.sample_interval,
        field_sample_interval=args.field_sample_interval,
        vacf_window=args.vacf_window,
        drift_burn_fraction=args.drift_burn_fraction,
        extra_capacity=args.extra_capacity,
        save_trajectories=not args.no_save_trajectories,
        equilibrium_thermostat=None if args.equilibrium_thermostat == "same" else args.equilibrium_thermostat,
        equilibrium_thermostat_mode=None if args.equilibrium_thermostat_mode == "same" else args.equilibrium_thermostat_mode,
        field_thermostat=None if args.field_thermostat == "same" else args.field_thermostat,
        field_thermostat_mode=None if args.field_thermostat_mode == "same" else args.field_thermostat_mode,
        adaptive_fields=args.adaptive_fields,
        field_min_samples=args.field_min_samples,
        field_max_samples=args.field_max_samples,
        steady_check_interval=args.steady_check_interval,
        steady_absolute_tolerance=args.steady_absolute_tolerance,
        steady_relative_tolerance=args.steady_relative_tolerance,
        steady_window_fraction=args.steady_window_fraction,
        max_all_temperature=args.max_all_temperature,
        rdf_smoothing_sigma=args.rdf_smoothing_sigma,
        rdf_jax_sigma=args.rdf_jax_sigma,
        rdf_sample_stride=args.rdf_sample_stride,
        include_solvent_rdf=not args.skip_solvent_rdf,
        solvent_rdf_subset_size=args.solvent_rdf_subset_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
