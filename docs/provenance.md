# Bulk Result Provenance

This document records the public provenance for the promoted bulk result.

## Target Protocol

- System: explicit-solvent bulk binary-solute reference liquid.
- Retained GLE particles: solute species `A` and `B`.
- Solvent handling: explicit in AA targets, eliminated in the retained GLE.
- Driven response: solute drift relative to the solvent reference frame.
- Production thermostat choice: Langevin-style transverse/peculiar thermostatting that avoids thermostatting the driven direction.

## Baseline Artifacts

- `aa_equilibrium_rdf.csv`: smoothed AA RDF targets, including `W-W` as solvent diagnostic.
- `aa_vacf.csv`: AA retained-solute VACF target.
- `ibi_potentials.csv` and `ibi_potentials.npz`: solute conservative interactions from RDF inversion.
- `memory_kernel.csv`: Volterra memory reconstruction on the physical ps axis.
- `gle_baseline_rdf.csv` and `gle_baseline_vacf.csv`: baseline retained-solute GLE validation outputs.
- `gle_baseline_mobility.csv`: baseline non-equilibrium response before adding GLE-NECK.

## Promoted GLE-NECK Candidate

- Corrective model: neural/asymptotic memory correction.
- Zero-field gate: `E^2`.
- Training fields: `E = 0.5, 1.0, 2.0`.
- Lag window: `l_max = 500`.
- Optimizer: Adam.
- Promoted artifacts:
  - `gleneck_mpt_mobility.csv`
  - `gleneck_mpt_training_loss.csv`
  - `gleneck_kernel_evolution.npz`

The corrective kernel gives a strong mobility match over the selected training regime. It also decays faster than the equilibrium memory kernel. This is retained as a current modeling caveat and should be discussed explicitly in any paper or thesis text that uses the result.

The SPT curves are included as comparison baselines. The `E = 2.0` SPT run completed and matched its training point, but its diagnostics include a minimum-pair-distance warning; the promoted result is therefore the MPT candidate, not the high-field SPT branch.

## Non-Public Material

Exploratory regularizer sweeps, alternative kernel parameterizations, raw trajectories, scheduler logs, and draft manuscript material are not included in the public repository. They belong in private working storage or a separate archival data release.
