# Bulk Result Provenance

This document records the reference data, training settings, and validation procedure for the bulk results.

## Target Protocol

- System: explicit-solvent bulk binary-solute reference liquid.
- Retained GLE particles: solute species `A` and `B`.
- Solvent handling: explicit in AA targets, eliminated in the retained GLE.
- Driven response: solute drift relative to the solvent reference frame.
- Production thermostat choice: Langevin-style transverse/peculiar thermostatting that avoids thermostatting the driven direction.

## Baseline Data

- `aa_equilibrium_rdf.csv`: smoothed AA RDF targets, including `W-W` as solvent diagnostic.
- `aa_vacf.csv`: AA retained-solute VACF target.
- `ibi_potentials.csv` and `ibi_potentials.npz`: solute conservative interactions from RDF inversion.
- `memory_kernel.csv`: Volterra memory reconstruction on the physical ps axis.
- `gle_baseline_rdf.csv` and `gle_baseline_vacf.csv`: baseline retained-solute GLE validation outputs.
- `gle_baseline_mobility.csv`: baseline non-equilibrium response before adding GLE-NECK.

The baseline VACF data span approximately `204 ps`; the figure displays the `0-200 ps` comparison window.
The baseline RDF data are smoothed histograms from a longer zero-field GLE run. Histograms were accumulated during simulation without storing the full position trajectory.

## Multi-Point GLE-NECK Model

- Corrective model: neural memory correction constrained to vanish at zero field.
- Zero-field gate: `E^2`.
- Training fields: `E = 0.5, 1.0, 2.0`.
- Lag window: `l_max = 500`.
- Optimizer: Adam.
- Result files:
  - `gleneck_mpt_mobility.csv`
  - `gleneck_mpt_training_loss.csv`
  - `gleneck_validation_loss.csv`
  - `gleneck_kernel_evolution.npz`

Mobility errors are reported in `gleneck_validation_loss.csv`. The learned corrective kernel decays faster than the equilibrium memory kernel. Matching drift velocities alone does not establish that the fitted memory timescale is unique or microscopically correct.

The validation-loss panel is computed from the saved final mobility curves. For each single-point (SPT) or multi-point (MPT) model, validation MSE is evaluated against AA mobility over `E <= 2` after excluding that model's training fields. No additional training is performed for this calculation.

The field-conditioned-kernel figure shows the exact trained kernels at `E = 0.5, 1.0, 2.0`, the zero-field limit imposed by the `E^2` gate, and linear field interpolation at intermediate 0.25-spaced visualization points. The intermediate curves are for presentation of the learned field dependence, not additional training simulations.

The SPT curves are comparison baselines. The `E = 2.0` SPT run completed and matched its training point, but triggered a minimum-pair-distance warning and is marked `unstable` in `bulk_result_summary.json`. The main bulk result uses the MPT model.
