# Scientific Workflow

This document records the intended end-to-end workflow for the cleaned GLE-NECK repository.

## 1. All-Atom Reference Targets

Run an explicit-solvent bulk reference system containing solutes `A` and `B` in solvent `W/C`. Equilibrium runs produce retained-solute RDF and VACF targets. Driven NEMD runs produce the all-atom mobility curve.

The retained GLE model propagates only the solute particles. The solvent is eliminated and represented through the conservative PMF, equilibrium memory/noise, and non-equilibrium corrective kernel.

## 2. Equilibrium Coarse-Graining

Use retained-solute RDF targets to build tabulated CG conservative interactions for `A-A`, `A-B`, and `B-B`. Use the equilibrium retained-solute VACF to reconstruct the memory kernel by solving the Volterra equation.

The public plotting convention is:

- lag time in `ps`;
- memory kernels in `ps^-2`;
- internal JAX-MD time units only inside simulation code.

## 3. Baseline GLE

Run the retained-solute GLE using:

- tabulated CG conservative force;
- equilibrium memory friction;
- colored noise consistent with the equilibrium memory;
- external driving force for transport validation.

The baseline must reproduce the AA equilibrium VACF and approximately reproduce retained-solute RDF targets before GLE-NECK training is interpreted.

## 4. GLE-NECK Training

Add a learned corrective memory contribution:

```text
M_total(E, tau) = M_eq(tau) + Delta M(E, tau)
Delta M(E, tau) = E^2 * N_theta(E, tau)
```

The `E^2` factor enforces smooth zero-field recovery. Training differentiates through the GLE rollout and minimizes drift-velocity error against AA mobility targets.

The current promoted candidate is the `neural_tau0p015_midlong` MPT run trained at `E = 0.5, 1.0, 2.0`.

## 5. Figure Reproduction

Regenerate artifact-backed chapter figures with:

```bash
python scripts/reproduce_figures.py --all --root . --output-dir figures/chapter5
```

This command does not rerun AA simulations or GLE-NECK training. It uses committed compact artifacts.
