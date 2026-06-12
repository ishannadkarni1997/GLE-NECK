# GLE-NECK

GLE-NECK stands for **Generalized Langevin Equation with Non-Equilibrium Corrective Kernel**. This repository contains a cleaned, publication-facing implementation of the bulk GLE-NECK workflow plus compact artifacts for reproducing the current thesis/manuscript figures.

The repository is intentionally smaller than the working research directory. Raw trajectories, exploratory notebooks, draft PDFs, reference libraries, SLURM logs, and private cluster paths are excluded.

## What This Repository Reproduces

The cleaned workflow is:

1. Generate equilibrium explicit-solvent all-atom reference data.
2. Generate driven all-atom NEMD mobility targets.
3. Compute RDF and VACF targets from equilibrium runs.
4. Compute the mobility curve from driven runs.
5. Build retained-solute CG pair potentials by RDF inversion.
6. Reconstruct the equilibrium memory kernel from the VACF using a Volterra equation.
7. Run the baseline retained-solute GLE and validate RDF/VACF against AA targets.
8. Train the GLE-NECK corrective memory kernel using SPT/MPT drift-velocity losses.
9. Reproduce the current chapter/paper figures and diagnostic result plots.

The current promoted bulk corrective-kernel candidate is documented in `RESULTS_PROVENANCE.md`:

```text
neural_tau0p015_midlong
training fields: E = 0.5, 1.0, 2.0
l_max = 500
kernel model: asym_softmax_poly
```

## Quick Start

For figure reproduction from committed processed artifacts:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/reproduce_figures.py --all --root . --output-dir figures/chapter5
pytest
```

For JAX/JAX-MD simulation and training work, install the optional JAX dependencies in an environment suitable for your CPU/GPU platform:

```bash
python -m pip install -e '.[dev,jax]'
```

Cluster-specific CUDA/JAX setup is summarized in `environment_cluster.md` and `environment_jax_cluster.yml`.

## Repository Layout

- `src/gleneck/`: reusable package code.
- `scripts/`: command-line workflows for target generation, GLE construction, training, validation, and plotting.
- `configs/`: locked publication-oriented model/training configurations.
- `data/processed/`: compact processed artifacts used by tests and figure reproduction.
- `figures/chapter5/`: regenerated chapter/paper figure images.
- `results/canonical/`: promoted current result snapshots.
- `results/diagnostics/`: selected diagnostics supporting unit and memory-kernel choices.
- `docs/`: method, unit, and figure-provenance notes.
- `slurm/`: generic cluster job template.
- `tests/`: smoke tests for artifact loading, plotting, and bulk workflow helpers.

## Data Policy

Committed data are compact CSV/NPY/NPZ artifacts needed for figure reproduction and smoke tests. The following should stay outside Git:

- raw trajectories and velocity histories;
- large simulation dumps;
- private cluster workspaces;
- draft PDFs/PPTs and reference-library PDFs;
- exploratory output directories.

If a manuscript requires full raw-data archival, deposit those files separately in a durable data repository and cite the DOI here.

## Status

This is a cleaned publication scaffold, not the complete exploratory history. The full working directory is kept separately as a private lab repository.
