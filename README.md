# GLE-NECK

**Generalized Langevin Equation with Non-Equilibrium Corrective Kernel**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20code-orange)](#status)

<p align="center">
  <img src="docs/assets/gleneck-banner.png" alt="GLE-NECK project graphic" width="380">
</p>

GLE-NECK is a differentiable coarse-grained transport workflow for learning non-equilibrium corrections to an equilibrium generalized Langevin equation (GLE). This public repository contains the **bulk binary-solute system**: explicit-solvent reference simulations define equilibrium structure, equilibrium dynamics, and driven mobility targets; a retained-solute GLE reproduces equilibrium RDF/VACF behavior; GLE-NECK then learns a field-conditioned corrective memory kernel from steady-state transport data.

The public repository is intentionally compact. It includes processed bulk artifacts, figure-generation code, tests, and minimal run scripts. Raw trajectories, exploratory notebooks, failed architecture sweeps, scheduler logs, and draft thesis material are excluded.

## Model

<p align="center">
  <img src="docs/assets/ch5_fig4_gleneck_framework.png" alt="GLE-NECK model construction workflow" width="900">
</p>

The starting point is a common tension in molecular coarse graining. Equilibrium coarse-graining can preserve structure through a potential of mean force and can preserve equilibrium dynamics through a GLE memory/noise pair. Transport under external forcing is harder: the eliminated solvent is no longer just an equilibrium bath. It reorganizes around the driven retained variables, and this field-dependent solvent response is generally not represented by an equilibrium memory kernel.

For retained solute coordinates, GLE-NECK writes the effective force as

```math
\mathbf F_{\mathrm{GLE}\text{-}\mathrm{NECK}}(t)
=
\mathbf F_{\mathrm{PMF}}(\mathbf R(t))
+
\int_0^t d\tau\,
\left[
\mathbf K(t-\tau)
+
\boldsymbol{\Xi}_{\phi}(t-\tau,\mathbf F_{\mathrm{ext}})
\right]\cdot \mathbf v(\tau)
+
\boldsymbol{\eta}(t)
+
\mathbf F_{\mathrm{ext}} .
```

Here $\mathbf F_{\mathrm{PMF}}$ is obtained from retained-solute RDF targets, $\mathbf K$ is the equilibrium memory reconstructed from the all-atom VACF through a Volterra equation, and $\boldsymbol{\eta}$ is the corresponding colored noise. The new object is the corrective kernel $\boldsymbol{\Xi}_{\phi}$: a learned, field-conditioned memory contribution that adjusts the solvent response under non-equilibrium driving.

The zero-field recovery constraint is enforced by the parameterization

```math
\boldsymbol{\Xi}_{\phi}(\tau,\mathbf F_{\mathrm{ext}})
=
\|\mathbf F_{\mathrm{ext}}\|^2
\mathcal N_{\phi}(\tau,\mathbf F_{\mathrm{ext}}),
\qquad
\boldsymbol{\Xi}_{\phi}(\tau,\mathbf 0)=\mathbf 0 .
```

Thus the corrective term vanishes smoothly as the external field approaches zero, leaving the audited equilibrium GLE unchanged.

## Framework

<p align="center">
  <img src="docs/assets/ch5_fig6_differentiable_training.png" alt="Differentiable GLE-NECK training framework" width="900">
</p>

GLE-NECK is trained top-down on transport while keeping the equilibrium construction fixed. The differentiable GLE integrator maps a corrective-kernel parameter vector $\phi$ and an external field $F$ to a predicted steady-state drift velocity,

```math
v_{\mathrm{ss}}^{\mathrm{GLE}\text{-}\mathrm{NECK}}(\phi,F)
=
\mathcal S_T\!\left[
\mathbf R_t,\mathbf v_t;
\mathbf F_{\mathrm{PMF}},\mathbf K,\boldsymbol{\eta},
\boldsymbol{\Xi}_{\phi},F
\right],
```

where $\mathcal S_T$ denotes time integration followed by a steady-state velocity estimator. For a set of training fields $\mathcal E_{\mathrm{train}}$, the drift-matching objective is

```math
\mathcal L(\phi)
=
\frac{1}{|\mathcal E_{\mathrm{train}}|}
\sum_{F\in\mathcal E_{\mathrm{train}}}
\left[
v_{\mathrm{ss}}^{\mathrm{GLE}\text{-}\mathrm{NECK}}(\phi,F)
-
v_{\mathrm{ss}}^{\mathrm{AA}}(F)
\right]^2
+
\lambda\,\mathcal R(\phi).
```

In the promoted bulk result, $\mathcal E_{\mathrm{train}}=\{0.5,1.0,2.0\}$. Single-point training (SPT) is retained as a diagnostic baseline, while multi-point training (MPT) is the promoted model because it constrains a shared field-conditioned correction across the transport curve.

The workflow is:

1. Sample an explicit-solvent all-atom reference at equilibrium and under steady external fields.
2. Compute equilibrium RDFs, the retained-solute VACF, and the all-atom mobility curve.
3. Invert retained-solute RDFs to obtain tabulated PMF/IBI conservative interactions.
4. Reconstruct the equilibrium memory kernel from the VACF and validate the baseline GLE against RDF/VACF targets.
5. Train $\boldsymbol{\Xi}_{\phi}$ through differentiable GLE simulation using drift-velocity loss.
6. Evaluate SPT and MPT mobility curves, training loss, held-out mobility loss, and corrective-kernel evolution.

## Results

Generate the complete bulk figure set with:

```bash
python scripts/make_bulk_figures.py --all
```

The generated figures are written to `figures/bulk/` as both PNG and PDF:

| Figure | Purpose |
| --- | --- |
| `01_aa_equilibrium_targets` | AA RDF targets, including solvent-solvent diagnostic RDF, and IBI/PMF solute potentials |
| `02_gle_baseline_benchmark` | Baseline GLE RDF, memory kernel, and long-window VACF validation |
| `03_baseline_mobility` | AA mobility target versus equilibrium GLE response |
| `04_spt_vs_mpt_loss` | SPT/MPT training loss and post-hoc held-out mobility validation loss |
| `05_spt_vs_mpt_mobility` | AA, baseline GLE, SPT, and MPT mobility curves over the promoted `E <= 2` regime |
| `06_mpt_kernel_evolution_logtau` | MPT corrective-kernel evolution on a logarithmic lag-time axis |
| `07_field_conditioned_kernel` | Final corrective kernel versus field, shown every 0.25 field units through `E = 2` |

<p align="center">
  <img src="figures/bulk/02_gle_baseline_benchmark.png" alt="Bulk GLE baseline benchmark" width="850">
</p>

<p align="center">
  <img src="figures/bulk/05_spt_vs_mpt_mobility.png" alt="Bulk GLE-NECK mobility comparison" width="650">
</p>

<p align="center">
  <img src="figures/bulk/06_mpt_kernel_evolution_logtau.png" alt="Bulk MPT corrective-kernel evolution" width="850">
</p>

## Installation

For artifact-backed figure reproduction and tests:

```bash
git clone https://github.com/ishannadkarni1997/GLE-NECK.git
cd GLE-NECK
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For full differentiable simulations and training, install the optional JAX stack in an environment appropriate for your CPU/GPU platform:

```bash
python -m pip install -e '.[dev,jax]'
```

## Reproducibility

Regenerate figures:

```bash
python scripts/make_bulk_figures.py --all
```

Run artifact checks and tests:

```bash
python scripts/check_bulk_reproducibility.py
pytest
```

Installed entry points:

```bash
gleneck-bulk-figures --all
gleneck-bulk-check
```

## Repository Layout

```text
src/gleneck/                 reusable Python package
scripts/                     public command-line workflows
configs/                     locked bulk run configuration
data/processed/bulk/         compact processed artifacts
figures/bulk/                generated public figures
docs/                        method, units, provenance, and cluster notes
examples/                    minimal reproduction workflow
slurm/                       generic GPU job template
tests/                       artifact, plotting, and hygiene tests
```

## Data Policy

Committed data are limited to compact processed artifacts required to reproduce the public figures and smoke tests. The repository does not include raw trajectories, position/velocity histories, scheduler logs, failed training branches, or draft manuscript materials. Those files should be archived separately for formal publication if they are needed for full reruns.

## Status

The bulk workflow is the maintained public path. The promoted GLE-NECK result uses a neural/asymptotic corrective kernel trained at `E = 0.5, 1.0, 2.0` with `l_max = 500`. The learned correction reproduces the selected mobility regime well but decays faster than the audited equilibrium memory kernel; this is documented as a current modeling caveat rather than hidden in the implementation.

## Citation

If this code is useful, please cite the associated thesis or manuscript when available. A placeholder citation file is provided in [`CITATION.cff`](CITATION.cff).
