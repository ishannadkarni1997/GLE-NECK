# GLE-NECK

**Generalized Langevin Equation with Non-Equilibrium Corrective Kernels**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research%20code-orange)](#status)

<p align="center">
  <img src="docs/assets/gleneck-banner.png" alt="GLE-NECK project graphic" width="400">
</p>

## Overview

GLE-NECK learns **non-equilibrium memory-kernel corrections** for coarse-grained molecular dynamics. Training differentiates through a generalized Langevin equation (GLE) simulation to match all-atom drift velocities under external forcing.

Equilibrium coarse-graining can preserve structure through an effective potential and can preserve equilibrium dynamics through a memory/noise pair. Transport under external driving is harder. When solvent degrees of freedom are removed, their **field-dependent response** is removed as well. Simply applying an external force to an equilibrium GLE assumes that the eliminated solvent remains an equilibrium bath, which can give the wrong mobility or drift response.

GLE-NECK addresses this by keeping the equilibrium GLE fixed and learning only the missing non-equilibrium response as a **field-conditioned corrective memory kernel**.

This repository contains the **bulk binary-solute transport model**, processed reference data, simulation and training scripts, and tests. The results below compare its drift response with all-atom simulations and show how the learned memory changes with field and training.

---

## Core idea

The equilibrium retained-solute GLE is first constructed from all-atom reference data:

- a conservative PMF/IBI interaction is obtained from retained-solute RDF targets,
- the equilibrium memory kernel is reconstructed from the all-atom VACF,
- colored noise is generated consistently with the equilibrium memory kernel,
- the resulting baseline GLE is validated against equilibrium RDF/VACF behavior.

GLE-NECK then augments this equilibrium model with a learned non-equilibrium correction.

With the sign convention used here, the retained-solute dynamics are written schematically as

```math
m\dot{\mathbf v}(t)
=
\mathbf F_{\mathrm{PMF}}(\mathbf R(t))
+
\mathbf F_{\mathrm{ext}}
-
\int_0^t d\tau\,
\left[
\mathbf K_{\mathrm{eq}}(t-\tau)
+
\boldsymbol{\Xi}_{\phi}(t-\tau,\mathbf F_{\mathrm{ext}})
\right]
\mathbf v(\tau)
+
\boldsymbol{\eta}_{\mathrm{eq}}(t).
```

Here:

- $\mathbf F_{\mathrm{PMF}}$ is the equilibrium conservative force,
- $\mathbf K_{\mathrm{eq}}$ is the equilibrium memory kernel,
- $\boldsymbol{\eta}_{\mathrm{eq}}$ is the corresponding equilibrium colored noise,
- $\mathbf F_{\mathrm{ext}}$ is the applied external field,
- $\boldsymbol{\Xi}_{\phi}$ is the learned non-equilibrium corrective kernel.

The correction is constrained to vanish in the zero-field limit:

```math
\boldsymbol{\Xi}_{\phi}(\tau,\mathbf F_{\mathrm{ext}})
=
\|\mathbf F_{\mathrm{ext}}\|^2
\mathcal N_{\phi}(\tau,\mathbf F_{\mathrm{ext}}),
\qquad
\boldsymbol{\Xi}_{\phi}(\tau,\mathbf 0)=\mathbf 0.
```

The equilibrium model is recovered at zero field. At finite fields, the learned kernel corrects the transport response.


---

## Model construction

<p align="center">
  <img src="docs/assets/ch5_fig4_gleneck_framework.png" alt="GLE-NECK model construction workflow" width="900">
</p>

The workflow separates equilibrium coarse-graining from transport correction:

1. **Equilibrium structure**  
   Compute retained-solute RDFs from explicit-solvent all-atom simulations.

2. **Conservative interaction**  
   Invert RDF targets to obtain tabulated PMF/IBI conservative interactions.

3. **Equilibrium dynamics**  
   Compute the retained-solute VACF and reconstruct the equilibrium memory kernel.

4. **Equilibrium GLE validation**  
   Validate the baseline GLE against RDF and VACF targets.

5. **Non-equilibrium transport data**  
   Run explicit-solvent driven simulations and compute steady-state drift velocities.

6. **Corrective-kernel training**  
   Learn $\boldsymbol{\Xi}_{\phi}$ through differentiable GLE simulation using drift-velocity loss.

7. **Transport validation**  
   Compare GLE-NECK mobility predictions against all-atom reference mobilities.

---

## Differentiable training

<p align="center">
  <img src="docs/assets/ch5_fig6_differentiable_training.png" alt="Differentiable GLE-NECK training framework" width="900">
</p>

The corrective kernel is trained top-down on transport observables. For a corrective-kernel parameter vector $\phi$ and external field $F$, the differentiable GLE integrator produces a trajectory and a steady-state drift estimate:

```math
v_{\mathrm{ss}}^{\mathrm{GLE\text{-}NECK}}(\phi,F)
=
\mathcal S_T
\left[
\mathbf R_t,\mathbf v_t;
\mathbf F_{\mathrm{PMF}},
\mathbf K_{\mathrm{eq}},
\boldsymbol{\eta}_{\mathrm{eq}},
\boldsymbol{\Xi}_{\phi},
F
\right],
```

where $\mathcal S_T$ denotes GLE time integration followed by a steady-state velocity estimator.

For training fields $\mathcal E_{\mathrm{train}}$, the loss is

```math
\mathcal L(\phi)
=
\frac{1}{|\mathcal E_{\mathrm{train}}|}
\sum_{F\in\mathcal E_{\mathrm{train}}}
\left[
v_{\mathrm{ss}}^{\mathrm{GLE\text{-}NECK}}(\phi,F)
-
v_{\mathrm{ss}}^{\mathrm{AA}}(F)
\right]^2
+
\lambda\,\mathcal R(\phi).
```

The multi-point model shown below uses the training fields

```math
\mathcal E_{\mathrm{train}}=\{0.5,1.0,2.0\}.
```

Single-point models are trained at one field each and serve as comparison baselines. The multi-point model learns one field-dependent correction from all three training fields.

### Training algorithm

```text
Algorithm: Transport-targeted training of the GLE-NECK corrective kernel

Inputs:
  F_train          training fields
  T_AA(F)          all-atom transport targets
  F_C              conservative PMF/IBI force
  Gamma_eq         equilibrium memory kernel
  eta_eq           equilibrium colored noise
  phi              initial neural-kernel parameters

Output:
  phi*             trained corrective-kernel parameters

for n = 1, ..., N_opt:
    L = 0

    for each F_i in F_train:
        # Construct gated corrective kernel
        Gamma_corr_phi(t; F_i) =
            g(F_i) * NN_phi(e_t(t), e_F(F_i))

        # Form total kernel
        Gamma_tot(t; F_i) =
            Gamma_eq(t) + Gamma_corr_phi(t; F_i)

        # Simulate CG-GLE trajectory
        X_CG[0:T](phi, F_i) =
            S_dt(X_0, F_C, Gamma_tot, eta_eq, F_i)

        # Estimate transport observable
        T_CG(F_i; phi) =
            O(X_CG[0:T](phi, F_i))

        # Accumulate transport loss
        L += w_i * ||T_CG(F_i; phi) - T_AA(F_i)||^2

    # Differentiate through GLE integrator and update kernel
    grad_phi = dL/dphi
    phi = Optimizer(phi, grad_phi)

return phi* = phi
```
The algorithm is written for the bulk force-conditioned kernel. For confined transport, the same workflow is used with `Gamma_corr_phi(t; F_i, z)` and with `T_CG` defined as the spatially resolved velocity profile.

---

## Results

Regenerate the bulk transport figures from the included processed data:

```bash
python scripts/make_bulk_figures.py --all
```

Expected outputs include:

- equilibrium RDF/VACF checks,
- reconstructed memory kernels,
- single-point training diagnostics,
- multi-point training diagnostics,
- corrective-kernel evolution,
- mobility comparison curves,
- held-out mobility loss summaries.

Generated figures are written to:

```text
figures/bulk/
```

The following animations summarize the field response, corrective-memory
emergence, differentiable training, and confined-system memory structure:

<table>
<tr>
<td align="center">
<img src="docs/assets/results_animations/gle_neck_mobility_response.gif" alt="GLE NECK mobility response" width="900">
</td>
</tr>
<tr>
<td align="center">
<img src="docs/assets/results_animations/emergence_non_equilibrium_memory.gif" alt="Emergence of non-equilibrium memory with field" width="900">
</td>
</tr>
<tr>
<td align="center">
<img src="docs/assets/results_animations/corrective_memory_training.gif" alt="Corrective memory emerging during training" width="900">
</td>
</tr>
<tr>
<td align="center">
<img src="docs/assets/results_animations/confinement_memory_emergence.gif" alt="Emergence of spatial non-equilibrium memory in confinement" width="900">
</td>
</tr>
</table>

---

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/ishannadkarni1997/GLE-NECK.git
cd GLE-NECK

python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For differentiable simulations and training, install the optional JAX stack in an environment appropriate for your CPU/GPU platform:

```bash
python -m pip install -e '.[dev,jax]'
```

---

## Reproducibility

Regenerate all public bulk figures:

```bash
python scripts/make_bulk_figures.py --all
```

Check the processed data and generated figures:

```bash
python scripts/check_bulk_reproducibility.py
```

Run tests:

```bash
pytest
```

Installed entry points:

```bash
gleneck-bulk-figures --all
gleneck-bulk-check
```

---

## Repository layout

```text
src/gleneck/                 simulation, training, and analysis code
scripts/                    command-line entry points
configs/                    bulk model and training settings
data/processed/bulk/         processed reference data and model results
figures/bulk/                generated bulk figures
docs/                       method, units, data provenance, and visualizations
examples/                   figure reproduction example
slurm/                      GPU job template
tests/                      data validation, plotting, and repository checks
```

## Status

Development is ongoing. The code and processed data reproduce the bulk binary-solute results. The confinement animation illustrates related work; confinement simulation and training code are maintained separately. Raw simulation trajectories are not included.

The corrective kernel is fitted to transport observables. Different optimization and regularization choices can produce different kernel shapes with similar drift predictions, so the fitted kernel should not be interpreted as a unique microscopic memory function. See [result provenance](docs/provenance.md) for validation details and limitations.

---

## Publication Status

The manuscript is in preparation.

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
