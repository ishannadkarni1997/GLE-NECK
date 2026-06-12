# Results Provenance

This repository promotes one current bulk GLE-NECK candidate for publication-prep work while preserving enough diagnostics to show why the choice is reasonable.

## Canonical Candidate

- Label: `neural_tau0p015_midlong`
- Model: `asym_softmax_poly`
- Training fields: `E = 0.5, 1.0, 2.0`
- Lag length: `l_max = 500`
- Optimizer: Adam
- Result directory in this public repo: `results/canonical/neural_tau0p015_midlong`

Promoted figures:

- `bulk_lmax500_mobility_compare.png`
- `bulk_lmax500_loss_compare.png`
- `bulk_lmax500_neural_tau0p015_midlong_kernel_evolution.png`
- `kernel_evolution_all_architectures_grid.png`
- `kernel_evolution_all_architectures_grid_logx.png`

## Interpretation

The `neural_tau0p015_midlong` branch gives the strongest mobility match among the current MPT candidates trained on the `E = 0.5, 1.0, 2.0` field set. Its learned correction decays faster than the audited equilibrium memory kernel. This is accepted as the current empirical candidate, but the theory discussion should explicitly note the timescale separation and not overclaim that the correction must share the equilibrium memory timescale.

## Supporting Diagnostics

Selected memory-unit and VACF diagnostics are committed under:

```text
results/diagnostics/equilibrium_memory_audit/
```

These diagnostics verify the physical lag-axis convention and distinguish plotting/unit mistakes from the current AA target's actual equilibrium memory timescale.

## Non-Promoted Results

Exploratory regularizers, alternative kernel bases, high-field extrapolation checks, and failed/unstable optimization branches are not part of the public claim. They should remain in the private lab repository unless a future manuscript section requires them as supplementary diagnostics.
