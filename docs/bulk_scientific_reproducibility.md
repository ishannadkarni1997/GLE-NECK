# Bulk Scientific Reproducibility

This document records the first clean extraction from `code/Bulk/clean_bulk_system.ipynb`. It is intentionally narrower than the legacy notebook: it captures the data contract, notebook-derived training defaults, and portable preprocessing needed before full JAX-MD/GLE-NECK reruns.

## Physical Bulk Model

The final bulk system is an explicit-solvent reference liquid with solute species `A` and `B` in solvent `W`/`C`. The GLE and GLE-NECK models retain only the solute coordinates:

- retained GLE degrees of freedom: `A` and `B` solutes.
- eliminated degrees of freedom: explicit solvent `W`/`C`.
- retained pair interactions: `A-A`, `A-B`, and `B-B`.
- solvent-only diagnostics: `W-W` RDFs validate the explicit reference liquid but do not enter the retained GLE state or the CG pair-potential table.

Some early cells in the legacy bulk notebook create a graphene/nanopore scaffold. In the final bulk interpretation this scaffold is legacy/inert: interactions involving the scaffold are zeroed in those cells, and the scientific model should not be described as graphene confinement. Clean code should preserve any required legacy indexing only when reading old arrays, while naming the physical model as bulk solute transport with implicit solvent.

## Current Bulk Data Contract

Full bulk reruns need:

- `CGpotentials_int_NVE242.pkl`: two-species tabulated retained-solute coarse-grained potential for `A-A`, `A-B`, and `B-B`.
- `traj_cg.npy`: retained-solute position trajectory, expected shape `(T, 242, 3)`.
- `vel_cg.npy`: retained-solute velocity trajectory, expected shape `(T, 242, 3)`.
- `fitted_memory_kernal.npy`: fitted equilibrium GLE memory kernel.

Only the CG potential and fitted memory kernel are currently present in the local repo. The trajectory and velocity arrays should stay private and should be placed under:

```text
data/private/bulk/traj_cg.npy
data/private/bulk/vel_cg.npy
```

or supplied with `--private-root`.

## Commands

Audit the current rerun inputs:

```bash
python scripts/bulk_reproducibility.py audit --root .
```

Write a machine-readable audit:

```bash
python scripts/bulk_reproducibility.py audit --root . --output data/processed/bulk/bulk_reproducibility_audit.json
```

Create portable compact inputs for future reruns:

```bash
python scripts/bulk_reproducibility.py prepare-inputs --root .
```

This writes:

- `data/processed/bulk/cg_potentials_NVE242.npz`
- `data/processed/bulk/bulk_reproducibility_audit.json`

Run the minimal JAX/JAX-MD/GLE-NECK smoke check with synthetic positions:

```bash
python scripts/run_bulk_jax_smoke.py --root . --synthetic
```

On the cluster this should normally be submitted through SLURM:

```bash
sbatch slurm/bulk_jax_smoke.sbatch
```

Run the baseline GLE mobility path using the legacy AA field grid. By default this generates a thermal retained-solute `A/B` state and performs a short zero-field warmup:

```bash
python scripts/run_bulk_gle_baseline.py --root . --init-mode generated --dt 1.0 --warmup-steps 50 --steps 100
```

On the cluster:

```bash
sbatch slurm/bulk_gle_baseline.sbatch
```

Run a smoke-size GLE-NECK training loop against the legacy drift targets:

```bash
python scripts/train_bulk_gleneck.py --root . --init-mode generated --dt 1.0 --warmup-steps 20 --mode mpt --epochs 2 --steps-per-loss 10
```

On the cluster:

```bash
sbatch slurm/bulk_gleneck_smoke.sbatch
```

To reproduce the legacy notebook initialization exactly, use a private directory containing `traj_cg.npy` and `vel_cg.npy`:

```bash
--init-mode legacy --private-root /path/to/directory/containing/traj_cg.npy/and/vel_cg.npy
```

The new AA target-generation job writes this retained-state pair when it completes, but it is no longer a blocker for baseline GLE or GLE-NECK runs.

Generated retained-state runs are smoke/debug runs, not final scientific mobility estimates. The final notebook-derived convention uses equilibrated retained-solute positions, velocities, velocity history, and random-force history. The clean scripts therefore write `GLE_stability_diagnostics.json` or `GLENECK_stability_diagnostics.json` and return a nonzero exit code when any run exceeds the configured hard-core, speed, or temperature checks. Use `--allow-unstable` only when inspecting an intentionally bad trajectory.

Run the minimal physics validation checks before interpreting mobility or training results:

```bash
python scripts/validate_bulk_physics.py --root . --init-mode generated --dt 10.0
```

When retained AA/CG history arrays are available, prefer:

```bash
python scripts/validate_bulk_physics.py --root . --init-mode legacy --private-root /path/to/traj_vel_directory --dt 10.0
```

The legacy initializer now uses `vel_cg.npy` as a lag-ordered velocity history (`current, previous, ...`) instead of tiling the last velocity frame. This is a methodological correction relative to smoke-mode initialization and should be used for production baseline and GLE-NECK runs.

Generate fresh explicit-solvent bulk target values on the cluster:

```bash
sbatch slurm/bulk_aa_targets.sbatch
```

The first SLURM target job is a full-size pilot with the final particle counts but modest sampling. It writes to:

```text
$GLENECK_WORKSPACE/data_private/bulk/aa_targets/job_${SLURM_JOB_ID}
```

Expected outputs are:

- `AA_retained_rdf_targets.csv`: retained-solute `A-A`, `B-B`, and `A-B` RDF targets.
- `AA_retained_vacf_target.csv`: drift-subtracted retained-solute VACF target.
- `AA_mobility_targets.csv`: drift velocity versus external field.
- `traj_cg.npy` and `vel_cg.npy`: retained-solute equilibrium position/velocity samples for IBI, Volterra, and GLE initialization.
- `AA_target_generation_report.json`: parameters and output paths.

On the SLURM cluster, the same scaffold is staged at:

```text
$GLENECK_WORKSPACE/repo
```

Private arrays for future full reruns should be staged outside the repo, for example:

```text
$GLENECK_WORKSPACE/data_private/bulk/traj_cg.npy
$GLENECK_WORKSPACE/data_private/bulk/vel_cg.npy
```

Then audit with:

```bash
cd $GLENECK_WORKSPACE/repo
. .venv/bin/activate
python scripts/bulk_reproducibility.py audit --root . --private-root ../data_private/bulk
```

## Notebook-Derived Training Targets

Single-force low-field training:

- field: `0.0347`
- target drift: `0.11145`
- nominal epochs: `150`

Single-force high-field training:

- field: `0.1027`
- target drift: `0.1729`
- nominal epochs: `150`

Asymptotic multi-force training:

- fields: `0.0347`, `0.06883`, `0.1027`
- target drifts: `0.11145`, `0.15468`, `0.1729`
- chapter artifact target: `900` epochs

## Extracted Model Defaults

- number of CG particles: `242`
- retained species: `A`, `B`
- eliminated solvent species: `W`/`C`
- retained potential pairs: `A-A`, `A-B`, `B-B`
- box size: `(100.0, 50.0, 50.0)`
- memory length: `L_max = 300`
- corrective-kernel embedding dimension: `32`
- polynomial time degree: `1`
- optimizer: Adam, learning rate `1e-2`
- GLE time step: `10.0` in the notebook unit system, multiplied at runtime by the JAX-MD real-unit time scalar when JAX-MD is available.
- temperature: `40.0` in the notebook unit system, multiplied at runtime by the JAX-MD real-unit temperature scalar when JAX-MD is available.
- zero-field gate: asymptotic corrective kernel is multiplied by `E^2`

On the current cluster JAX-MD environment, this gives an effective GLE time step of approximately `0.204548` for `dt = 10.0`, and an effective temperature of approximately `0.079488` for `temperature = 40.0`. Earlier generated-state instability came from accidentally using raw `10.0` and raw `40.0` in the clean scaffold.

## Final Bulk Reproduction Ladder

1. Recompute explicit-reference observables from staged arrays: solute RDFs, solvent `W-W` diagnostic RDF if available, and solute VACF.
2. Recompute the explicit-reference mobility curve from driven-field velocity arrays.
3. Recompute the retained-solute CG conservative model from `A-A`, `A-B`, and `B-B` RDF targets.
4. Recompute the equilibrium memory kernel by solving the Volterra equation from the equilibrium VACF.
5. Run the baseline GLE using retained solutes only, the tabulated CG potential, the equilibrium memory kernel, and colored noise.
6. Train single-force and multi-force GLE-NECK corrective kernels through differentiable GLE simulation using drift-velocity targets.
7. Validate the mobility curve, learned kernel evolution, and zero-field recovery against the current processed artifacts.

## What Is Still Missing

The clean package does not yet run full JAX-MD/GLE-NECK training. The next implementation step is to build the JAX-MD bulk runner around the extracted data contract and model defaults, then run CPU-scale smoke tests before submitting SLURM GPU reruns.

The missing private/staged inputs for complete reruns are the explicit-reference equilibrium and driven-field trajectories/velocities, plus the retained-solute `traj_cg.npy` and `vel_cg.npy` arrays used to initialize clean GLE-NECK training.
