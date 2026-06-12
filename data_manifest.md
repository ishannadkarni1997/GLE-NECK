# Data Manifest

This project separates compact processed artifacts from raw/private simulation data.

## Included or Intended for GitHub

`data/processed/bulk/`

- RDF and VACF validation CSVs.
  - `A-A`, `A-B`, and `B-B` RDFs are retained-solute observables used for the CG/GLE conservative model.
  - `W-W` RDFs are explicit-solvent diagnostics and are not propagated in the retained GLE state.
- Equilibrium memory-kernel NPY files.
- Bulk mobility, training-loss, drift, kernel-evolution, corrective-kernel, and friction-distribution CSVs.
- `asym_MPT_kernel_evolution_900epochs.npz`, a portable NumPy snapshot bundle derived from the two legacy multi-force JAX-array pickle files.
- `cg_potentials_NVE242.npz`, a portable two-species tabulated CG potential derived from the legacy JAX-array pickle. This is for future bulk scientific reruns, not Chapter 5 figure plotting.
- `bulk_reproducibility_audit.json`, a machine-readable audit of bulk rerun inputs and extracted training defaults.

`data/processed/confinement/`

- Confined RDF, density, wall/fluid potential, VACF, velocity-profile, training-loss, and learned-kernel CSV/NPY artifacts.

## Excluded from GitHub

- Raw trajectories and velocities.
- Private bulk rerun arrays such as `data/private/bulk/traj_cg.npy` and `data/private/bulk/vel_cg.npy`, which contain retained-solute positions and velocities.
- Fresh explicit-solvent target-generation outputs under `data_private/bulk/aa_targets/` on the cluster.
- Large all-atom simulation dumps.
- Private cluster paths and user-specific scratch directories.
- SLURM logs and generated figures.

## Legacy Provenance

The current source artifacts live in:

- `code/Bulk/`
- `code/Confinement/`

The notebooks in those folders are exploratory and are preserved unchanged during the first reproducibility milestone.
