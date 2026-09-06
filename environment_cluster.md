# Cluster Environment Notes

These notes record the environment used for the bulk GPU checks on May 31, 2026. Set the workspace location for your cluster with:

```text
$GLENECK_WORKSPACE
```

## Recorded Environment

- Login host: `splpslurm04`
- Workspace: `$GLENECK_WORKSPACE`
- Default Python: `Python 3.13.12` at `$CONDA_PREFIX_ROOT/bin/python`
- Default Python did not include NumPy or Matplotlib before creating the figure-reproduction venv.
- Figure generation was tested with Python `3.13.12`, NumPy `2.4.6`, and Matplotlib `3.10.9`.
- Simulation and training environment: conda env `gleneck-jax`
- `gleneck-jax` path: `$CONDA_PREFIX_ROOT/envs/gleneck-jax`
- `gleneck-jax` Python: `3.11.15`
- `gleneck-jax` package versions after setup:
  - `jax==0.10.1`
  - `jaxlib==0.10.1`
  - `jax-md==0.2.28`
  - `dm-haiku==0.0.16`
  - `optax==0.2.8`
  - `flax==0.12.7`
  - `numpy==2.4.6`

SLURM/GPU snapshot:

- `nvidia-smi` is not available on the login node.
- `sinfo` reports GPU partition `gpu*` with H100 nodes using `gpu:h100:8`.
- `hpcf_test` also reports `gpu:h100:8`.
- Probe job `87782` ran on `nodegpu303` and saw:
  - GPU: `NVIDIA H100 80GB HBM3`
  - Driver: `595.58.03`
  - GPU memory: `81559 MiB`
  - `CUDA_VISIBLE_DEVICES=0`
  - JAX default backend: `gpu`
  - JAX devices: `[CudaDevice(id=0)]`
  - Small JAX matrix multiplication completed successfully.
- Bulk synthetic smoke job `87783` ran on `nodegpu303` and wrote `$GLENECK_WORKSPACE/runs/bulk_smoke/bulk_jax_smoke.json`.
  - JAX backend: `gpu`
  - JAX device: `cuda:0`
  - Synthetic state shape: `(242, 3)`
  - JAX-MD tabulated-potential energy: `-0.5030135733395918`
  - JAX-MD force L2 norm: `0.021503159776329994`
  - Memory and noise-filter shapes: `(300,)`
  - Corrective-kernel L2 norm at field `0.0347`: `2.1974286765359535e-06`
  - Note: JAX-MD emitted a non-fatal `FutureWarning` about int dtype promotion in scatter internals.

Available CUDA module families observed:

- `cuda11.8/toolkit/11.8.0`
- `cuda12.3/toolkit/12.3.2`
- `cuda12.4/toolkit/12.4.1`
- `cuda12.5/toolkit/12.5.1`
- `cuda12.6/toolkit/12.6.3`
- `cuda12.8/toolkit/12.8.1`
- `cuda12.9/toolkit/12.9.0`

## Workspace Layout

```text
$GLENECK_WORKSPACE/
  repo/              # Git checkout
  data_private/      # raw/private trajectories, not committed
  runs/              # long-running training outputs
  slurm_logs/        # scheduler stdout/stderr
```

## Cluster-Specific Setup

Before running training jobs, check:

- Whether long training jobs need explicit CUDA modules or can rely on the pip-bundled CUDA libraries from `jax[cuda12]`.
- Whether multi-GPU jobs require NCCL-specific SLURM settings.
- Runtime memory use for the intended system size.

Figure generation and processed-data checks run on CPU. JAX/JAX-MD are required for simulation and training.
