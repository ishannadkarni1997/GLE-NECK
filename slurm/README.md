# SLURM Templates

The files in this directory are generic starting points for running GLE-NECK jobs on a GPU cluster. They intentionally avoid user-specific paths.

Before submitting jobs, set:

```bash
export GLENECK_WORKSPACE=/path/to/GLE-NECK/workspace
```

Expected layout:

```text
$GLENECK_WORKSPACE/
  repo/          # Git checkout
  data_private/  # raw trajectories and large private artifacts
  runs/          # generated model outputs
  slurm_logs/    # scheduler logs
```

Use `gleneck_gpu_job.sbatch.template` as the base and customize the partition, CUDA/conda setup, and command line for your cluster.
