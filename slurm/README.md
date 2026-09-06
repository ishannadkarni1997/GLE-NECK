# SLURM Templates

Use the template in this directory to run GLE-NECK jobs on a GPU cluster. Set the workspace path and adapt the resource requests to your cluster.

Before submitting jobs, set:

```bash
export GLENECK_WORKSPACE=/path/to/GLE-NECK/workspace
```

Expected layout:

```text
$GLENECK_WORKSPACE/
  repo/          # Git checkout
  data_private/  # raw trajectories and large checkpoints
  runs/          # generated model outputs
  slurm_logs/    # scheduler logs
```

Use `gleneck_gpu_job.sbatch.template` as the base and customize the partition, CUDA/conda setup, and command line for your cluster.
