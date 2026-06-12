# Bulk GLE/GLE-NECK Unit Conventions

This project keeps one convention for the clean bulk workflow.

## Simulation Internals

- JAX-MD real-unit timestep scalar:
  `tu = 0.0204548282835039` internal time per femtosecond.
- A notebook timestep `dt=1.0` means:
  `dt_internal = dt * tu = 0.0204548282835039`
  and `dt_ps = dt * 0.001 = 0.001 ps`.
- GLE integrator arrays use internal JAX-MD time.
- Memory kernels used by the GLE integrator are in `internal_time^-2`.

## Physical Output

- User-facing time axes are physical picoseconds:
  `tau_ps = tau_internal / tu * 0.001`.
- User-facing memory-kernel axes are `ps^-2`:
  `M_ps2 = M_internal * (tu / 0.001)^2`.
- The CLI flag `--kernel-tau-prior-ps` is literal physical ps.
  Do not pre-convert it to internal time in SLURM scripts.

## Output File Rules

- CSVs may keep legacy compatibility columns such as `time_lag`, but must also
  include explicit columns such as `time_lag_internal`, `time_lag_ps`,
  `*_memory_internal`, and `*_memory_ps2`.
- Training NPZ files keep internal arrays for reruns and add `*_per_ps2`
  arrays for plotting.
- Plotting scripts should prefer `*_per_ps2` arrays/columns when present.
