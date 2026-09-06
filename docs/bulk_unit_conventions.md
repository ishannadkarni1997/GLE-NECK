# Bulk GLE/GLE-NECK Unit Conventions

The bulk simulations use internal JAX-MD time units; figures report time in picoseconds. The conversions below apply to the integrator, saved data, and plotting scripts.

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

- CSVs should include explicit physical-time columns such as `time_ps` and
  memory columns such as `*_memory_ps2`.
- Training NPZ files keep internal arrays for reruns and add `*_per_ps2`
  arrays for plotting.
- Plotting scripts should prefer `*_per_ps2` arrays/columns when present.
