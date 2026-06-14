# Data Manifest

This repository commits only compact processed artifacts for the bulk transport workflow. They are sufficient to regenerate public figures and run smoke tests; they are not a substitute for archived raw trajectories.

## Bulk Processed Artifacts

All files live in `data/processed/bulk/`.

| File | Contents |
| --- | --- |
| `aa_equilibrium_rdf.csv` | Equilibrium AA RDF targets for `A-A`, `B-B`, `A-B`, and solvent diagnostic `W-W` |
| `aa_vacf.csv` | Equilibrium retained-solute AA VACF target on a physical ps axis |
| `aa_mobility.csv` | Driven AA mobility target, reported as relative solute drift versus field |
| `ibi_potentials.csv` | IBI/PMF solute potentials for `A-A`, `A-B`, and `B-B` |
| `ibi_potentials.npz` | Portable NumPy copy of the same solute potential tables |
| `memory_kernel.csv` | Volterra memory reconstruction with raw and fitted kernels in `ps^-2` |
| `gle_baseline_rdf.csv` | RDFs from the retained-solute baseline GLE |
| `gle_baseline_vacf.csv` | VACF from the retained-solute baseline GLE |
| `gle_baseline_mobility.csv` | Mobility response of the equilibrium baseline GLE |
| `gleneck_mpt_mobility.csv` | Promoted multi-point GLE-NECK mobility response |
| `gleneck_mpt_training_loss.csv` | Promoted multi-point training loss |
| `gleneck_kernel_evolution.npz` | Downsampled corrective-kernel history for public figures |
| `gleneck_spt_mobility.csv` | Optional single-point mobility curves used in SPT/MPT comparison |
| `gleneck_spt_training_loss.csv` | Optional single-point training losses used in SPT/MPT comparison |
| `bulk_result_summary.json` | Machine-readable summary of the promoted bulk result |

## Excluded Data

The public repository excludes raw AA trajectories, retained-solute position/velocity histories, large training checkpoints, scheduler logs, exploratory notebooks, and draft manuscript files. Those data should be archived separately for publication if they are needed for full reruns.
