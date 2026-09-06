# Data Manifest

The included processed data can be used to regenerate the bulk figures and run the data-validation tests. Full simulation reruns also require the raw trajectories described below.

## Bulk Processed Data

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
| `gleneck_mpt_mobility.csv` | Multi-point GLE-NECK mobility response |
| `gleneck_mpt_training_loss.csv` | Multi-point training loss |
| `gleneck_validation_loss.csv` | Post-hoc train/held-out/full mobility MSE for SPT and MPT models over `E <= 2` |
| `gleneck_kernel_evolution.npz` | Downsampled corrective-kernel history for figures |
| `gleneck_spt_mobility.csv` | Optional single-point mobility curves used in SPT/MPT comparison |
| `gleneck_spt_training_loss.csv` | Optional single-point training losses used in SPT/MPT comparison |
| `bulk_result_summary.json` | Model settings, training metrics, and diagnostic warnings |

## Inputs for Full Reruns

Raw AA trajectories, retained-solute position and velocity histories, and large training checkpoints are stored separately. They are not required to regenerate figures from the processed data in this repository.
