# Chapter 5 Figure Audit

Current chapter target: `chap_5GLE_NECK_pdf.pdf`.

Running `python scripts/reproduce_figures.py --all` writes a fresh machine-readable provenance table to `outputs/figures/figure_provenance.csv` alongside the generated PNGs.

| Chapter figure | Content | Status | Output | Plotting function | Input artifact(s) | Legacy source |
|---|---|---|---|---|---|---|
| 5.1 | GLE-NECK delta-learning framework | Missing schematic asset | n/a | n/a | n/a | PPT/chapter draft only |
| 5.2 | Neural corrective-kernel architecture and asymptotic gating | Missing schematic asset | n/a | n/a | n/a | PPT/chapter draft only |
| 5.3 | Bulk differentiable training workflow | Missing schematic asset | n/a | n/a | n/a | PPT/chapter draft only |
| 5.4 | Bulk single-force corrective-kernel training | Artifact-backed | `outputs/figures/ch5_fig04_bulk_single_force_training.png` | `plot_bulk_single_force_training` | `asym_SPT_kernel_evolutionE0p1027V0p1729.csv`, `asym_SPT_training_loss_E0p1027V0p1729.csv` | `code/Bulk/clean_bulk_system.ipynb` E0p1027 exports; `figures.ipynb` cell 4 plotting pattern |
| 5.5 | Bulk multi-force kernel training | Artifact-backed from normalized legacy MPT kernel snapshots | `outputs/figures/ch5_fig05_bulk_multifield_kernels.png` | `plot_bulk_multifield_kernels` | `asym_MPT_kernel_evolution_900epochs.npz`, `fitted_memory_kernal.npy` | `code/Bulk/clean_bulk_system.ipynb` cells 173-178; `figures.ipynb` cell 5 plotting pattern |
| 5.6 | Bulk mobility response | Artifact-backed | `outputs/figures/ch5_fig06_bulk_mobility_response.png` | `plot_bulk_mobility` | `AA_mobility_data.csv`, `GLE_mobility_data.csv`, `asym_MPT_GLENECK_mobility_data_900Epoch_E0p0347_0p06883_0p1027.csv`, `SPT_GLENECK_mobility_dataE0p0347V0p11145.csv`, `SPT_GLENECK_mobility_dataE0p1027V0p1729.csv` | `code/Bulk/clean_bulk_system.ipynb` cells 180-181; `figures.ipynb` cell 6 |
| 5.7 | Bulk zero-field corrective-kernel behavior | Artifact-backed | `outputs/figures/ch5_fig07_bulk_corrective_kernels.png` | `plot_bulk_corrective_kernels` | `asym_MPT_900epoch_kernel_corrections_vs_field.csv` | `code/Bulk/clean_bulk_system.ipynb` cell 177; `figures.ipynb` cell 7 |
| 5.8 | Confined differentiable training workflow | Missing schematic asset | n/a | n/a | n/a | PPT/chapter draft only |
| 5.9 | Confined plug-like profile and loss | Artifact-backed | `outputs/figures/ch5_fig09_confinement_semiplug_profile.png` | `plot_confinement_profile` | `epoch0_160_E150_parabolic_semiplug_profile_evolution.csv`, `epoch0_160_E150_semiplug_training_loss.csv` | `code/Confinement/clean_confined_system.ipynb` cells 139-143; `figures/Bulk_confinement` figure3 pattern |
| 5.10 | Confined parabolic profile and loss | Artifact-backed | `outputs/figures/ch5_fig10_confinement_parabolic_profile.png` | `plot_confinement_profile` | `epoch0_200_E150_parabolic_velocity_profile_evolution.csv`, `epoch0_200_E150_parabolic_training_loss.csv` | `code/Confinement/clean_confined_system.ipynb` cells 139-143; `conf_figures.ipynb` cell 3 |
| 5.11 | Position-dependent confined kernels | Artifact-backed | `outputs/figures/ch5_fig11_confinement_kernels.png` | `plot_confinement_kernels` | `E_150_semiplug_epoch0_160_learnt_kernel.csv`, `E_150_parabolic_epoch0_200_learnt_kernel.csv` | `code/Confinement/clean_confined_system.ipynb` cells 147-148; `conf_figures.ipynb` cell 4 |

Supporting validation figures are also reproducible from processed artifacts:

- Bulk RDF and VACF/memory validation. The exact bulk IBI-potential panel currently depends on the legacy `CGpotentials_int_NVE242.pkl` object and should be converted in a JAX-capable environment before it is treated as portable.
- Confinement RDF, density, potential, VACF/memory validation.

The first code milestone does not recreate schematic figures from scratch; it records them as missing standalone assets.
