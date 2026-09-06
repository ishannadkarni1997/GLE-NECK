from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_CLUSTER_WORKSPACE = Path(os.environ.get("GLENECK_WORKSPACE", "gleneck-workspace"))


@dataclass(frozen=True)
class BulkSpeciesContract:
    """Physical retained/eliminated split for the bulk GLE-NECK system."""

    retained_species: tuple[str, ...] = ("A", "B")
    eliminated_species: tuple[str, ...] = ("W",)
    retained_particle_count: int = 242
    retained_pair_labels: tuple[str, ...] = ("A-A", "A-B", "B-B")
    solvent_diagnostic_label: str = "W-W"
    notes: str = (
        "The explicit reference system contains retained solutes A/B in explicit solvent W/C. "
        "The GLE/CG model propagates only A/B solute coordinates. Solvent structure such "
        "as W-W RDFs characterizes the explicit reference liquid; solvent effects "
        "enter the GLE through the PMF, equilibrium memory/noise, and learned NECK correction."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BulkModelConfig:
    """Default parameters for the public bulk GLE-NECK model."""

    n_cg: int = 242
    n_a: int = 121
    n_b: int = 121
    box_size: tuple[float, float, float] = (100.0, 50.0, 50.0)
    mass_a: float = 22.5
    mass_b: float = 35.5
    r_min: float = 0.1
    r_max: float = 10.0
    num_bins: int = 100
    r_onset: float = 1.0
    r_cutoff: float = 10.0
    dr_threshold: float = 1.0
    l_max: int = 300
    memory_orig_interval: float = 1.0
    embed_dim: int = 32
    poly_degree: int = 1
    kernel_model: str = "asym_softmax_poly"
    kernel_field_scale: float = 1.0
    kernel_tau_prior_ps: float = 0.0
    kernel_exp_basis_taus_ps: tuple[float, ...] = (0.05, 0.1, 0.15, 0.25, 0.4)
    kernel_shape_regularization_weight: float = 0.0
    kernel_l2_regularization_weight: float = 0.0
    kernel_time_moment_regularization_weight: float = 0.0
    kernel_derivative_regularization_weight: float = 0.0
    kernel_second_derivative_regularization_weight: float = 0.0
    memory_history_scaling: str = "force-units"
    learning_rate: float = 1e-2
    dt: float = 10.0
    temperature: float = 40.0
    init_velocity_scale: float = 1.0
    noise_scale: float = 1.1
    drift_window: int = 10
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BulkTrainingConfig:
    """Training fields, target drifts, and settings for one bulk GLE-NECK run."""

    name: str
    mode: str
    fields: tuple[float, ...]
    target_drifts: tuple[float, ...]
    epochs: int
    steps_per_loss: int = 1000
    checkpoint_stride: int = 1
    seed: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.fields) != len(self.target_drifts):
            raise ValueError("fields and target_drifts must have matching lengths.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BulkAATargetConfig:
    """Explicit-solvent bulk reference defaults for target generation."""

    n_solvent: int = 6508
    n_a: int = 121
    n_b: int = 121
    box_size: tuple[float, float, float] = (100.0, 50.0, 50.0)
    temperature: float = 40.0
    dt: float = 1.0
    gamma: float = 0.02
    thermostat: str = "nose-hoover"
    langevin_mode: str = "solvent-yz"
    force_mode: str = "solute"
    drift_reference: str = "solvent"
    r_cutoff: float = 4.0
    dr_threshold: float = 1.0
    rdf_cutoff: float = 10.0
    rdf_dr: float = 0.1
    mass_w: float = 18.5
    mass_a: float = 22.5
    mass_b: float = 35.5
    sigma_w: float = 2.4
    sigma_a: float = 2.8
    sigma_b: float = 3.4
    epsilon_w: float = 0.0138
    epsilon_a: float = 0.0130
    epsilon_b: float = 0.0230
    default_fields: tuple[float, ...] = (
        0.0,
        0.011111111111111112,
        0.022222222222222223,
        0.03333333333333333,
        0.044444444444444446,
        0.05555555555555556,
        0.06666666666666667,
        0.07777777777777778,
        0.08888888888888889,
        0.1,
    )
    seed: int = 0
    notes: str = (
        "Explicit-solvent reference model: "
        "A/B solutes in W/C solvent. The solute-counter-solvent force mode is a "
        "zero-net-force diagnostic control."
    )

    @property
    def n_retained(self) -> int:
        return self.n_a + self.n_b

    @property
    def n_total(self) -> int:
        return self.n_solvent + self.n_retained

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_retained"] = self.n_retained
        data["n_total"] = self.n_total
        return data


DEFAULT_BULK_MODEL = BulkModelConfig()
DEFAULT_BULK_SPECIES = BulkSpeciesContract()
DEFAULT_BULK_AA_TARGET = BulkAATargetConfig()

DEFAULT_SPT_LOW_TRAINING = BulkTrainingConfig(
    name="bulk_spt_E0p5",
    mode="single-force",
    fields=(0.5,),
    target_drifts=(0.260510302147,),
    epochs=300,
    notes="Single-point comparison at E = 0.5.",
)

DEFAULT_SPT_HIGH_TRAINING = BulkTrainingConfig(
    name="bulk_spt_E1",
    mode="single-force",
    fields=(1.0,),
    target_drifts=(0.414644441332,),
    epochs=300,
    notes="Single-point comparison at E = 1.0.",
)

DEFAULT_SPT_MAX_TRAINING = BulkTrainingConfig(
    name="bulk_spt_E2",
    mode="single-force",
    fields=(2.0,),
    target_drifts=(0.615561361077,),
    epochs=300,
    notes="Single-point comparison at E = 2.0.",
)

DEFAULT_MPT_TRAINING = BulkTrainingConfig(
    name="bulk_mpt_E0p5_1_2",
    mode="multi-force",
    fields=(0.5, 1.0, 2.0),
    target_drifts=(0.260510302147, 0.414644441332, 0.615561361077),
    epochs=800,
    notes="Multi-point training at E = 0.5, 1.0, and 2.0.",
)


DEFAULT_TRAINING_CONFIGS = (
    DEFAULT_SPT_LOW_TRAINING,
    DEFAULT_SPT_HIGH_TRAINING,
    DEFAULT_SPT_MAX_TRAINING,
    DEFAULT_MPT_TRAINING,
)
