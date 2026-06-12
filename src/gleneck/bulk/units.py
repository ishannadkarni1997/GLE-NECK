from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any


FEMTOSECOND_TO_PS = 1.0e-3


@dataclass(frozen=True)
class NotebookUnitScalars:
    """JAX-MD real-unit scalars used by the legacy notebooks."""

    distance: float = 1.0
    time: float = 0.0204548282835039
    energy: float = 1.0
    mass: float = 1.0
    temperature: float = 0.0019872035997726056


@lru_cache(maxsize=1)
def notebook_unit_scalars() -> NotebookUnitScalars:
    """Return JAX-MD real-unit scalars used by the notebooks.

    JAX-MD is not installed in the lightweight Mac plotting environment, so the
    fallback must match ``units.real_unit_system()`` rather than silently
    becoming dimensionless. Otherwise physical-time plot axes differ between
    local diagnostics and cluster training.
    """
    try:
        from jax_md import units
    except ModuleNotFoundError:
        return NotebookUnitScalars()

    raw = units.real_unit_system()
    return NotebookUnitScalars(
        distance=float(raw["distance"]),
        time=float(raw["time"]),
        energy=float(raw["energy"]),
        mass=float(raw["mass"]),
        temperature=float(raw["temperature"]),
    )


def effective_dt(config: Any) -> float:
    """Return the JAX-MD internal timestep used by the integrator."""
    return float(config.dt) * notebook_unit_scalars().time


def effective_dt_internal(config: Any) -> float:
    """Explicit alias for the integrator timestep in JAX-MD internal time."""
    return effective_dt(config)


def effective_dt_ps(config: Any) -> float:
    """Return the physical timestep in picoseconds.

    The legacy notebooks specify ``config.dt`` in femtoseconds before applying
    the JAX-MD real-unit scalar.  Keep this conversion independent of whether
    JAX-MD is importable so exported tables have stable physical axes.
    """
    return float(config.dt) * FEMTOSECOND_TO_PS


def internal_time_to_ps(value: Any) -> Any:
    """Convert JAX-MD internal time values to physical picoseconds."""
    return value / notebook_unit_scalars().time * FEMTOSECOND_TO_PS


def ps_to_internal_time(value: Any) -> Any:
    """Convert physical picoseconds to JAX-MD internal time values."""
    return value / FEMTOSECOND_TO_PS * notebook_unit_scalars().time


def memory_internal_to_per_ps2(value: Any) -> Any:
    """Convert a memory kernel from internal-time^-2 to ps^-2."""
    factor = (notebook_unit_scalars().time / FEMTOSECOND_TO_PS) ** 2
    return value * factor


def memory_per_ps2_to_internal(value: Any) -> Any:
    """Convert a memory kernel from ps^-2 to internal-time^-2."""
    factor = (FEMTOSECOND_TO_PS / notebook_unit_scalars().time) ** 2
    return value * factor


def effective_temperature(config: Any) -> float:
    return float(config.temperature) * notebook_unit_scalars().temperature


def effective_mass(value: float) -> float:
    return float(value) * notebook_unit_scalars().mass


def effective_field(value: float) -> float:
    units = notebook_unit_scalars()
    return float(value) * units.energy / units.distance
