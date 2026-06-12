#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as md
import os
import platform
import subprocess
import sys


def _version(package: str) -> str:
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return "not installed"


def _run(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=30)
    except FileNotFoundError:
        return 127, f"{command[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(command)} timed out"
    output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    return completed.returncode, output


def main() -> int:
    print("=== System ===")
    print(f"python={sys.version}")
    print(f"executable={sys.executable}")
    print(f"platform={platform.platform()}")
    print(f"hostname={platform.node()}")

    print("\n=== SLURM Environment ===")
    for name in (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NODELIST",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
    ):
        print(f"{name}={os.environ.get(name, '')}")

    print("\n=== nvidia-smi ===")
    code, output = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    print(f"returncode={code}")
    print(output)

    print("\n=== Package Versions ===")
    for package in ("jax", "jaxlib", "jax-md", "dm-haiku", "optax", "flax", "numpy"):
        print(f"{package}={_version(package)}")

    print("\n=== JAX Devices ===")
    import jax
    import jax.numpy as jnp

    print(f"jax_platform={jax.default_backend()}")
    print(f"jax_devices={jax.devices()}")
    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    y = x @ x
    print(f"matmul_sum={float(jnp.sum(y)):.6f}")

    print("\n=== Optional Imports ===")
    import haiku as hk
    import jax_md
    import optax

    print(f"haiku_module={hk.__name__}")
    print(f"jax_md_module={jax_md.__name__}")
    print(f"optax_module={optax.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

