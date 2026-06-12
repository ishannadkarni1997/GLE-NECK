#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-gleneck-jax}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then
  CONDA_SOLVER=mamba
else
  CONDA_SOLVER=conda
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment ${ENV_NAME} already exists; updating it."
  "${CONDA_SOLVER}" env update -y -n "${ENV_NAME}" -f "${ROOT_DIR}/environment_jax_cluster.yml" --prune
else
  echo "Creating conda environment ${ENV_NAME}."
  "${CONDA_SOLVER}" env create -y -n "${ENV_NAME}" -f "${ROOT_DIR}/environment_jax_cluster.yml"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

# Official JAX GPU wheels bundle CUDA/cuDNN runtime libraries and avoid tying the
# environment to a particular cluster module. Keep this installation step here
# rather than in the conda YAML so it can be adjusted independently if needed.
python -m pip install --upgrade \
  "jax[cuda12]==0.10.1" \
  "jax-md==0.2.28" \
  "dm-haiku==0.0.16" \
  "optax==0.2.8" \
  "flax==0.12.7"
python -m pip install -e "${ROOT_DIR}[dev]"

python - <<'PY'
import importlib.metadata as md
import sys

packages = ["jax", "jaxlib", "jax-md", "dm-haiku", "optax", "flax", "numpy"]
print("Python", sys.version)
for package in packages:
    try:
        print(f"{package}=={md.version(package)}")
    except md.PackageNotFoundError:
        print(f"{package}: not installed")
PY
