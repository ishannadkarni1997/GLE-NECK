# Minimal Bulk Reproduction

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python scripts/make_bulk_figures.py --all
python scripts/check_bulk_reproducibility.py
pytest
```

The main outputs are written to `figures/bulk/`. The committed processed data used by these commands are documented in `data_manifest.md`.
