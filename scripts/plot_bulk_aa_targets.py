#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


KB_KCAL_PER_MOL_K = 0.00198720425864083


def load_csv(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        raise ValueError(f"{path} does not contain tabular data.")
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names or ()}


def pmf_from_rdf(g_r: np.ndarray, temperature: float) -> np.ndarray:
    safe = np.clip(np.asarray(g_r, dtype=float), 1.0e-8, None)
    potential = -KB_KCAL_PER_MOL_K * temperature * np.log(safe)
    tail = potential[-20:]
    finite = tail[np.isfinite(tail)]
    if finite.size:
        potential = potential - float(np.mean(finite))
    return potential


def plot_targets(target_dir: Path, output: Path | None, temperature: float) -> Path:
    rdf_path = target_dir / "AA_rdf_targets_smooth.csv"
    if not rdf_path.exists():
        rdf_path = target_dir / "AA_retained_rdf_targets.csv"
    vacf_path = target_dir / "AA_retained_vacf_target.csv"
    if not rdf_path.exists():
        raise FileNotFoundError(f"Missing RDF target file under {target_dir}")
    if not vacf_path.exists():
        raise FileNotFoundError(f"Missing VACF target file under {target_dir}")

    rdf = load_csv(rdf_path)
    vacf = load_csv(vacf_path)
    r = rdf["r_distance"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    colors = {"g_r_AA": "tab:blue", "g_r_BB": "tab:orange", "g_r_AB": "tab:green", "g_r_WW": "0.55"}
    labels = {"g_r_AA": "A-A", "g_r_BB": "B-B", "g_r_AB": "A-B", "g_r_WW": "Solvent-solvent"}
    for key in ("g_r_AA", "g_r_BB", "g_r_AB", "g_r_WW"):
        if key in rdf and np.any(np.isfinite(rdf[key])):
            axes[0].plot(r, rdf[key], color=colors[key], lw=2.0, label=labels[key])
    axes[0].axhline(1.0, color="0.6", lw=1.2, ls=":")
    axes[0].set_title("AA equilibrium RDF targets")
    axes[0].set_xlabel("r (A)")
    axes[0].set_ylabel("g(r)")
    axes[0].legend(frameon=True)

    for key in ("g_r_AA", "g_r_BB", "g_r_AB"):
        if key in rdf and np.any(np.isfinite(rdf[key])):
            axes[1].plot(r, pmf_from_rdf(rdf[key], temperature), color=colors[key], lw=2.0, label=labels[key])
    axes[1].axhline(0.0, color="0.6", lw=1.2, ls=":")
    axes[1].set_title("Boltzmann-inverted solute potentials")
    axes[1].set_xlabel("r (A)")
    axes[1].set_ylabel("U(r) (kcal/mol)")
    axes[1].set_ylim(-0.12, 0.22)
    axes[1].legend(frameon=True)

    axes[2].plot(vacf["time_lag"], vacf["vacf_solute_norm"], color="black", lw=2.0)
    axes[2].axhline(0.0, color="0.6", lw=1.2, ls=":")
    axes[2].set_title("AA retained-solute VACF")
    axes[2].set_xlabel("tau (ps)")
    axes[2].set_ylabel("Normalized VACF")

    fig.suptitle(target_dir.name)
    if output is None:
        output = target_dir / "AA_equilibrium_target_diagnostics.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot bulk AA equilibrium RDF/VACF target diagnostics.")
    parser.add_argument("target_dir", type=Path, help="Directory containing AA_rdf_targets_smooth.csv and AA_retained_vacf_target.csv.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path. Defaults inside target_dir.")
    parser.add_argument("--temperature", type=float, default=40.0, help="Temperature used for Boltzmann inversion.")
    args = parser.parse_args()
    print(plot_targets(args.target_dir, args.output, args.temperature))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
