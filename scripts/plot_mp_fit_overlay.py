#!/usr/bin/env python3
"""Overlay empirical J_l eigenvalue spectra with their fitted Marchenko-Pastur
density, one grid of subplots (all layers) per checkpoint.

Reuses the least-squares MP fit (mp_bin_masses / fit_mp) from
plot_mp_fit_r2.py so the fitted sigma^2, q, R^2 match that script exactly.
Intended to visually answer "how good is the MP fit" for the first and last
checkpoint of a run.

Uses matplotlib (not plotly/kaleido) because kaleido's bundled headless
Chrome is missing system shared libraries (libglib-2.0) in this environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_mp_fit_r2 import BINS, fit_mp, mp_bin_masses
from plot_spectra_dashboard import spectrum_steps

GRID_COLS = 6


def layer_eigenvalues(step_dir: Path, layer: int) -> np.ndarray | None:
    with h5py.File(step_dir / "spectra.h5", "r") as handle:
        group = handle.get(f"layer_{layer}")
        if group is None or "eigenvalues" not in group:
            return None
        values = np.asarray(group["eigenvalues"], dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        return values if values.size >= 100 else None


def density_from_masses(masses: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Convert per-bin probability mass to density per decade (log10 bin width)."""
    return masses / np.diff(np.log10(bins))


def make_checkpoint_figure(results_dir: Path, step: int, n_layers: int, corpus_label: str):
    centers = np.sqrt(BINS[:-1] * BINS[1:])
    rows = -(-n_layers // GRID_COLS)
    fig, axes = plt.subplots(rows, GRID_COLS, figsize=(GRID_COLS * 2.4, rows * 2.0), squeeze=False)
    fit_rows: list[dict] = []
    step_dir = results_dir / f"step{step:04d}"

    for layer in range(n_layers):
        ax = axes[layer // GRID_COLS][layer % GRID_COLS]
        eigenvalues = layer_eigenvalues(step_dir, layer)
        if eigenvalues is None:
            ax.axis("off")
            continue
        sigma2, q, r2 = fit_mp(eigenvalues)
        fit_rows.append({"layer": layer, "sigma2": sigma2, "q": q, "r2": r2, "n": eigenvalues.size})

        empirical_counts, _ = np.histogram(eigenvalues, bins=BINS)
        empirical_density = density_from_masses(empirical_counts / eigenvalues.size, BINS)
        fit_density = density_from_masses(mp_bin_masses(sigma2, q, BINS), BINS)

        ax.fill_between(centers, empirical_density, step="mid", color="#1f77b4", alpha=0.3, linewidth=0)
        ax.plot(centers, empirical_density, color="#1f77b4", linewidth=0.8, drawstyle="steps-mid")
        ax.plot(centers, fit_density, color="#d62728", linewidth=1.3)
        ax.set_xscale("log")
        ax.set_xlim(BINS[0], BINS[-1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.97, 0.90, f"L{layer}\nR\u00b2={r2:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=7)

    for layer in range(n_layers, rows * GRID_COLS):
        axes[layer // GRID_COLS][layer % GRID_COLS].axis("off")

    mean_r2 = float(np.mean([r["r2"] for r in fit_rows])) if fit_rows else float("nan")
    fig.suptitle(
        f"Marchenko-Pastur fit vs. empirical J_l spectra -- {corpus_label}, step {step:,} "
        f"(mean R\u00b2 = {mean_r2:.3f})",
        fontsize=13,
    )
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", lw=3, label="Empirical"),
            plt.Line2D([0], [0], color="#d62728", lw=2, label="MP fit"),
        ],
        loc="upper right", bbox_to_anchor=(0.99, 0.985), fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, fit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("/data/jlens/pythia-12b-deduped"))
    parser.add_argument("--label", default="wikitext")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/pythia-12b"))
    args = parser.parse_args()

    checkpoints = spectrum_steps(args.results_dir)
    steps = sorted(checkpoints)
    if not steps:
        raise ValueError(f"No completed spectra.h5 checkpoints found in {args.results_dir}.")
    first_step, last_step = steps[0], steps[-1]

    with h5py.File(checkpoints[first_step] / "spectra.h5", "r") as handle:
        n_layers = int(handle.attrs["n_layers"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in (first_step, last_step):
        fig, fit_rows = make_checkpoint_figure(args.results_dir, step, n_layers, args.label)
        output = args.output_dir / f"mp_fit_overlay_step{step:04d}.png"
        fig.savefig(output, dpi=150)
        plt.close(fig)
        mean_r2 = np.mean([r["r2"] for r in fit_rows])
        worst = min(fit_rows, key=lambda r: r["r2"])
        print(f"step {step:>6,}: mean R\u00b2 = {mean_r2:.4f}, worst layer = L{worst['layer']} (R\u00b2={worst['r2']:.4f})")
        print(f"  Wrote {output}")


if __name__ == "__main__":
    main()
