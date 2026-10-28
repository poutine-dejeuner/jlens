#!/usr/bin/env python3
"""Plot raw spectra with fitted three-component mixture curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
import numpy as np

from fit_pythia_three_component_mixture import mixture_masses
from plot_logskewnormal_fits import BINS, density_per_decade, histogram_masses, overlay_limits
from plot_pythia_skewnormal_heatmap import checkpoint_paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

GRID_COLUMNS = 6


def load_rows(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="") as handle:
        return {
            int(row["layer"]): {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        }


def plot_step(step: int, spectra_path: Path, rows: dict[int, dict[str, float]], output: Path) -> None:
    with h5py.File(spectra_path, "r") as handle:
        n_layers = int(handle.attrs["n_layers"]) - 1
        n_rows = int(np.ceil(n_layers / GRID_COLUMNS))
        figure, axes = plt.subplots(n_rows, GRID_COLUMNS, figsize=(15, n_rows * 2.45), squeeze=False)
        centers = np.sqrt(BINS[:-1] * BINS[1:])
        for layer in range(n_layers):
            axis = axes[layer // GRID_COLUMNS][layer % GRID_COLUMNS]
            row = rows[layer]
            values = np.asarray(handle[f"layer_{layer}/eigenvalues"], dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            parameters = np.array([
                np.log(row["mp_sigma2"]), row["mp_q"], row["semicircle_center"],
                np.log(row["semicircle_radius"]), row["lognormal_location"],
                row["lognormal_log_scale"],
                np.log(row["mp_weight"] / row["lognormal_weight"]),
                np.log(row["semicircle_weight"] / row["lognormal_weight"]),
            ])
            empirical = density_per_decade(histogram_masses(values), BINS)
            fitted = density_per_decade(np.clip(mixture_masses(parameters), 0.0, None), BINS)
            x_low, x_high = overlay_limits(empirical, fitted)
            axis.fill_between(centers, empirical, step="mid", color="#1f77b4", alpha=0.25, linewidth=0)
            axis.plot(centers, empirical, color="#1f77b4", linewidth=0.75, drawstyle="steps-mid")
            axis.plot(centers, fitted, color="#111111", linewidth=1.2)
            axis.set_xscale("log")
            axis.set_xlim(x_low, x_high)
            axis.set_yticks([])
            axis.tick_params(axis="x", which="both", labelsize=5.5, pad=1)
            axis.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=3))
            axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.text(
                0.03, 0.96,
                f"L{layer}  3-component mixture\n$R^2$={row['r2']:.3f}\n"
                f"MP={row['mp_weight']:.2f}  semi={row['semicircle_weight']:.2f}\n"
                f"LN={row['lognormal_weight']:.2f}\n"
                f"MP: s2={row['mp_sigma2']:.3g}, q={row['mp_q']:.3f}\n"
                f"Semi: c={row['semicircle_center']:.3f}, R={row['semicircle_radius']:.3f}\n"
                f"LN: mu={row['lognormal_location']:.3f}, "
                f"sigma={np.exp(row['lognormal_log_scale']):.3f}",
                transform=axis.transAxes, ha="left", va="top", fontsize=6.3,
            )
        for layer in range(n_layers, n_rows * GRID_COLUMNS):
            axes[layer // GRID_COLUMNS][layer % GRID_COLUMNS].axis("off")
    figure.suptitle(f"Pythia-12B-deduped three-component mixture -- step {step:,}", fontsize=13)
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    paths = checkpoint_paths(args.results_dir)
    for step in args.steps:
        plot_step(
            step, paths[step], load_rows(args.fit_dir / f"step_{step}.csv"),
            args.output_dir / f"pythia_three_component_overlay_step{step}.png",
        )


if __name__ == "__main__":
    main()
