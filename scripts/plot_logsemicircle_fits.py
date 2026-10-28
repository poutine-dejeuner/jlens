#!/usr/bin/env python3
"""Fit a semicircle distribution in log eigenvalue to a Pythia spectrum."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.optimize import least_squares

from plot_logskewnormal_fits import BINS, density_per_decade, histogram_masses, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

GRID_COLUMNS = 6


def semicircle_cdf(log_eigenvalues: np.ndarray, center: float, radius: float) -> np.ndarray:
    """Return the CDF of a semicircle distribution in log eigenvalue."""
    scaled = (np.asarray(log_eigenvalues, dtype=float) - center) / radius
    cdf = np.empty_like(scaled)
    cdf[scaled <= -1.0] = 0.0
    cdf[scaled >= 1.0] = 1.0
    inside = (scaled > -1.0) & (scaled < 1.0)
    z = scaled[inside]
    cdf[inside] = 0.5 + (z * np.sqrt(1.0 - z**2) + np.arcsin(z)) / np.pi
    return cdf


def fit_log_semicircle(eigenvalues: np.ndarray) -> tuple[float, float, float]:
    """Fit log-semicircle center and radius by binned least squares."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("Spectrum contains no finite positive eigenvalues.")
    logs = np.log(values)
    target = histogram_masses(values)
    initial_radius = max(2.0 * logs.std(), 1e-3)
    starts = [
        np.array([logs.mean(), np.log(initial_radius * scale)])
        for scale in (0.5, 1.0, 2.0, 4.0)
    ]
    best = None
    for start in starts:
        result = least_squares(
            lambda parameters: np.diff(
                semicircle_cdf(
                    np.log(BINS),
                    center=parameters[0],
                    radius=np.exp(parameters[1]),
                )
            )
            - target,
            start,
            bounds=(np.array([-30.0, -8.0]), np.array([10.0, 8.0])),
            max_nfev=3_000,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    center, log_radius = best.x
    radius = float(np.exp(log_radius))
    predicted = np.diff(semicircle_cdf(np.log(BINS), center, radius))
    return r2_score(target, predicted), float(center), radius


def spectra_path_for_step(results_dir: Path, step: int) -> Path:
    """Find a checkpoint spectrum regardless of zero padding in its directory."""
    matches = [
        path / "spectra.h5"
        for path in results_dir.glob("step*")
        if path.is_dir()
        and path.name.removeprefix("step").isdigit()
        and int(path.name.removeprefix("step")) == step
        and (path / "spectra.h5").exists()
    ]
    if len(matches) != 1:
        raise ValueError(f"Missing checkpoint spectra for step {step}: {results_dir}")
    return matches[0]


def plot_step(
    spectra_path: Path,
    step: int,
    output: Path,
    csv_output: Path,
) -> None:
    """Fit and plot every nontrivial layer in one checkpoint."""
    rows: list[dict[str, float | int]] = []
    with h5py.File(spectra_path, "r") as handle:
        n_layers = int(handle.attrs["n_layers"]) - 1
        n_rows = int(np.ceil(n_layers / GRID_COLUMNS))
        figure, axes = plt.subplots(
            n_rows,
            GRID_COLUMNS,
            figsize=(GRID_COLUMNS * 2.5, n_rows * 2.45),
            squeeze=False,
        )
        centers = np.sqrt(BINS[:-1] * BINS[1:])
        for layer in range(n_layers):
            axis = axes[layer // GRID_COLUMNS][layer % GRID_COLUMNS]
            values = np.asarray(handle[f"layer_{layer}/eigenvalues"], dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            r2, center, radius = fit_log_semicircle(values)
            fit_masses = np.diff(semicircle_cdf(np.log(BINS), center, radius))
            empirical = density_per_decade(histogram_masses(values), BINS)
            fitted = density_per_decade(fit_masses, BINS)
            rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "r2_logsemicircle": r2,
                    "log_center": center,
                    "log_radius": radius,
                }
            )
            axis.fill_between(
                centers, empirical, step="mid", color="#1f77b4", alpha=0.25, linewidth=0
            )
            axis.plot(
                centers,
                empirical,
                color="#1f77b4",
                linewidth=0.75,
                drawstyle="steps-mid",
            )
            axis.plot(centers, fitted, color="#d62728", linewidth=1.2)
            axis.set_xscale("log")
            axis.set_yticks([])
            axis.tick_params(axis="x", which="both", labelsize=5.5, pad=1)
            axis.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=3))
            axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.text(
                0.03,
                0.96,
                f"L{layer}\n$R^2$={r2:.3f}\n$c$={center:.2f}\n$R$={radius:.2f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
            )

        for layer in range(n_layers, n_rows * GRID_COLUMNS):
            axes[layer // GRID_COLUMNS][layer % GRID_COLUMNS].axis("off")

    figure.suptitle(
        f"Log-semicircle fit vs. empirical J_l spectra -- "
        f"Pythia-12B-deduped, step {step:,}",
        fontsize=13,
    )
    figure.legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", lw=3, label="Empirical"),
            plt.Line2D([0], [0], color="#d62728", lw=2, label="Log-semicircle"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/pythia_logsemicircle_overlay_step1.png"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/paper/logsemicircle_step1.csv"),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Fit the selected checkpoint and write its overlay grid."""
    args = parse_args()
    spectra_path = spectra_path_for_step(args.results_dir, args.step)
    plot_step(spectra_path, args.step, args.output, args.csv)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
