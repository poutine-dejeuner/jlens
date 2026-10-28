#!/usr/bin/env python3
"""Plot raw Pythia spectra with MP, log-semicircle, and log-skew-normal fits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib
import numpy as np

from plot_logsemicircle_fits import semicircle_cdf
from plot_logskewnormal_fits import (
    BINS,
    FitResult,
    bin_masses,
    density_per_decade,
    histogram_masses,
    overlay_limits,
)
from fit_pythia_mp_lognormal_mixture import lognormal_bin_masses
from plot_pythia_skewnormal_heatmap import checkpoint_paths, mp_bin_masses

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

GRID_COLUMNS = 6

MODEL_LABELS = {
    "mp": "MP",
    "semicircle": "Log-semicircle",
    "logskewnormal": "Log-skew-normal",
    "mp_lognormal_mixture": "MP + log-normal",
}


def load_fit_rows(path: Path) -> dict[tuple[int, int], dict[str, float | str]]:
    """Load merged model-fit rows keyed by checkpoint step and layer."""
    with path.open(newline="") as handle:
        rows = {}
        for row in csv.DictReader(handle):
            rows[(int(row["step"]), int(row["layer"]))] = {
                key: value if key == "best_model" else float(value)
                for key, value in row.items()
            }
    return rows


def winning_curve(
    row: dict[str, float | str],
) -> tuple[np.ndarray, str, str, str, float]:
    """Return the winning fitted masses, display metadata, and R²."""
    model = str(row["best_model"])
    if model == "mp":
        return (
            mp_bin_masses(float(row["mp_sigma2"]), float(row["mp_q"])),
            "#9467bd",
            MODEL_LABELS[model],
            rf"$\sigma^2$={float(row['mp_sigma2']):.3g}" "\n"
            rf"$q$={float(row['mp_q']):.3f}",
            float(row["r2_mp"]),
        )
    if model == "semicircle":
        return (
            np.diff(
                semicircle_cdf(
                    np.log(BINS),
                    float(row["semicircle_log_center"]),
                    float(row["semicircle_log_radius"]),
                )
            ),
            "#d62728",
            MODEL_LABELS[model],
            rf"$c$={float(row['semicircle_log_center']):.3f}" "\n"
            rf"$R$={float(row['semicircle_log_radius']):.3f}",
            float(row["r2_semicircle"]),
        )
    if model == "logskewnormal":
        log_scale = float(row["skew_log_scale"])
        return (
            np.clip(
                bin_masses(
                    FitResult(
                        r2=float(row["r2_logskewnormal"]),
                        parameters=(
                            float(row["skew_shape"]),
                            float(row["skew_location"]),
                            log_scale,
                        ),
                    ),
                    "log-skew-normal",
                ),
                0.0,
                None,
            ),
            "#2ca02c",
            MODEL_LABELS[model],
            rf"$\alpha$={float(row['skew_shape']):.3f}" "\n"
            rf"$\mu$={float(row['skew_location']):.3f}" "\n"
            rf"$\sigma$={np.exp(log_scale):.3f}",
            float(row["r2_logskewnormal"]),
        )
    if model == "mp_lognormal_mixture":
        mp_weight = float(row["mixture_mp_weight"])
        mp = mp_bin_masses(
            float(row["mixture_mp_sigma2"]),
            float(row["mixture_mp_q"]),
        )
        lognormal = lognormal_bin_masses(
            float(row["mixture_lognormal_location"]),
            float(row["mixture_lognormal_log_scale"]),
        )
        return (
            np.clip(mp_weight * mp + (1.0 - mp_weight) * lognormal, 0.0, None),
            "#ff7f0e",
            MODEL_LABELS[model],
            rf"$w_\mathrm{{MP}}$={mp_weight:.3f}" "\n"
            rf"$\sigma^2_\mathrm{{MP}}$={float(row['mixture_mp_sigma2']):.3g}" "\n"
            rf"$q_\mathrm{{MP}}$={float(row['mixture_mp_q']):.3f}" "\n"
            rf"$\mu_\mathrm{{LN}}$={float(row['mixture_lognormal_location']):.3f}" "\n"
            rf"$\sigma_\mathrm{{LN}}$={np.exp(float(row['mixture_lognormal_log_scale'])):.3f}",
            float(row["r2_mp_lognormal_mixture"]),
        )
    raise ValueError(f"Unknown winning model: {model}")


def plot_step(
    step: int,
    spectra_path: Path,
    fit_rows: dict[tuple[int, int], dict[str, float | str]],
    output: Path,
) -> None:
    """Render one raw empirical-versus-three-model overlay grid."""
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
            row = fit_rows[(step, layer)]
            values = np.asarray(handle[f"layer_{layer}/eigenvalues"], dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            empirical = density_per_decade(histogram_masses(values), BINS)
            masses, color, model_label, parameters, r2 = winning_curve(row)
            fitted = density_per_decade(np.clip(masses, 0.0, None), BINS)
            x_low, x_high = overlay_limits(empirical, fitted)

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
            axis.plot(centers, fitted, color=color, linewidth=1.2)
            axis.set_xscale("log")
            axis.set_xlim(x_low, x_high)
            axis.set_yticks([])
            axis.tick_params(axis="x", which="both", labelsize=5.5, pad=1)
            axis.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=3))
            axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.text(
                0.03,
                0.96,
                f"L{layer}  {model_label}\n"
                f"$R^2$={r2:.3f}\n"
                f"{parameters}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.3,
            )
        for layer in range(n_layers, n_rows * GRID_COLUMNS):
            axes[layer // GRID_COLUMNS][layer % GRID_COLUMNS].axis("off")

    figure.suptitle(
        f"Pythia-12B-deduped spectra and fitted distributions -- step {step:,}",
        fontsize=13,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--fit-csv", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args(argv)


def main() -> None:
    """Render model overlays for selected checkpoints."""
    args = parse_args()
    paths = checkpoint_paths(args.results_dir)
    fit_rows = load_fit_rows(args.fit_csv)
    missing = set(args.steps) - paths.keys()
    if missing:
        raise ValueError(f"Missing checkpoint spectra for steps: {sorted(missing)}")
    for step in args.steps:
        output = args.output_dir / f"pythia_model_overlay_step{step}.png"
        plot_step(step, paths[step], fit_rows, output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
