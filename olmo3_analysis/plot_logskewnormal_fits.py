#!/usr/bin/env python3
"""Compare log-normal and log-skew-normal fits of Jacobian Gram spectra.

The paper grid bins positive eigenvalues over [1e-6, 1e3] and retains mass
outside that interval in the normalization. This script uses the same
bin-mass least-squares objective for both nested models:

  log-normal:      log(lambda) ~ Normal(location, scale)
  log-skew-normal: log(lambda) ~ SkewNormal(shape, location, scale)

It is intended to diagnose the late-training shallow-layer regime where a
symmetric log-normal distribution visibly misses the skewed bulk.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm, skewnorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

# Olmo-3 spectra are analysed in normalized units λ/d_model. The fit window is
# the paper's raw [1e-6, 1e3] window carried into those units, so normalized
# fits equal raw fits with log-locations shifted by -log(d_model).
D_MODEL = 5120  # Olmo-3-1125-32B hidden size
BINS = np.geomspace(1e-6, 1e3, 811) / D_MODEL
LOG_BIN_EDGES = np.log(BINS)
SKEW_STARTS = (-12.0, -4.0, 0.0, 4.0, 12.0)
OVERLAY_COLUMNS = 5
OVERLAY_DISPLAY_BINS = 60


@dataclass(frozen=True)
class FitResult:
    """Least-squares fit quality for a spectral density model."""

    r2: float
    parameters: tuple[float, ...]


def spectrum_steps(results_dir: Path) -> dict[int, Path]:
    """Return all checkpoint directories containing spectra, indexed by step."""
    return {
        int(path.name.removeprefix("step")): path
        for path in results_dir.glob("step*")
        if (path / "spectra.h5").exists()
    }


def normalized_eigenvalues(handle: h5py.File, layer: int) -> np.ndarray:
    """Return finite positive eigenvalues of one layer as λ/d_model."""
    d_model = int(handle.attrs["d_model"])
    if d_model != D_MODEL:
        raise ValueError(f"Spectra have d_model={d_model}, fit window assumes {D_MODEL}.")
    values = np.asarray(handle[f"layer_{layer}/eigenvalues"], dtype=float)
    return values[np.isfinite(values) & (values > 0)] / d_model


def histogram_masses(eigenvalues: np.ndarray) -> np.ndarray:
    """Return fitted-window bin masses normalized by all positive eigenvalues."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("Spectrum contains no finite positive eigenvalues.")
    counts, _ = np.histogram(values, bins=BINS)
    return counts / values.size


def r2_score(target: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination for probability masses."""
    sst = float(np.sum((target - target.mean()) ** 2))
    if sst == 0.0:
        raise ValueError("Cannot calculate R² for constant bin masses.")
    return 1.0 - float(np.sum((target - predicted) ** 2)) / sst


def bin_masses(fit: FitResult, model: str, bins: np.ndarray = BINS) -> np.ndarray:
    """Return fitted probability mass in each eigenvalue bin."""
    log_edges = np.log(bins)
    if model == "log-normal":
        location, log_scale = fit.parameters
        cdf = norm.cdf(log_edges, loc=location, scale=np.exp(log_scale))
    elif model == "log-skew-normal":
        shape, location, log_scale = fit.parameters
        cdf = skewnorm.cdf(
            log_edges,
            a=shape,
            loc=location,
            scale=np.exp(log_scale),
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    return np.diff(cdf)


def _fit(
    target: np.ndarray,
    cdf,
    starts: list[np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
) -> FitResult:
    best = None
    for start in starts:
        result = least_squares(
            lambda parameters: np.diff(cdf(LOG_BIN_EDGES, parameters)) - target,
            start,
            bounds=(lower, upper),
            max_nfev=3_000,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    predicted = np.diff(cdf(LOG_BIN_EDGES, best.x))
    return FitResult(r2=r2_score(target, predicted), parameters=tuple(best.x))


def fit_log_normal(eigenvalues: np.ndarray) -> FitResult:
    """Fit a normal distribution to log eigenvalues by binned least squares."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    logs = np.log(values)
    target = histogram_masses(values)
    return _fit(
        target,
        lambda edges, parameters: norm.cdf(
            edges, loc=parameters[0], scale=np.exp(parameters[1])
        ),
        [np.array([logs.mean(), np.log(logs.std())])],
        np.array([-30.0, -8.0]),
        np.array([10.0, 8.0]),
    )


def fit_log_skew_normal(eigenvalues: np.ndarray) -> FitResult:
    """Fit a skew-normal distribution to log eigenvalues by binned least squares."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    logs = np.log(values)
    target = histogram_masses(values)
    return _fit(
        target,
        lambda edges, parameters: skewnorm.cdf(
            edges,
            a=parameters[0],
            loc=parameters[1],
            scale=np.exp(parameters[2]),
        ),
        [np.array([shape, logs.mean(), np.log(logs.std())]) for shape in SKEW_STARTS],
        np.array([-30.0, -30.0, -8.0]),
        np.array([30.0, 10.0, 8.0]),
    )


def fit_checkpoint(
    step: int, spectra_path: Path, max_layer: int | None
) -> list[dict[str, float | int]]:
    """Fit every layer in one checkpoint."""
    rows = []
    with h5py.File(spectra_path, "r") as handle:
        for layer in range(int(handle.attrs["n_layers"])):
            if max_layer is not None and layer > max_layer:
                continue
            group = handle.get(f"layer_{layer}")
            if group is None or "eigenvalues" not in group:
                continue
            values = normalized_eigenvalues(handle, layer)
            log_normal = fit_log_normal(values)
            log_skew_normal = fit_log_skew_normal(values)
            rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "r2_lognormal": log_normal.r2,
                    "r2_logskewnormal": log_skew_normal.r2,
                    "r2_gain": log_skew_normal.r2 - log_normal.r2,
                    "skew_shape": log_skew_normal.parameters[0],
                    "skew_location": log_skew_normal.parameters[1],
                    "skew_log_scale": log_skew_normal.parameters[2],
                }
            )
    return rows


def plot_grid(rows: list[dict[str, float | int]], output: Path) -> None:
    """Plot skew-normal fit quality and its gain over log-normal."""
    steps = sorted({int(row["step"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    r2 = np.full((len(layers), len(steps)), np.nan)
    gain = np.full_like(r2, np.nan)
    step_index = {step: index for index, step in enumerate(steps)}
    layer_index = {layer: index for index, layer in enumerate(layers)}
    for row in rows:
        index = layer_index[int(row["layer"])], step_index[int(row["step"])]
        r2[index] = float(row["r2_logskewnormal"])
        gain[index] = float(row["r2_gain"])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, values, title, cmap, lower, upper in (
        (
            axes[0],
            r2,
            r"Log-skew-normal $R^2$",
            "viridis",
            np.nanmin(r2),
            1.0,
        ),
        (
            axes[1],
            gain,
            r"$\Delta R^2$: log-skew-normal minus log-normal",
            "magma",
            0.0,
            np.nanmax(gain),
        ),
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=lower,
            vmax=upper,
        )
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.set_ylabel("layer")
        tick_indices = np.linspace(0, len(steps) - 1, min(8, len(steps)), dtype=int)
        axis.set_xticks(tick_indices, [str(steps[index]) for index in tick_indices], rotation=45)
        layer_ticks = np.linspace(0, len(layers) - 1, min(8, len(layers)), dtype=int)
        axis.set_yticks(layer_ticks, [str(layers[index]) for index in layer_ticks])
        figure.colorbar(image, ax=axis, shrink=0.85)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def density_per_decade(masses: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Convert bin masses to a density with respect to log10 eigenvalue."""
    return masses / np.diff(np.log10(bins))


def overlay_limits(*densities: np.ndarray) -> tuple[float, float]:
    """Return x limits spanning visible fitted or empirical density support."""
    centers = np.sqrt(BINS[:-1] * BINS[1:])
    retained = np.zeros(centers.size, dtype=bool)
    for density in densities:
        peak = density.max()
        if peak > 0:
            retained |= density > 0.01 * peak
    indices = np.flatnonzero(retained)
    if indices.size == 0:
        return float(BINS[0]), float(BINS[-1])
    return float(centers[indices[0]] / 1.4), float(centers[indices[-1]] * 1.4)


def plot_overlay(
    step: int,
    spectra_path: Path,
    max_layer: int | None,
    output: Path,
) -> None:
    """Plot empirical spectra with log-normal and log-skew-normal fits."""
    with h5py.File(spectra_path, "r") as handle:
        available_layers = int(handle.attrs["n_layers"])
        layers = list(range(available_layers))
        if max_layer is not None:
            layers = layers[: max_layer + 1]
        rows = int(np.ceil(len(layers) / OVERLAY_COLUMNS))
        figure, axes = plt.subplots(
            rows,
            OVERLAY_COLUMNS,
            figsize=(OVERLAY_COLUMNS * 2.5, rows * 2.45),
            squeeze=False,
        )

        for layer in layers:
            axis = axes[layer // OVERLAY_COLUMNS][layer % OVERLAY_COLUMNS]
            group = handle.get(f"layer_{layer}")
            if group is None or "eigenvalues" not in group:
                axis.axis("off")
                continue
            values = normalized_eigenvalues(handle, layer)
            if values.size == 0:
                axis.axis("off")
                continue

            normal_fit = fit_log_normal(values)
            skew_fit = fit_log_skew_normal(values)
            coarse_empirical = density_per_decade(histogram_masses(values), BINS)
            coarse_normal = density_per_decade(
                bin_masses(normal_fit, "log-normal"), BINS
            )
            coarse_skew = density_per_decade(
                bin_masses(skew_fit, "log-skew-normal"), BINS
            )
            x_low, x_high = overlay_limits(
                coarse_empirical, coarse_normal, coarse_skew
            )
            display_bins = np.geomspace(x_low, x_high, OVERLAY_DISPLAY_BINS + 1)
            centers = np.sqrt(display_bins[:-1] * display_bins[1:])
            counts, _ = np.histogram(values, bins=display_bins)
            empirical = density_per_decade(counts / values.size, display_bins)
            normal = density_per_decade(
                bin_masses(normal_fit, "log-normal", display_bins),
                display_bins,
            )
            skew = density_per_decade(
                bin_masses(skew_fit, "log-skew-normal", display_bins),
                display_bins,
            )

            axis.fill_between(
                centers, empirical, step="mid", color="#1f77b4", alpha=0.25, linewidth=0
            )
            axis.plot(
                centers,
                empirical,
                color="#1f77b4",
                linewidth=0.8,
                drawstyle="steps-mid",
            )
            axis.plot(centers, normal, color="#ff7f0e", linewidth=1.1)
            axis.plot(centers, skew, color="#d62728", linewidth=1.2)
            axis.set_xscale("log")
            axis.set_xlim(x_low, x_high)
            axis.set_yticks([])
            axis.tick_params(axis="x", which="both", labelsize=5.5, pad=1)
            axis.xaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=3))
            axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            axis.text(
                0.03,
                0.96,
                f"L{layer}\n"
                f"LN $R^2$={normal_fit.r2:.3f}\n"
                f"LSN $R^2$={skew_fit.r2:.3f}\n"
                f"$\\alpha$={skew_fit.parameters[0]:.1f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=6.5,
            )

        for layer in range(len(layers), rows * OVERLAY_COLUMNS):
            axes[layer // OVERLAY_COLUMNS][layer % OVERLAY_COLUMNS].axis("off")

    figure.suptitle(
        f"Log-spectrum fits vs. empirical J_l spectra -- Pythia-12B-deduped, "
        f"step {step:,}",
        fontsize=13,
    )
    figure.legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", lw=3, label="Empirical"),
            plt.Line2D([0], [0], color="#ff7f0e", lw=2, label="Log-normal"),
            plt.Line2D([0], [0], color="#d62728", lw=2, label="Log-skew-normal"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.985),
        fontsize=9,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=2_000)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Explicit checkpoint steps to fit (default: all steps after --min-step).",
    )
    parser.add_argument(
        "--max-layer",
        type=int,
        default=None,
        help="Fit layers through this inclusive index (default: all layers).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/paper/logskewnormal_comparison_pythia12b.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/pythia_logskewnormal_heatmap.png"),
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        default=None,
        help=(
            "Write empirical-versus-fit overlay grids for the first and last "
            "selected checkpoints to this directory."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Fit and plot all requested checkpoints."""
    args = parse_args()
    checkpoints = spectrum_steps(args.results_dir)
    if args.steps is None:
        selected_steps = [step for step in sorted(checkpoints) if step >= args.min_step]
    else:
        selected_steps = sorted(set(args.steps))
        missing_steps = [step for step in selected_steps if step not in checkpoints]
        if missing_steps:
            raise ValueError(f"Missing checkpoint spectra for steps: {missing_steps}")
    if not selected_steps:
        raise ValueError(f"No checkpoints at or after step {args.min_step}.")

    rows = []
    for step in selected_steps:
        rows.extend(
            fit_checkpoint(
                step,
                checkpoints[step] / "spectra.h5",
                args.max_layer,
            )
        )
        print(f"Fitted step {step}", flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_grid(rows, args.figure)
    print(f"Wrote {args.csv} and {args.figure}")
    if args.overlay_dir is not None:
        for step in (selected_steps[0], selected_steps[-1]):
            output = args.overlay_dir / f"pythia_logskewnormal_overlay_step{step}.png"
            plot_overlay(
                step,
                checkpoints[step] / "spectra.h5",
                args.max_layer,
                output,
            )
            print(f"Wrote {output}")


if __name__ == "__main__":
    main()
