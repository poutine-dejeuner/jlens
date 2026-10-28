#!/usr/bin/env python3
"""Rebuild the Pythia model-comparison grid with log-skew-normal fits.

Every cell is fit against the same 810 log-spaced eigenvalue-bin masses:
Marchenko--Pastur, a bounded semicircle in log eigenvalue, and a
log-skew-normal.  This makes the best-model heatmap internally comparable and
replaces the historical log-normal baseline rather than mixing fit protocols.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.optimize import least_squares

from plot_logsemicircle_fits import fit_log_semicircle
from plot_logskewnormal_fits import BINS, fit_log_skew_normal, histogram_masses, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

MP_QUADRATURE_NODES, MP_QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(4)
MP_BIN_MIDPOINTS = (BINS[:-1] + BINS[1:]) / 2
MP_BIN_HALF_WIDTHS = (BINS[1:] - BINS[:-1]) / 2
MP_QUADRATURE_POINTS = (
    MP_BIN_MIDPOINTS[:, None]
    + MP_BIN_HALF_WIDTHS[:, None] * MP_QUADRATURE_NODES[None, :]
)

MODEL_NAMES = ("mp", "logskewnormal", "semicircle", "mp_lognormal_mixture")
MODEL_COLORS = ("#1f77b4", "#2ca02c", "#d62728", "#ff7f0e")


def mp_bin_masses(sigma2: float, q: float) -> np.ndarray:
    """Approximate MP(sigma2, q) probability masses in the fixed log bins."""
    lower = sigma2 * (1.0 - np.sqrt(q)) ** 2
    upper = sigma2 * (1.0 + np.sqrt(q)) ** 2
    points = MP_QUADRATURE_POINTS
    radicand = np.clip((upper - points) * (points - lower), 0.0, None)
    density = np.sqrt(radicand) / (2.0 * np.pi * q * sigma2 * points)
    density[(points < lower) | (points > upper)] = 0.0
    return MP_BIN_HALF_WIDTHS * (density * MP_QUADRATURE_WEIGHTS).sum(axis=1)


def fit_mp(eigenvalues: np.ndarray) -> tuple[float, float, float]:
    """Fit Marchenko--Pastur scale and aspect ratio by binned least squares."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        raise ValueError("Spectrum contains no finite positive eigenvalues.")
    target = histogram_masses(values)
    median = float(np.median(values))
    best = None
    for scale in (0.5, 1.0, 2.0):
        for q_start in (0.1, 0.5, 0.95):
            result = least_squares(
                lambda parameters: mp_bin_masses(
                    np.exp(parameters[0]), parameters[1]
                )
                - target,
                x0=np.array([np.log(median * scale), q_start]),
                bounds=(np.array([-30.0, 1e-3]), np.array([10.0, 1.0])),
                max_nfev=3_000,
            )
            if best is None or result.cost < best.cost:
                best = result
    assert best is not None
    sigma2, q = float(np.exp(best.x[0])), float(best.x[1])
    return r2_score(target, mp_bin_masses(sigma2, q)), sigma2, q


def fit_checkpoint(
    step: int,
    spectra_path: Path,
    models: tuple[str, ...] = MODEL_NAMES,
) -> list[dict[str, float | int | str]]:
    """Fit selected models for all nontrivial transport spectra in one checkpoint."""
    rows = []
    with h5py.File(spectra_path, "r") as handle:
        for layer in range(int(handle.attrs["n_layers"]) - 1):
            values = np.asarray(handle[f"layer_{layer}/eigenvalues"], dtype=float)
            row: dict[str, float | int | str] = {"step": step, "layer": layer}
            if "mp" in models:
                mp_r2, mp_sigma2, mp_q = fit_mp(values)
                row.update(
                    {
                    "r2_mp": mp_r2,
                    "mp_sigma2": mp_sigma2,
                    "mp_q": mp_q,
                    }
                )
            if "logskewnormal" in models:
                skew_fit = fit_log_skew_normal(values)
                row.update(
                    {
                        "r2_logskewnormal": skew_fit.r2,
                        "skew_shape": skew_fit.parameters[0],
                        "skew_location": skew_fit.parameters[1],
                        "skew_log_scale": skew_fit.parameters[2],
                    }
                )
            if "semicircle" in models:
                semicircle_r2, semicircle_center, semicircle_radius = (
                    fit_log_semicircle(values)
                )
                row.update(
                    {
                        "r2_semicircle": semicircle_r2,
                        "semicircle_log_center": semicircle_center,
                        "semicircle_log_radius": semicircle_radius,
                    }
                )
            rows.append(row)
    return rows


def heatmap_arrays(
    rows: list[dict[str, float | int | str]],
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Return axes, categorical best-model grid, and best-fit-quality grid."""
    steps = sorted({int(row["step"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    model_index = {name: index for index, name in enumerate(MODEL_NAMES)}
    best_model = np.full((len(layers), len(steps)), np.nan)
    best_r2 = np.full_like(best_model, np.nan)
    step_index = {step: index for index, step in enumerate(steps)}
    layer_index = {layer: index for index, layer in enumerate(layers)}
    for row in rows:
        index = layer_index[int(row["layer"])], step_index[int(row["step"])]
        best_model[index] = model_index[str(row["best_model"])]
        best_r2[index] = max(
            float(row["r2_mp"]),
            float(row["r2_logskewnormal"]),
            float(row["r2_semicircle"]),
        )
    return steps, layers, best_model, best_r2


def plot_heatmap(rows: list[dict[str, float | int | str]], output: Path) -> None:
    """Render best model and best R² across the full Pythia grid."""
    steps, layers, best_model, best_r2 = heatmap_arrays(rows)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    categorical = axes[0].imshow(
        best_model,
        aspect="auto",
        origin="lower",
        cmap=ListedColormap(MODEL_COLORS),
        vmin=-0.5,
        vmax=len(MODEL_NAMES) - 0.5,
    )
    colorbar = figure.colorbar(categorical, ax=axes[0], ticks=range(len(MODEL_NAMES)))
    colorbar.ax.set_yticklabels(
        ("MP", "log-skew-normal", "log-semicircle", "MP + log-normal")
    )
    quality = axes[1].imshow(
        best_r2,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=np.nanmin(best_r2),
        vmax=1.0,
    )
    figure.colorbar(quality, ax=axes[1], label=r"best $R^2$")
    tick_indices = np.linspace(0, len(steps) - 1, min(10, len(steps)), dtype=int)
    layer_ticks = np.linspace(0, len(layers) - 1, min(8, len(layers)), dtype=int)
    for axis, title in zip(
        axes,
        ("Best-fitting model", r"Best fit quality ($R^2$)"),
    ):
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.set_ylabel("layer")
        axis.set_xticks(tick_indices, [f"{steps[index]:,}" for index in tick_indices])
        axis.set_yticks(layer_ticks, [str(layers[index]) for index in layer_ticks])
        axis.tick_params(axis="x", rotation=45)
    figure.suptitle(
        "Pythia-12B-deduped: MP vs. log-skew-normal vs. log-semicircle",
        fontsize=13,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def checkpoint_paths(results_dir: Path) -> dict[int, Path]:
    """Return every checkpoint spectrum indexed by its numeric training step."""
    return {
        int(path.name.removeprefix("step")): path / "spectra.h5"
        for path in results_dir.glob("step*")
        if path.name.removeprefix("step").isdigit() and (path / "spectra.h5").exists()
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("docs/paper/model_comparison_grid_pythia12b_logskewnormal.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/pythia_skewnormal_heatmap.png"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=None,
        help="Fit only these checkpoint steps (default: every available checkpoint).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
        help="Models to fit (default: all three, which also renders the heatmap).",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Fit the full grid, write the comparison CSV, and render its heatmap."""
    args = parse_args()
    checkpoints = checkpoint_paths(args.results_dir)
    if args.steps is not None:
        requested = set(args.steps)
        missing = requested - checkpoints.keys()
        if missing:
            raise ValueError(f"Missing checkpoint spectra for steps: {sorted(missing)}")
        checkpoints = {step: checkpoints[step] for step in sorted(requested)}
    if not checkpoints:
        raise ValueError(f"No checkpoint spectra found in {args.results_dir}.")
    models = tuple(args.models)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fit_checkpoint, step, spectra_path, models): step
            for step, spectra_path in checkpoints.items()
        }
        for future in as_completed(futures):
            step = futures[future]
            rows.extend(future.result())
            print(f"Fitted step {step}", flush=True)
    rows.sort(key=lambda row: (int(row["step"]), int(row["layer"])))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if set(models) == set(MODEL_NAMES):
        plot_heatmap(rows, args.figure)
        print(f"Wrote {args.csv} and {args.figure}")
    else:
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
