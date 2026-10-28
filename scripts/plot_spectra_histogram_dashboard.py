#!/usr/bin/env python3
"""Create an interactive histogram dashboard for Pythia-12B J-Lens spectra.

Each animation frame contains overlaid, log-binned raw eigenvalue-density curves
for every J_l. Wikitext and random-token prompt averages are shown side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard_playback import write_dashboard

from plot_spectra_dashboard import layer_colorbar_trace, spectrum_steps


def read_eigenvalues(step_dir: Path, n_layers: int) -> list[np.ndarray | None]:
    """Read raw positive J_l^T J_l eigenvalues (no normalization)."""
    spectra: list[np.ndarray | None] = []
    with h5py.File(step_dir / "spectra.h5", "r") as handle:
        for layer in range(n_layers):
            group = handle.get(f"layer_{layer}")
            if group is None or "eigenvalues" not in group:
                spectra.append(None)
                continue
            values = np.asarray(group["eigenvalues"], dtype=float)
            values = values[np.isfinite(values) & (values > 0)]
            if not values.size:
                spectra.append(None)
                continue
            spectra.append(values)
    return spectra


HISTOGRAM_SMOOTHING_BINS = 0.0  # Gaussian width in bins; 0 = raw histogram
HISTOGRAM_X_MIN = 1e-6
HISTOGRAM_X_MAX = 1e3


def histogram_bins(n_bins: int) -> np.ndarray:
    """Shared logarithmic bins on [HISTOGRAM_X_MIN, HISTOGRAM_X_MAX] for raw eigenvalues."""
    return np.geomspace(HISTOGRAM_X_MIN, HISTOGRAM_X_MAX, n_bins + 1)


def density_curve(eigenvalues: np.ndarray | None, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the probability density of log10(lambda) on a fixed log grid.

    Probabilities use the whole spectrum as denominator, so eigenvalues outside
    the plotted window do not contribute and empty bins are exactly zero. A
    optional Gaussian smoothing (HISTOGRAM_SMOOTHING_BINS, off by default) reduces sampling noise.
    """
    centers = np.sqrt(bins[:-1] * bins[1:])
    if eigenvalues is None:
        return centers, np.full_like(centers, np.nan)
    counts, _ = np.histogram(eigenvalues, bins=bins)
    mass = counts / eigenvalues.size
    if HISTOGRAM_SMOOTHING_BINS > 0:
        radius = int(np.ceil(4 * HISTOGRAM_SMOOTHING_BINS))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / HISTOGRAM_SMOOTHING_BINS) ** 2)
        kernel /= kernel.sum()
        mass = np.convolve(mass, kernel, mode="same")
    return centers, mass / np.diff(np.log10(bins))


def layer_trace(layer: int, eigenvalues: np.ndarray | None, bins: np.ndarray, color: str, column: int) -> go.Scatter:
    x, y = density_curve(eigenvalues, bins)
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": color, "width": 1.25},
        showlegend=False,
        hovertemplate=f"J<sub>{layer}</sub><br>eigenvalue=%{{x:.3e}}<br>density per decade=%{{y:.3f}}<extra></extra>",
    )


def make_dashboard(wikitext_dir: Path, random_dir: Path, output: Path, n_bins: int) -> None:
    wikitext_steps = spectrum_steps(wikitext_dir)
    random_steps = spectrum_steps(random_dir)
    steps = sorted(set(wikitext_steps) & set(random_steps))
    if not steps:
        raise ValueError("No common completed spectra.h5 checkpoints found.")

    with h5py.File(wikitext_steps[steps[0]] / "spectra.h5", "r") as handle:
        n_layers = int(handle.attrs["n_layers"])
        d_model = int(handle.attrs["d_model"])
    with h5py.File(random_steps[steps[0]] / "spectra.h5", "r") as handle:
        if int(handle.attrs["n_layers"]) != n_layers or int(handle.attrs["d_model"]) != d_model:
            raise ValueError("Wikitext and random-token spectra have incompatible dimensions.")

    print(f"Loading {len(steps)} common checkpoints, {n_layers} layers, d={d_model}...")
    data_by_step: dict[int, tuple[list[np.ndarray | None], list[np.ndarray | None]]] = {}
    minimum = np.inf
    maximum = 0.0
    for index, step in enumerate(steps, start=1):
        wikitext_spectra = read_eigenvalues(wikitext_steps[step], n_layers)
        random_spectra = read_eigenvalues(random_steps[step], n_layers)
        data_by_step[step] = (wikitext_spectra, random_spectra)
        for spectra in (wikitext_spectra, random_spectra):
            for eigenvalues in spectra:
                if eigenvalues is not None:
                    minimum = min(minimum, float(eigenvalues.min()))
                    maximum = max(maximum, float(eigenvalues.max()))
        print(f"  [{index:2d}/{len(steps)}] step {step}")
    if not np.isfinite(minimum) or maximum <= minimum:
        raise ValueError("No usable positive eigenvalues found.")

    bins = histogram_bins(n_bins)

    colors = [f"hsl({260 - 260 * layer / max(n_layers - 1, 1):.1f}, 70%, 45%)" for layer in range(n_layers)]
    first_wikitext, first_random = data_by_step[steps[0]]
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Wikitext-averaged J̄<sub>l</sub>", "Random-token-averaged J̄<sub>l</sub>"),
        horizontal_spacing=0.06,
    )
    for layer, color in enumerate(colors):
        figure.add_trace(layer_trace(layer, first_wikitext[layer], bins, color, 1), row=1, col=1)
        figure.add_trace(layer_trace(layer, first_random[layer], bins, color, 2), row=1, col=2)
    figure.add_trace(layer_colorbar_trace(n_layers, HISTOGRAM_X_MIN, 0.0), row=1, col=1)

    peaks = [
        np.nanmax(density_curve(eigenvalues, bins)[1])
        for spectra in (s for pair in data_by_step.values() for s in pair)
        for eigenvalues in spectra
        if eigenvalues is not None
    ]
    density_max = 1.05 * float(np.max(peaks))

    figure.frames = [
        go.Frame(
            name=str(step),
            data=[
                trace
                for layer, color in enumerate(colors)
                for trace in (
                    layer_trace(layer, data_by_step[step][0][layer], bins, color, 1),
                    layer_trace(layer, data_by_step[step][1][layer], bins, color, 2),
                )
            ]
            + [layer_colorbar_trace(n_layers, HISTOGRAM_X_MIN, 0.0)],
        )
        for step in steps
    ]
    step_buttons = [
        {
            "args": [[str(step)], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}],
            "label": f"{step:,}",
            "method": "animate",
        }
        for step in steps
    ]
    figure.update_layout(
        title={"text": "Pythia-12B J-Lens eigenvalue distributions: every layer at every completed common checkpoint", "x": 0.5},
        template="plotly_white",
        height=780,
        width=1500,
        hovermode="closest",
        margin={"l": 80, "r": 140, "t": 125, "b": 110},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Training step: ", "font": {"size": 16}}, "pad": {"t": 55}, "steps": step_buttons}],
        updatemenus=[
            {"type": "buttons", "direction": "left", "x": 0.0, "y": 1.18, "xanchor": "left", "yanchor": "top", "buttons": [
                {"label": "Loop checkpoints", "method": "skip", "args": []},
                {"label": "Pause", "method": "skip", "args": []},
            ]},
        ],
    )
    figure.add_annotation(
        text="Raw (unnormalized) eigenvalue histograms on shared log bins, 1e-6 to 1e3; eigenvalues outside this window are not shown." + ("Unsmoothed." if HISTOGRAM_SMOOTHING_BINS <= 0 else "Lightly smoothed across adjacent bins."),
        xref="paper", yref="paper", x=0.5, y=-0.14, showarrow=False, font={"size": 12},
    )
    for column in (1, 2):
        figure.update_xaxes(title_text="Eigenvalue λ of J<sub>l</sub><sup>T</sup>J<sub>l</sub>", type="log", range=[np.log10(HISTOGRAM_X_MIN), np.log10(HISTOGRAM_X_MAX)], exponentformat="power", showexponent="all", row=1, col=column)
        figure.update_yaxes(title_text="Probability density per decade", range=[0, density_max], row=1, col=column)

    write_dashboard(figure, output, log_space=False)
    print(f"Wrote {output} ({len(steps)} checkpoints × {n_layers} layers × 2 corpora)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikitext-dir", type=Path, default=Path("/data/jlens/pythia-12b-deduped"))
    parser.add_argument("--random-dir", type=Path, default=Path("/data/jlens/pythia-12b-deduped-random"))
    parser.add_argument("--output", type=Path, default=Path("figures/pythia-12b/spectra_histogram_dashboard.html"))
    parser.add_argument("--bins", type=int, default=90, help="Number of shared log bins per checkpoint (default: 90).")
    args = parser.parse_args()
    if args.bins < 10:
        parser.error("--bins must be at least 10")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    make_dashboard(arguments.wikitext_dir, arguments.random_dir, arguments.output, arguments.bins)
