#!/usr/bin/env python3
"""Create one interactive empirical J-Lens spectrum histogram dashboard.

Supports both a flat `step*` checkpoint layout (e.g. Pythia-12B) and a nested
`stage{1..7}/step{N}` layout with repeating step numbers across stages (e.g.
K2-Horizon-32B) via plot_spectra_dashboard.discover_checkpoints. Pass
--model-label to set the title for non-Pythia models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go

from dashboard_playback import write_dashboard

from plot_spectra_dashboard import checkpoint_label, discover_checkpoints, layer_colorbar_trace
import plot_spectra_histogram_dashboard as histogram_module
from plot_spectra_histogram_dashboard import HISTOGRAM_X_MAX, HISTOGRAM_X_MIN, density_curve, histogram_bins, read_eigenvalues


def layer_trace(layer: int, eigenvalues: np.ndarray | None, bins: np.ndarray, color: str) -> go.Scatter:
    x, y = density_curve(eigenvalues, bins)
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": color, "width": 1.25},
        showlegend=False,
        hovertemplate=f"J<sub>{layer}</sub><br>eigenvalue=%{{x:.3e}}<br>density per decade=%{{y:.3f}}<extra></extra>",
    )


def make_dashboard(results_dir: Path, corpus_label: str, model_label: str, output: Path, n_bins: int) -> None:
    checkpoints = discover_checkpoints(results_dir)
    keys = sorted(checkpoints)
    if not keys:
        raise ValueError(f"No completed spectra.h5 checkpoints found in {results_dir}.")
    labels = [checkpoint_label(key) for key in keys]
    with h5py.File(checkpoints[keys[0]] / "spectra.h5", "r") as handle:
        n_layers = int(handle.attrs["n_layers"])
        d_model = int(handle.attrs["d_model"])

    print(f"Loading {len(keys)} {corpus_label} checkpoints, {n_layers} layers, d={d_model}...")
    spectra_by_key: dict[tuple[int, int], list[np.ndarray | None]] = {}
    minimum = np.inf
    maximum = 0.0
    for index, key in enumerate(keys, start=1):
        spectra = read_eigenvalues(checkpoints[key], n_layers)
        spectra_by_key[key] = spectra
        for eigenvalues in spectra:
            if eigenvalues is not None:
                minimum = min(minimum, float(eigenvalues.min()))
                maximum = max(maximum, float(eigenvalues.max()))
        print(f"  [{index:2d}/{len(keys)}] {labels[index - 1]}")
    if not np.isfinite(minimum) or maximum <= minimum:
        raise ValueError("No usable positive eigenvalues found.")

    bins = histogram_bins(n_bins)
    peaks = [
        np.nanmax(density_curve(eigenvalues, bins)[1])
        for spectra in spectra_by_key.values()
        for eigenvalues in spectra
        if eigenvalues is not None
    ]
    density_max = 1.05 * float(np.max(peaks))
    colors = [f"hsl({260 - 260 * layer / max(n_layers - 1, 1):.1f}, 70%, 45%)" for layer in range(n_layers)]

    figure = go.Figure()
    first_key = keys[0]
    for layer, color in enumerate(colors):
        figure.add_trace(layer_trace(layer, spectra_by_key[first_key][layer], bins, color))
    figure.add_trace(layer_colorbar_trace(n_layers, HISTOGRAM_X_MIN, 0.0))
    figure.frames = [
        go.Frame(
            name=label,
            data=[layer_trace(layer, spectra_by_key[key][layer], bins, color) for layer, color in enumerate(colors)]
            + [layer_colorbar_trace(n_layers, HISTOGRAM_X_MIN, 0.0)],
        )
        for key, label in zip(keys, labels)
    ]
    step_buttons = [
        {"args": [[label], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}], "label": label, "method": "animate"}
        for label in labels
    ]
    figure.update_layout(
        title={"text": f"{model_label} J-Lens eigenvalue distributions — {corpus_label}  (normalized: λ → λ/d_model, d_model={d_model})", "x": 0.5},
        template="plotly_white",
        height=810,
        width=1100,
        hovermode="closest",
        margin={"l": 90, "r": 140, "t": 125, "b": 140},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Checkpoint: ", "font": {"size": 16}}, "pad": {"t": 85}, "steps": step_buttons}],
        updatemenus=[
            {"type": "buttons", "direction": "left", "x": 0.0, "y": 1.18, "xanchor": "left", "yanchor": "top", "buttons": [
                {"label": "Loop checkpoints", "method": "skip", "args": []},
                {"label": "Pause", "method": "skip", "args": []},
            ]},
        ],
    )
    figure.add_annotation(
        text=f"Normalized eigenvalue (λ/d_model) histograms on shared log bins, {HISTOGRAM_X_MIN:.2g} to {HISTOGRAM_X_MAX:.2g} (raw 1e-6 to 1e3); eigenvalues outside this window are not shown. " + ("Unsmoothed." if histogram_module.HISTOGRAM_SMOOTHING_BINS <= 0 else "Lightly smoothed across adjacent bins."),
        xref="paper", yref="paper", x=0.5, y=-0.14, showarrow=False, font={"size": 12},
    )
    figure.update_xaxes(title_text="Eigenvalue λ / d_model (λ → λ/hidden_dim)", type="log", range=[np.log10(HISTOGRAM_X_MIN), np.log10(HISTOGRAM_X_MAX)], exponentformat="power", showexponent="all")
    figure.update_yaxes(title_text="Probability density per decade", range=[0, density_max])
    write_dashboard(figure, output, log_space=False)
    print(f"Wrote {output} ({len(keys)} checkpoints × {n_layers} layers)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Corpus label shown in the title.")
    parser.add_argument("--model-label", default="Pythia-12B", help="Model name shown in the title (default: Pythia-12B).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=90, help="Log bins per checkpoint (default: 90).")
    parser.add_argument("--smoothing", type=float, default=None, help="Gaussian smoothing width in bins (default: 0 = unsmoothed).")
    args = parser.parse_args()
    if args.bins < 10:
        parser.error("--bins must be at least 10")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoothing is not None:
        histogram_module.HISTOGRAM_SMOOTHING_BINS = arguments.smoothing
    make_dashboard(arguments.results_dir, arguments.label, arguments.model_label, arguments.output, arguments.bins)
