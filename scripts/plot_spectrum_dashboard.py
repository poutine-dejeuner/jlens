#!/usr/bin/env python3
"""Create one interactive rank-spectrum dashboard for a Pythia-12B J-Lens run."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go

from dashboard_playback import write_dashboard

from plot_spectra_dashboard import layer_colorbar_trace, read_spectra, spectrum_steps


def layer_trace(layer: int, spectrum: tuple[np.ndarray, np.ndarray] | None, color: str) -> go.Scatter:
    x, y = ([], []) if spectrum is None else spectrum
    return go.Scatter(
        x=x, y=y, mode="lines", line={"color": color, "width": 1.25},
        showlegend=False,
        hovertemplate=f"J<sub>{layer}</sub><br>rank=%{{x:,}}<br>eigenvalue=%{{y:.3e}}<extra></extra>",
    )


def make_dashboard(results_dir: Path, corpus_label: str, output: Path, max_points: int) -> None:
    checkpoints = spectrum_steps(results_dir)
    steps = sorted(checkpoints)
    if not steps:
        raise ValueError(f"No completed spectra.h5 checkpoints found in {results_dir}.")
    with h5py.File(checkpoints[steps[0]] / "spectra.h5", "r") as handle:
        n_layers = int(handle.attrs["n_layers"])
        d_model = int(handle.attrs["d_model"])

    print(f"Loading {len(steps)} {corpus_label} checkpoints, {n_layers} layers, d={d_model}...")
    spectra_by_step = {}
    minimum, maximum = np.inf, 0.0
    for index, step in enumerate(steps, start=1):
        spectra = read_spectra(checkpoints[step], n_layers, max_points)
        spectra_by_step[step] = spectra
        for spectrum in spectra:
            if spectrum is not None:
                minimum = min(minimum, float(np.min(spectrum[1])))
                maximum = max(maximum, float(np.max(spectrum[1])))
        print(f"  [{index:2d}/{len(steps)}] step {step}")
    if not np.isfinite(minimum) or maximum <= minimum:
        raise ValueError("No usable positive eigenvalues found.")

    colors = [f"hsl({260 - 260 * layer / max(n_layers - 1, 1):.1f}, 70%, 45%)" for layer in range(n_layers)]
    figure = go.Figure(data=[layer_trace(layer, spectra_by_step[steps[0]][layer], color) for layer, color in enumerate(colors)] + [layer_colorbar_trace(n_layers, 1.0, minimum)])
    figure.frames = [
        go.Frame(name=str(step), data=[layer_trace(layer, spectra_by_step[step][layer], color) for layer, color in enumerate(colors)] + [layer_colorbar_trace(n_layers, 1.0, minimum)])
        for step in steps
    ]
    step_buttons = [{"args": [[str(step)], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}], "label": f"{step:,}", "method": "animate"} for step in steps]
    figure.update_layout(
        title={"text": f"Pythia-12B J-Lens spectra — {corpus_label}", "x": 0.5}, template="plotly_white", height=810, width=1100,
        hovermode="closest", margin={"l": 90, "r": 140, "t": 125, "b": 140},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Training step: ", "font": {"size": 16}}, "pad": {"t": 85}, "steps": step_buttons}],
        updatemenus=[
            {"type": "buttons", "direction": "left", "x": 0.0, "y": 1.18, "xanchor": "left", "yanchor": "top", "buttons": [
                {"label": "Loop checkpoints", "method": "skip", "args": []},
                {"label": "Pause", "method": "skip", "args": []},
            ]},
        ],
    )
    figure.add_annotation(text="Raw eigenvalues of J̄<sub>l</sub><sup>T</sup>J̄<sub>l</sub>. All layers are loaded; ranks are sampled for responsive rendering.", xref="paper", yref="paper", x=0.5, y=-0.14, showarrow=False, font={"size": 12})
    figure.update_xaxes(title_text="Eigenvalue rank", type="log", exponentformat="power", showexponent="all")
    figure.update_yaxes(title_text="Eigenvalue λ", type="log", tickformat=".1e", exponentformat="power", showexponent="all", range=[np.log10(minimum), np.log10(maximum)])
    write_dashboard(figure, output)
    print(f"Wrote {output} ({len(steps)} checkpoints × {n_layers} layers)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=384)
    args = parser.parse_args()
    if args.max_points < 64:
        parser.error("--max-points must be at least 64")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    make_dashboard(arguments.results_dir, arguments.label, arguments.output, arguments.max_points)
