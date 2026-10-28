#!/usr/bin/env python3
"""Create an interactive Pythia-12B Jacobian-spectrum comparison dashboard.

Each animation frame is one training checkpoint.  Within a frame, every
available J_l spectrum is shown as a rank--eigenvalue curve: layers are color
coded and the two prompt distributions occupy side-by-side panels.  The
rendered curves are rank-decimated only; their values are the stored
J_l^T J_l eigenvalues on the original scale.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard_playback import write_dashboard


def spectrum_steps(results_dir: Path) -> dict[int, Path]:
    """Return valid checkpoint directories indexed by integer training step."""
    checkpoints = {}
    for step_dir in results_dir.glob("step*"):
        spectra_path = step_dir / "spectra.h5"
        if not step_dir.is_dir() or not spectra_path.exists():
            continue
        try:
            step = int(step_dir.name.removeprefix("step"))
            with h5py.File(spectra_path, "r") as handle:
                if "n_layers" not in handle.attrs or "layer_0" not in handle:
                    continue
                if "eigenvalues" not in handle["layer_0"]:
                    continue
            checkpoints[step] = step_dir
        except (OSError, ValueError):
            continue
    return checkpoints


_STAGE_RE = re.compile(r"^stage(\d+)$")


def discover_checkpoints(results_dir: Path) -> dict[tuple[int, int], Path]:
    """Return checkpoint directories indexed by (stage, step).

    Supports Pythia-12B's flat `step*` layout (stage is always 0) and
    K2-Horizon-32B's nested `stage{1..7}/step{N}` layout, where step numbers
    repeat across stages, so (stage, step) -- not step alone -- is unique.
    """
    stage_dirs = sorted(d for d in results_dir.glob("stage*") if d.is_dir() and _STAGE_RE.match(d.name))
    if not stage_dirs:
        return {(0, step): path for step, path in spectrum_steps(results_dir).items()}
    checkpoints: dict[tuple[int, int], Path] = {}
    for stage_dir in stage_dirs:
        stage = int(_STAGE_RE.match(stage_dir.name).group(1))
        for step, path in spectrum_steps(stage_dir).items():
            checkpoints[(stage, step)] = path
    return checkpoints


def checkpoint_label(key: tuple[int, int]) -> str:
    """Frame label for a (stage, step) checkpoint key."""
    stage, step = key
    return str(step) if stage == 0 else f"stage{stage} step{step}"


def layer_colorscale(n_layers: int) -> list[list[float]]:
    """Continuous colorscale matching the per-layer HSL gradient (hsl(260 - 260*t, 70%, 45%))
    used to color one line per layer across all spectra dashboards."""
    stops = np.linspace(0.0, 1.0, min(21, max(n_layers, 2)))
    return [[float(t), f"hsl({260 - 260 * t:.1f}, 70%, 45%)"] for t in stops]


def layer_colorbar_trace(n_layers: int, x: float, y: float):
    """Invisible marker trace rendering a continuous colorbar in place of one
    legend entry per layer, which is unreadable once there are dozens of layers.

    Uses a real, finite (zero-size/zero-opacity) point at (x, y) -- which must
    satisfy each panel's axis domain/positivity requirements -- rather than a
    None coordinate, because dashboard_playback.write_dashboard's cross-frame
    x-equality check fails when comparing arrays containing NaN.
    """
    tick_count = min(6, n_layers)
    tick_layers = sorted({int(round(t)) for t in np.linspace(0, n_layers - 1, tick_count)})
    return go.Scatter(
        x=[x],
        y=[y],
        mode="markers",
        marker={
            "size": 0,
            "opacity": 0,
            "color": [0],
            "cmin": 0,
            "cmax": max(n_layers - 1, 1),
            "colorscale": layer_colorscale(n_layers),
            "showscale": True,
            "colorbar": {
                "title": {"text": "Layer"},
                "tickvals": tick_layers,
                "ticktext": [f"L{layer}" for layer in tick_layers],
                "len": 0.75,
            },
        },
        hoverinfo="skip",
        showlegend=False,
    )


def rank_indices(n_values: int, max_points: int) -> np.ndarray:
    """Sample the spectral head densely and the bulk at logarithmic ranks."""
    if n_values <= max_points:
        return np.arange(n_values)
    head = np.arange(min(64, n_values))
    logarithmic = np.geomspace(1, n_values, max_points).astype(int) - 1
    linear = np.linspace(0, n_values - 1, max_points // 3, dtype=int)
    return np.unique(np.clip(np.concatenate((head, logarithmic, linear)), 0, n_values - 1))


def read_spectra(step_dir: Path, n_layers: int, max_points: int) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Read all layer spectra for a checkpoint, retaining their raw scale."""
    spectra: list[tuple[np.ndarray, np.ndarray] | None] = []
    with h5py.File(step_dir / "spectra.h5", "r") as handle:
        for layer in range(n_layers):
            name = f"layer_{layer}"
            if name not in handle or "eigenvalues" not in handle[name]:
                spectra.append(None)
                continue
            eigenvalues = np.asarray(handle[name]["eigenvalues"], dtype=float)
            positive = eigenvalues[np.isfinite(eigenvalues) & (eigenvalues > 0)]
            if not positive.size:
                spectra.append(None)
                continue
            # Keep the full d_model rank grid in every frame. Tiny negative or
            # non-finite numerical artifacts are displayed at the spectral floor
            # instead of being removed, which would shift later sampled ranks.
            floor = float(positive.min())
            eigenvalues = np.where(np.isfinite(eigenvalues) & (eigenvalues > 0), eigenvalues, floor)
            eigenvalues = np.sort(eigenvalues)[::-1]
            indices = rank_indices(eigenvalues.size, max_points)
            spectra.append((indices + 1, eigenvalues[indices]))
    return spectra


def layer_trace(layer: int, spectrum: tuple[np.ndarray, np.ndarray] | None, color: str, column: int) -> go.Scatter:
    if spectrum is None:
        x = y = []
    else:
        x, y = spectrum
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        line={"color": color, "width": 1.25},
        showlegend=False,
        hovertemplate=(
            f"J<sub>{layer}</sub><br>rank=%{{x:,}}<br>eigenvalue=%{{y:.3e}}<extra></extra>"
        ),
    )


def make_dashboard(
    wikitext_dir: Path,
    random_dir: Path,
    output: Path,
    max_points: int,
) -> None:
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

    colors = [f"hsl({260 - 260 * layer / max(n_layers - 1, 1):.1f}, 70%, 45%)" for layer in range(n_layers)]
    print(f"Loading {len(steps)} common checkpoints, {n_layers} layers, d={d_model}...")
    data_by_step = {}
    minimum = np.inf
    maximum = 0.0
    for index, step in enumerate(steps, start=1):
        data_by_step[step] = (
            read_spectra(wikitext_steps[step], n_layers, max_points),
            read_spectra(random_steps[step], n_layers, max_points),
        )
        for spectra in data_by_step[step]:
            for spectrum in spectra:
                if spectrum is not None:
                    minimum = min(minimum, float(np.min(spectrum[1])))
                    maximum = max(maximum, float(np.max(spectrum[1])))
        print(f"  [{index:2d}/{len(steps)}] step {step}")
    if not np.isfinite(minimum) or maximum <= minimum:
        raise ValueError("No usable positive eigenvalues found.")

    first_wikitext, first_random = data_by_step[steps[0]]
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Wikitext-averaged J̄<sub>l</sub>", "Random-token-averaged J̄<sub>l</sub>"),
        horizontal_spacing=0.06,
    )
    for layer, color in enumerate(colors):
        figure.add_trace(layer_trace(layer, first_wikitext[layer], color, 1), row=1, col=1)
        figure.add_trace(layer_trace(layer, first_random[layer], color, 2), row=1, col=2)
    figure.add_trace(layer_colorbar_trace(n_layers, 1.0, minimum), row=1, col=1)

    frames = []
    for step in steps:
        wikitext_spectra, random_spectra = data_by_step[step]
        frame_data = []
        for layer, color in enumerate(colors):
            frame_data.append(layer_trace(layer, wikitext_spectra[layer], color, 1))
            frame_data.append(layer_trace(layer, random_spectra[layer], color, 2))
        frame_data.append(layer_colorbar_trace(n_layers, 1.0, minimum))
        frames.append(go.Frame(name=str(step), data=frame_data))
    figure.frames = frames

    steps_menu = [
        {
            "args": [[str(step)], {"frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}],
            "label": f"{step:,}",
            "method": "animate",
        }
        for step in steps
    ]
    figure.update_layout(
        title={
            "text": "Pythia-12B J-Lens spectra: every layer at every completed common checkpoint",
            "x": 0.5,
        },
        template="plotly_white",
        height=780,
        width=1500,
        hovermode="closest",
        margin={"l": 80, "r": 140, "t": 125, "b": 110},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Training step: ", "font": {"size": 16}},
                "pad": {"t": 55},
                "steps": steps_menu,
            }
        ],
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 1.18,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Loop checkpoints",
                        "method": "skip",
                        "args": [],
                    },
                    {
                        "label": "Pause",
                        "method": "skip",
                        "args": [],
                    },
                ],
            },
        ],
    )
    figure.add_annotation(
        text=(
            "Raw eigenvalues of J̄<sub>l</sub><sup>T</sup>J̄<sub>l</sub>. "
            "All layers are loaded; ranks are sampled for responsive rendering."
        ),
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.14,
        showarrow=False,
        font={"size": 12},
    )

    for column in (1, 2):
        figure.update_xaxes(title_text="Eigenvalue rank", type="log", exponentformat="power", showexponent="all", row=1, col=column)
        figure.update_yaxes(
            title_text="Eigenvalue λ",
            type="log",
            tickformat=".1e",
            exponentformat="power",
            showexponent="all",
            range=[np.log10(minimum), np.log10(maximum)],
            row=1,
            col=column,
        )

    write_dashboard(figure, output)
    print(f"Wrote {output} ({len(steps)} checkpoints × {n_layers} layers × 2 corpora)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikitext-dir", type=Path, default=Path("/data/jlens/pythia-12b-deduped"))
    parser.add_argument("--random-dir", type=Path, default=Path("/data/jlens/pythia-12b-deduped-random"))
    parser.add_argument("--output", type=Path, default=Path("figures/pythia-12b/spectra_dashboard.html"))
    parser.add_argument("--max-points", type=int, default=384, help="Maximum rendered rank samples per layer spectrum (default: 384).")
    args = parser.parse_args()
    if args.max_points < 64:
        parser.error("--max-points must be at least 64")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    make_dashboard(arguments.wikitext_dir, arguments.random_dir, arguments.output, arguments.max_points)
