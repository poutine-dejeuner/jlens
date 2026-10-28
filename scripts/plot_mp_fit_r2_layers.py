#!/usr/bin/env python3
"""Plot per-layer Marchenko-Pastur fit R^2 versus training step, one file per corpus.

Reads the CSV written by plot_mp_fit_r2.py; all layers are superposed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

CORPUS_TITLES = {"wikitext": "Wikitext-averaged", "random": "Random-token-averaged"}


PANELS = (("r2", "R²"), ("sigma2", "Fitted σ² (scale)"), ("q", "Fitted q (aspect ratio)"))


def make_figure(rows: list[dict], corpus: str) -> go.Figure:
    n_layers = 1 + max(int(r["layer"]) for r in rows)
    figure = make_subplots(rows=len(PANELS), cols=1, shared_xaxes=True, vertical_spacing=0.05)
    for layer in range(n_layers):
        subset = sorted((r for r in rows if r["corpus"] == corpus and int(r["layer"]) == layer), key=lambda r: int(r["step"]))
        if not subset:
            continue
        for index, (key, label) in enumerate(PANELS, start=1):
            figure.add_trace(
                go.Scatter(
                    x=[int(r["step"]) for r in subset], y=[float(r[key]) for r in subset], mode="lines",
                    line={"color": f"hsl({260 - 260 * layer / max(n_layers - 1, 1):.1f}, 70%, 45%)", "width": 1.25},
                    name=f"L{layer}", legendgroup=f"layer-{layer}", showlegend=index == 1,
                    hovertemplate=f"J<sub>{layer}</sub><br>step %{{x}}<br>{label}=%{{y:.4g}}<extra></extra>",
                ),
                row=index, col=1,
            )
    figure.update_layout(
        title={"text": f"Marchenko–Pastur fits of each J<sub>l</sub> spectrum across training — {CORPUS_TITLES[corpus]}", "x": 0.5},
        template="plotly_white", width=1100, height=1100,
        legend={"title": {"text": "Layer"}, "itemsizing": "constant", "font": {"size": 10}},
    )
    figure.update_xaxes(type="log", exponentformat="power", showexponent="all")
    figure.update_xaxes(title_text="Training step", row=len(PANELS), col=1)
    figure.update_yaxes(title_text="R²", range=[-0.3, 1.02], row=1, col=1)
    figure.update_yaxes(title_text="σ²", type="log", exponentformat="power", showexponent="all", row=2, col=1)
    figure.update_yaxes(title_text="q", range=[0, 1.05], row=3, col=1)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("figures/pythia-12b/mp_fit_r2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures/pythia-12b"))
    args = parser.parse_args()

    with args.csv.open() as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for corpus in CORPUS_TITLES:
        output = args.output_dir / f"mp_fit_r2_layers_{corpus}.png"
        make_figure(rows, corpus).write_image(output, scale=2)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
