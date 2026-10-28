#!/usr/bin/env python3
"""Fit a Marchenko-Pastur law to every J_l eigenvalue spectrum and report R^2.

For each (corpus, checkpoint, layer) the empirical spectrum is binned into
fixed log-spaced bins covering all eigenvalues (probability mass per bin). The
MP law with free scale sigma^2 and aspect ratio q in (0, 1] is integrated over
the same bins and fitted by least squares on the bin masses.
R^2 = 1 - SSE / SST over those masses.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import least_squares

from plot_spectra_dashboard import spectrum_steps

BINS = np.geomspace(1e-12, 1e5, 121)
THETA = np.linspace(0.0, np.pi, 4001)


def mp_bin_masses(sigma2: float, q: float, bins: np.ndarray = BINS) -> np.ndarray:
    """Probability mass of MP(sigma2, q) in each bin, via lambda = s2(1+q-2 sqrt(q) cos t)."""
    lam = sigma2 * (1.0 + q - 2.0 * np.sqrt(q) * np.cos(THETA))
    density = 2.0 * sigma2 * np.sin(THETA) ** 2 / np.pi / np.maximum(lam, 1e-300)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(THETA))])
    cdf /= cdf[-1]
    return np.diff(np.interp(bins, lam, cdf, left=0.0, right=1.0))


def fit_mp(eigenvalues: np.ndarray) -> tuple[float, float, float]:
    """Return (sigma2, q, R^2) of the best least-squares MP fit to bin masses."""
    counts, _ = np.histogram(eigenvalues, bins=BINS)
    target = counts / eigenvalues.size
    sst = float(np.sum((target - target.mean()) ** 2))
    median = float(np.median(eigenvalues))

    def residual(params: np.ndarray) -> np.ndarray:
        return mp_bin_masses(float(np.exp(params[0])), float(params[1])) - target

    best = None
    for scale in (0.5, 1.0, 2.0):
        for q0 in (0.1, 0.5, 1.0):
            result = least_squares(
                residual, x0=[np.log(median * scale), q0], bounds=([-30.0, 1e-3], [15.0, 1.0]), x_scale=[1.0, 0.3]
            )
            if best is None or result.cost < best.cost:
                best = result
    sse = 2.0 * best.cost
    return float(np.exp(best.x[0])), float(best.x[1]), 1.0 - sse / sst


def fit_corpus(results_dir: Path, label: str) -> list[dict]:
    checkpoints = spectrum_steps(results_dir)
    rows = []
    for step in sorted(checkpoints):
        with h5py.File(checkpoints[step] / "spectra.h5", "r") as handle:
            n_layers = int(handle.attrs["n_layers"])
            for layer in range(n_layers):
                group = handle.get(f"layer_{layer}")
                if group is None or "eigenvalues" not in group:
                    continue
                values = np.asarray(group["eigenvalues"], dtype=float)
                values = values[np.isfinite(values) & (values > 0)]
                if values.size < 100:
                    continue
                sigma2, q, r2 = fit_mp(values)
                rows.append({"corpus": label, "step": step, "layer": layer, "sigma2": sigma2, "q": q, "r2": r2})
        print(f"  {label} step {step}: mean R2 = {np.mean([r['r2'] for r in rows if r['step'] == step]):.3f}", flush=True)
    return rows


def make_figure(rows: list[dict]) -> go.Figure:
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07)
    panels = (("r2", np.mean, "R²"), ("sigma2", np.median, "σ² (median over layers)"), ("q", np.mean, "q"))
    for label, color in (("wikitext", "#1f77b4"), ("random", "#d62728")):
        subset = [r for r in rows if r["corpus"] == label]
        steps = sorted({r["step"] for r in subset})
        for index, (key, reduce, title) in enumerate(panels, start=1):
            values = [reduce([r[key] for r in subset if r["step"] == s]) for s in steps]
            figure.add_trace(
                go.Scatter(
                    x=steps, y=values, mode="lines+markers", name=label, legendgroup=label, showlegend=index == 1,
                    line={"color": color},
                    hovertemplate="step %{x}<br>" + title + "=%{y:.4g}<extra>" + label + "</extra>",
                ),
                row=index, col=1,
            )
    figure.update_layout(
        title={"text": "Marchenko–Pastur fits of J<sub>l</sub> spectra (mean R² and q, median σ², over layers)", "x": 0.5},
        template="plotly_white", width=1000, height=1000,
    )
    figure.update_xaxes(type="log", exponentformat="power", showexponent="all")
    figure.update_xaxes(title_text="Training step", row=3, col=1)
    figure.update_yaxes(title_text="R²", range=[0.75, 1.0], row=1, col=1)
    figure.update_yaxes(title_text="σ²", type="log", exponentformat="power", showexponent="all", row=2, col=1)
    figure.update_yaxes(title_text="q", range=[0, 1.05], row=3, col=1)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wikitext-dir", type=Path, default=Path("/data/jlens/pythia-12b-deduped"))
    parser.add_argument("--random-dir", type=Path, default=Path("/data/jlens/pythia-12b-deduped-random"))
    parser.add_argument("--output", type=Path, default=Path("figures/pythia-12b/mp_fit_r2.png"))
    parser.add_argument("--csv", type=Path, default=Path("figures/pythia-12b/mp_fit_r2.csv"))
    args = parser.parse_args()
    rows = fit_corpus(args.wikitext_dir, "wikitext") + fit_corpus(args.random_dir, "random")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corpus", "step", "layer", "sigma2", "q", "r2"])
        writer.writeheader()
        writer.writerows(rows)
    make_figure(rows).write_image(args.output, scale=2)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
