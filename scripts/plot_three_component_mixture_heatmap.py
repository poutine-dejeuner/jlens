#!/usr/bin/env python3
"""Render RGB Pythia mixture-composition and fit-quality heatmaps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_grid(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = sorted({int(row["step"]) for row in rows})
    layers = sorted({int(row["layer"]) for row in rows})
    composition = np.zeros((len(layers), len(steps), 3))
    r2 = np.zeros((len(layers), len(steps)))
    for row in rows:
        layer, step = layers.index(int(row["layer"])), steps.index(int(row["step"]))
        composition[layer, step] = [
            float(row["semicircle_weight"]),
            float(row["lognormal_weight"]),
            float(row["mp_weight"]),
        ]
        r2[layer, step] = float(row["r2"])
    return steps, layers, composition, r2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps, layers, composition, r2 = load_grid(args.csv)
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].imshow(composition, aspect="auto", origin="lower")
    axes[0].set_title("Mixture composition (red: log-semicircle; green: log-normal; blue: MP)")
    image = axes[1].imshow(r2, aspect="auto", origin="lower", cmap="viridis", vmin=r2.min(), vmax=1)
    axes[1].set_title("Three-component mixture fit quality (R²)")
    figure.colorbar(image, ax=axes[1], label="R²")
    ticks = np.linspace(0, len(steps) - 1, min(10, len(steps)), dtype=int)
    for axis in axes:
        axis.set_xlabel("training step")
        axis.set_ylabel("layer")
        axis.set_xticks(ticks, [f"{steps[i]:,}" for i in ticks], rotation=45)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
