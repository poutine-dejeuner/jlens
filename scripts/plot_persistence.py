#!/usr/bin/env python3
"""Plot persistence diagrams and barcodes for token manifold homology.

Reads persistence_all.npz from /data/jlens/qwen3.6-27b/ and produces:
  - Persistence diagrams (birth vs death) for H0-H4, side-by-side
  - Barcode plots (lifespan bars sorted by birth time)
  - Bar count / mean lifespan vs dimension for dimensionality inference

Usage:
    uv run python scripts/plot_persistence.py
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

DATA_PATH = Path("/data/jlens/qwen3.6-27b/persistence_all.npz")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = np.load(DATA_PATH)

MANIFOLDS = {
    "singleton": "Option A (Singleton)",
    "contextual": "Option B (Contextualized)",
}
COLORS = {"singleton": "#2166ac", "contextual": "#b2182b"}

# Load all diagrams, subsampling H1 to 50K bars for plotting
MAX_H1_PLOT = 50000

np.random.seed(42)
diagrams = {}
for key in MANIFOLDS:
    diagrams[key] = {}
    for dim in range(5):
        k = f"{key}_h{dim}"
        dgm = data[k]
        # Remove infinite deaths
        finite = dgm[dgm[:, 1] < np.inf]
        # Subsample H1 for manageable plotting
        if dim == 1 and len(finite) > MAX_H1_PLOT:
            # Keep the longest-lived bars (most topologically significant)
            lifespans = finite[:, 1] - finite[:, 0]
            top_idx = np.argpartition(-lifespans, MAX_H1_PLOT - 1)[:MAX_H1_PLOT]
            finite = finite[top_idx]
        diagrams[key][dim] = finite


# ---------------------------------------------------------------------------
# Figure 1: Persistence diagrams (birth vs death)
# ---------------------------------------------------------------------------

def plot_persistence_diagrams():
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=False, sharey=False)

    for row, (key, label) in enumerate(MANIFOLDS.items()):
        color = COLORS[key]
        for dim in range(5):
            ax = axes[row, dim]
            dgm = diagrams[key][dim]

            if len(dgm) == 0:
                ax.text(0.5, 0.5, "∅", transform=ax.transAxes,
                        ha="center", va="center", fontsize=18, color="gray")
                ax.set_title(f"H{dim}", fontsize=12, fontweight="bold")
                continue

            births, deaths = dgm[:, 0], dgm[:, 1]
            lifespans = deaths - births

            # Color by lifespan
            sc = ax.scatter(births, deaths, c=lifespans, cmap="YlOrRd",
                           s=3 if dim > 0 else 1, alpha=0.6, edgecolors="none",
                           vmin=0, vmax=np.percentile(lifespans, 95) if len(lifespans) > 5 else lifespans.max())

            # Diagonal line (birth = death)
            lims = [0, max(births.max(), deaths.max()) * 1.05]
            ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)

            # Stats annotation
            mean_ls = np.mean(lifespans)
            n_bars = len(dgm)
            ax.set_title(f"H{dim}  (n={n_bars})", fontsize=11, fontweight="bold")
            ax.text(0.95, 0.05, f"μ={mean_ls:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color="gray")

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")

        axes[row, 0].set_ylabel(f"{label}\nDeath", fontsize=11)

    for dim in range(5):
        axes[1, dim].set_xlabel("Birth", fontsize=11)

    fig.suptitle("Token Manifold Persistence Diagrams", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "persistence_diagrams.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: Barcodes
# ---------------------------------------------------------------------------

def plot_barcodes():
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    for row, (key, label) in enumerate(MANIFOLDS.items()):
        ax = axes[row]
        color = COLORS[key]
        y_offset = 0
        y_ticks = []
        y_labels = []

        for dim in range(5):
            dgm = diagrams[key][dim]
            if len(dgm) == 0:
                y_offset += 1  # spacer
                continue

            # Sort by birth time
            order = np.argsort(dgm[:, 0])
            dgm_sorted = dgm[order]

            for i, (birth, death) in enumerate(dgm_sorted):
                lifespan = death - birth
                ax.barh(y_offset + i, lifespan, left=birth, height=0.8,
                        color=color, alpha=0.15 + 0.15 * dim, edgecolor="none")

            # Label this homology group
            mid_y = y_offset + len(dgm_sorted) / 2
            y_ticks.append(mid_y)
            y_labels.append(f"H{dim}")
            y_offset += len(dgm_sorted)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=10, fontweight="bold")
        ax.set_ylabel(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Filtration value" if row == 1 else "")
        ax.set_xlim(0, None)
        ax.grid(axis="x", alpha=0.2)

    fig.suptitle("Token Manifold Persistence Barcodes", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "persistence_barcodes.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Dimensionality inference
# ---------------------------------------------------------------------------

def plot_dimensionality():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    dims = list(range(5))
    width = 0.35
    x = np.arange(len(dims))

    # Panel A: bar count vs dimension
    ax = axes[0]
    for i, (key, label) in enumerate(MANIFOLDS.items()):
        counts = [len(diagrams[key][d]) for d in dims]
        bars = ax.bar(x + i * width, counts, width, color=COLORS[key],
                      alpha=0.7, label=label, edgecolor="white", linewidth=0.5)
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        str(count), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"H{d}" for d in dims])
    ax.set_ylabel("Number of bars")
    ax.set_title("Bar Count vs Homological Dimension", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.2)

    # Panel B: mean lifespan vs dimension
    ax = axes[1]
    for i, (key, label) in enumerate(MANIFOLDS.items()):
        lifespans = []
        for d in dims:
            dgm = diagrams[key][d]
            if len(dgm) > 0:
                ls = dgm[:, 1] - dgm[:, 0]
                lifespans.append(np.mean(ls))
            else:
                lifespans.append(0)
        bars = ax.bar(x + i * width, lifespans, width, color=COLORS[key],
                      alpha=0.7, label=label, edgecolor="white", linewidth=0.5)
        for bar, ls in zip(bars, lifespans):
            if ls > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{ls:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f"H{d}" for d in dims])
    ax.set_ylabel("Mean lifespan")
    ax.set_title("Mean Lifespan vs Homological Dimension", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Manifold Dimensionality from Persistent Homology", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "persistence_dimensionality.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: Persistence images (2D histograms of birth vs death)
# ---------------------------------------------------------------------------

def plot_persistence_images():
    """Density heatmaps of birth vs death for H1-H4."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for row, (key, label) in enumerate(MANIFOLDS.items()):
        for col, dim in enumerate([1, 2, 3, 4]):
            ax = axes[row, col]
            dgm = diagrams[key][dim]

            if len(dgm) == 0:
                ax.text(0.5, 0.5, "∅", transform=ax.transAxes,
                        ha="center", va="center", fontsize=24, color="gray")
                ax.set_title(f"H{dim}", fontsize=12, fontweight="bold")
                continue

            births, deaths = dgm[:, 0], dgm[:, 1]

            # 2D histogram
            h, xedges, yedges = np.histogram2d(births, deaths, bins=40)
            ax.pcolormesh(xedges, yedges, h.T, cmap="YlOrRd", shading="auto")

            lims = [0, max(births.max(), deaths.max()) * 1.05]
            ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_aspect("equal")

            ax.set_title(f"H{dim} (n={len(dgm)})", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{label}\nDeath", fontsize=10)
            if row == 1:
                ax.set_xlabel("Birth", fontsize=10)

    fig.suptitle("Token Manifold Persistence Images", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "persistence_images.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found. Run save_persistence_all.sh first.")
        sys.exit(1)

    print(f"Loading from {DATA_PATH}")
    print(f"  Singleton H0: {len(diagrams['singleton'][0])} bars")
    print(f"  Singleton H1: {len(diagrams['singleton'][1])} bars")
    print(f"  Singleton H2: {len(diagrams['singleton'][2])} bars")
    print(f"  Singleton H3: {len(diagrams['singleton'][3])} bars")
    print(f"  Singleton H4: {len(diagrams['singleton'][4])} bars")
    print(f"  Contextual H0: {len(diagrams['contextual'][0])} bars")
    print(f"  Contextual H1: {len(diagrams['contextual'][1])} bars")
    print(f"  Contextual H2: {len(diagrams['contextual'][2])} bars")
    print(f"  Contextual H3: {len(diagrams['contextual'][3])} bars")
    print(f"  Contextual H4: {len(diagrams['contextual'][4])} bars")

    plot_persistence_diagrams()
    plot_barcodes()
    plot_dimensionality()
    plot_persistence_images()
    print("All plots generated.")
