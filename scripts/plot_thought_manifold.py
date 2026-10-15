#!/usr/bin/env python3
"""Plot thought manifold eigenvectors: 2D scatter of token projections
onto eigenvector pairs (1,2) and (3,4).  Annotates cluster outposts
with their token strings so the semantic meaning of each cluster is visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from transformers import AutoTokenizer


def load_manifold(h5path: str) -> dict:
    with h5py.File(h5path, "r") as f:
        return {
            "eigenvectors": f["eigenvectors"][:],
            "eigenvalues": f["eigenvalues"][:],
            "explained_variance_ratio": f["explained_variance_ratio"][:],
        }


def _find_cluster_label(
    x: np.ndarray, y: np.ndarray, tokens: list[str], xlim: tuple,
    ylim: tuple, margin: float = 0.0,
) -> tuple[str, float, float] | None:
    """Pick the token closest to one extreme of the cluster bounds."""
    mask = (
        (x >= xlim[0] - margin) & (x <= xlim[1] + margin) &
        (y >= ylim[0] - margin) & (y <= ylim[1] + margin)
    )
    if mask.sum() < 3:
        return None

    # Pick the token closest to the bounding-box corner
    cx = xlim[1] if abs(xlim[1] - np.median(x[mask])) > abs(xlim[0] - np.median(x[mask])) else xlim[0]
    cy = ylim[1] if abs(ylim[1] - np.median(y[mask])) > abs(ylim[0] - np.median(y[mask])) else ylim[0]

    dists = (x[mask] - cx) ** 2 + (y[mask] - cy) ** 2
    best = np.where(mask)[0][np.argmin(dists)]
    return tokens[best], x[best], y[best]


def _is_special(tok: str) -> bool:
    return tok.startswith("<") and tok.endswith(">")


def _clean_tok(tok: str) -> str:
    # Replace newlines with ↵
    return tok.replace("\n", "↵").replace("\r", "⏎")


def plot_pair(ax, U: np.ndarray, ev: np.ndarray, pair: tuple[int, int],
              tokenizer, token_strings: list[str],
              n_subsample: int = 20000, n_annotate: int = 18):
    """Scatter tokens + annotate cluster outposts."""
    i, j = pair
    V = U.shape[0]

    rng = np.random.default_rng(123)
    idx_plot = rng.choice(V, min(V, n_subsample), replace=False)
    x = U[idx_plot, i]
    y = U[idx_plot, j]

    # Color by position in vocab (blue=low, red=high)
    colors = idx_plot / V

    # Annotate: pick extreme points from a grid of bins
    ax.scatter(x, y, c=colors, s=1, alpha=0.25, cmap="plasma", rasterized=True)

    # ---- Find cluster outposts ----
    # Bin space into a coarse grid and pick one token per inhabited bin
    n_bins = 7
    x_edges = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins + 1)
    y_edges = np.linspace(np.percentile(y, 1), np.percentile(y, 99), n_bins + 1)

    labels = []
    for bx in range(n_bins):
        for by in range(n_bins):
            xl = x_edges[bx], x_edges[bx + 1]
            yl = y_edges[by], y_edges[by + 1]
            result = _find_cluster_label(x, y, token_strings, xl, yl, margin=0.0)
            if result is not None:
                labels.append(result)

    # Sort by distance from origin (most extreme first) and keep top N
    labels.sort(key=lambda t: t[1] ** 2 + t[2] ** 2, reverse=True)
    labels = labels[:n_annotate]

    # Deduplicate nearby labels
    kept = []
    for tok, tx, ty in labels:
        too_close = False
        for _, kx, ky in kept:
            if abs(tx - kx) < 0.001 and abs(ty - ky) < 0.001:
                too_close = True
                break
        if not too_close:
            kept.append((tok, tx, ty))
    labels = kept

    for tok, tx, ty in labels:
        disp = _clean_tok(tok)
        if _is_special(disp):
            color = "#cc0000"  # red for special tokens
        elif disp.startswith(("Ġ", " ")):
            color = "#006600"  # green for space-prefixed (word-start)
        else:
            color = "#0000cc"  # blue for continuation tokens
        ax.annotate(
            disp, (tx, ty),
            fontsize=6, color=color, alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
        )

    var_i = 100 * ev[i]
    var_j = 100 * ev[j]
    ax.set_xlabel(f"PC {i+1} ({var_i:.1f}%)")
    ax.set_ylabel(f"PC {j+1} ({var_j:.1f}%)")
    ax.set_title(f"Eigenvectors {i+1} × {j+1}")
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Plot thought manifold eigenvectors with token labels")
    parser.add_argument("manifold", help="Path to manifold.h5")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B",
                        help="Model ID for tokenizer (default: Qwen/Qwen3.6-27B)")
    parser.add_argument("--no-subsample", action="store_true")
    parser.add_argument("--n-annotate", type=int, default=18,
                        help="Max annotation labels per panel")
    args = parser.parse_args()

    h5path = Path(args.manifold)
    outpath = Path(args.output) if args.output else h5path.with_name(
        "thought_manifold_2d_labeled.png")

    print("Loading manifold...")
    data = load_manifold(str(h5path))
    U = data["eigenvectors"]
    ev = data["explained_variance_ratio"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build token strings for all vocab IDs
    V = U.shape[0]
    print(f"Building {V} token strings...")
    token_strings = [tokenizer.decode([i]) for i in range(V)]

    n_subsample = 0 if args.no_subsample else 20000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    plot_pair(ax1, U, ev, (0, 1), tokenizer, token_strings,
              n_subsample, args.n_annotate)
    plot_pair(ax2, U, ev, (2, 3), tokenizer, token_strings,
              n_subsample, args.n_annotate)

    fig.suptitle("Qwen3.6-27B  Thought Manifold  —  Token-space eigenvectors with cluster labels",
                 fontsize=13, y=0.97)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved → {outpath}")


if __name__ == "__main__":
    main()
