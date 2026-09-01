#!/usr/bin/env python3
"""Plot eigenvalue distributions for all layers of each finished checkpoint on one figure."""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jlens.storage import load_checkpoint_result


def plot_eigendist_grid(results_dir: Path, save_path: Path, max_cols: int = 4):
    step_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("step")
        and (d / "spectra.h5").exists()
    )
    steps = [int(d.name.replace("step", "")) for d in step_dirs]

    if not steps:
        print("No results with spectra.h5 found.")
        return

    n_checkpoints = len(steps)
    n_cols = min(max_cols, n_checkpoints)
    n_rows = (n_checkpoints + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle("Eigenvalue distributions — all layers per checkpoint",
                 fontsize=14, y=0.98)

    cmap = plt.colormaps["plasma"]
    n_layers_global = None

    for idx, step in enumerate(steps):
        ax = axes[idx // n_cols][idx % n_cols]
        r = load_checkpoint_result(results_dir, step)
        eigenvalues = r["spectral"]["eigenvalues"]
        n_layers = r["spectral"]["n_layers"]
        if n_layers_global is None:
            n_layers_global = n_layers

        for layer_i in range(n_layers):
            eigvals = eigenvalues[layer_i]
            if eigvals is None:
                continue
            eigvals_pos = eigvals[eigvals > 1e-10]
            if len(eigvals_pos) < 2:
                continue
            sorted_vals = np.sort(eigvals_pos)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            color = cmap(layer_i / max(n_layers - 1, 1))
            ax.loglog(sorted_vals, cdf, color=color, linewidth=0.8)

        n_plotted = sum(1 for e in eigenvalues if e is not None)
        ax.set_title(f"step {step} ({n_plotted} layers)", fontsize=10)
        ax.set_xlabel("λ")
        ax.set_ylabel("CDF(λ)")
        ax.grid(True, alpha=0.3, which="both")

    # Hide unused subplots
    for idx in range(n_checkpoints, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    if n_layers_global:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_layers_global - 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.6, pad=0.02)
        cbar.set_label("Layer")

    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    fig.savefig(save_path, bbox_inches="tight", dpi=200)
    print(f"Saved {save_path}  ({n_checkpoints} checkpoints)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="eigendist_grid.png")
    parser.add_argument("--max-cols", type=int, default=4)
    args = parser.parse_args()
    plot_eigendist_grid(Path(args.results_dir), Path(args.output), args.max_cols)
