"""Visualization for checkpoint-ladder spectral analysis."""

import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps

from .storage import load_checkpoint_result

logger = logging.getLogger(__name__)

# Style configuration
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.figsize": (10, 6),
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def load_all_results(results_dir: Path) -> dict:
    """Load all checkpoint results from a results directory.

    Returns:
        Dict mapping step (int) → result dict.
    """
    step_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("step")
    )

    results = {}
    for step_dir in step_dirs:
        step = int(step_dir.name.replace("step", ""))
        try:
            results[step] = load_checkpoint_result(results_dir, step)
        except Exception as e:
            logger.warning(f"Failed to load step {step}: {e}")

    logger.info(f"Loaded {len(results)} checkpoint results")
    return results


def plot_coherence_evolution(
    results: dict,
    save_path: Path | None = None,
):
    """Plot coherence κ_ℓ vs depth, colored by training step phase."""
    steps = sorted(results.keys())
    n_layers = results[steps[0]]["spectral"]["n_layers"]

    # Group steps into phases
    phases = {
        "early (step 1-512)": [],
        "mid (step 1K-32K)": [],
        "late (step 33K-143K)": [],
    }
    for step in steps:
        if step <= 512:
            phases["early (step 1-512)"].append(step)
        elif step <= 32000:
            phases["mid (step 1K-32K)"].append(step)
        else:
            phases["late (step 33K-143K)"].append(step)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"early (step 1-512)": "blue",
              "mid (step 1K-32K)": "orange",
              "late (step 33K-143K)": "green"}

    for phase_name, phase_steps in phases.items():
        if not phase_steps:
            continue
        # Average over steps in this phase
        all_kappa = []
        for step in phase_steps:
            kappa = results[step]["spectral"]["coherence"]
            all_kappa.append(kappa)

        mean_kappa = np.mean(all_kappa, axis=0)
        std_kappa = np.std(all_kappa, axis=0)
        layers = np.arange(n_layers)

        ax.plot(layers, mean_kappa, color=colors[phase_name],
                label=phase_name, linewidth=2)
        ax.fill_between(layers,
                         mean_kappa - std_kappa,
                         mean_kappa + std_kappa,
                         color=colors[phase_name], alpha=0.15)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Coherence κ_ℓ")
    ax.set_title("Coherence fraction vs depth by training phase")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved to {save_path}")
    else:
        return fig


def plot_spectral_heatmap(
    results: dict,
    quantity: str = "q_eff",
    save_path: Path | None = None,
):
    """Plot a 2D heatmap: quantity vs (layer, training step).

    Args:
        results: Dict of step → result
        quantity: Which quantity to plot:
            'q_eff', 'coherence', 'n_spikes', 'power_law_alpha',
            'effective_rank', 'a_coeff', 'r_norm'
    """
    steps = sorted(results.keys())
    n_layers = results[steps[0]]["spectral"]["n_layers"]

    # Build matrix: (n_steps, n_layers)
    data = np.zeros((len(steps), n_layers))
    for i, step in enumerate(steps):
        r = results[step]
        if quantity in ["coherence", "a_coeff", "r_norm"]:
            data[i] = r["stats"][quantity]
        else:
            data[i] = r["spectral"][quantity]

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, n_layers - 1, steps[0], steps[-1]],
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Training step")
    ax.set_title(f"{quantity} vs layer × training step")

    # Log scale for y-axis (steps are log-spaced early, linear later)
    ax.set_yscale("log")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(quantity)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved to {save_path}")
    else:
        return fig


def plot_eigenvalue_distribution(
    results: dict,
    step: int,
    save_path: Path | None = None,
):
    """Plot eigenvalue CDF for all layers at a given checkpoint."""
    if step not in results:
        raise ValueError(f"Step {step} not in results")

    r = results[step]
    eigenvalues = r["spectral"]["eigenvalues"]
    n_layers = r["spectral"]["n_layers"]

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = colormaps["plasma"]
    for i in range(n_layers):
        eigvals = eigenvalues[i]
        eigvals_pos = eigvals[eigvals > 1e-10]

        # Empirical CDF
        sorted_vals = np.sort(eigvals_pos)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        color = cmap(i / (n_layers - 1))
        ax.loglog(sorted_vals, cdf, color=color, linewidth=1,
                  label=f"L{i}" if i % 2 == 0 else "")

    # Power-law reference lines
    for alpha, label in [(0.45, "α=0.45 (aligned)"), (0.08, "α=0.08 (uncorrelated)")]:
        x_ref = np.logspace(-3, 2, 100)
        y_ref = x_ref ** alpha
        y_ref = y_ref / y_ref[-1] * 0.1  # scale for visibility
        ax.loglog(x_ref, y_ref, "--", linewidth=1, alpha=0.5, label=label)

    ax.set_xlabel("Eigenvalue λ")
    ax.set_ylabel("CDF(λ)")
    ax.set_title(f"Eigenvalue distribution — step {step}")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved to {save_path}")
    else:
        return fig


def plot_summary_dashboard(
    results: dict,
    save_dir: Path | None = None,
):
    """Generate a full dashboard of plots."""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    figs = []

    # 1. Coherence evolution
    fig = plot_coherence_evolution(results)
    if save_dir:
        fig.savefig(save_dir / "coherence_evolution.png", bbox_inches="tight")
    figs.append(fig)
    plt.close(fig)

    # 2-6. Heatmaps
    for quantity in ["coherence", "q_eff", "power_law_alpha", "n_spikes",
                      "effective_rank", "a_coeff"]:
        fig = plot_spectral_heatmap(results, quantity=quantity)
        if save_dir:
            fig.savefig(save_dir / f"heatmap_{quantity}.png", bbox_inches="tight")
        figs.append(fig)
        plt.close(fig)

    # 7. Eigenvalue distribution for key checkpoints
    steps = sorted(results.keys())
    key_steps = [steps[0]]  # first
    # Find steps near phase transitions
    for target in [512, 1000, 32000]:
        closest = min(steps, key=lambda s: abs(s - target))
        if closest not in key_steps:
            key_steps.append(closest)
    if steps[-1] not in key_steps:
        key_steps.append(steps[-1])

    for step in key_steps:
        fig = plot_eigenvalue_distribution(results, step)
        if save_dir:
            fig.savefig(save_dir / f"eigenvalues_step{step}.png",
                       bbox_inches="tight")
        figs.append(fig)
        plt.close(fig)

    logger.info(f"Generated {len(figs)} plots")
    return figs
