#!/usr/bin/env python
"""Cross-correlate WMT14 fr-en scores with J-lens metrics across training checkpoints.

Generates:
  - Correlation heatmaps (Spearman + Pearson)
  - Scatter plots for strongest wmt-metric pairs
  - Evolution plot of chrF/TER alongside top J-lens metrics
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Paths
WMT_FILE = Path("results/wmt_eval/all_results.json")
JLENS_DIR = Path("results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def load_jlens_metrics():
    """Load J-lens layer-0 metrics from all checkpoint results."""
    steps = []
    metrics = {
        "n_spikes_L0": [],
        "tr_jtj_L0": [],
        "a_coeff_L0": [],
        "kappa_L0": [],
        "alpha_L0": [],
        "eff_rank_L0": [],
        "fcr_L0": [],
    }

    for step_dir in sorted(JLENS_DIR.glob("step*")):
        step_str = step_dir.name.replace("step", "")
        try:
            step = int(step_str)
        except ValueError:
            continue

        stats_file = step_dir / "spectra.h5"
        if not stats_file.exists():
            continue

        try:
            import h5py
            with h5py.File(stats_file, "r") as f:
                layer_grp = f["layer_0"]

                def get_attr(grp, name):
                    return grp.attrs[name] if name in grp.attrs else np.nan

                steps.append(step)
                metrics["n_spikes_L0"].append(get_attr(layer_grp, "n_spikes"))
                metrics["tr_jtj_L0"].append(get_attr(layer_grp, "tr_jtj_mean"))
                metrics["a_coeff_L0"].append(get_attr(layer_grp, "a_coeff"))
                metrics["kappa_L0"].append(get_attr(layer_grp, "coherence"))
                metrics["alpha_L0"].append(get_attr(layer_grp, "power_law_alpha"))
                metrics["eff_rank_L0"].append(get_attr(layer_grp, "effective_rank"))
                metrics["fcr_L0"].append(get_attr(layer_grp, "fcr"))
        except Exception:
            continue

    # Sort by step
    order = np.argsort(steps)
    steps = np.array(steps)[order]
    for k in metrics:
        metrics[k] = np.array(metrics[k])[order]

    return steps, metrics


def load_wmt_metrics():
    """Load WMT14 fr-en scores."""
    with open(WMT_FILE) as f:
        data = json.load(f)

    steps = []
    bleu = []
    chrf = []
    ter = []

    for step_str, vals in data.items():
        steps.append(int(step_str))
        bleu.append(vals["bleu"])
        chrf.append(vals["chrf"])
        ter.append(vals["ter"])

    order = np.argsort(steps)
    return (
        np.array(steps)[order],
        np.array(bleu)[order],
        np.array(chrf)[order],
        np.array(ter)[order],
    )


def align_metrics(wmt_steps, wmt_vals, jlens_steps, jlens_vals):
    """Align two time series by matching steps."""
    common_steps = np.intersect1d(wmt_steps, jlens_steps)

    wmt_aligned = np.array([wmt_vals[np.where(wmt_steps == s)[0][0]] for s in common_steps])
    jlens_aligned = np.array([jlens_vals[np.where(jlens_steps == s)[0][0]] for s in common_steps])

    # Filter out NaN
    mask = ~np.isnan(wmt_aligned) & ~np.isnan(jlens_aligned)
    return common_steps[mask], wmt_aligned[mask], jlens_aligned[mask]


def main():
    print("Loading J-lens metrics...")
    jlens_steps, jlens = load_jlens_metrics()
    print(f"  {len(jlens_steps)} checkpoints")

    print("Loading WMT14 fr-en scores...")
    wmt_steps, bleu, chrf, ter = load_wmt_metrics()
    print(f"  {len(wmt_steps)} checkpoints")
    print(f"  chrF range: {chrf.min():.1f} - {chrf.max():.1f}")
    print(f"  TER range:  {ter.min():.1f} - {ter.max():.1f}")

    wmt_names = ["chrF (↑)", "TER (↓)", "BLEU (↑)"]
    wmt_arrays = [chrf, ter, bleu]
    jlens_names = list(jlens.keys())
    jlens_labels = {
        "n_spikes_L0": "Spike count",
        "tr_jtj_L0": "tr(JᵀJ)",
        "a_coeff_L0": "Identity a",
        "kappa_L0": "Coherence κ",
        "alpha_L0": "Power-law α",
        "eff_rank_L0": "Eff. rank",
        "fcr_L0": "FCR",
    }

    # === 1. Correlation heatmaps ===
    print("\nComputing correlations...")
    n_wmt = len(wmt_names)
    n_jl = len(jlens_names)
    spearman = np.full((n_wmt, n_jl), np.nan)
    pearson = np.full((n_wmt, n_jl), np.nan)
    spearman_p = np.full((n_wmt, n_jl), np.nan)

    for i, (wn, wv) in enumerate(zip(wmt_names, wmt_arrays)):
        for j, (jn, jv) in enumerate(zip(jlens_names, jlens.values())):
            steps_a, wv_a, jv_a = align_metrics(wmt_steps, wv, jlens_steps, jv)
            if len(steps_a) < 5:
                continue
            rho, p = stats.spearmanr(wv_a, jv_a)
            r, _ = stats.pearsonr(wv_a, jv_a)
            spearman[i, j] = rho
            pearson[i, j] = r
            spearman_p[i, j] = p

    # Plot heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax_idx, (title, mat) in enumerate([
        ("Spearman ρ", spearman),
        ("Pearson r", pearson),
        ("Δ = |ρ| − |r|", np.abs(spearman) - np.abs(pearson)),
    ]):
        ax = axes[ax_idx]
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(n_jl))
        ax.set_xticklabels([jlens_labels.get(n, n) for n in jlens_names], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_wmt))
        ax.set_yticklabels(wmt_names, fontsize=8)
        ax.set_title(title, fontsize=11)

        for i in range(n_wmt):
            for j in range(n_jl):
                val = mat[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("WMT14 fr-en × J-Lens L0 metric correlations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "wmt_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → figures/wmt_correlation_heatmap.png")

    # === 2. Scatter plots for top pairs ===
    # Find strongest correlations
    pairs = []
    for i, wn in enumerate(wmt_names):
        for j, jn in enumerate(jlens_names):
            if not np.isnan(spearman[i, j]):
                pairs.append((wn, jn, spearman[i, j], i, j))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Top 6 pairs
    top_pairs = pairs[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (wn, jn, rho, wi, ji) in enumerate(top_pairs):
        ax = axes[idx]
        wmt_vals = wmt_arrays[wi]
        jlens_val = list(jlens.values())[ji]

        steps_a, wv_a, jv_a = align_metrics(wmt_steps, wmt_vals, jlens_steps, jlens_val)
        r, _ = stats.pearsonr(wv_a, jv_a)

        # Color by training step (log scale)
        log_steps = np.log10(steps_a + 1)
        sc = ax.scatter(jv_a, wv_a, c=log_steps, cmap="viridis", s=15, alpha=0.7, edgecolors="none")

        # Trend line
        z = np.polyfit(jv_a, wv_a, 1)
        x_line = np.linspace(jv_a.min(), jv_a.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.5, lw=1)

        ax.set_title(f"ρ = {rho:+.3f}, r = {r:+.3f}", fontsize=10)
        # Also show on xlabel
        ax.set_xlabel(f"{jlens_labels.get(jn, jn)}\nρ={rho:+.3f}  r={r:+.3f}", fontsize=8)
        ax.set_ylabel(wn, fontsize=9)

    # Remove unused axes
    for idx in range(len(top_pairs), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("WMT14 fr-en × J-Lens: Strongest Correlations", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "wmt_correlation_scatters.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → figures/wmt_correlation_scatters.png")

    # === 3. Evolution plot: chrF + TER vs top metric ===
    # Dual-axis time series
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Panel A: chrF vs n_spikes_L0
    ax1 = axes[0, 0]
    steps_a, chrf_a, spikes_a = align_metrics(wmt_steps, chrf, jlens_steps, jlens["n_spikes_L0"])
    ax1.plot(steps_a, chrf_a, "b.-", alpha=0.6, markersize=3, label="chrF")
    ax1.set_ylabel("chrF", color="b", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="b")

    ax1b = ax1.twinx()
    ax1b.plot(steps_a, spikes_a, "r.-", alpha=0.6, markersize=3, label="Spike count")
    ax1b.set_ylabel("Spike count L0", color="r", fontsize=10)
    ax1b.tick_params(axis="y", labelcolor="r")
    rho, _ = stats.spearmanr(chrf_a, spikes_a)
    r, _ = stats.pearsonr(chrf_a, spikes_a)
    ax1.set_title(f"chrF vs Spike count L0 (ρ={rho:+.3f}, r={r:+.3f})", fontsize=10)
    ax1.set_xlabel("Training step")

    # Panel B: TER vs tr_jtj_L0
    ax2 = axes[0, 1]
    steps_a, ter_a, tr_a = align_metrics(wmt_steps, ter, jlens_steps, jlens["tr_jtj_L0"])
    ax2.plot(steps_a, ter_a, "b.-", alpha=0.6, markersize=3, label="TER")
    ax2.set_ylabel("TER (↓)", color="b", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="b")

    ax2b = ax2.twinx()
    ax2b.plot(steps_a, tr_a, "r.-", alpha=0.6, markersize=3, label="tr(JᵀJ)")
    ax2b.set_ylabel("tr(JᵀJ) L0", color="r", fontsize=10)
    ax2b.tick_params(axis="y", labelcolor="r")
    rho, _ = stats.spearmanr(ter_a, tr_a)
    r, _ = stats.pearsonr(ter_a, tr_a)
    ax2.set_title(f"TER vs tr(JᵀJ) L0 (ρ={rho:+.3f}, r={r:+.3f})", fontsize=10)
    ax2.set_xlabel("Training step")

    # Panel C: chrF vs kappa_L0
    ax3 = axes[1, 0]
    steps_a, chrf_a, kappa_a = align_metrics(wmt_steps, chrf, jlens_steps, jlens["kappa_L0"])
    ax3.plot(steps_a, chrf_a, "b.-", alpha=0.6, markersize=3, label="chrF")
    ax3.set_ylabel("chrF", color="b", fontsize=10)
    ax3.tick_params(axis="y", labelcolor="b")

    ax3b = ax3.twinx()
    ax3b.plot(steps_a, kappa_a, "r.-", alpha=0.6, markersize=3, label="κ")
    ax3b.set_ylabel("Coherence κ L0", color="r", fontsize=10)
    ax3b.tick_params(axis="y", labelcolor="r")
    rho, _ = stats.spearmanr(chrf_a, kappa_a)
    r, _ = stats.pearsonr(chrf_a, kappa_a)
    ax3.set_title(f"chrF vs Coherence κ L0 (ρ={rho:+.3f}, r={r:+.3f})", fontsize=10)
    ax3.set_xlabel("Training step")

    # Panel D: TER vs alpha_L0
    ax4 = axes[1, 1]
    steps_a, ter_a, alpha_a = align_metrics(wmt_steps, ter, jlens_steps, jlens["alpha_L0"])
    ax4.plot(steps_a, ter_a, "b.-", alpha=0.6, markersize=3, label="TER")
    ax4.set_ylabel("TER (↓)", color="b", fontsize=10)
    ax4.tick_params(axis="y", labelcolor="b")

    ax4b = ax4.twinx()
    ax4b.plot(steps_a, alpha_a, "r.-", alpha=0.6, markersize=3, label="α")
    ax4b.set_ylabel("Power-law α L0", color="r", fontsize=10)
    ax4b.tick_params(axis="y", labelcolor="r")
    rho, _ = stats.spearmanr(ter_a, alpha_a)
    r, _ = stats.pearsonr(ter_a, alpha_a)
    ax4.set_title(f"TER vs Power-law α L0 (ρ={rho:+.3f}, r={r:+.3f})", fontsize=10)
    ax4.set_xlabel("Training step")

    plt.suptitle("WMT14 fr-en Translation vs J-Lens Metrics (Pythia-160M)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "wmt_vs_jlens_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  → figures/wmt_vs_jlens_evolution.png")

    # === 4. Bar chart: mean |ρ| across translation vs existing benchmarks ===
    # Load existing benchmark correlations for comparison
    print("\n=== WMT14 fr-en × J-Lens top correlations ===")
    for wn, jn, rho, wi, ji in pairs[:10]:
        print(f"  {wn:>12s} × {jlens_labels.get(jn, jn):>15s}: ρ = {rho:+.4f}")

    print(f"\nDone! {len(pairs)} pairs evaluated.")


if __name__ == "__main__":
    main()
