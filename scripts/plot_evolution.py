#!/usr/bin/env python3
"""Generate all Direction IV spectral evolution plots from jspace-research.txt.

Metrics from the paper (Section 5, "Proposed measurement"):
  - CKA block structure across (layer, step)
  - Readout kurtosis
  - Effective dimensionality / effective rank of WU·J̄ℓ
  - Coherence fraction κℓ
  - Spectral quantities: qeff drainage, power-law exponent, spike count
  - MLP-gain curve

Plus the key diagnostic from Direction I (Section 2):
  - aℓ = tr(J̄)/d   (identity coefficient)
  - Rℓ norm (off-identity energy)
  - Power-law exponent α of the eigenvalue tail

Output: summary_evolution.png (multi-panel dashboard) + individual figures.
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jlens.storage import load_checkpoint_result
from jlens.spectra import eigen_decompose, effective_rank, fit_power_law_tail


# ── helpers ──────────────────────────────────────────────────────────────────

def load_all(results_dir: Path) -> dict[int, dict]:
    step_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("step")
        and (d / "metadata.json").exists()
        and (d / "spectra.h5").exists()
    )
    results = {}
    for sd in step_dirs:
        step = int(sd.name.replace("step", ""))
        try:
            results[step] = load_checkpoint_result(results_dir, step)
        except Exception as e:
            print(f"  [skip step {step}: {e}]")
    return results


def steps_array(results: dict) -> np.ndarray:
    return np.array(sorted(results.keys()))


def layer_matrix(results, steps, key: str, source: str = "stats"):
    """Extract (n_steps, n_layers) matrix for a scalar quantity."""
    n_layers = results[steps[0]]["spectral"]["n_layers"]
    M = np.full((len(steps), n_layers), np.nan)
    for i, s in enumerate(steps):
        r = results[s]
        if source == "stats":
            vals = r["stats"][key]
        else:
            vals = r["spectral"][key]
        for j, v in enumerate(vals):
            M[i, j] = v if v is not None else np.nan
    return M


# ── power-law exponent from stored eigenvalues ───────────────────────────────

def power_law_alpha_matrix(results, steps):
    """Compute power-law exponent α for every (step, layer) from eigenvalue tails."""
    n_layers = results[steps[0]]["spectral"]["n_layers"]
    M = np.full((len(steps), n_layers), np.nan)
    for i, s in enumerate(steps):
        r = results[s]
        eigvals_list = r["spectral"]["eigenvalues"]
        for j, ev in enumerate(eigvals_list):
            if ev is None or len(ev) < 20:
                continue
            try:
                result = fit_power_law_tail(ev)
                M[i, j] = result.get("alpha", np.nan) if result else np.nan
            except Exception:
                M[i, j] = np.nan
    return M


def spike_count_matrix(results, steps, n_prompts=1000, d_model=768):
    """Count spikes (> 3× MP bulk edge) for every (step, layer)."""
    q = n_prompts / d_model
    bulk_edge = (1 + np.sqrt(q)) ** 2
    threshold = 3 * bulk_edge  # ≈ 12.8 for n=1000, d=768

    n_layers = results[steps[0]]["spectral"]["n_layers"]
    M = np.full((len(steps), n_layers), np.nan)
    for i, s in enumerate(steps):
        r = results[s]
        eigvals_list = r["spectral"]["eigenvalues"]
        for j, ev in enumerate(eigvals_list):
            if ev is None:
                continue
            M[i, j] = int(np.sum(ev > threshold))
    return M


def effective_rank_matrix(results, steps):
    """Compute effective rank from eigenvalues."""
    n_layers = results[steps[0]]["spectral"]["n_layers"]
    M = np.full((len(steps), n_layers), np.nan)
    for i, s in enumerate(steps):
        r = results[s]
        eigvals_list = r["spectral"]["eigenvalues"]
        for j, ev in enumerate(eigvals_list):
            if ev is None or len(ev) == 0:
                continue
            M[i, j] = effective_rank(ev)
    return M


# ── main dashboard ───────────────────────────────────────────────────────────

def make_dashboard(results_dir: Path, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    results = load_all(results_dir)
    steps = steps_array(results)
    n_steps = len(steps)
    n_layers = results[steps[0]]["spectral"]["n_layers"]
    print(f"  {n_steps} checkpoints, {n_layers} layers")

    # ── Precompute all matrices ──────────────────────────────────────────
    print("Precomputing metrics...")

    coh = layer_matrix(results, steps, "coherence", "stats")
    a_coeff = layer_matrix(results, steps, "a_coeff", "stats")
    r_norm = layer_matrix(results, steps, "r_norm", "stats")
    tr_jtj = layer_matrix(results, steps, "tr_jtj_mean", "stats")
    q_eff = layer_matrix(results, steps, "q_eff", "spectral")
    spikes = spike_count_matrix(results, steps)
    eff_rank = effective_rank_matrix(results, steps)
    pl_alpha = power_law_alpha_matrix(results, steps)

    # Derived: identity coefficient a = tr(J̄)/d
    # Already stored as a_coeff from accumulator

    # ── Figure 1: Coherence κℓ vs depth colored by training phase ────────
    print("Plotting: coherence evolution...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 12))
    fig1.suptitle("Spectral Evolution of the Jacobian Lens (J̄ℓ) — Pythia-160M, 153 checkpoints",
                  fontsize=15, fontweight="bold", y=0.99)

    # Panel A: Coherence κℓ vs layer, grouped by training phase
    ax = axes1[0, 0]
    phase_defs = [
        ("Init (step 1–64)", 1, 64, "blue"),
        ("Collapse (step 128–512)", 128, 512, "red"),
        ("Early structuration (1K–8K)", 1000, 8000, "orange"),
        ("Mid structuration (16K–64K)", 16000, 64000, "green"),
        ("Late (100K–143K)", 100000, 143000, "purple"),
    ]
    for label, lo, hi, color in phase_defs:
        mask = (steps >= lo) & (steps <= hi)
        if not mask.any():
            continue
        phase_coh = coh[mask]
        mean_c = np.nanmean(phase_coh, axis=0)
        std_c = np.nanstd(phase_coh, axis=0)
        layers_arr = np.arange(n_layers)
        ax.plot(layers_arr, mean_c, color=color, linewidth=2, label=label)
        ax.fill_between(layers_arr, mean_c - std_c, mean_c + std_c,
                        color=color, alpha=0.1)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Coherence κℓ")
    ax.set_title("A: Coherence fraction κℓ = tr(J̄ᵀJ̄) / tr(E[JᵀJ])")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel B: Identity coefficient aℓ = tr(J̄)/d vs layer
    ax = axes1[0, 1]
    for label, lo, hi, color in phase_defs:
        mask = (steps >= lo) & (steps <= hi)
        if not mask.any():
            continue
        phase_a = a_coeff[mask]
        mean_a = np.nanmean(phase_a, axis=0)
        ax.plot(np.arange(n_layers), mean_a, color=color, linewidth=2,
                label=label)
    # Add reference lines
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="a=1 (identity)")
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Identity coefficient aℓ = tr(J̄)/d")
    ax.set_title("B: Identity coefficient aℓ (J̄ℓ = aℓ·I + Rℓ)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel C: Off-identity norm ‖Rℓ‖/√d vs layer
    ax = axes1[1, 0]
    for label, lo, hi, color in phase_defs:
        mask = (steps >= lo) & (steps <= hi)
        if not mask.any():
            continue
        phase_r = r_norm[mask]
        mean_r = np.nanmean(phase_r, axis=0)
        ax.plot(np.arange(n_layers), mean_r, color=color, linewidth=2,
                label=label)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("‖Rℓ‖ / √d")
    ax.set_title("C: Off-identity norm ‖Rℓ‖/√d  (where J̄ℓ = aℓ·I + Rℓ)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel D: Effective rank vs layer
    ax = axes1[1, 1]
    for label, lo, hi, color in phase_defs:
        mask = (steps >= lo) & (steps <= hi)
        if not mask.any():
            continue
        phase_er = eff_rank[mask]
        mean_er = np.nanmean(phase_er, axis=0)
        ax.plot(np.arange(n_layers), mean_er, color=color, linewidth=2,
                label=label)
    ax.set_xlabel("Layer ℓ")
    ax.set_ylabel("Effective rank")
    ax.set_title("D: Effective rank of J̄ℓᵀJ̄ℓ")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(output_dir / "summary_evolution.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig1)
    print(f"  → {output_dir / 'summary_evolution.png'}")

    # ── Figure 2: Heatmaps (coherence, a_coeff, eff_rank, pl_alpha) ──────
    print("Plotting: heatmaps...")
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle("Spectral metrics vs (layer, training step) — Pythia-160M",
                  fontsize=14, fontweight="bold")

    heatmap_configs = [
        (coh, "Coherence κℓ", "viridis", (0, 1)),
        (a_coeff, "Identity coefficient aℓ", "RdBu_r", None),
        (eff_rank, "Effective rank", "plasma", None),
        (pl_alpha, "Power-law exponent α", "cividis", None),
    ]

    for (ax, (data, title, cmap, vrange)) in zip(axes2.flat, heatmap_configs):
        masked = np.ma.masked_invalid(data)
        kwargs = dict(aspect="auto", origin="lower", cmap=cmap)
        if vrange:
            kwargs["vmin"], kwargs["vmax"] = vrange
        im = ax.imshow(masked, extent=[0, n_layers - 1, steps[0], steps[-1]],
                       **kwargs)
        ax.set_xlabel("Layer ℓ")
        ax.set_ylabel("Training step")
        ax.set_title(title)
        ax.set_yscale("log")
        # Log y ticks at powers of 10
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs='all', numticks=10))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        cbar = fig2.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(title)

    fig2.tight_layout()
    fig2.savefig(output_dir / "heatmap_dashboard.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"  → {output_dir / 'heatmap_dashboard.png'}")

    # ── Figure 3: Spectral evolution per layer vs step ───────────────────
    print("Plotting: spectral evolution by layer...")
    key_layers = [0, 2, 4, 6, 8, 10]
    n_k = len(key_layers)
    fig3, axes3 = plt.subplots(3, 2, figsize=(14, 16))
    fig3.suptitle("Per-layer spectral metrics across training — Pythia-160M",
                  fontsize=14, fontweight="bold")

    for idx, layer in enumerate(key_layers):
        ax = axes3[idx // 2, idx % 2]

        # Twin axis for a_coeff
        ax2 = ax.twinx()

        (line_coh,) = ax.plot(steps, coh[:, layer], color="blue",
                              linewidth=1, alpha=0.7, label="κℓ")
        (line_er,) = ax.plot(steps, eff_rank[:, layer] / d_model, color="green",
                             linewidth=1, alpha=0.7, label="eff_rank / d")
        (line_a,) = ax2.plot(steps, a_coeff[:, layer], color="red",
                             linewidth=1, alpha=0.7, label="aℓ")
        ax2.plot(steps, np.ones_like(steps), color="red", linestyle="--",
                 alpha=0.3, linewidth=0.8)

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Coherence / eff_rank/d")
        ax2.set_ylabel("aℓ", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        # Combined legend
        lines = [line_coh, line_er, line_a]
        labels = ["Coherence κℓ", "eff_rank/d", "aℓ = tr(J̄)/d"]
        ax.legend(lines, labels, fontsize=8, loc="center right")

    fig3.tight_layout()
    fig3.savefig(output_dir / "spectral_evolution.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig3)
    print(f"  → {output_dir / 'spectral_evolution.png'}")

    # ── Figure 4: Early-layer power law (key diagnostic from paper) ──────
    print("Plotting: power-law tail analysis...")
    fig4, axes4 = plt.subplots(2, 2, figsize=(14, 12))
    fig4.suptitle("Early-layer power-law tail — the scalar alignment order parameter",
                  fontsize=14, fontweight="bold")

    # Panel A: α vs step for early layers (0-4)
    ax = axes4[0, 0]
    early_layers = [0, 1, 2, 3, 4]
    cmap_el = plt.colormaps["plasma"]
    for li in early_layers:
        color = cmap_el(li / 10)
        ax.plot(steps, pl_alpha[:, li], color=color, linewidth=1.5,
                label=f"Layer {li}")
    # Reference lines from paper:
    ax.axhline(y=0.45, color="green", linestyle="--", alpha=0.5,
               label="α≈0.45 (aligned; trained Qwen)")
    ax.axhline(y=0.08, color="gray", linestyle="--", alpha=0.5,
               label="α≈0.08 (uncorrelated factors; Fuss–Catalan)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Power-law exponent α")
    ax.set_title("A: Power-law exponent α of eigenvalue tail")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel B: α averaged over early layers vs step
    ax = axes4[0, 1]
    mean_alpha_early = np.nanmean(pl_alpha[:, :5], axis=1)
    mean_alpha_mid = np.nanmean(pl_alpha[:, 5:8], axis=1)
    mean_alpha_late = np.nanmean(pl_alpha[:, 8:], axis=1)
    ax.plot(steps, mean_alpha_early, color="red", linewidth=2, label="Early (L0-4)")
    ax.plot(steps, mean_alpha_mid, color="orange", linewidth=2, label="Mid (L5-7)")
    ax.plot(steps, mean_alpha_late, color="blue", linewidth=2, label="Late (L8-10)")
    ax.axhline(y=0.45, color="green", linestyle="--", alpha=0.5)
    ax.axhline(y=0.08, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean power-law exponent α")
    ax.set_title("B: Band-averaged power-law exponent")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel C: spike count evolution
    ax = axes4[1, 0]
    for li in early_layers:
        color = cmap_el(li / 10)
        ax.plot(steps, spikes[:, li], color=color, linewidth=1.5,
                label=f"Layer {li}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Spike count (> 3×bulk edge)")
    ax.set_title("C: Spike population (workspace dimensionality proxy)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel D: tr(J̄ᵀJ̄) mean for early layers (the raw energy)
    ax = axes4[1, 1]
    for li in early_layers:
        color = cmap_el(li / 10)
        ax.plot(steps, tr_jtj[:, li], color=color, linewidth=1.5,
                label=f"Layer {li}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("tr(J̄ᵀJ̄) / 1000 (energy)")
    ax.set_title("D: Average transport energy tr(J̄ᵀJ̄)")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4.savefig(output_dir / "power_law_analysis.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig4)
    print(f"  → {output_dir / 'power_law_analysis.png'}")

    # ── Figure 5: Coherence ignition — κ vs step for each layer ──────────
    print("Plotting: coherence ignition...")
    fig5, ax5 = plt.subplots(figsize=(12, 7))
    fig5.suptitle("Transport Coherence Ignition — κℓ vs training step",
                  fontsize=14, fontweight="bold")

    cmap_l = plt.colormaps["plasma"]
    for li in range(n_layers):
        if li == n_layers - 1:  # target layer = NaN
            continue
        color = cmap_l(li / (n_layers - 1))
        ax5.plot(steps, coh[:, li], color=color, linewidth=1.2,
                 label=f"L{li}")

    # Mark genuine phase transitions derived from data:
    # Step 128: a drops below 0.9 (identity breaks)
    # Step 512→1000: a collapses 0.62→0.42, largest drop
    # Step 2000: coherence drops below 0.8, spikes collapse below 20
    # Step ~53000: eff_rank drops below 200 for good
    transitions = [
        (128, "a<0.9\n(identity breaks)", "red"),
        (1000, "a collapses\n0.62→0.42", "darkorange"),
        (2000, "κ<0.8\nspikes<20", "brown"),
        (53000, "eff_rank\n<200", "purple"),
    ]
    for tx, label, color in transitions:
        ax5.axvline(x=tx, color=color, linestyle="--", alpha=0.5, linewidth=1.2)
        ax5.text(tx * 1.15, 1.02, label, fontsize=7, color=color,
                 rotation=0, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax5.set_xlabel("Training step (log scale)")
    ax5.set_ylabel("Coherence κℓ")
    ax5.set_title("Coherence fraction across training — ignition diagnostic")
    ax5.set_xscale("log")
    ax5.set_ylim(0, 1.05)
    ax5.legend(fontsize=7, ncol=2, loc="lower left")
    ax5.grid(True, alpha=0.3)

    fig5.tight_layout()
    fig5.savefig(output_dir / "coherence_ignition.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig5)
    print(f"  → {output_dir / 'coherence_ignition.png'}")

    # ── Figure 6: Effective rank drainage (qeff analog) ──────────────────
    print("Plotting: effective rank drainage...")
    fig6, ax6 = plt.subplots(figsize=(12, 7))
    fig6.suptitle("Effective Rank Drainage — eff_rank vs (layer, step)",
                  fontsize=14, fontweight="bold")

    for li in range(n_layers):
        if li == n_layers - 1:
            continue
        color = cmap_l(li / (n_layers - 1))
        ax6.plot(steps, eff_rank[:, li] / d_model, color=color,
                 linewidth=1.2, label=f"L{li}")

    ax6.set_xlabel("Training step (log scale)")
    ax6.set_ylabel("Effective rank / d")
    ax6.set_title("Effective rank drainage — workspace dimensionality collapse")
    ax6.set_xscale("log")
    ax6.legend(fontsize=7, ncol=2, loc="upper right")
    ax6.grid(True, alpha=0.3)

    fig6.tight_layout()
    fig6.savefig(output_dir / "eff_rank_drainage.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig6)
    print(f"  → {output_dir / 'eff_rank_drainage.png'}")

    # ── Print key numerical results ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("KEY NUMERICAL RESULTS")
    print("=" * 70)

    # Phase summaries
    for label, lo, hi, _ in phase_defs:
        mask = (steps >= lo) & (steps <= hi)
        if not mask.any():
            continue
        print(f"\n{label}:")
        for li in [0, 5, 10]:
            c = np.nanmean(coh[mask, li])
            a = np.nanmean(a_coeff[mask, li])
            er = np.nanmean(eff_rank[mask, li])
            al = np.nanmean(pl_alpha[mask, li])
            sp = int(np.nanmean(spikes[mask, li]))
            print(f"  L{li:2d}: κ={c:.3f}  a={a:.3f}  eff_rank={er:.0f}  α={al:.3f}  spikes={sp}")

    # Best/worst coherence
    best_idx = np.unravel_index(np.nanargmax(coh[:, :10]), coh[:, :10].shape)
    worst_idx = np.unravel_index(np.nanargmin(coh[:, :10]), coh[:, :10].shape)
    print(f"\nMax coherence: κ={coh[best_idx]:.4f} at step {steps[best_idx[0]]}, layer {best_idx[1]}")
    print(f"Min coherence: κ={coh[worst_idx]:.4f} at step {steps[worst_idx[0]]}, layer {worst_idx[1]}")

    print(f"\nAll plots saved to {output_dir}/")
    print("Done.")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()

    d_model = 768
    make_dashboard(Path(args.results_dir), Path(args.output_dir))
