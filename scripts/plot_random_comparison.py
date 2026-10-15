#!/usr/bin/env python3
"""Plot wikitext vs random-token comparison for the Pythia-12B JLens sweep.

Reads spectral metrics from two results directories and produces overlay plots
showing how metrics differ between structured (wikitext) and unstructured
(random token) prompts.

Output: figures/pythia-12b-random/
"""
import sys, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py


def load_all_light(results_dir: Path) -> dict[int, dict]:
    """Load only scalar attrs from stats.h5 + spectra.h5 — no jbar matrices."""
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
            stats = {}
            spectral = {}
            with h5py.File(sd / "stats.h5", "r") as f:
                n_layers = f.attrs["n_layers"]
                stats["coherence"] = []
                stats["a_coeff"] = []
                stats["r_norm"] = []
                stats["tr_jtj_mean"] = []
                for i in range(n_layers):
                    grp = f[f"layer_{i}"]
                    stats["coherence"].append(grp.attrs.get("coherence", None))
                    stats["a_coeff"].append(grp.attrs.get("a_coeff", None))
                    stats["r_norm"].append(grp.attrs.get("r_norm", None))
                    stats["tr_jtj_mean"].append(grp.attrs.get("tr_jtj_mean", None))
            with h5py.File(sd / "spectra.h5", "r") as f:
                n_layers_s = f.attrs["n_layers"]
                d_model = f.attrs["d_model"]
                spectral["n_layers"] = n_layers_s
                spectral["d_model"] = d_model
                spectral["q_eff"] = []
                for i in range(n_layers_s):
                    grp = f[f"layer_{i}"]
                    spectral["q_eff"].append(grp.attrs.get("q_eff", None))
            results[step] = {"stats": stats, "spectral": spectral}
        except Exception as e:
            print(f"  [skip step {step}: {e}]")
    return results


def common_steps(a: dict, b: dict) -> np.ndarray:
    return np.array(sorted(set(a.keys()) & set(b.keys())))


def layer_matrix(results, steps, key: str, source: str = "stats"):
    n_layers = results[steps[0]]["spectral"]["n_layers"]
    M = np.full((len(steps), n_layers), np.nan)
    for i, s in enumerate(steps):
        r = results[s]
        vals = r[source][key]
        for j, v in enumerate(vals):
            M[i, j] = v if v is not None else np.nan
    return M


def make_comparison(wikitext_dir: Path, random_dir: Path, output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading wikitext results (attrs only)...")
    wt = load_all_light(wikitext_dir)
    print(f"  {len(wt)} checkpoints")
    print("Loading random-token results (attrs only)...")
    rnd = load_all_light(random_dir)
    print(f"  {len(rnd)} checkpoints")

    steps = common_steps(wt, rnd)
    n_steps = len(steps)
    sample = wt[steps[0]]
    n_layers = sample["spectral"]["n_layers"]
    d_model = sample["spectral"]["d_model"]
    print(f"  {n_steps} common checkpoints, {n_layers} layers, d={d_model}")

    if n_steps < 3:
        print("ERROR: Need at least 3 common checkpoints for a plot.")
        return

    # Extract metrics for both
    print("Extracting metrics...")
    # Wikitext
    wt_coh = layer_matrix(wt, steps, "coherence", "stats")
    wt_a = layer_matrix(wt, steps, "a_coeff", "stats")
    wt_r = layer_matrix(wt, steps, "r_norm", "stats")
    wt_tr = layer_matrix(wt, steps, "tr_jtj_mean", "stats")
    wt_qeff = layer_matrix(wt, steps, "q_eff", "spectral")
    # Random
    rnd_coh = layer_matrix(rnd, steps, "coherence", "stats")
    rnd_a = layer_matrix(rnd, steps, "a_coeff", "stats")
    rnd_r = layer_matrix(rnd, steps, "r_norm", "stats")
    rnd_tr = layer_matrix(rnd, steps, "tr_jtj_mean", "stats")
    rnd_qeff = layer_matrix(rnd, steps, "q_eff", "spectral")

    # ── Figure 1: Identity coefficient a_l across layers ────────────────
    print("Plotting: identity coefficient comparison...")
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle("Wikitext vs Random Tokens — Pythia-12B Jacobian Lens",
                  fontsize=16, fontweight="bold")

    def phase_mask(lo, hi):
        return (steps >= lo) & (steps <= hi)

    phases = [
        ("Init (step 1-64)", 1, 64),
        ("Collapse (128-512)", 128, 512),
        ("Early (1K-8K)", 1000, 8000),
        ("Mid (16K-64K)", 16000, 64000),
        ("Late (100K-143K)", 100000, 143000),
    ]

    # Map phase to color
    phase_colors = ["blue", "red", "orange", "green", "purple"]

    # Panel A: a_coeff vs layer (phase-averaged, wikitext vs random)
    ax = axes1[0, 0]
    for (label, lo, hi), color in zip(phases, phase_colors):
        mask = phase_mask(lo, hi)
        if not mask.any():
            continue
        ax.plot(np.arange(n_layers), np.nanmean(wt_a[mask], axis=0),
                color=color, linewidth=2, linestyle="-", label=f"WT {label}")
        ax.plot(np.arange(n_layers), np.nanmean(rnd_a[mask], axis=0),
                color=color, linewidth=2, linestyle="--", label=f"RND {label}")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("aℓ = tr(J̄)/d")
    ax.set_title("A: Identity coefficient (solid=wikitext, dashed=random)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel B: a_coeff scatter (WT vs RND, all layers, all steps)
    ax = axes1[0, 1]
    for li in [0, 5, 10, 15, 20, 25, 30]:
        ax.scatter(wt_a[:, li], rnd_a[:, li], s=15, alpha=0.6, label=f"L{li}")
    lims = [0, 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("aℓ (wikitext)")
    ax.set_ylabel("aℓ (random)")
    ax.set_title("B: aℓ scatter (WT vs RND)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Panel C: coherence scatter
    ax = axes1[0, 2]
    for li in [0, 5, 10, 15, 20, 25, 30]:
        ax.scatter(wt_coh[:, li], rnd_coh[:, li], s=15, alpha=0.6, label=f"L{li}")
    lims = [0, 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("κℓ (wikitext)")
    ax.set_ylabel("κℓ (random)")
    ax.set_title("C: Coherence scatter (WT vs RND)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Panel D: a_coeff difference Δ = random - wikitext (heatmap)
    ax = axes1[1, 0]
    delta_a = rnd_a - wt_a
    im = ax.imshow(delta_a.T, aspect="auto", origin="lower",
                   cmap="RdBu_r", extent=[steps[0], steps[-1], 0, n_layers - 1],
                   vmin=-0.3, vmax=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_title("D: Δaℓ = aℓ(random) − aℓ(wikitext)")
    cbar = fig1.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δaℓ")
    # Redder = random is higher (random closer to identity)

    # Panel E: coherence difference Δκ = random - wikitext (heatmap)
    ax = axes1[1, 1]
    delta_coh = rnd_coh - wt_coh
    im = ax.imshow(delta_coh.T, aspect="auto", origin="lower",
                   cmap="RdBu_r", extent=[steps[0], steps[-1], 0, n_layers - 1],
                   vmin=-0.3, vmax=0.3)
    ax.set_xscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_title("E: Δκℓ = κℓ(random) − κℓ(wikitext)")
    cbar = fig1.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Δκℓ")

    # Panel F: tr(JTJ) ratio
    ax = axes1[1, 2]
    for li in [0, 5, 10, 15, 20, 25, 30]:
        ax.semilogx(steps, rnd_tr[:, li] / wt_tr[:, li], linewidth=1,
                    alpha=0.7, label=f"L{li}")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("tr(JTJ) ratio (random/wikitext)")
    ax.set_title("F: Energy ratio tr(J̄ᵀJ̄)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(output_dir / "identity_comparison.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig1)
    print(f"  → {output_dir / 'identity_comparison.png'}")

    # ── Figure 2: Per-layer evolution comparison ────────────────────────
    print("Plotting: per-layer evolution comparison...")
    key_layers = [0, 5, 10, 15, 20, 25, 30]
    n_k = len(key_layers)
    ncols = 4
    nrows = (n_k + ncols - 1) // ncols

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(20, 5 * nrows))
    fig2.suptitle("Per-Layer Spectral Evolution — Wikitext vs Random Tokens",
                  fontsize=15, fontweight="bold")

    for idx, li in enumerate(key_layers):
        ax = axes2[idx // ncols, idx % ncols] if nrows > 1 else axes2[idx]

        ax.plot(steps, wt_a[:, li], color="blue", linewidth=1.5,
                label="aℓ (wikitext)")
        ax.plot(steps, rnd_a[:, li], color="red", linewidth=1.5,
                label="aℓ (random)")
        ax.plot(steps, wt_coh[:, li], color="cyan", linewidth=1,
                alpha=0.5, label="κℓ (wikitext)")
        ax.plot(steps, rnd_coh[:, li], color="orange", linewidth=1,
                alpha=0.5, label="κℓ (random)")
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Value")
        ax.set_title(f"Layer {li}")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    # Hide empty subplots
    for idx in range(n_k, nrows * ncols):
        ax = axes2[idx // ncols, idx % ncols] if nrows > 1 else axes2[idx]
        ax.set_visible(False)

    fig2.tight_layout()
    fig2.savefig(output_dir / "per_layer_evolution.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"  → {output_dir / 'per_layer_evolution.png'}")

    # ── Figure 3: Correlation of metrics across steps ───────────────────
    print("Plotting: step-wise correlations...")
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 12))
    fig3.suptitle("Step-wise Correlation: Does training trajectory match?",
                  fontsize=14, fontweight="bold")

    # Panel A: aℓ(wikitext) vs aℓ(random) per step, colored by layer
    ax = axes3[0, 0]
    for li in range(n_layers):
        if li == n_layers - 1:
            continue
        # Spearman rho for this layer across all steps
        valid = ~(np.isnan(wt_a[:, li]) | np.isnan(rnd_a[:, li]))
        if valid.sum() < 3:
            continue
        from scipy.stats import spearmanr
        rho, _ = spearmanr(wt_a[valid, li], rnd_a[valid, li])
        color = plt.cm.plasma(li / n_layers)
        ax.scatter(wt_a[:, li], rnd_a[:, li], s=10, alpha=0.3, color=color,
                   label=f"L{li} ρ={rho:.2f}" if li % 5 == 0 else "")
    lims = [0, 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("aℓ (wikitext)")
    ax.set_ylabel("aℓ (random)")
    ax.set_title("A: aℓ cross-prompt type")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Panel B: κℓ(wikitext) vs κℓ(random)
    ax = axes3[0, 1]
    for li in range(n_layers):
        if li == n_layers - 1:
            continue
        valid = ~(np.isnan(wt_coh[:, li]) | np.isnan(rnd_coh[:, li]))
        if valid.sum() < 3:
            continue
        from scipy.stats import spearmanr
        rho, _ = spearmanr(wt_coh[valid, li], rnd_coh[valid, li])
        color = plt.cm.plasma(li / n_layers)
        ax.scatter(wt_coh[:, li], rnd_coh[:, li], s=10, alpha=0.3, color=color,
                   label=f"L{li} ρ={rho:.2f}" if li % 5 == 0 else "")
    lims = [0, 1.05]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("κℓ (wikitext)")
    ax.set_ylabel("κℓ (random)")
    ax.set_title("B: κℓ cross-prompt type")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Panel C: Spearman ρ per layer (aℓ correlation across steps between WT and RND)
    ax = axes3[1, 0]
    from scipy.stats import spearmanr
    a_corrs = []
    coh_corrs = []
    for li in range(n_layers - 1):
        valid_a = ~(np.isnan(wt_a[:, li]) | np.isnan(rnd_a[:, li]))
        valid_c = ~(np.isnan(wt_coh[:, li]) | np.isnan(rnd_coh[:, li]))
        if valid_a.sum() >= 3:
            rho_a, _ = spearmanr(wt_a[valid_a, li], rnd_a[valid_a, li])
            a_corrs.append(rho_a)
        else:
            a_corrs.append(np.nan)
        if valid_c.sum() >= 3:
            rho_c, _ = spearmanr(wt_coh[valid_c, li], rnd_coh[valid_c, li])
            coh_corrs.append(rho_c)
        else:
            coh_corrs.append(np.nan)
    ax.plot(range(n_layers - 1), a_corrs, "o-", color="blue", linewidth=1.5,
            label="aℓ correlation")
    ax.plot(range(n_layers - 1), coh_corrs, "s-", color="green", linewidth=1.5,
            label="κℓ correlation")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ (WT vs RND across steps)")
    ax.set_title("C: Per-layer step-trajectory correlation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Ratio of random/wikitext for aℓ and κℓ across layers
    ax = axes3[1, 1]
    ratio_a = np.nanmean(rnd_a, axis=0) / np.nanmean(wt_a, axis=0)
    ratio_coh = np.nanmean(rnd_coh, axis=0) / np.nanmean(wt_coh, axis=0)
    ax.plot(range(n_layers - 1), ratio_a[:n_layers-1], "o-", color="red",
            linewidth=2, label="mean(a_random) / mean(a_wikitext)")
    ax.plot(range(n_layers - 1), ratio_coh[:n_layers-1], "s-", color="orange",
            linewidth=2, label="mean(κ_random) / mean(κ_wikitext)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Ratio (random / wikitext)")
    ax.set_title("D: Mean ratio random/wikitext by layer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(output_dir / "correlation_analysis.png", dpi=200,
                 bbox_inches="tight")
    plt.close(fig3)
    print(f"  → {output_dir / 'correlation_analysis.png'}")

    # ── Print summary stats ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    for li in [0, 5, 10, 15, 20, 25, 30]:
        wa = np.nanmean(wt_a[:, li])
        ra = np.nanmean(rnd_a[:, li])
        wc = np.nanmean(wt_coh[:, li])
        rc = np.nanmean(rnd_coh[:, li])
        print(f"  L{li:2d}: aℓ: WT={wa:.3f} RND={ra:.3f} Δ={ra-wa:+.3f}  "
              f"κℓ: WT={wc:.3f} RND={rc:.3f} Δ={rc-wc:+.3f}")

    print(f"\nAll plots saved to {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikitext-dir", default="/data/jlens/pythia-12b-deduped")
    parser.add_argument("--random-dir", default="/data/jlens/pythia-12b-deduped-random")
    parser.add_argument("--output-dir", default="figures/pythia-12b-random")
    args = parser.parse_args()

    make_comparison(Path(args.wikitext_dir), Path(args.random_dir),
                    Path(args.output_dir))
