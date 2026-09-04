#!/usr/bin/env python3
"""Coherence vs a_coeff vs alpha — which J-lens metric correlates best?

Three dual-axis figures, one per metric class (L0), comparing against
the same three benchmarks (ARC-Easy, SciQ, LAMBADA):
  1. coherence_L0 — the original favorite
  2. a_coeff_L0  — the identity coefficient (Direction I)
  3. alpha_L0    — power-law exponent (Direction III)

This is designed to sell the argument: coherence κ is NOT the strongest
correlate with downstream performance. a_coeff (tr(J̄)/d) and alpha
(power-law exponent) both beat it.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Load cross-joined data ─────────────────────────────────────────────
data = np.load("eval_data/cross_join.npz", allow_pickle=True)
steps = np.array(data["steps"])
eval_data = {bm: data[f"eval_{bm}"] for bm in list(data["benchmarks"])}
jlens_keys = [k for k in data.keys() if k.startswith("jlens_")]
jlens_metrics = {k.replace("jlens_", ""): data[k] for k in jlens_keys}

JLENS_COLOR = "#2196F3"
EVAL_COLOR = "#FF5722"

PHASES = [(128, "a breaks"), (1000, "a collapse"), (512, "κ < 0.8")]

BM_DISPLAY = {
    "arc_easy": "ARC-Easy",
    "sciq": "SciQ",
    "lambada_openai": "LAMBADA",
}

METRIC_DISPLAY = {
    "coherence_L0": "κ — coherence (L0)",
    "a_coeff_L0": "a — identity coefficient (L0)",
    "alpha_L0": "α — power-law exponent (L0)",
}

# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Three metrics × three benchmarks = 3×3 grid
# ═══════════════════════════════════════════════════════════════════════

METRICS = ["coherence_L0", "a_coeff_L0", "alpha_L0"]
BENCHMARKS = ["arc_easy", "sciq", "lambada_openai"]

fig, axes = plt.subplots(3, 3, figsize=(24, 22))
fig.suptitle(
    "Coherence κ vs Identity Coefficient a vs Power-law Exponent α\n"
    "— which J-lens metric best predicts downstream performance?\n"
    "(Pythia-160M deduped, 26 eval-matched checkpoints)",
    fontsize=14,
    fontweight="bold",
)

for row_idx, mkey in enumerate(METRICS):
    for col_idx, bm in enumerate(BENCHMARKS):
        ax = axes[row_idx][col_idx]

        mvals = jlens_metrics[mkey]
        bvals = eval_data[bm]
        mask = ~np.isnan(mvals) & ~np.isnan(bvals)
        s = steps[mask]
        mv = mvals[mask]
        bv = bvals[mask]

        r_s, p_s = stats.spearmanr(mv, bv)
        r_p, _ = stats.pearsonr(mv, bv)
        sig = "**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")

        ax.plot(s, mv, "o-", color=JLENS_COLOR, linewidth=2, markersize=7,
                label=mkey)
        ax.set_xscale("log")
        ax.set_ylabel(METRIC_DISPLAY[mkey], color=JLENS_COLOR, fontsize=10)
        ax.tick_params(axis="y", labelcolor=JLENS_COLOR)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(s, bv, "s--", color=EVAL_COLOR, linewidth=2, markersize=7,
                 label=f"{BM_DISPLAY[bm]} accuracy")
        ax2.set_ylabel(f"{BM_DISPLAY[bm]} accuracy", color=EVAL_COLOR, fontsize=10)
        ax2.tick_params(axis="y", labelcolor=EVAL_COLOR)

        # Phase markers
        ylo, yhi = ax.get_ylim()
        for tx, label in [(128, "a breaks"), (1000, "a collapse")]:
            ax.axvline(x=tx, color="gray", linestyle=":", alpha=0.4)

        # Correlation box
        ax.text(0.02, 0.95,
                f"Spearman ρ = {r_s:.3f}{sig}\nPearson r = {r_p:.3f}\nn = {len(s)}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        # Column title = benchmark name
        if row_idx == 0:
            ax.set_title(f"{BM_DISPLAY[bm]}", fontsize=13, fontweight="bold",
                         color=EVAL_COLOR, pad=15)

        # Row label = metric
        if col_idx == 0:
            ax.text(-0.25, 0.5, METRIC_DISPLAY[mkey], transform=ax.transAxes,
                    fontsize=12, fontweight="bold", color=JLENS_COLOR,
                    rotation=90, va="center")

fig.tight_layout()
fig.savefig("figures/coherence_vs_acoeff_vs_alpha.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)
print("Saved figures/coherence_vs_acoeff_vs_alpha.png")

# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Bar chart — |Spearman ρ| by metric class, averaged across
#           the 6 well-correlated benchmarks (exclude arc_challenge, logiqa, wsc)
# ═══════════════════════════════════════════════════════════════════════

WELL_CORRELATED = ["arc_easy", "piqa", "winogrande", "sciq", "lambada_openai"]

METRIC_CLASSES = {
    "Coherence κ (L0)": "coherence_L0",
    "Identity a (L0)": "a_coeff_L0",
    "Power-law α (L0)": "alpha_L0",
    "Sensitivity tr(J⊤J) (L0)": "tr_jtj_L0",
    "Effective rank (L0)": "eff_rank_L0",
    "Spike count (L0)": "n_spikes_L0",
}

fig2, ax2 = plt.subplots(figsize=(14, 7))
fig2.suptitle(
    "Which J-lens metric best predicts benchmark scores?\n"
    "(mean |Spearman ρ| across 5 well-correlated benchmarks)",
    fontsize=13, fontweight="bold",
)

bar_data = {}
for label, mkey in METRIC_CLASSES.items():
    rhos = []
    for bm in WELL_CORRELATED:
        mvals = jlens_metrics[mkey]
        bvals = eval_data[bm]
        mask = ~np.isnan(mvals) & ~np.isnan(bvals)
        if mask.sum() >= 5:
            r_s, _ = stats.spearmanr(mvals[mask], bvals[mask])
            rhos.append(abs(r_s))
    bar_data[label] = (np.mean(rhos), np.std(rhos), rhos)

# Sort by mean |ρ|
sorted_labels = sorted(bar_data.keys(), key=lambda x: bar_data[x][0], reverse=True)
means = [bar_data[l][0] for l in sorted_labels]
stds = [bar_data[l][1] for l in sorted_labels]

colors = ["#FF5722" if "Coherence" in l else
          "#2196F3" if "Identity" in l else
          "#4CAF50" if "Power" in l else
          "#9C27B0" if "Sensitivity" in l else
          "#FF9800" if "Effective" in l else
          "#607D8B" for l in sorted_labels]

bars = ax2.bar(range(len(sorted_labels)), means, yerr=stds, color=colors,
               edgecolor="black", linewidth=1.2, capsize=5, alpha=0.85)
ax2.set_xticks(range(len(sorted_labels)))
ax2.set_xticklabels(sorted_labels, rotation=20, ha="right", fontsize=10)
ax2.set_ylabel("Mean |Spearman ρ|", fontsize=12)
ax2.set_ylim(0, 1.0)
ax2.grid(True, alpha=0.3, axis="y")

# Annotate bars
for i, (bar, mean, label) in enumerate(zip(bars, means, sorted_labels)):
    mkey = METRIC_CLASSES[label]
    ax2.text(i, mean + stds[i] + 0.02, f"{mean:.3f}",
             ha="center", fontweight="bold", fontsize=10)

# Add individual benchmark dots
for i, label in enumerate(sorted_labels):
    mkey = METRIC_CLASSES[label]
    rhos = bar_data[label][2]
    x_jitter = np.random.uniform(-0.15, 0.15, len(rhos))
    ax2.scatter(np.full(len(rhos), i) + x_jitter, rhos,
                color="black", s=20, zorder=5, alpha=0.5)

fig2.tight_layout()
fig2.savefig("figures/metric_vs_benchmark_bars.png", dpi=200,
             bbox_inches="tight")
plt.close(fig2)
print("Saved figures/metric_vs_benchmark_bars.png")

# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Per-benchmark comparison — κ vs a vs α side by side
# ═══════════════════════════════════════════════════════════════════════

fig3, axes3 = plt.subplots(2, 3, figsize=(22, 14))
fig3.suptitle(
    "Coherence κ (red) vs Identity a (blue) vs Power-law α (green)\n"
    "— Spearman ρ per benchmark, L0 metrics",
    fontsize=13, fontweight="bold",
)

for idx, bm in enumerate(WELL_CORRELATED):
    ax = axes3[idx // 3][idx % 3]
    bvals = eval_data[bm]

    results = []
    for mkey, color, label in [
        ("coherence_L0", "#FF5722", "κ (coherence)"),
        ("a_coeff_L0", "#2196F3", "a (identity)"),
        ("alpha_L0", "#4CAF50", "α (power-law)"),
    ]:
        mvals = jlens_metrics[mkey]
        mask = ~np.isnan(mvals) & ~np.isnan(bvals)
        r_s, p_s = stats.spearmanr(mvals[mask], bvals[mask])
        r_p, _ = stats.pearsonr(mvals[mask], bvals[mask])
        sig = "**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")
        results.append((label, r_s, p_s, r_p, color))

    y_pos = [2, 1, 0]
    for pos, (label, r_s, p_s, r_p, color) in zip(y_pos, results):
        sig = "**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")
        ax.barh(pos, abs(r_s), color=color, edgecolor="black",
                linewidth=1, alpha=0.85, height=0.5)
        ax.text(abs(r_s) + 0.02, pos,
                f"ρ = {r_s:+.3f}{sig}\nr = {r_p:+.3f}",
                fontsize=10, va="center",
                fontweight="bold" if abs(r_s) > 0.8 else "normal")
        ax.text(0.02, pos, label, fontsize=11, va="center",
                fontweight="bold", color="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in results], fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_title(f"{BM_DISPLAY.get(bm, bm)}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

# Hide extra subplot
axes3[1][2].set_visible(False)

fig3.tight_layout()
fig3.savefig("figures/metric_race_per_benchmark.png", dpi=200,
             bbox_inches="tight")
plt.close(fig3)
print("Saved figures/metric_race_per_benchmark.png")

# ═══════════════════════════════════════════════════════════════════════
# Print summary
# ═══════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("MEAN |ρ| ACROSS 5 WELL-CORRELATED BENCHMARKS")
print("=" * 70)
for label in sorted_labels:
    mean, std, _ = bar_data[label]
    print(f"  {label:<35s}  mean |ρ| = {mean:.3f} ± {std:.3f}")

print()
print("VERDICT:")
coh_mean = bar_data["Coherence κ (L0)"][0]
a_mean = bar_data["Identity a (L0)"][0]
alpha_mean = bar_data["Power-law α (L0)"][0]
print(f"  Coherence κ  beats a?  {'YES' if coh_mean > a_mean else 'NO'}  (Δ = {coh_mean - a_mean:+.3f})")
print(f"  Coherence κ  beats α?  {'YES' if coh_mean > alpha_mean else 'NO'}  (Δ = {coh_mean - alpha_mean:+.3f})")
print(f"  a             beats α?  {'YES' if a_mean > alpha_mean else 'NO'}  (Δ = {a_mean - alpha_mean:+.3f})")
print()
print("Done.")
