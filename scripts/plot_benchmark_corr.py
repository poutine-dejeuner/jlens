#!/usr/bin/env python3
"""Plot J-lens metrics vs benchmark scores — correlation analysis."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# Load cross-joined data
data = np.load("eval_data/cross_join.npz", allow_pickle=True)
steps = np.array(data["steps"])
benchmarks = list(data["benchmarks"])

# Extract arrays
eval_data = {bm: data[f"eval_{bm}"] for bm in benchmarks}
jlens_keys = [k for k in data.keys() if k.startswith("jlens_")]
jlens_metrics = {k.replace("jlens_", ""): data[k] for k in jlens_keys}

# ── Figure 1: Correlation heatmap (Spearman + Pearson) ─────────────────
plot_metrics = [
    "coherence_L0", "coherence_mean",
    "a_coeff_L0", "a_coeff_mean",
    "eff_rank_L0", "eff_rank_L5", "eff_rank_mean",
    "tr_jtj_L0", "tr_jtj_mean",
    "alpha_L0", "alpha_mean",
    "n_spikes_L0",
]

bm_labels = ["ARC-Easy", "ARC-Chal.", "PIQA", "WinoGrande",
             "SciQ", "LAMBADA", "LogiQA", "WSC"]

# Compute both correlation matrices
spearman_mat = np.zeros((len(plot_metrics), len(benchmarks)))
pearson_mat = np.zeros((len(plot_metrics), len(benchmarks)))
p_matrix = np.zeros((len(plot_metrics), len(benchmarks)))

for i, mkey in enumerate(plot_metrics):
    mvals = jlens_metrics[mkey]
    for j, bm in enumerate(benchmarks):
        bvals = eval_data[bm]
        mask = ~np.isnan(mvals) & ~np.isnan(bvals)
        if mask.sum() >= 5:
            r_s, p_s = stats.spearmanr(mvals[mask], bvals[mask])
            r_p, _ = stats.pearsonr(mvals[mask], bvals[mask])
            spearman_mat[i, j] = r_s
            pearson_mat[i, j] = r_p
            p_matrix[i, j] = p_s

# ── Figure 1a: Spearman heatmap ────────────────────────────────────────
fig1a, ax1a = plt.subplots(figsize=(14, 10))
fig1a.suptitle("Spearman ρ: J-lens metrics vs Benchmark scores\n(Pythia-160M deduped, 26 checkpoints)",
               fontsize=13, fontweight="bold")

im = ax1a.imshow(spearman_mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax1a.set_xticks(range(len(benchmarks)))
ax1a.set_xticklabels(bm_labels, rotation=45, ha="right", fontsize=10)
ax1a.set_yticks(range(len(plot_metrics)))
ax1a.set_yticklabels(plot_metrics, fontsize=9)
ax1a.set_title("Spearman rank correlation (ρ)", fontsize=11)

for i in range(len(plot_metrics)):
    for j in range(len(benchmarks)):
        r = spearman_mat[i, j]
        p = p_matrix[i, j]
        if not np.isnan(r):
            color = "white" if abs(r) > 0.6 else "black"
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            ax1a.text(j, i, f"{r:.2f}{sig}", ha="center", va="center",
                      fontsize=9, color=color,
                      fontweight="bold" if abs(r) > 0.7 else "normal")

cbar = plt.colorbar(im, ax=ax1a, shrink=0.8)
cbar.set_label("Spearman ρ", fontsize=10)
fig1a.tight_layout()
fig1a.savefig("figures/correlation_heatmap.png", dpi=200, bbox_inches="tight")
plt.close(fig1a)
print("Saved figures/correlation_heatmap.png")

# ── Figure 1b: Pearson heatmap ─────────────────────────────────────────
fig1b, ax1b = plt.subplots(figsize=(14, 10))
fig1b.suptitle("Pearson r: J-lens metrics vs Benchmark scores\n(Pythia-160M deduped, 26 checkpoints)",
               fontsize=13, fontweight="bold")

im2 = ax1b.imshow(pearson_mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
ax1b.set_xticks(range(len(benchmarks)))
ax1b.set_xticklabels(bm_labels, rotation=45, ha="right", fontsize=10)
ax1b.set_yticks(range(len(plot_metrics)))
ax1b.set_yticklabels(plot_metrics, fontsize=9)
ax1b.set_title("Pearson linear correlation (r)", fontsize=11)

for i in range(len(plot_metrics)):
    for j in range(len(benchmarks)):
        r = pearson_mat[i, j]
        if not np.isnan(r):
            color = "white" if abs(r) > 0.6 else "black"
            ax1b.text(j, i, f"{r:.2f}", ha="center", va="center",
                      fontsize=9, color=color,
                      fontweight="bold" if abs(r) > 0.7 else "normal")

cbar2 = plt.colorbar(im2, ax=ax1b, shrink=0.8)
cbar2.set_label("Pearson r", fontsize=10)
fig1b.tight_layout()
fig1b.savefig("figures/correlation_heatmap_pearson.png", dpi=200, bbox_inches="tight")
plt.close(fig1b)
print("Saved figures/correlation_heatmap_pearson.png")

# ── Figure 1c: Δ = |Spearman| - |Pearson| (non-linearity diagnostic) ───
delta_mat = np.abs(spearman_mat) - np.abs(pearson_mat)

fig1c, ax1c = plt.subplots(figsize=(14, 10))
fig1c.suptitle("Δ = |Spearman ρ| - |Pearson r| — non-linearity diagnostic\n(positive = non-linear relationship)",
               fontsize=13, fontweight="bold")

im3 = ax1c.imshow(delta_mat, cmap="RdYlGn_r", aspect="auto", vmin=-0.15, vmax=0.15)
ax1c.set_xticks(range(len(benchmarks)))
ax1c.set_xticklabels(bm_labels, rotation=45, ha="right", fontsize=10)
ax1c.set_yticks(range(len(plot_metrics)))
ax1c.set_yticklabels(plot_metrics, fontsize=9)
ax1c.set_title("Δ = |ρ| - |r|  (red = more rank-correlated, green = more linear)", fontsize=11)

for i in range(len(plot_metrics)):
    for j in range(len(benchmarks)):
        d = delta_mat[i, j]
        if not np.isnan(d):
            ax1c.text(j, i, f"{d:+.2f}", ha="center", va="center",
                      fontsize=9, color="black")

cbar3 = plt.colorbar(im3, ax=ax1c, shrink=0.8)
cbar3.set_label("Δ = |ρ| - |r|", fontsize=10)
fig1c.tight_layout()
fig1c.savefig("figures/correlation_delta.png", dpi=200, bbox_inches="tight")
plt.close(fig1c)
print("Saved figures/correlation_delta.png")

# ── Figure 2: Best scatter plots ──────────────────────────────────────
top_pairs = [
    ("a_coeff_L0", "arc_easy", "Identity coefficient a (L0)"),
    ("tr_jtj_L0", "piqa", "Sensitivity tr(J⊤J) (L0)"),
    ("n_spikes_L0", "sciq", "Spike count (L0)"),
]

fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("J-lens metrics vs Benchmark accuracy — scatter plots",
              fontsize=13, fontweight="bold")

for idx, (mkey, bm, mlabel) in enumerate(top_pairs):
    ax = axes[idx]
    mvals = jlens_metrics[mkey]
    bvals = eval_data[bm]
    mask = ~np.isnan(mvals) & ~np.isnan(bvals)
    x, y = mvals[mask], bvals[mask]

    r_s, p_s = stats.spearmanr(x, y)
    r_p, _ = stats.pearsonr(x, y)
    sig = "**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")

    colors = np.log10(steps[mask])
    sc = ax.scatter(x, y, c=colors, cmap="viridis", s=60, edgecolors="k",
                    linewidth=0.5, alpha=0.85)

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="gray",
            alpha=0.6, linewidth=1.5)

    ax.set_xlabel(mlabel, fontsize=10)
    ax.set_ylabel(f"{bm} accuracy", fontsize=10)
    ax.set_title(f"ρ = {r_s:.3f}{sig}, r = {r_p:.3f}, n = {len(x)}",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log10(step)", fontsize=8)

fig2.tight_layout()
fig2.savefig("figures/correlation_scatters.png", dpi=200, bbox_inches="tight")
plt.close(fig2)
print("Saved figures/correlation_scatters.png")

# ── Figure 3: Dual-axis time series — a_coeff vs ARC-Easy ─────────────
fig3, ax1 = plt.subplots(figsize=(14, 6))
fig3.suptitle("Training Dynamics: J-lens Identity Coefficient vs ARC-Easy Accuracy",
              fontsize=13, fontweight="bold")

mkey = "a_coeff_L0"
bm = "arc_easy"
mvals = jlens_metrics[mkey]
bvals = eval_data[bm]
mask = ~np.isnan(mvals) & ~np.isnan(bvals)
s = steps[mask]
mv = mvals[mask]
bv = bvals[mask]

color_jlens = "#2196F3"
color_eval = "#FF5722"

ax1.plot(s, mv, "o-", color=color_jlens, linewidth=2, markersize=7,
         label=f"J-lens {mkey}")
ax1.set_xlabel("Training step (log)", fontsize=11)
ax1.set_ylabel("a — identity coefficient (L0)", color=color_jlens, fontsize=11)
ax1.tick_params(axis="y", labelcolor=color_jlens)
ax1.set_xscale("log")
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(s, bv, "s--", color=color_eval, linewidth=2, markersize=7,
         label="ARC-Easy accuracy")
ax2.set_ylabel("ARC-Easy accuracy", color=color_eval, fontsize=11)
ax2.tick_params(axis="y", labelcolor=color_eval)

for tx, label in [(128, "a breaks"), (1000, "a collapse"), (512, "κ<0.8")]:
    ax1.axvline(x=tx, color="gray", linestyle=":", alpha=0.4)
    ax1.text(tx * 1.2, ax1.get_ylim()[0] + 0.05, label, fontsize=8,
             color="gray", rotation=90)

r_s, p_s = stats.spearmanr(mv, bv)
r_p, _ = stats.pearsonr(mv, bv)
sig = "**" if p_s < 0.01 else "*"
ax1.text(0.02, 0.95,
         f"Spearman ρ = {r_s:.3f}{sig}\nPearson r = {r_p:.3f}\nn = {len(s)}",
         transform=ax1.transAxes, fontsize=11, verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

fig3.tight_layout()
fig3.savefig("figures/identity_vs_arc_easy.png", dpi=200, bbox_inches="tight")
plt.close(fig3)
print("Saved figures/identity_vs_arc_easy.png")

# ── Figure 4: Small multiples ──────────────────────────────────────────
fig4, axes = plt.subplots(3, 3, figsize=(18, 18), constrained_layout=True)
fig4.suptitle("J-lens metrics vs Benchmark scores — scatter matrix (Spearman + Pearson)",
              fontsize=14, fontweight="bold")

top_metrics = ["a_coeff_L0", "coherence_L0", "n_spikes_L0",
               "eff_rank_L5", "alpha_mean", "tr_jtj_mean",
               "eff_rank_mean", "a_coeff_L5", "coherence_L5"]

plot_bms = ["arc_easy", "arc_challenge", "piqa", "winogrande", "sciq",
            "lambada_openai", "logiqa", "wsc"]

for row_idx, mkey in enumerate(top_metrics[:3]):
    for col_idx, bm in enumerate(plot_bms[:3]):
        ax = axes[row_idx, col_idx]
        mvals = jlens_metrics[mkey]
        bvals = eval_data[bm]
        mask = ~np.isnan(mvals) & ~np.isnan(bvals)
        x, y = mvals[mask], bvals[mask]
        if len(x) >= 5:
            r_s, p_s = stats.spearmanr(x, y)
            r_p, _ = stats.pearsonr(x, y)
            sig = "**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")
            ax.scatter(x, y, c=np.log10(steps[mask]), cmap="viridis",
                       s=50, edgecolors="k", linewidth=0.5)
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, slope * x_line + intercept, "--", color="gray", alpha=0.6)
            ax.set_title(f"ρ = {r_s:.3f}{sig}, r = {r_p:.3f}", fontsize=10)
        ax.set_xlabel(mkey, fontsize=8)
        ax.set_ylabel(bm, fontsize=8)
        ax.grid(True, alpha=0.3)

for idx in range(9, len(axes.flat)):
    axes.flat[idx].set_visible(False)

fig4.savefig("figures/correlation_small_multiples.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("Saved figures/correlation_small_multiples.png")

# ── Print summary ──────────────────────────────────────────────────────
print()
print("=" * 70)
print("TOP CORRELATIONS (|ρ| > 0.80)")
print("=" * 70)
results = []
for mkey in jlens_metrics:
    mvals = jlens_metrics[mkey]
    for bm in benchmarks:
        bvals = eval_data[bm]
        mask = ~np.isnan(mvals) & ~np.isnan(bvals)
        if mask.sum() >= 5:
            r_s, p_s = stats.spearmanr(mvals[mask], bvals[mask])
            r_p, _ = stats.pearsonr(mvals[mask], bvals[mask])
            if abs(r_s) > 0.80:
                results.append((abs(r_s), r_s, r_p, p_s, mkey, bm))

results.sort(reverse=True)
for abs_r, r_s, r_p, p_s, mkey, bm in results:
    sig = "**" if p_s < 0.01 else "*"
    print(f"  ρ = {r_s:+.4f}{sig}  r = {r_p:+.4f}  |  {mkey:25s}  ×  {bm}")

print()
print("Done.")
