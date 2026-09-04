#!/usr/bin/env python3
"""Dual-axis time-series plots: top J-lens metrics vs benchmark scores.

Mirrors the identity_vs_arc_easy.png style — one figure per pair, each with:
- Left y-axis: J-lens metric over log(training step)
- Right y-axis: benchmark accuracy over log(training step)
- Phase transition markers (a-breaks, κ<0.8, a-collapse)
- Spearman ρ annotation
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Load cross-joined data ─────────────────────────────────────────────
data = np.load("eval_data/cross_join.npz", allow_pickle=True)
steps = np.array(data["steps"])
benchmarks = list(data["benchmarks"])

eval_data = {bm: data[f"eval_{bm}"] for bm in benchmarks}
jlens_keys = [k for k in data.keys() if k.startswith("jlens_")]
jlens_metrics = {k.replace("jlens_", ""): data[k] for k in jlens_keys}

# ── Benchmark display names ────────────────────────────────────────────
BM_DISPLAY = {
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "piqa": "PIQA",
    "winogrande": "WinoGrande",
    "sciq": "SciQ",
    "lambada_openai": "LAMBADA",
    "logiqa": "LogiQA",
    "wsc": "WSC",
}

# ── J-lens metric display names ────────────────────────────────────────
METRIC_DISPLAY = {
    "a_coeff_L0": "a — identity coefficient (L0)",
    "a_coeff_L5": "a — identity coefficient (L5)",
    "a_coeff_mean": "a — identity coefficient (mean)",
    "coherence_L0": "κ — coherence (L0)",
    "coherence_L5": "κ — coherence (L5)",
    "coherence_mean": "κ — coherence (mean)",
    "alpha_L0": "α — power-law exponent (L0)",
    "alpha_L5": "α — power-law exponent (L5)",
    "alpha_mean": "α — power-law exponent (mean)",
    "eff_rank_L0": "Effective rank (L0)",
    "eff_rank_L5": "Effective rank (L5)",
    "eff_rank_mean": "Effective rank (mean)",
    "n_spikes_L0": "Spike count (L0)",
    "n_spikes_L5": "Spike count (L5)",
    "n_spikes_mean": "Spike count (mean)",
    "tr_jtj_L0": "tr(J⊤J) — sensitivity (L0)",
    "tr_jtj_L5": "tr(J⊤J) — sensitivity (L5)",
    "tr_jtj_mean": "tr(J⊤J) — sensitivity (mean)",
}

# ── Top metric-benchmark pairs (by |ρ|, picking most insightful combos) ─
# Best pair per benchmark, L0-preferred, avoiding metric repeats where possible.
TOP_PAIRS = [
    ("a_coeff_L0", "arc_easy"),          # ρ = -0.901 **
    ("n_spikes_L0", "sciq"),             # ρ = -0.894 **
    ("tr_jtj_L0", "piqa"),               # ρ = -0.847 **
    ("tr_jtj_L0", "winogrande"),         # ρ = -0.847 **  (same metric, diff benchmark)
    ("n_spikes_L0", "lambada_openai"),   # ρ = -0.862 **
    ("alpha_L5", "lambada_openai"),      # ρ = +0.820 **  (positive correlation!)
]

# ── Colors ─────────────────────────────────────────────────────────────
JLENS_COLOR = "#2196F3"  # blue
EVAL_COLOR = "#FF5722"  # deep orange

# Phase milestones
PHASES = [
    (128, "a breaks"),
    (1000, "a collapse"),
    (512, "κ < 0.8"),
]


def plot_pair(mkey: str, bm: str, ax, show_legend: bool = True):
    """Plot a single metric-benchmark pair on the given axes."""
    mvals = jlens_metrics[mkey]
    bvals = eval_data[bm]
    mask = ~np.isnan(mvals) & ~np.isnan(bvals)
    s = steps[mask]
    mv = mvals[mask]
    bv = bvals[mask]

    # Correlation
    r_s, p_s = stats.spearmanr(mv, bv)
    sig = "**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")
    r_pe, p_pe = stats.pearsonr(mv, bv)

    # Left axis: J-lens metric
    ax.plot(s, mv, "o-", color=JLENS_COLOR, linewidth=2, markersize=7,
            label=f"J-lens {mkey}")
    ax.set_xlabel("Training step (log)", fontsize=11)
    ax.set_ylabel(METRIC_DISPLAY.get(mkey, mkey), color=JLENS_COLOR, fontsize=11)
    ax.tick_params(axis="y", labelcolor=JLENS_COLOR)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    # Right axis: benchmark
    ax2 = ax.twinx()
    ax2.plot(s, bv, "s--", color=EVAL_COLOR, linewidth=2, markersize=7,
             label=f"{BM_DISPLAY.get(bm, bm)} accuracy")
    ax2.set_ylabel(f"{BM_DISPLAY.get(bm, bm)} accuracy", color=EVAL_COLOR, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=EVAL_COLOR)

    # Phase markers
    ylo, yhi = ax.get_ylim()
    for tx, label in PHASES:
        ax.axvline(x=tx, color="gray", linestyle=":", alpha=0.4)
        ax.text(tx * 1.3, ylo + (yhi - ylo) * 0.08, label,
                fontsize=8, color="gray", rotation=90)

    # Correlation box
    ax.text(
        0.02, 0.95,
        f"Spearman ρ = {r_s:.3f}{sig}\nPearson r = {r_pe:.3f}\nn = {len(s)}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Legend
    if show_legend:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    return r_s, p_s, r_pe, len(s)


# ── Figure: 3×2 grid of dual-axis plots ────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(22, 26))
fig.suptitle(
    "Training Dynamics: J-lens Metrics vs Benchmark Scores\n"
    "(Pythia-160M deduped, 26 eval-matched checkpoints)",
    fontsize=15,
    fontweight="bold",
)

for idx, (mkey, bm) in enumerate(TOP_PAIRS):
    row, col = divmod(idx, 2)
    ax = axes[row][col]
    r_s, p_s, r_pe, n = plot_pair(mkey, bm, ax, show_legend=(idx == 0))

    # Subtitle with ρ
    sig = "**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")
    bm_name = BM_DISPLAY.get(bm, bm)
    metric_name = METRIC_DISPLAY.get(mkey, mkey)
    ax.set_title(f"ρ = {r_s:.3f}{sig}, r = {r_pe:.3f}  |  {metric_name}  ×  {bm_name}",
                 fontsize=12, fontweight="bold")

fig.tight_layout()
fig.savefig("figures/dual_axis_dashboard.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved figures/dual_axis_dashboard.png")

# ── Individual high-res figures ────────────────────────────────────────
for mkey, bm in TOP_PAIRS:
    bm_name = BM_DISPLAY.get(bm, bm).lower().replace(" ", "_")
    fig_i, ax_i = plt.subplots(figsize=(14, 6))

    r_s, p_s, r_pe, n = plot_pair(mkey, bm, ax_i, show_legend=True)
    sig = "**" if p_s < 0.01 else ("*" if p_s < 0.05 else "")

    fig_i.suptitle(
        f"J-lens {METRIC_DISPLAY.get(mkey, mkey)} vs {BM_DISPLAY.get(bm, bm)}",
        fontsize=13,
        fontweight="bold",
    )
    fig_i.tight_layout()
    fname = f"figures/{mkey}_vs_{bm_name}.png"
    fig_i.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig_i)
    print(f"Saved {fname}  (ρ = {r_s:.4f}{sig}, n = {n})")

print()
print("Done.")
