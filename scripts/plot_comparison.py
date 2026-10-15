#!/usr/bin/env python3
"""Side-by-side comparison plots: wikitext vs random-token JLens metrics.

Output: figures/pythia-12b-random-full/comparison/{metric}.png
  Each plot shows two panels:
    Left: wikitext (blue/orange) — layer-mean curve + per-layer heat
    Right: random (green/red) — same layout
    Bottom: delta curve (random − wikitext)

Also generates a 3×4 summary grid.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_LABELS = {
    "coherence": "Coherence κ",
    "a_coeff": "Identity Coefficient a",
    "effective_rank": "Effective Rank",
    "fcr": "Fisher Condition Ratio",
    "mp_sigma2": "MP σ²",
    "n_spikes": "N Spikes",
    "power_law_alpha": "Power-Law α",
    "q_eff": "Q Effective",
    "r_norm": "‖R‖ / √d",
    "sigma_eff_rank": "Σ Effective Rank",
    "tr_jtj_mean": "Tr(JᵀJ)/d",
}


def load_all(results_dir: Path) -> dict:
    """{metric_name: (steps, layer_means, layer_matrix)}"""
    dlist = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("step")],
        key=lambda d: int(d.name.replace("step", "")),
    )
    n_layers = 0
    # Gather raw data
    raw = {}  # {metric: {step: np.array([n_layers])}}
    steps_list = []
    for d in dlist:
        step = int(d.name.replace("step", ""))
        sf = d / "spectra.h5"
        if not sf.exists():
            continue
        steps_list.append(step)
        with h5py.File(sf, "r") as f:
            if n_layers == 0:
                n_layers = f.attrs.get(
                    "n_layers", sum(1 for k in f.keys() if k.startswith("layer_"))
                )
            for li in range(n_layers):
                key = f"layer_{li}"
                if key not in f:
                    continue
                for attr in f[key].attrs:
                    if attr == "eigenvector_overlap":
                        continue
                    if attr not in raw:
                        raw[attr] = {}
                    if step not in raw[attr]:
                        raw[attr][step] = np.full(n_layers, np.nan)
                    raw[attr][step][li] = float(f[key].attrs[attr])

    result = {}
    for mname, mdata in raw.items():
        steps = np.array(sorted(mdata.keys()))
        mat = np.array([mdata[s] for s in steps])  # [n_steps, n_layers]
        means = np.nanmean(mat, axis=1)
        result[mname] = (steps, means, mat)
    return result


def plot_comparison(wt_data, rd_data, mname, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    wt_steps, wt_means, wt_mat = wt_data[mname]
    rd_steps, rd_means, rd_mat = rd_data[mname]

    # Limit steps to common range
    min_step = max(wt_steps[0], rd_steps[0])
    max_step = min(wt_steps[-1], rd_steps[-1])

    wt_mask = (wt_steps >= min_step) & (wt_steps <= max_step)
    rd_mask = (rd_steps >= min_step) & (rd_steps <= max_step)

    wt_steps_c = wt_steps[wt_mask]
    rd_steps_c = rd_steps[rd_mask]
    wt_means_c = wt_means[wt_mask]
    rd_means_c = rd_means[rd_mask]
    wt_mat_c = wt_mat[wt_mask]
    rd_mat_c = rd_mat[rd_mask]

    label = METRIC_LABELS.get(mname, mname)
    cmap = plt.cm.plasma

    fig = plt.figure(figsize=(16, 12))

    # ── Panel A: Wikitext layer-mean + per-layer curves ──
    ax_wt = fig.add_subplot(2, 3, 1)
    for li in range(wt_mat_c.shape[1]):
        vals = wt_mat_c[:, li]
        mask = ~np.isnan(vals)
        if not mask.any():
            continue
        color = cmap(li / (wt_mat_c.shape[1] - 1))
        ax_wt.plot(wt_steps_c[mask], vals[mask], color=color,
                   linewidth=0.4, alpha=0.5)
    ax_wt.plot(wt_steps_c, wt_means_c, color="blue", linewidth=2,
               label="Layer mean", zorder=10)
    ax_wt.set_title("Wikitext Prompts", fontsize=12, fontweight="bold")
    ax_wt.set_ylabel(label, fontsize=11)
    ax_wt.set_xscale("log")
    ax_wt.grid(True, alpha=0.3)
    ax_wt.legend(fontsize=8)

    # ── Panel B: Random layer-mean + per-layer curves ──
    ax_rd = fig.add_subplot(2, 3, 2)
    for li in range(rd_mat_c.shape[1]):
        vals = rd_mat_c[:, li]
        mask = ~np.isnan(vals)
        if not mask.any():
            continue
        color = cmap(li / (rd_mat_c.shape[1] - 1))
        ax_rd.plot(rd_steps_c[mask], vals[mask], color=color,
                   linewidth=0.4, alpha=0.5)
    ax_rd.plot(rd_steps_c, rd_means_c, color="green", linewidth=2,
               label="Layer mean", zorder=10)
    ax_rd.set_title("Random Tokens", fontsize=12, fontweight="bold")
    ax_rd.set_xscale("log")
    ax_rd.grid(True, alpha=0.3)
    ax_rd.legend(fontsize=8)

    # ── Panel C: Overlaid layer means ──
    ax_overlay = fig.add_subplot(2, 3, 3)
    ax_overlay.plot(wt_steps_c, wt_means_c, color="blue", linewidth=2.5,
                    label="Wikitext")
    ax_overlay.plot(rd_steps_c, rd_means_c, color="green", linewidth=2.5,
                    label="Random")
    ax_overlay.set_title("Layer-Mean Overlay", fontsize=12, fontweight="bold")
    ax_overlay.set_xscale("log")
    ax_overlay.legend(fontsize=10)
    ax_overlay.grid(True, alpha=0.3)

    # ── Panel D: Delta (random − wikitext) layer-mean ──
    ax_delta = fig.add_subplot(2, 3, 4)
    # Interpolate to common steps if needed
    if len(wt_steps_c) == len(rd_steps_c) and np.allclose(wt_steps_c, rd_steps_c):
        delta = rd_means_c - wt_means_c
        delta_steps = wt_steps_c
    else:
        # Use the shorter one
        common_steps = np.intersect1d(wt_steps_c, rd_steps_c)
        wt_d = np.array([wt_means_c[list(wt_steps_c).index(s)] for s in common_steps])
        rd_d = np.array([rd_means_c[list(rd_steps_c).index(s)] for s in common_steps])
        delta = rd_d - wt_d
        delta_steps = common_steps

    ax_delta.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    colors_d = ["red" if d > 0 else "blue" for d in delta]
    ax_delta.bar(range(len(delta)), delta, color=colors_d, alpha=0.6, width=0.8)
    # Annotate x ticks with steps
    tick_indices = np.linspace(0, len(delta_steps) - 1, min(10, len(delta_steps))).astype(int)
    ax_delta.set_xticks(tick_indices)
    ax_delta.set_xticklabels([str(s) for s in delta_steps[tick_indices]],
                             rotation=45, fontsize=7)
    ax_delta.set_title("Δ = Random − Wikitext (layer mean)", fontsize=12, fontweight="bold")
    ax_delta.set_ylabel(f"Δ {label}", fontsize=11)
    ax_delta.grid(True, alpha=0.3, axis="y")

    # ── Panel E: Correlation per layer ──
    ax_corr = fig.add_subplot(2, 3, 5)
    layer_corrs = []
    for li in range(wt_mat_c.shape[1]):
        w = wt_mat_c[:, li]
        r = rd_mat_c[:, li]
        mask = ~(np.isnan(w) | np.isnan(r))
        if mask.sum() < 5:
            layer_corrs.append(0)
            continue
        c = np.corrcoef(w[mask], r[mask])[0, 1]
        layer_corrs.append(c)
    layer_corrs = np.array(layer_corrs)
    colors_c = ["green" if c >= 0 else "red" for c in layer_corrs]
    ax_corr.bar(range(len(layer_corrs)), layer_corrs, color=colors_c, alpha=0.7)
    ax_corr.axhline(y=0, color="gray", linestyle="--")
    ax_corr.set_xlabel("Layer", fontsize=11)
    ax_corr.set_ylabel("Pearson r (WT vs RD)", fontsize=11)
    ax_corr.set_title("Per-Layer Correlation\n(Wikitext vs Random)", fontsize=12, fontweight="bold")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.grid(True, alpha=0.3, axis="y")

    # ── Panel F: Per-layer delta at last step ──
    ax_final = fig.add_subplot(2, 3, 6)
    w_final = wt_mat_c[-1]
    r_final = rd_mat_c[-1]
    delta_final = r_final - w_final
    colors_f = ["green" if d >= 0 else "red" for d in delta_final]
    ax_final.bar(range(len(delta_final)), delta_final, color=colors_f, alpha=0.7)
    ax_final.axhline(y=0, color="gray", linestyle="--")
    ax_final.set_xlabel("Layer", fontsize=11)
    ax_final.set_ylabel(f"Δ {label}", fontsize=11)
    ax_final.set_title(f"Per-Layer Δ at Step {max_step}"
                       f"\n(mean Δ = {np.nanmean(delta_final):+.4g})",
                       fontsize=12, fontweight="bold")
    ax_final.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Pythia-12B: {label} — Wikitext vs Random Tokens",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fname = mname.replace("/", "_")
    fig.savefig(out_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def plot_summary_grid(wt_data, rd_data, out_dir):
    """3×4 grid: layer-mean overlay for all 11 metrics."""
    metrics = sorted(set(wt_data.keys()) & set(rd_data.keys()))
    n_cols = 4
    n_rows = (len(metrics) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 4.5))
    fig.suptitle("Pythia-12B: Wikitext (blue) vs Random Tokens (green) — All Metrics",
                 fontsize=16, fontweight="bold", y=0.995)

    for idx, mname in enumerate(metrics):
        ax = axes[idx // n_cols, idx % n_cols]
        label = METRIC_LABELS.get(mname, mname)

        wt_steps, wt_means, _ = wt_data[mname]
        rd_steps, rd_means, _ = rd_data[mname]

        ax.plot(wt_steps, wt_means, color="blue", linewidth=1.5, alpha=0.8,
                label="Wikitext")
        ax.plot(rd_steps, rd_means, color="green", linewidth=1.5, alpha=0.8,
                label="Random")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for idx in range(len(metrics), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "summary_comparison_all.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  summary_comparison_all.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-dir", default="/data/jlens/pythia-12b-deduped")
    parser.add_argument("--rd-dir", default="/data/jlens/pythia-12b-deduped-random")
    parser.add_argument("--out-dir", default="figures/pythia-12b-random-full/comparison")
    args = parser.parse_args()

    wt_dir = Path(args.wt_dir)
    rd_dir = Path(args.rd_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading wikitext data...")
    wt_data = load_all(wt_dir)
    print(f"  {len(wt_data)} metrics, {len(wt_data[list(wt_data.keys())[0]][0])} steps")

    print("Loading random data...")
    rd_data = load_all(rd_dir)
    print(f"  {len(rd_data)} metrics, {len(rd_data[list(rd_data.keys())[0]][0])} steps")

    metrics = sorted(set(wt_data.keys()) & set(rd_data.keys()))
    print(f"\nGenerating {len(metrics)} comparison plots...")
    for mname in metrics:
        fname = plot_comparison(wt_data, rd_data, mname, out_dir)
        print(f"  {fname}.png")

    print("\nSummary grid...")
    plot_summary_grid(wt_data, rd_data, out_dir)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} files in {out_dir}/")


if __name__ == "__main__":
    main()
