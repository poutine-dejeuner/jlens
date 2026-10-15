#!/usr/bin/env python3
"""Generate individual figures for Pythia-12B J-lens analysis.

Output:
  figures/pythia-12b/metrics/{metric}.png  — one plot per J-lens metric, all layers
  figures/pythia-12b/benchmarks/           — one plot per benchmark, dual y-axis
"""
import json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_val_loss(results_dir: Path) -> dict:
    with open(results_dir / "val_loss_all.json") as f:
        entries = json.load(f)
    return {int(e["step"].replace("step", "")): e for e in entries}


def load_spectra_attrs(results_dir: Path) -> dict:
    """{metric_name: {step: np.array([n_layers])}}"""
    import h5py
    step_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("step")],
        key=lambda d: int(d.name.replace("step", "")),
    )
    metrics = {}
    n_layers = 0
    for d in step_dirs:
        step = int(d.name.replace("step", ""))
        sf = d / "spectra.h5"
        if not sf.exists():
            continue
        with h5py.File(sf, "r") as f:
            if n_layers == 0:
                n_layers = f.attrs.get("n_layers", sum(1 for k in f.keys() if k.startswith("layer_")))
            for layer_idx in range(n_layers):
                lg = f[f"layer_{layer_idx}"]
                for attr_name in lg.attrs:
                    if attr_name not in metrics:
                        metrics[attr_name] = {}
                    if step not in metrics[attr_name]:
                        metrics[attr_name][step] = np.full(n_layers, np.nan)
                    metrics[attr_name][step][layer_idx] = float(lg.attrs[attr_name])
    return metrics, n_layers


LABELS = {
    "coherence": "Coherence ρ",
    "effective_rank": "Effective Rank",
    "power_law_alpha": "Power-Law α",
    "n_spikes": "N Spikes",
    "fcr": "Fisher Condition Ratio",
    "r_norm": "||R|| (residual norm)",
    "a_coeff": "A Coefficient",
    "tr_jtj_mean": "Tr(JᵀJ) / d",
    "sigma_eff_rank": "Σ Eff. Rank",
}


def plot_per_metric(attrs, n_layers, val_loss, steps_sorted, out_dir):
    """One figure per J-lens metric, showing all layers + overlaid loss (normalized)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_steps = np.array(sorted(val_loss.keys()))
    loss_vals = np.array([val_loss[s]["loss"] for s in loss_steps])

    cmap = plt.cm.plasma
    for mname in sorted(attrs.keys()):
        if mname == "eigenvector_overlap":
            continue  # flat metric, produces no usable plot
        mdata = attrs[mname]
        msteps = sorted(mdata.keys())
        label = LABELS.get(mname, mname.replace("_", " ").title())

        fig, ax = plt.subplots(figsize=(10, 5.5))

        # Gather all metric values to compute global min/max for normalization
        all_vals = []
        for li in range(n_layers):
            vals = np.array([mdata[s][li] for s in msteps])
            all_vals.append(vals[~np.isnan(vals)])
        if all_vals:
            all_cat = np.concatenate(all_vals)
            vmin, vmax = all_cat.min(), all_cat.max()
        else:
            vmin, vmax = 0, 1

        # Individual layers (thin, colored by depth)
        for li in range(n_layers):
            vals = np.array([mdata[s][li] for s in msteps])
            mask = ~np.isnan(vals)
            if not mask.any():
                continue
            vals_norm = (vals[mask] - vmin) / (vmax - vmin + 1e-10)
            color = cmap(li / (n_layers - 1))
            ax.plot(np.array(msteps)[mask], vals_norm,
                    color=color, linewidth=0.5, alpha=1)

        # Overlay loss (normalized, black dashed, same visual weight as layers)
        loss_min, loss_max = loss_vals.min(), loss_vals.max()
        loss_norm = (loss_vals - loss_min) / (loss_max - loss_min + 1e-10)
        ax.plot(loss_steps, loss_norm,
                color="black", linewidth=1.5, linestyle="--", alpha=0.7,
                label=f"Val Loss ({loss_min:.2f}→{loss_max:.2f})", zorder=10)

        # Colorbar for layer depth
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_layers - 1))
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label("Layer", fontsize=9)

        ax.set_ylabel(f"{label}\n(normalized [{vmin:.3g}, {vmax:.3g}])", fontsize=10)
        ax.set_title(f"Pythia-12B: {label}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        fname = mname.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}.png")


def plot_per_benchmark(cross_join_path, out_dir):
    """One figure per benchmark: score + J-lens metrics (dual axis)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cj = np.load(cross_join_path, allow_pickle=True)
    steps = cj["steps"]
    benchmarks = [b.decode() if isinstance(b, bytes) else b for b in cj["benchmarks"]]
    benchmark_cols = [k for k in cj.keys() if k.startswith("eval_")]
    jlens_cols = [k for k in cj.keys() if k.startswith("jlens_")]

    # J-lens display names: strip jlens_ prefix, pretty format
    def jlens_label(col):
        name = col.replace("jlens_", "").replace("_", " ")
        parts = name.rsplit(" ", 1)
        if parts[-1] in ("L0", "L5", "mean"):
            return f"{parts[0].title()} ({parts[1]})"
        return name.title()

    from scipy.stats import pearsonr

    colors_jlens = plt.cm.tab10(np.linspace(0, 1, len(jlens_cols)))

    for bi, bcol in enumerate(benchmark_cols):
        bname = bcol.replace("eval_", "")
        bvals = cj[bcol]
        mask_b = ~np.isnan(bvals)

        # Compute Pearson r of each J-lens metric with this benchmark
        correlations = {}  # signed r
        for jcol in jlens_cols:
            jvals = cj[jcol]
            mask = ~(np.isnan(bvals) | np.isnan(jvals))
            if mask.sum() < 3:
                correlations[jcol] = 0.0
            else:
                r, _ = pearsonr(bvals[mask], jvals[mask])
                correlations[jcol] = r

        # Sort by |r| for legend ordering; display signed r
        sorted_jlens = sorted(jlens_cols, key=lambda c: abs(correlations[c]), reverse=True)
        best = sorted_jlens[0]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Benchmark scores (primary y)
        ax1.plot(steps[mask_b], bvals[mask_b] * 100,
                 color="black", linewidth=2.5, marker="o", markersize=5,
                 label=f"{bname} (%)", zorder=10)
        ax1.set_ylabel(f"{bname} Accuracy (%)", fontsize=13, color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.set_xscale("log")

        # J-lens metrics (secondary y, normalized)
        ax2 = ax1.twinx()
        for ji, jcol in enumerate(sorted_jlens):
            jvals = cj[jcol]
            mask_j = ~np.isnan(jvals)
            if not mask_j.any():
                continue
            j_norm = (jvals[mask_j] - jvals[mask_j].min()) / (jvals[mask_j].max() - jvals[mask_j].min() + 1e-10)
            is_best = (jcol == best)
            r_val = correlations[jcol]
            lbl = f"{jlens_label(jcol)}  (r={r_val:+.3f})"
            ax2.plot(steps[mask_j], j_norm,
                     color=colors_jlens[ji % len(colors_jlens)],
                     linewidth=3.0 if is_best else 1.2,
                     alpha=1.0 if is_best else 0.65,
                     label=lbl, zorder=9 if is_best else 1)

        ax2.set_ylabel("J-Lens Metric (normalized)", fontsize=11)
        ax2.set_ylim(-0.05, 1.05)

        ax1.set_title(f"Pythia-12B: {bname} vs J-Lens Metrics", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Training Step", fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Legend: benchmark first, then best metric bold, then pale ones
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # Reorder: best metric first among J-lens
        best_idx = sorted_jlens.index(best)
        lines2 = [lines2[best_idx]] + [l for i, l in enumerate(lines2) if i != best_idx]
        labels2 = [labels2[best_idx]] + [l for i, l in enumerate(labels2) if i != best_idx]
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
                   loc="upper left", ncol=2)

        fig.tight_layout()
        fig.savefig(out_dir / f"{bname}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {bname}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data/jlens/pythia-12b-deduped")
    parser.add_argument("--cross-join", default="eval_data/12b_cross_join.npz")
    parser.add_argument("--out-dir", default="figures/pythia-12b")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    val_loss = load_val_loss(results_dir)
    loss_steps = sorted(val_loss.keys())
    print(f"{len(loss_steps)} checkpoints with val loss")

    attrs, n_layers = load_spectra_attrs(results_dir)
    print(f"{len(attrs)} metrics, {n_layers} layers")

    # 1. Per-metric plots
    print("\n=== Per-Metric Plots ===")
    plot_per_metric(attrs, n_layers, val_loss, loss_steps,
                    out_dir / "metrics")

    # 2. Per-benchmark plots
    cj_path = Path(args.cross_join)
    if cj_path.exists():
        print("\n=== Per-Benchmark Plots ===")
        plot_per_benchmark(cj_path, out_dir / "benchmarks")
    else:
        print(f"\nNo cross-join at {cj_path}, skipping benchmarks")

    print(f"\nDone. Figures in {out_dir}/metrics/ and {out_dir}/benchmarks/")


if __name__ == "__main__":
    main()
