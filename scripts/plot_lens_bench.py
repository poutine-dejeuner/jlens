#!/usr/bin/env python3
"""Plot lens benchmark pass@1 over training steps with J-Lens metric overlay.

Output: figures/pythia-12b/lens_bench/{benchmark}_vs_metrics.png
  One figure per benchmark: pass@1 (%) on left y-axis, normalized JL metrics
  on right y-axis. Best Pearson r highlighted (bold, red), others pale.
  Also generates a correlation matrix and per-benchmark scatter (via cross_join_lens).
"""
import json, argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

BENCH_NAMES_DISPLAY = {
    "multihop": "Multi-Hop Reasoning",
    "multilingual": "Multilingual Transfer",
    "order-ops": "Order of Operations",
    "association": "Concept Association",
    "typo": "Typo Robustness",
    "poetry": "Poetry Generation",
}

METRIC_LABELS = {
    "coherence": "Coherence ρ",
    "effective_rank": "Effective Rank",
    "power_law_alpha": "Power-Law α",
    "n_spikes": "N Spikes",
    "fcr": "Fisher Condition Ratio",
    "r_norm": "‖R‖",
    "a_coeff": "A Coefficient",
    "tr_jtj_mean": "Tr(JᵀJ)/d",
    "q_eff": "Q Effective",
    "mp_sigma2": "MP σ²",
    "sigma_eff_rank": "Σ Eff. Rank",
}


def load_lens_bench(results_dir: Path) -> dict:
    """{bench_name: (steps, pass_at_1, median_rank)} sorted by step."""
    out_dir = results_dir / "lens_bench"
    files = sorted(
        out_dir.glob("step*.json"),
        key=lambda p: int(p.stem[4:]),
    )
    data = {}
    for fp in files:
        d = json.loads(fp.read_text())
        step = d["step"]
        for bname, bdata in d["benchmarks"].items():
            if bname not in data:
                data[bname] = {"steps": [], "pass_at_1": [], "median_rank": []}
            data[bname]["steps"].append(step)
            data[bname]["pass_at_1"].append(bdata["pass_at"]["1"])
            data[bname]["median_rank"].append(bdata["median_rank"])
    for bname in data:
        idx = np.argsort(data[bname]["steps"])
        for key in ("steps", "pass_at_1", "median_rank"):
            data[bname][key] = np.array(data[bname][key])[idx]
    return data


def load_spectral_attrs(results_dir: Path) -> dict:
    """{metric_name: (steps, layer_means)} skipping eigenvector_overlap."""
    dlist = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("step")],
        key=lambda d: int(d.name.replace("step", "")),
    )
    metrics = {}
    n_layers = 0
    for d in dlist:
        step = int(d.name.replace("step", ""))
        sf = d / "spectra.h5"
        if not sf.exists():
            continue
        with h5py.File(sf, "r") as f:
            if n_layers == 0:
                n_layers = f.attrs.get(
                    "n_layers", sum(1 for k in f.keys() if k.startswith("layer_"))
                )
            for li in range(n_layers):
                key = f"layer_{li}"
                if key not in f:
                    continue
                lg = f[key]
                for attr in lg.attrs:
                    if attr == "eigenvector_overlap":
                        continue
                    if attr not in metrics:
                        metrics[attr] = {}
                    if step not in metrics[attr]:
                        metrics[attr][step] = np.full(n_layers, np.nan)
                    metrics[attr][step][li] = float(lg.attrs[attr])

    result = {}
    for mname, mdata in metrics.items():
        steps = sorted(mdata.keys())
        means = np.array([np.nanmean(mdata[s]) for s in steps])
        result[mname] = (np.array(steps), np.array(means))
    return result


def interpolate_to_steps(target_steps, src_steps, src_vals):
    """Linearly interpolate src_vals to target_steps grid."""
    mask = ~np.isnan(src_vals)
    if mask.sum() < 2:
        return np.full_like(target_steps, np.nan, dtype=float)
    return np.interp(target_steps, src_steps[mask], src_vals[mask],
                     left=np.nan, right=np.nan)


def plot_lens_vs_metrics(lens_data, spectral, out_dir):
    """One figure per lens benchmark: pass@1 + best JL metrics overlay."""
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_names = sorted(spectral.keys())
    colors_jlens = plt.cm.tab10(np.linspace(0, 1, len(spec_names)))

    for bname in sorted(lens_data.keys()):
        bdata = lens_data[bname]
        steps = bdata["steps"]
        bvals = bdata["pass_at_1"] * 100  # percentage
        mask_b = ~np.isnan(bvals)

        # Compute Pearson r for each JL metric vs this benchmark
        correlations = {}
        for sn in spec_names:
            src_steps, src_vals = spectral[sn]
            interp = interpolate_to_steps(steps, src_steps, src_vals)
            mask = ~(np.isnan(bvals) | np.isnan(interp))
            if mask.sum() < 3:
                correlations[sn] = 0.0
            else:
                r, _ = pearsonr(bvals[mask], interp[mask])
                correlations[sn] = r

        # Sort by |r|
        sorted_spec = sorted(spec_names, key=lambda s: abs(correlations[s]), reverse=True)
        best = sorted_spec[0]

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Benchmark score (primary y, black, bold)
        ax1.plot(steps[mask_b], bvals[mask_b],
                 color="black", linewidth=2.5, marker="o", markersize=5,
                 label=f"pass@1 (%)", zorder=10)
        ax1.set_ylabel(f"{BENCH_NAMES_DISPLAY.get(bname, bname)} pass@1 (%)",
                       fontsize=13, color="black")
        ax1.tick_params(axis="y", labelcolor="black")
        ax1.set_xscale("log")

        # JL metrics (secondary y, normalized)
        ax2 = ax1.twinx()
        for ji, sn in enumerate(sorted_spec):
            src_steps, src_vals = spectral[sn]
            interp = interpolate_to_steps(steps, src_steps, src_vals)
            mask_j = ~np.isnan(interp)
            if not mask_j.any():
                continue
            j_norm = ((interp[mask_j] - interp[mask_j].min())
                      / (interp[mask_j].max() - interp[mask_j].min() + 1e-10))
            is_best = (sn == best)
            r_val = correlations[sn]
            lbl = f"{METRIC_LABELS.get(sn, sn)}  (r={r_val:+.3f})"
            ax2.plot(steps[mask_j], j_norm,
                     color=colors_jlens[ji % len(colors_jlens)],
                     linewidth=3.0 if is_best else 1.2,
                     alpha=1.0 if is_best else 0.65,
                     label=lbl, zorder=9 if is_best else 1)

        ax2.set_ylabel("J-Lens Metric (normalized)", fontsize=11)
        ax2.set_ylim(-0.05, 1.05)

        ax1.set_title(
            f"Pythia-12B: {BENCH_NAMES_DISPLAY.get(bname, bname)} vs J-Lens Metrics",
            fontsize=14, fontweight="bold",
        )
        ax1.set_xlabel("Training Step", fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Legend: benchmark first, then best metric bold, then pale
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        best_idx = sorted_spec.index(best)
        lines2 = [lines2[best_idx]] + [l for i, l in enumerate(lines2) if i != best_idx]
        labels2 = [labels2[best_idx]] + [l for i, l in enumerate(labels2) if i != best_idx]
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7,
                   loc="upper left", ncol=2)

        fig.tight_layout()
        fname = bname.replace("/", "_")
        fig.savefig(out_dir / f"{fname}_vs_metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname}_vs_metrics.png  (best: {METRIC_LABELS.get(best, best)}, r={correlations[best]:+.3f})")


def plot_summary_grid(lens_data, spectral, out_path):
    """2×3 summary: pass@1 + best metric + loss overlay for each benchmark."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    bench_order = ["multihop", "multilingual", "order-ops", "association", "typo", "poetry"]

    for ax, bname in zip(axes.flat, bench_order):
        bdata = lens_data[bname]
        steps = bdata["steps"]
        bvals = bdata["pass_at_1"] * 100

        # Find best metric
        best_sn = None
        best_r = 0
        for sn in sorted(spectral.keys()):
            src_steps, src_vals = spectral[sn]
            interp = interpolate_to_steps(steps, src_steps, src_vals)
            mask = ~(np.isnan(bvals) | np.isnan(interp))
            if mask.sum() < 3:
                continue
            r, _ = pearsonr(bvals[mask], interp[mask])
            if abs(r) > abs(best_r):
                best_r = r
                best_sn = sn

        ax.plot(steps, bvals, color="black", linewidth=2, marker="o",
                markersize=4, alpha=0.9)
        ax.set_title(
            f"{BENCH_NAMES_DISPLAY.get(bname, bname)}\n"
            f"vs {METRIC_LABELS.get(best_sn, best_sn)} (r={best_r:+.2f})",
            fontsize=11, fontweight="bold",
        )
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

        # Best metric overlay on twin axis
        if best_sn is not None:
            src_steps, src_vals = spectral[best_sn]
            interp = interpolate_to_steps(steps, src_steps, src_vals)
            mask = ~np.isnan(interp)
            if mask.any():
                j_norm = ((interp[mask] - interp[mask].min())
                          / (interp[mask].max() - interp[mask].min() + 1e-10))
                ax2 = ax.twinx()
                ax2.plot(steps[mask], j_norm, color="red", linewidth=1.2,
                         linestyle="--", alpha=0.7)
                ax2.set_ylim(-0.05, 1.05)
                ax2.set_yticks([])

    fig.suptitle(
        "Pythia-12B: Lens Benchmark pass@1 × Best J-Lens Metric",
        fontsize=16, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="/data/jlens/pythia-12b-deduped")
    parser.add_argument("--out-dir", default="figures/pythia-12b")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) / "lens_bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lens benchmark data...")
    lens_data = load_lens_bench(results_dir)
    print(f"  {len(lens_data)} benchmarks")

    print("Loading spectral metrics...")
    spectral = load_spectral_attrs(results_dir)
    print(f"  {len(spectral)} metrics")

    print("\n=== Lens Benchmark vs JL Metrics ===")
    plot_lens_vs_metrics(lens_data, spectral, out_dir)

    print("\n=== Summary Grid ===")
    plot_summary_grid(lens_data, spectral, out_dir / "summary_vs_metrics.png")

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == "__main__":
    main()
