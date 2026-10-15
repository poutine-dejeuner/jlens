#!/usr/bin/env python3
"""Cross-join lens benchmark results with J-lens spectral metrics and compute correlations.

Output:
  figures/pythia-12b/lens_bench/lens_bench_corr.png  — correlation matrix
  figures/pythia-12b/lens_bench/aggr_{bench}.png     — per-benchmark vs best metric
"""
import json, argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


BENCH_NAMES_DISPLAY = {
    "multihop": "Multi-Hop",
    "multilingual": "Multilingual",
    "order-ops": "Order-Ops",
    "association": "Association",
    "typo": "Typo",
    "poetry": "Poetry",
}

METRIC_LABELS = {
    "coherence": "Coherence ρ",
    "effective_rank": "Effective Rank",
    "power_law_alpha": "Power-Law α",
    "n_spikes": "N Spikes",
    "fcr": "Fisher Condition Ratio",
    "r_norm": "||R||",
    "a_coeff": "A Coefficient",
    "tr_jtj_mean": "Tr(JᵀJ)/d",
    "q_eff": "Q Effective",
}


def load_lens_bench(results_dir: Path) -> dict:
    """{bench_name: ([steps], [pass_at_1], [median_rank])}"""
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
    """{metric_name: {step: [per-layer values]}} — mean over layers."""
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
                n_layers = f.attrs.get(
                    "n_layers", sum(1 for k in f.keys() if k.startswith("layer_"))
                )
            for layer_idx in range(n_layers):
                key = f"layer_{layer_idx}"
                if key not in f:
                    continue
                lg = f[key]
                for attr_name in lg.attrs:
                    if attr_name == "eigenvector_overlap":
                        continue  # flat metric, no signal
                    if attr_name not in metrics:
                        metrics[attr_name] = {}
                    if step not in metrics[attr_name]:
                        metrics[attr_name][step] = np.full(n_layers, np.nan)
                    metrics[attr_name][step][layer_idx] = float(lg.attrs[attr_name])

    # Mean over layers for each step
    result = {}
    for mname, mdata in metrics.items():
        steps = sorted(mdata.keys())
        means = []
        for s in steps:
            vals = mdata[s][~np.isnan(mdata[s])]
            means.append(vals.mean() if len(vals) > 0 else np.nan)
        result[mname] = (np.array(steps), np.array(means))
    return result


def cross_join(lens_data, spectral, eval_points_only=True):
    """
    If eval_points_only: only use steps where we have lens benchmark data.
    Returns dict with aligned arrays.
    """
    # Get common steps
    bench_steps = set()
    for bdata in lens_data.values():
        bench_steps |= set(bdata["steps"].astype(int))

    if eval_points_only:
        common_steps = bench_steps
    else:
        # Use all spectral steps that overlap
        spec_steps = set()
        for _, (steps, _) in spectral.items():
            spec_steps |= set(steps.astype(int))
        common_steps = bench_steps & spec_steps

    common_steps = sorted(common_steps)
    step_to_idx = {s: i for i, s in enumerate(common_steps)}

    # Build arrays: [n_steps, n_features]
    bench_features = {}
    for bname, bdata in lens_data.items():
        b_steps = bdata["steps"].astype(int)
        arr = np.full(len(common_steps), np.nan)
        for i, s in enumerate(b_steps):
            if s in step_to_idx:
                arr[step_to_idx[s]] = bdata["pass_at_1"][i]
        bench_features[bname] = arr

    spec_features = {}
    for mname, (steps, means) in spectral.items():
        arr = np.full(len(common_steps), np.nan)
        for i, s in enumerate(steps.astype(int)):
            if s in step_to_idx:
                arr[step_to_idx[s]] = means[i]
        spec_features[mname] = arr

    return common_steps, bench_features, spec_features


def plot_correlation_matrix(bench_features, spec_features, out_path):
    """Pearson r matrix: rows=benchmarks, cols=JL metrics.  Annotate signed r; color by |r|."""
    bench_names = sorted(bench_features.keys())
    spec_names = sorted(spec_features.keys())

    r_matrix = np.zeros((len(bench_names), len(spec_names)))
    absr_matrix = np.zeros((len(bench_names), len(spec_names)))

    for i, bn in enumerate(bench_names):
        bvals = bench_features[bn]
        for j, sn in enumerate(spec_names):
            svals = spec_features[sn]
            mask = ~(np.isnan(bvals) | np.isnan(svals))
            if mask.sum() < 3:
                r_matrix[i, j] = 0
                absr_matrix[i, j] = 0
            else:
                r, _ = pearsonr(bvals[mask], svals[mask])
                r_matrix[i, j] = r
                absr_matrix[i, j] = abs(r)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(absr_matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")

    # Annotate with signed r
    for i in range(len(bench_names)):
        for j in range(len(spec_names)):
            r_val = r_matrix[i, j]
            abs_val = absr_matrix[i, j]
            if abs_val > 0.8:
                color = "white"
            else:
                color = "black"
            ax.text(j, i, f"{r_val:+.3f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if abs_val > 0.8 else "normal")

    ax.set_xticks(range(len(spec_names)))
    ax.set_xticklabels(
        [METRIC_LABELS.get(s, s) for s in spec_names],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_yticks(range(len(bench_names)))
    ax.set_yticklabels(
        [BENCH_NAMES_DISPLAY.get(b, b) for b in bench_names],
        fontsize=10,
    )
    ax.set_title(
        "Pythia-12B: Lens Benchmark pass@1 × J-Lens Metrics\n"
        "Pearson |r| Correlation",
        fontsize=14, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Pearson |r|", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path.name}")


def plot_per_benchmark_best(common_steps, bench_features, spec_features, out_dir):
    """Scatter + trend: pass@1 vs best-correlating JL metric for each benchmark."""
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_names = sorted(bench_features.keys())
    spec_names = sorted(spec_features.keys())

    for bn in bench_names:
        bvals = bench_features[bn]
        mask_b = ~np.isnan(bvals)

        # Find best metric (by |r|)
        best_absr = 0
        best_r = 0
        best_sn = spec_names[0]
        for sn in spec_names:
            svals = spec_features[sn]
            mask = ~(np.isnan(bvals) | np.isnan(svals))
            if mask.sum() < 3:
                continue
            r, _ = pearsonr(bvals[mask], svals[mask])
            if abs(r) > best_absr:
                best_absr = abs(r)
                best_r = r
                best_sn = sn

        best_vals = spec_features[best_sn]
        mask = ~(np.isnan(bvals) | np.isnan(best_vals))

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            best_vals[mask], bvals[mask] * 100,
            c=np.array(common_steps)[mask], cmap="plasma",
            s=50, alpha=0.8, edgecolors="black", linewidth=0.5,
        )
        ax.set_xlabel(METRIC_LABELS.get(best_sn, best_sn), fontsize=12)
        ax.set_ylabel(
            f"{BENCH_NAMES_DISPLAY.get(bn, bn)} pass@1 (%)", fontsize=12,
        )
        ax.set_title(
            f"Pythia-12B: {BENCH_NAMES_DISPLAY.get(bn, bn)} vs {METRIC_LABELS.get(best_sn, best_sn)}\n"
            f"Pearson r = {best_r:+.4f}",
            fontsize=14, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Training Step", fontsize=9)

        fig.tight_layout()
        fig.savefig(out_dir / f"aggr_{bn}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  aggr_{bn}.png  (best: {best_sn}, r={best_r:+.4f})")


def print_correlations(bench_features, spec_features):
    """Print top correlations for each benchmark."""
    bench_names = sorted(bench_features.keys())
    spec_names = sorted(spec_features.keys())

    print("\nTop 3 J-Lens correlations per benchmark (Pearson r):")
    for bn in bench_names:
        bvals = bench_features[bn]
        results = []
        for sn in spec_names:
            svals = spec_features[sn]
            mask = ~(np.isnan(bvals) | np.isnan(svals))
            if mask.sum() < 3:
                continue
            r, p = pearsonr(bvals[mask], svals[mask])
            results.append((abs(r), r, sn, p))
        results.sort(reverse=True)
        print(f"\n  {BENCH_NAMES_DISPLAY.get(bn, bn)}:")
        for ri, (absr, r, sn, p) in enumerate(results[:3]):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    {r:+.4f}  {METRIC_LABELS.get(sn, sn):20s}  p={p:.4f} {sig}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", default="/data/jlens/pythia-12b-deduped",
    )
    parser.add_argument(
        "--out-dir", default="figures/pythia-12b/lens_bench",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lens benchmark data...")
    lens_data = load_lens_bench(results_dir)
    print(f"  {len(lens_data)} benchmarks")

    print("Loading spectral metrics...")
    spectral = load_spectral_attrs(results_dir)
    print(f"  {len(spectral)} metrics")

    print("Cross-joining...")
    common_steps, bench_features, spec_features = cross_join(lens_data, spectral)
    print(f"  {len(common_steps)} common checkpoints")

    # Print detailed correlations
    print_correlations(bench_features, spec_features)

    # Correlation matrix
    print("\n=== Correlation Matrix ===")
    plot_correlation_matrix(bench_features, spec_features, out_dir / "lens_bench_corr.png")

    # Per-benchmark best metric scatter
    print("\n=== Per-Benchmark Scatter Plots ===")
    plot_per_benchmark_best(common_steps, bench_features, spec_features, out_dir)

    print(f"\nDone. Figures in {out_dir}/")


if __name__ == "__main__":
    main()
