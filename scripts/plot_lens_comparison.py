#!/usr/bin/env python3
"""Lens benchmark comparison: wikitext vs random-token JLens pass@1 curves.

Output: per-benchmark 2-panel figure (top: overlay with fill, bottom: Δ ribbon)
        plus a summary grid.
"""
import argparse, json
from pathlib import Path

import h5py, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

BENCH_NAMES = {
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

# High-contrast colorblind-safe: blue vs orange
COLOR_WT = "#2166ac"   # deep blue
COLOR_RD = "#d95f02"   # burnt orange
COLOR_WT_LIGHT = "#67a9cf"
COLOR_RD_LIGHT = "#fc8d59"


def load_lens_bench(results_dir: Path) -> dict:
    lb_dir = results_dir / "lens_bench"
    files = sorted(lb_dir.glob("step*.json"), key=lambda p: int(p.stem[4:]))
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


def interpolate_to_steps(target_steps, src_steps, src_vals):
    mask = ~np.isnan(src_vals)
    if mask.sum() < 2:
        return np.full_like(target_steps, np.nan, dtype=float)
    return np.interp(target_steps, src_steps[mask], src_vals[mask],
                     left=np.nan, right=np.nan)


def plot_bench_overlay(lens_wt, lens_rd, bname, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    title = BENCH_NAMES.get(bname, bname)

    bdata_wt = lens_wt[bname]
    bdata_rd = lens_rd[bname]

    steps_common = np.union1d(bdata_wt["steps"], bdata_rd["steps"])

    val_wt = interpolate_to_steps(steps_common, bdata_wt["steps"],
                                  bdata_wt["pass_at_1"]) * 100
    val_rd = interpolate_to_steps(steps_common, bdata_rd["steps"],
                                  bdata_rd["pass_at_1"]) * 100
    delta = val_wt - val_rd

    mask = ~(np.isnan(val_wt) | np.isnan(val_rd))
    steps = steps_common[mask]
    vw = val_wt[mask]
    vr = val_rd[mask]
    d = delta[mask]

    # 2-panel layout: top = overlay, bottom = difference
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 8),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.08},
        sharex=True,
    )

    # ── Top panel: overlay with semi-transparent fill ──
    # Draw random first (behind), wikitext on top
    ax_top.plot(steps / 1000, vr, color=COLOR_RD, linewidth=2.0,
                marker="s", markersize=4, alpha=0.85,
                label="random JLens", zorder=3)
    ax_top.plot(steps / 1000, vw, color=COLOR_WT, linewidth=2.5,
                marker="o", markersize=5, alpha=0.90,
                label="wikitext JLens", zorder=5)

    # Fill between curves to highlight the gap
    # Use color of whichever is higher at each point
    fill_color = np.where(vw >= vr, COLOR_WT_LIGHT, COLOR_RD_LIGHT)
    # Can't do per-segment fill easily — use a neutral gray fill
    ax_top.fill_between(steps / 1000, vw, vr,
                        color="#888888", alpha=0.18, linewidth=0, zorder=1)

    ax_top.set_ylabel("pass@1 (%)", fontsize=13)
    ax_top.set_title(f"{title} — Wikitext vs Random JLens", fontsize=15,
                     fontweight="bold")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(fontsize=11, loc="lower right")

    # ── Bottom panel: Δ ribbon ──
    # Fill above/below zero with different colors
    ax_bot.fill_between(steps / 1000, 0, d,
                        where=(d >= 0), color=COLOR_WT_LIGHT,
                        alpha=0.4, linewidth=0)
    ax_bot.fill_between(steps / 1000, 0, d,
                        where=(d < 0), color=COLOR_RD_LIGHT,
                        alpha=0.4, linewidth=0)
    ax_bot.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax_bot.plot(steps / 1000, d, color="#333333", linewidth=1.2,
                marker=".", markersize=2, alpha=0.7)
    ax_bot.set_ylabel("Δ (pp)  WT − RD", fontsize=12)
    ax_bot.set_xlabel("Training Step (k)", fontsize=12)
    ax_bot.grid(True, alpha=0.25)

    # Add text label in delta panel
    delta_max = d.max()
    delta_min = d.min()
    delta_last = d[-1] if len(d) > 0 else 0
    leader = "wikitext" if delta_last > 0 else "random"
    ax_bot.text(0.98, 0.95,
                f"Final: {leader} by {abs(delta_last):.1f} pp",
                transform=ax_bot.transAxes, fontsize=10,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.85))

    # Symmetric y for bottom
    max_abs = max(abs(delta_max), abs(delta_min)) * 1.2
    ax_bot.set_ylim(-max_abs, max_abs)

    fig.tight_layout()
    fname = f"lens_{bname}"
    fig.savefig(out_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}.png  final: {leader} Δ={delta_last:+.1f}pp  "
          f"range=[{delta_min:+.1f}, {delta_max:+.1f}]")


def plot_summary_grid(lens_wt, lens_rd, out_dir):
    """2×3 grid with fill-between on each subplot."""
    bench_order = ["multihop", "multilingual", "order-ops",
                   "association", "typo", "poetry"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for ax, bname in zip(axes.flat, bench_order):
        bdata_wt = lens_wt.get(bname)
        bdata_rd = lens_rd.get(bname)
        if bdata_wt is None or bdata_rd is None:
            continue

        steps = bdata_wt["steps"]
        vw = np.array(bdata_wt["pass_at_1"]) * 100
        vr = interpolate_to_steps(steps, bdata_rd["steps"],
                                  np.array(bdata_rd["pass_at_1"]) * 100)

        ax.plot(steps / 1000, vr, color=COLOR_RD, linewidth=1.6,
                marker="s", markersize=2.5, alpha=0.75, zorder=2,
                label="random")
        ax.plot(steps / 1000, vw, color=COLOR_WT, linewidth=2.0,
                marker="o", markersize=3, alpha=0.85, zorder=3,
                label="wikitext")
        ax.fill_between(steps / 1000, vw, vr,
                        color="#888888", alpha=0.14, linewidth=0, zorder=1)

        ax.set_title(BENCH_NAMES.get(bname, bname), fontsize=12, fontweight="bold")
        ax.set_ylabel("pass@1 (%)", fontsize=10)
        ax.grid(True, alpha=0.25)

        # Label final delta
        if len(vw) > 0 and not np.isnan(vr[-1]):
            d_last = vw[-1] - vr[-1]
            leader = "WT" if d_last > 0 else "RD"
            ax.text(0.97, 0.92, f"{leader} {abs(d_last):.1f}pp",
                    transform=ax.transAxes, fontsize=8, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="gray", alpha=0.8))

        if bname == bench_order[0]:
            legend = ax.legend(fontsize=9, loc="upper left")
            if legend:
                legend.set_title("JLens", prop={"size": 8})

    fig.suptitle("Pythia-12B: Lens Benchmarks — Wikitext vs Random JLens",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "lens_summary_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  lens_summary_grid.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wt-dir", default="/data/jlens/pythia-12b-deduped")
    parser.add_argument("--rd-dir", default="/data/jlens/pythia-12b-deduped-random")
    parser.add_argument("--out-dir",
                        default="figures/pythia-12b-random-full/comparison/lens-bench")
    args = parser.parse_args()

    wt_dir = Path(args.wt_dir)
    rd_dir = Path(args.rd_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading lens bench (wikitext)...")
    lens_wt = load_lens_bench(wt_dir)
    print(f"  {sum(len(v['steps']) for v in lens_wt.values())} data points")
    print("Loading lens bench (random)...")
    lens_rd = load_lens_bench(rd_dir)
    print(f"  {sum(len(v['steps']) for v in lens_rd.values())} data points")

    benchmarks = sorted(set(lens_wt.keys()) & set(lens_rd.keys()))
    print(f"\nGenerating {len(benchmarks)} benchmark comparison plots...")
    for bname in benchmarks:
        plot_bench_overlay(lens_wt, lens_rd, bname, out_dir)

    print("\nSummary grid...")
    plot_summary_grid(lens_wt, lens_rd, out_dir)

    print(f"\nDone. → {out_dir}/")


if __name__ == "__main__":
    main()
