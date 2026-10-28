#!/usr/bin/env python3
"""Plot circle-probe results (jlens.circle_probe score output).

  figures/circle_probe_resid.png        circ_minus_lin vs layer, residual stream
  figures/circle_probe_<rep>.png        head x layer heatmaps of circ_minus_lin
                                        (loop > 0 blue, open path < 0 red)
  figures/circle_probe_top_pca.png      word centroids of the best units, in
                                        their top-2 PCs, joined in true order
Also prints the top units per concept and the significant-unit rate vs controls.

Usage:
  python scripts/plot_circle_probe.py --scores X.csv --probe X.h5 [--out-dir figures]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from jlens.circle_probe import (CONCEPTS, load_probe, pca_scores,
                                template_demean, unit_array, word_centroids)

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
DIV = LinearSegmentedColormap.from_list(
    "div", ["#a32c2b", "#e34948", "#f0a3a2", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"])
METRIC_LABEL = "circ − lin rank corr.\n(>0 loop, <0 open path)"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "figure.dpi": 150,
})

CONCEPT_ORDER = [c.name for c in CONCEPTS]
REP_LABEL = {"resid": "residual stream", "resid_norm": "normed residual (mixer input)",
             "head_out": "attention head output (pre o_proj, gated)",
             "value": "attention value head (v_proj)",
             "dn_head_out": "DeltaNet head output (pre out_proj)"}


def plot_resid(df, full_layers, out):
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, CONCEPT_ORDER):
        for l in full_layers:
            ax.axvline(l, color=GRID, lw=0.8, zorder=0)
        for rep, col in (("resid", BLUE), ("resid_norm", ORANGE)):
            s = df[(df.concept == name) & (df.rep == rep)].sort_values("layer")
            ax.plot(s.layer, s.circ_minus_lin, color=col, lw=2, label=REP_LABEL[rep])
            sig = s[(s.circ_minus_lin_p < 0.01) & (s.circ_minus_lin > 0)]
            ax.scatter(sig.layer, sig.circ_minus_lin, s=10, color=col, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ctrl = "" if next(c for c in CONCEPTS if c.name == name).cyclic else " (control)"
        ax.set_title(name + ctrl, color=INK, loc="left")
        ax.set_ylim(-1.02, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("layer")
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABEL)
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("Does the ordered concept close into a loop?  "
                 "(dots: >0 with p<0.01 label-permutation; grey lines: full-attention layers)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_heads(df, rep, alpha, out):
    s = df[df.rep == rep]
    if s.empty:
        return
    layers = sorted(s.layer.unique())
    heads = sorted(s["head"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, CONCEPT_ORDER):
        c = s[s.concept == name]
        grid = np.full((len(heads), len(layers)), np.nan)
        sig = np.zeros_like(grid, dtype=bool)
        li = {l: i for i, l in enumerate(layers)}
        for r in c.itertuples():
            grid[r.head, li[r.layer]] = r.circ_minus_lin
            sig[r.head, li[r.layer]] = (r.circ_minus_lin_p < alpha) & (r.circ_minus_lin > 0)
        im = ax.imshow(grid, cmap=DIV, vmin=-1, vmax=1, aspect="auto",
                       interpolation="nearest")
        yy, xx = np.where(sig)
        ax.scatter(xx, yy, s=3, color=INK, marker=".", linewidths=0)
        ax.grid(False)
        ax.set_title(f"{name}  ({sig.sum()}/{sig.size} loop, p<{alpha:g})",
                     color=INK, loc="left")
        step = max(1, len(layers) // 8)
        ax.set_xticks(range(0, len(layers), step))
        ax.set_xticklabels([layers[i] for i in range(0, len(layers), step)])
    for ax in axes[1]:
        ax.set_xlabel("layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("head")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label(METRIC_LABEL + "\ndot: loop with p<%.3g" % alpha)
    fig.suptitle(f"{REP_LABEL[rep]}: per-head circle fit", color=INK, x=0.01, ha="left")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_top_pca(df, d, out, k=2):
    cyc = [c for c in CONCEPTS if c.cyclic]
    fig, axes = plt.subplots(k, len(cyc), figsize=(3 * len(cyc), 3 * k))
    for j, c in enumerate(cyc):
        top = df[df.concept == c.name].sort_values("circ_minus_lin", ascending=False).head(k)
        for i, r in enumerate(top.itertuples()):
            ax = axes[i, j]
            m = d["concept"] == c.name
            X = unit_array(d, r.rep, r.layer, r.head)[m].astype(np.float64)
            Xd = template_demean(X, d["template_idx"][m])
            M = word_centroids(Xd, d["word_idx"][m], len(c.words))
            P = pca_scores(M, 2)
            loop = np.r_[np.arange(len(c.words)), 0]
            ax.plot(P[loop, 0], P[loop, 1], color=GRID, lw=1.5, zorder=1)
            ax.scatter(P[:, 0], P[:, 1], s=40, color=BLUE, zorder=2,
                       edgecolor="white", linewidth=1.5)
            for w, (x, y) in zip(c.words, P):
                ax.annotate(w[:3], (x, y), xytext=(3, 3), textcoords="offset points",
                            fontsize=7, color=INK)
            h = "" if r.head < 0 else f" h{r.head}"
            ax.set_title(f"{c.name}: {r.rep} L{r.layer}{h}\ncirc−lin={r.circ_minus_lin:.2f} "
                         f"closure={r.closure:.1f} loo={r.loo:.2f}", fontsize=8,
                         color=INK, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", "datalim")
    fig.suptitle("Best units per concept: word centroids in top-2 PCs, "
                 "joined in true cyclic order", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def summary(df, alpha):
    df = df.assign(
        ordinal=(df.loo_p < alpha) & (df.loo > 0.5),
        loop=(df.circ_minus_lin_p < alpha) & (df.circ_minus_lin > 0) & (df.closure < 2),
        loop_sup=(df.loo_circ_minus_lin_p < alpha) & (df.loo_circ_minus_lin > 0))
    for col, desc in (("ordinal", "ordinal/smooth: loo_p < a, loo > 0.5"),
                      ("loop", "LOOP: circ_minus_lin > 0, p < a, closure < 2"),
                      ("loop_sup", "LOOP (supervised): loo_circ_minus_lin > 0, p < a")):
        print(f"\n=== Unit rate, {desc} (a={alpha:g}) ===")
        print(df.pivot_table(index="concept", columns="rep", values=col, aggfunc="mean")
              .reindex(CONCEPT_ORDER).round(3).to_string())
    print("\n=== Median over units ===")
    print(df.groupby("concept")[["closure", "circ_minus_lin", "loo_circ_minus_lin", "loo"]].median()
          .reindex(CONCEPT_ORDER).round(3).to_string())
    print("\n=== Top 8 units per concept (by circ_minus_lin) ===")
    for name in CONCEPT_ORDER:
        t = df[df.concept == name].sort_values("circ_minus_lin", ascending=False).head(8)
        print(f"\n{name}")
        print(t[["rep", "layer", "head", "circ_minus_lin", "circ_minus_lin_p",
                 "closure", "loo_circ_minus_lin", "loo", "h1"]]
              .to_string(index=False, float_format="%.3f"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--alpha", type=float, default=0.01)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.scores)
    d = load_probe(a.probe)
    plot_resid(df, d["full_layers"], out / "circle_probe_resid.png")
    for rep in ("head_out", "value", "dn_head_out"):
        plot_heads(df, rep, a.alpha, out / f"circle_probe_{rep}.png")
    plot_top_pca(df, d, out / "circle_probe_top_pca.png")
    summary(df, a.alpha)
    print(f"\nfigures written to {out}/circle_probe_*.png")


if __name__ == "__main__":
    main()
