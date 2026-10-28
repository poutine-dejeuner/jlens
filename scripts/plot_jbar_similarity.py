#!/usr/bin/env python3
"""Plot similarity matrix between wikitext and random Jacobian matrices.

<J_l>_wiki and <J_m>_rand: compute cosine similarity (normalized Frobenius
inner product) for every pair of layers (l, m) at the final training step.

Uses matrix multiplication for efficiency: flattens all jbar matrices into
vectors, normalizes by Frobenius norm, then computes S = W @ R^T.

Output:
  figures/pythia-12b/jbar_similarity.png - greyscale matrix
  figures/pythia-12b/jbar_similarity_annotated.png - with numerical annotations
"""
import argparse, json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_jbars(results_dir: Path, step: int, n_layers: int):
    """Load jbar matrices, skipping layers without jbar. Returns (jbars, layer_indices)."""
    step_dir = results_dir / f"step{step:04d}"
    stats_path = step_dir / "stats.h5"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}")

    jbars = []
    layer_indices = []
    with h5py.File(stats_path, "r") as f:
        for i in range(n_layers):
            grp = f[f"layer_{i}"]
            if "jbar" in grp:
                # Load directly as float64 for inner products
                jbars.append(np.array(grp["jbar"], dtype=np.float64))
                layer_indices.append(i)
    return jbars, layer_indices


def compute_similarity_matrix_fast(jbars_a: list, jbars_b: list) -> np.ndarray:
    """Efficient matrix-multiply approach: flattens + normalizes + dot product.

    Returns LxM cosine similarity matrix.
    """
    L, M = len(jbars_a), len(jbars_b)
    d = jbars_a[0].size  # total elements per jbar

    # Flatten and normalize: each row is flattened jbar / ||jbar||_F
    A = np.zeros((L, d), dtype=np.float64)
    B = np.zeros((M, d), dtype=np.float64)

    for l in range(L):
        flat = jbars_a[l].ravel()
        norm = np.sqrt(flat @ flat)
        A[l] = flat / norm if norm > 0 else 0.0

    for m in range(M):
        flat = jbars_b[m].ravel()
        norm = np.sqrt(flat @ flat)
        B[m] = flat / norm if norm > 0 else 0.0

    # S = A @ B^T  => S[l,m] = cos_sim(A_l, B_m)
    S = A @ B.T
    return S


def plot_similarity_matrix(S, out_dir, step, wiki_layers, rand_layers):
    """Greyscale heatmap with optional annotated version."""
    L, M = S.shape

    # ---- Simple greyscale ----
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(S, cmap="gray", vmin=-1, vmax=1, aspect="equal",
                   interpolation="nearest")
    ax.set_xlabel("Random J-Lens Layer (m)", fontsize=12)
    ax.set_ylabel("Wikitext J-Lens Layer (l)", fontsize=12)
    ax.set_title(
        f"Cosine Similarity <J_l>_wiki vs <J_m>_rand\n"
        f"Pythia-12B step {step:,}",
        fontsize=14, fontweight="bold",
    )

    xtick_pos = list(range(0, M, 5))
    ytick_pos = list(range(0, L, 5))
    ax.set_xticks(xtick_pos)
    ax.set_yticks(ytick_pos)
    ax.set_xticklabels([str(rand_layers[i]) for i in xtick_pos])
    ax.set_yticklabels([str(wiki_layers[i]) for i in ytick_pos])

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=10)
    cbar.ax.yaxis.set_ticks_position("left")
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.xaxis.set_label_position("top")

    fig.tight_layout()
    out_path = out_dir / "jbar_similarity.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out_path}")

    # ---- Annotated version ----
    fig2, ax2 = plt.subplots(figsize=(12, 11))
    im2 = ax2.imshow(S, cmap="gray", vmin=-1, vmax=1, aspect="equal",
                     interpolation="nearest")
    for l in range(L):
        for m in range(M):
            val = S[l, m]
            tc = "white" if abs(val) > 0.7 else "black"
            ax2.text(m, l, f"{val:.2f}", ha="center", va="center",
                     fontsize=5.5, color=tc)

    ax2.set_xlabel("Random J-Lens Layer (m)", fontsize=12)
    ax2.set_ylabel("Wikitext J-Lens Layer (l)", fontsize=12)
    ax2.set_title(
        f"Cosine Similarity <J_l>_wiki vs <J_m>_rand (annotated)\n"
        f"Pythia-12B step {step:,}",
        fontsize=14, fontweight="bold",
    )
    ax2.set_xticks(xtick_pos)
    ax2.set_yticks(ytick_pos)
    ax2.set_xticklabels([str(rand_layers[i]) for i in xtick_pos])
    ax2.set_yticklabels([str(wiki_layers[i]) for i in ytick_pos])

    cbar2 = fig2.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cbar2.set_label("Cosine Similarity", fontsize=10)
    cbar2.ax.yaxis.set_ticks_position("left")
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.xaxis.set_label_position("top")

    fig2.tight_layout()
    out_path2 = out_dir / "jbar_similarity_annotated.png"
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  {out_path2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-dir", default="/data/jlens/pythia-12b-deduped")
    parser.add_argument("--rand-dir", default="/data/jlens/pythia-12b-deduped-random")
    parser.add_argument("--step", type=int, default=143000)
    parser.add_argument("--out-dir", default="figures/pythia-12b")
    args = parser.parse_args()

    wiki_dir = Path(args.wiki_dir)
    rand_dir = Path(args.rand_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = args.step

    # Check both checkpoints exist
    wiki_meta = wiki_dir / f"step{step:04d}" / "metadata.json"
    rand_meta = rand_dir / f"step{step:04d}" / "metadata.json"
    if not wiki_meta.exists():
        print(f"Error: wikitext step {step} not found at {wiki_dir}")
        return
    if not rand_meta.exists():
        print(f"Error: random step {step} not found at {rand_dir}")
        return

    with open(wiki_meta) as f:
        wiki_md = json.load(f)
    with open(rand_meta) as f:
        rand_md = json.load(f)

    n_layers = wiki_md.get("n_layers", 36)
    d_model = wiki_md.get("d_model", 5120)
    print(f"Wikitext: {n_layers} layers, d={d_model}")
    print(f"Random:   {rand_md.get('n_layers', 36)} layers, d={rand_md.get('d_model', 5120)}")

    print(f"\nLoading jbar matrices for step {step:,}...")
    wiki_jbars, wiki_layers = load_jbars(wiki_dir, step, n_layers)
    rand_jbars, rand_layers = load_jbars(rand_dir, step, n_layers)
    print(f"  wiki: {len(wiki_jbars)} jbars (layers {wiki_layers[0]}-{wiki_layers[-1]})")
    print(f"  rand: {len(rand_jbars)} jbars (layers {rand_layers[0]}-{rand_layers[-1]})")

    n_w = len(wiki_jbars)
    n_r = len(rand_jbars)

    # Self-similarity stats
    print("\nSelf-similarity within wikitext...")
    wiki_self = compute_similarity_matrix_fast(wiki_jbars, wiki_jbars)
    dw = np.eye(n_w, dtype=bool)
    print(f"  diagonal:   {np.diag(wiki_self).min():.4f} - {np.diag(wiki_self).max():.4f}")
    print(f"  off-diag:   {wiki_self[~dw].min():.4f} - {wiki_self[~dw].max():.4f}")

    print("Self-similarity within random...")
    rand_self = compute_similarity_matrix_fast(rand_jbars, rand_jbars)
    dr = np.eye(n_r, dtype=bool)
    print(f"  diagonal:   {np.diag(rand_self).min():.4f} - {np.diag(rand_self).max():.4f}")
    print(f"  off-diag:   {rand_self[~dr].min():.4f} - {rand_self[~dr].max():.4f}")

    # Cross-similarity
    print(f"\nCross-similarity: <J_l>_wiki vs <J_m>_rand...")
    S = compute_similarity_matrix_fast(wiki_jbars, rand_jbars)
    print(f"  range: [{S.min():.4f}, {S.max():.4f}], mean={S.mean():.4f}, std={S.std():.4f}")

    # Also cross-similarity within the other direction for symmetry check
    print("\nCross-similarity: <J_l>_rand vs <J_m>_wiki...")
    S2 = compute_similarity_matrix_fast(rand_jbars, wiki_jbars)
    print(f"  match: {np.allclose(S, S2.T)} (should be True)")

    print("\nPlotting...")
    plot_similarity_matrix(S, out_dir, step, wiki_layers, rand_layers)
    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
