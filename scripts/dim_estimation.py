#!/usr/bin/env python3
"""Apply TDA dimension estimation methods to token embedding point clouds.

Methods:
  1. PCA effective rank (participation ratio):
     d_eff = (Σ λᵢ)² / Σ λᵢ²
  2. Two-NN: ratio of 2nd to 1st nearest-neighbor distances
  3. MLE / Levina-Bickel: local Poisson process estimator
  4. Persistent homology: max dimension with significant Betti numbers

Datasets:
  - singleton_embeddings.npy  (50688 tokens × 5120) — embedding layer weight matrix
  - context_layer{N}.npy      (6041 tokens × 5120)    — hidden states per layer
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import pareto

# DIY Two-NN and MLE — no skdim dependency (not installed on cluster)
# But we DO have scipy, numpy


def effective_rank_pca(eigvals: np.ndarray) -> float:
    """Participation ratio: (Σ λ)² / Σ λ²."""
    total = eigvals.sum()
    if total == 0:
        return 0.0
    return float(total**2 / (eigvals**2).sum())


def two_nn_dim(data: np.ndarray, n_samples: int = 5000) -> dict:
    """Two-NN intrinsic dimension estimator.

    Facco et al. (2017). The ratio μ = r₂/r₁ follows a Pareto distribution
    with parameter d+1 where d is the intrinsic dimension.

    We subsample to n_samples for efficiency on large point clouds.
    """
    n = min(n_samples, len(data))
    if n < len(data):
        idx = np.random.default_rng(42).choice(len(data), n, replace=False)
        pts = data[idx]
    else:
        pts = data

    tree = KDTree(pts)
    # k=3 nearest neighbors (neighbor 0 is self)
    dists, _ = tree.query(pts, k=3)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mu = r2 / r1
    mu = mu[np.isfinite(mu) & (mu > 0)]

    # ML estimate: d = n / Σ log(μᵢ) — but this is biased for small n.
    # Use the formula from Facco et al.:  log(μ) ~ Pareto(d+1)
    # d = n / Σ log(μ) — but that gives d+1, so:
    d_est = float(len(mu) / np.sum(np.log(mu)) - 1)
    if d_est <= 0:
        d_est = float(len(mu) / np.sum(np.log(mu)))  # fallback, drop the -1

    return {
        "method": "Two-NN",
        "d": d_est,
        "n_points": len(mu),
        "mu_median": float(np.median(mu)),
        "mu_mean": float(np.mean(mu)),
    }


def levina_bickel_dim(data: np.ndarray, k: int = 10, n_samples: int = 5000) -> dict:
    """Levina-Bickel MLE intrinsic dimension estimator.

    For each point i, given its distance Tₖ to the k-th nearest neighbor:
      d̂_i = (1/(k-2) Σ_{j=1}^{k-1} log(Tₖ/Tⱼ))⁻¹

    The global estimate is the harmonic mean of local estimates.
    """
    n = min(n_samples, len(data))
    if n < len(data):
        idx = np.random.default_rng(42).choice(len(data), n, replace=False)
        pts = data[idx]
    else:
        pts = data

    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k + 1)  # +1 for self

    local_dims = np.zeros(n)
    for i in range(n):
        Tk = dists[i, k]
        Tj = dists[i, 1:k]  # skip self
        mask = Tj > 0
        if mask.sum() < 2:
            local_dims[i] = np.nan
            continue
        log_ratios = np.log(Tk / Tj[mask])
        local_dims[i] = (k - 2) / np.sum(log_ratios) if np.sum(log_ratios) > 0 else np.nan

    local_dims = local_dims[np.isfinite(local_dims) & (local_dims > 0)]

    return {
        "method": f"Levina-Bickel (k={k})",
        "d": float(np.mean(local_dims)),
        "d_median": float(np.median(local_dims)),
        "d_std": float(np.std(local_dims)),
        "n_points": len(local_dims),
    }


def compute_pointcloud_dimensions(dataset_name: str, data: np.ndarray, n_pca: int = 1000) -> dict:
    """Run all dimension estimators on a point cloud."""
    n, d_ambient = data.shape
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  (n={n}, d={d_ambient})")

    results = {"dataset": dataset_name, "n_points": n, "d_ambient": d_ambient}

    # 1. PCA effective rank (use random subset for efficiency)
    n_pca_actual = min(n_pca, n)
    if n > n_pca_actual:
        rng = np.random.default_rng(42)
        subset = data[rng.choice(n, n_pca_actual, replace=False)]
    else:
        subset = data

    centered = subset - subset.mean(axis=0)
    cov = centered.T @ centered / (n_pca_actual - 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    eigvals = np.clip(eigvals, 0, None)
    d_pca = effective_rank_pca(eigvals)
    explained_95 = np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), 0.95) + 1
    explained_99 = np.searchsorted(np.cumsum(eigvals) / eigvals.sum(), 0.99) + 1
    results["pca_eff_dim"] = d_pca
    results["pca_95_components"] = explained_95
    results["pca_99_components"] = explained_99
    print(f"  PCA effective rank:     {d_pca:8.1f}")
    print(f"  PCA 95% variance:       {explained_95:8d} components")
    print(f"  PCA 99% variance:       {explained_99:8d} components")

    # 2. Two-NN
    nn2 = two_nn_dim(data)
    results["twonn_dim"] = nn2["d"]
    print(f"  Two-NN:                 {nn2['d']:8.1f}  (μ median={nn2['mu_median']:.2f})")

    # 3. Levina-Bickel with multiple k
    for k in [5, 10, 20, 50]:
        lb = levina_bickel_dim(data, k=k)
        results[f"levina_bickel_k{k}"] = lb["d"]
        print(f"  Levina-Bickel (k={k:2d}):   {lb['d']:8.1f}  (σ={lb['d_std']:.1f})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir",
                        default="/data/jlens/pythia-12b-deduped/embeddings_step143000")
    parser.add_argument("--layers", type=str, default="0,1,12,24,35,36",
                        help="Which context layers to analyze")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    layers = [int(x) for x in args.layers.split(",")]

    all_results = []

    # 1. Singleton embeddings
    singleton_path = emb_dir / "singleton_embeddings.npy"
    if singleton_path.exists():
        data = np.load(singleton_path).astype(np.float64)
        r = compute_pointcloud_dimensions("Singleton (embedding weights)", data)
        all_results.append(r)
    else:
        print(f"WARNING: {singleton_path} not found")

    # 2. Context layers
    for layer in layers:
        ctx_path = emb_dir / f"context_layer{layer}.npy"
        if ctx_path.exists():
            data = np.load(ctx_path).astype(np.float64)
            r = compute_pointcloud_dimensions(f"Context layer {layer}", data)
            all_results.append(r)
        else:
            print(f"WARNING: {ctx_path} not found")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Dataset':<30} {'PCA eff':>8} {'PCA95':>6} {'Two-NN':>8} {'LB(k=10)':>10} {'LB(k=50)':>10}")
    print("-" * 80)
    for r in all_results:
        name = r["dataset"]
        pca = r.get("pca_eff_dim", np.nan)
        pca95 = r.get("pca_95_components", np.nan)
        twonn = r.get("twonn_dim", np.nan)
        lb10 = r.get("levina_bickel_k10", np.nan)
        lb50 = r.get("levina_bickel_k50", np.nan)
        print(f"{name:<30} {pca:8.1f} {pca95:6d} {twonn:8.1f} {lb10:10.1f} {lb50:10.1f}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
