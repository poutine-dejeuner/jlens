#!/usr/bin/env python3
"""For each PC axis, produce a table showing tokens at each discretization
step along the axis (binned coordinate values), so the semantic concept
captured by that PC can be inferred by reading across the table.

Output: a TSV file figures/thought_manifold_axis_labels.tsv with columns:
  PC  bin_center  token_id  u_value  decoded_token
"""

from __future__ import annotations

import argparse, csv
from pathlib import Path

import h5py, numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Create human-readable PC axis label tables")
    parser.add_argument("manifold", help="Path to manifold.h5")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--n-pcs", type=int, default=6,
                        help="Number of PCs to tabulate")
    parser.add_argument("--n-bins", type=int, default=12,
                        help="Number of discretization bins per PC")
    parser.add_argument("--n-per-bin", type=int, default=5,
                        help="Tokens to show per bin")
    args = parser.parse_args()

    h5path = Path(args.manifold)
    outpath = Path(args.output) if args.output else Path(
        "figures/thought_manifold_axis_labels.tsv")

    print("Loading manifold...")
    with h5py.File(str(h5path), "r") as f:
        U = f["eigenvectors"][:]  # (V, k)
        ev = f["eigenvalues"][:]

    # Normalize each PC column to [-1, 1] for easier binning
    U_norm = U / np.abs(U).max(axis=0, keepdims=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    V = U.shape[0]
    token_strings = [tokenizer.decode([i]) for i in range(V)]

    # Write TSV
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["PC", "variance_pct", "eigenvalue",
                          "bin_center", "token_id", "u_value",
                          "decoded_token"])

        for pc in range(min(args.n_pcs, U.shape[1])):
            col = U_norm[:, pc]  # normalized to [-1, 1]
            var_pct = 100 * ev[pc] / ev[:50].sum()

            # Create equally-spaced bins covering the data range
            bins = np.linspace(col.min(), col.max(), args.n_bins + 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

            for b_idx in range(args.n_bins):
                lo, hi = bins[b_idx], bins[b_idx + 1]
                mask = (col >= lo) & (col < hi)
                # For the last bin, include the max value
                if b_idx == args.n_bins - 1:
                    mask = (col >= lo) & (col <= hi)

                idx_in_bin = np.where(mask)[0]
                if len(idx_in_bin) == 0:
                    continue

                # Pick tokens closest to bin center
                center = bin_centers[b_idx]
                dists = np.abs(col[idx_in_bin] - center)
                top_n = np.argsort(dists)[:args.n_per_bin]

                for rank in top_n:
                    tid = int(idx_in_bin[rank])
                    tok = token_strings[tid]
                    u_val = col[tid]
                    writer.writerow([
                        f"PC{pc+1}", f"{var_pct:.1f}", f"{ev[pc]:.0f}",
                        f"{center:.3f}", tid, f"{u_val:.4f}", tok
                    ])

    print(f"Saved → {outpath}")
    print(f"Open in a spreadsheet or use 'column -t -s \$'\\t' {outpath} | less -S'")


if __name__ == "__main__":
    main()
