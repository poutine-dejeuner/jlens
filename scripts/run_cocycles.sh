#!/usr/bin/env bash
#SBATCH --job-name=rip-coc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=logs/rip_coc_%j.out
#SBATCH --error=logs/rip_coc_%j.err

# Extract cocycle vertex lists from Ripser and map to token order.
# The cocycles array from do_cocycles=True gives the edge indices
# in cyclic order. We need to decode them into vertex sequences.

set -euo pipefail
echo "=== Cocycle Extraction: Qwen3.6-27B contextual ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, json, pickle
from pathlib import Path
from ripser import ripser
from collections import defaultdict

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10
R_THRESH = 1.2
TOP_N = 200  # Number of top cycles to extract cocycles for

# ── Load & preprocess (same pipeline) ──
print("Loading contextual representations...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)

with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_token_ids = manifest['token_ids'][:X_raw.shape[0]]

print("Preprocessing...")
t0 = time.time()
X_c = X_raw - X_raw.mean(axis=0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X_proj = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X_proj = X_proj / (np.linalg.norm(X_proj, axis=1, keepdims=True) + 1e-8)
print(f"  Done ({time.time()-t0:.1f}s)")

# ── Ripser with do_cocycles ──
print(f"\nRipser(maxdim=1, thresh={R_THRESH}, do_cocycles=True) on {X_proj.shape[0]} points...")
t0 = time.time()
result = ripser(X_proj, maxdim=1, thresh=R_THRESH, do_cocycles=True)
elapsed = time.time() - t0
dgms = result['dgms']
cocycles = result['cocycles']
print(f"  Done: {elapsed:.1f}s")
print(f"  H1 cocycles: {len(cocycles[1])}")
print(f"  cocycle[1][0] type: {type(cocycles[1][0])}")

# Inspect cocycle format
h1_coc = cocycles[1]  # list of cocycles for H1
print(f"\nInspecting H1 cocycle format (first 5):")
for i in range(min(5, len(h1_coc))):
    cc = h1_coc[i]
    if cc is None:
        print(f"  [{i}] None")
        continue
    # cocycle is typically a list of (coeff, simplex_index) pairs or similar
    print(f"  [{i}] type={type(cc)}, len={len(cc) if hasattr(cc, '__len__') else 'N/A'}")
    if hasattr(cc, '__len__') and len(cc) > 0:
        print(f"       first element: {cc[0] if isinstance(cc, list) else type(cc)}")

# For Ripser, cocycles[1][i] is a list of (coefficient, edge_index) pairs.
# But we need to map edge_index → (vertex_a, vertex_b).
# Ripser doesn't expose this mapping directly in the Python API.
# 
# Alternative: use the distance matrix to reconstruct.
# For each cocycle, find all edges at distance < death_radius, then
# extract the 1-cycle vertex sequence from the graph structure.

# ── Approach: build graph of edges within death radius ──
from scipy.sparse import coo_matrix

print("\nBuilding edge index for death-radius reconstruction...")
# We need a mapping from Ripser's internal edge index to vertex pairs.
# Strategy: use the birth radius graph and find minimal cycles.

# For each top cycle, extract all edges within the death radius,
# build a graph, and find a cycle.

h1_fin = dgms[1]
pers = h1_fin[:, 1] - h1_fin[:, 0]
fin_mask = np.isfinite(h1_fin[:, 1])
top_order = np.argsort(-pers[fin_mask])[:TOP_N]
fin_idx = np.where(fin_mask)[0]
top_full = fin_idx[top_order]

# For each top cycle, we know its birth and death.
# The cocycle list at the same index has the edge data.
# Let's decode using the distance matrix.

# Build sparse matrix of pairwise distances up to max death
max_death = h1_fin[fin_mask, 1].max()
print(f"Max death radius for top cycles: {max_death:.4f}")

# Use the cocycle data directly — it IS the set of edges in the cycle
print("\nDecoding cocycles into vertex sequences...")

# For each top cycle, the cocycle[i] is a list of (coefficient, edge_index) pairs
# The edge indices reference Ripser's internal simplex enumeration.
# We need to convert edge_index → vertex pair.
# 
# Ripser enumerates edges in order of increasing distance.
# We can reconstruct by sorting all pairwise distances and indexing.
# But this is expensive for 42K points.
#
# Better approach: use the cocycle as a set of edges on the birth-radius
# subcomplex. Since cocycles are cohomology generators, each entry is
# an edge (1-simplex). We can build the graph from those edges.

# First, let's check if cocycle entries have vertex info
print("\nDetailed cocycle inspection:")
for i in [0, 1, 2]:
    cc = h1_coc[i]
    if cc is None:
        print(f"  [{i}] None")
        continue
    if isinstance(cc, (list, tuple)):
        print(f"  [{i}] list/tuple, len={len(cc)}")
        if len(cc) > 0:
            item = cc[0]
            print(f"       item type: {type(item)}")
            if isinstance(item, tuple):
                print(f"       item: {item}")
    else:
        # Might be a scipy sparse matrix
        print(f"  [{i}] type={type(cc)}, shape={cc.shape if hasattr(cc, 'shape') else 'N/A'}")
        if hasattr(cc, 'nonzero'):
            nz = cc.nonzero()
            print(f"       nnz: {len(nz[0])}")
            if len(nz[0]) > 0:
                print(f"       first few indices: {nz[0][:10]}")

print("\nDone inspecting. Saving raw cocycle data...")
# Save the dgms and cocycles for now, decode later with better approach
out = {
    'dgms_h1': dgms[1],
    'cocycles_h1': h1_coc,  # May be very large
    'persistence': pers,
    'top_indices': top_full,
}
# cocycles can be huge — just save top N info
del out['cocycles_h1']  # too large for pickle
with open(DATA_DIR / 'cocycle_dgms.pkl', 'wb') as f:
    pickle.dump(out, f)

print("Done")
PYEOF
