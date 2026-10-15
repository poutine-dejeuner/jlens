#!/usr/bin/env bash
#SBATCH --job-name=rip-coc2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rip_coc2_%j.out
#SBATCH --error=logs/rip_coc2_%j.err

# Extract cocycle vertex lists: subsample 5000 points for faster iteration,
# use do_cocycles=True, decode edge indices to vertex pairs, 
# then map to token strings.

set -euo pipefail
echo "=== Cocycle Decode: Qwen3.6-27B contextual (subsample) ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, json
from pathlib import Path
from ripser import ripser
from scipy.sparse import csr_matrix, tril, find as sparse_find
import pickle
import gc

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10
R_THRESH = 1.2
N_SUBSAMPLE = 10000  # Start with 10K points
N_TOP = 50

# ── Load & preprocess ──
print("Loading contextual representations...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)

with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_token_ids = manifest['token_ids'][:X_raw.shape[0]]

# Subsample
rng = np.random.RandomState(42)
idx_sub = rng.choice(len(X_raw), N_SUBSAMPLE, replace=False)
X_sub = X_raw[idx_sub]
token_sub = [all_token_ids[i] for i in idx_sub]
print(f"Subsampled: {N_SUBSAMPLE} / {len(X_raw)}")

print("Preprocessing...")
t0 = time.time()
X_c = X_sub - X_sub.mean(axis=0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X_proj = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X_proj = X_proj / (np.linalg.norm(X_proj, axis=1, keepdims=True) + 1e-8)
print(f"  Done ({time.time()-t0:.1f}s)")

# ── Ripser with cocycles ──
print(f"\nRipser(maxdim=1, thresh={R_THRESH}, do_cocycles=True) on {N_SUBSAMPLE} points...")
t0 = time.time()
result = ripser(X_proj, maxdim=1, thresh=R_THRESH, do_cocycles=True)
elapsed = time.time() - t0
dgms = result['dgms']
cocycles = result['cocycles']
print(f"  Done: {elapsed:.1f}s")
print(f"  H0: {len(dgms[0])}, H1: {len(dgms[1])}")

# ── Inspect cocycle format ──
h1_coc = cocycles[1]
h1_dgm = dgms[1]
pers = h1_dgm[:,1] - h1_dgm[:,0]
fin = np.isfinite(h1_dgm[:,1])
top_order = np.argsort(-pers[fin])[:N_TOP]
fin_idx = np.where(fin)[0]
top_full = fin_idx[top_order]

print(f"\nH1 cocycles: {len(h1_coc)}")
print(f"H1 dgm shape: {h1_dgm.shape}, finite: {fin.sum()}")

# Check cocycle format for first top cycle
# Ripser's do_cocycles returns:
#   cocycles[dim][i] = list of (coefficient, simplex_index) 
#   where simplex_index maps to Ripser's internal edge enumeration.
# We need to reconstruct the vertex-edge mapping.

# Approach: build full distance matrix (triangular), get edge indices
# in order of increasing distance, map simplex_index → (u,v)

print("\nBuilding edge index mapping...")
# Build all pairwise distances (lower triangle, within thresh)
dists_full = np.linalg.norm(X_proj[:, None] - X_proj[None, :], axis=2)
n = X_proj.shape[0]

# Extract edges in order of distance (Ripser's enumeration order)
# Ripser enumerates edges in lexicographic order of (distance, i, j)
# Actually Ripser uses a colexicographic order: first by distance, 
# then by the index in the union-find.
# The exact order depends on implementation, but for small threshold 
# we can approximate.

# Actually, let's just use the cocycle edges as an edge set and 
# reconstruct the graph. For H1 cocycles, each edge creates a 
# dimension-1 cocycle.

# Simpler: Use the birth-radius to find the subgraph, then find
# shortest cycle containing the cocycle edges.

print("Inspecting cocycle contents...")
for ci in range(min(5, len(top_full))):
    i = top_full[ci]
    cc = h1_coc[i]
    birth = h1_dgm[i, 0]
    death = h1_dgm[i, 1]
    p = death - birth
    print(f"\n  Cycle {ci}: birth={birth:.4f}, death={death:.4f}, pers={p:.4f}")
    print(f"    cocycle type: {type(cc)}")
    if hasattr(cc, '__len__'):
        print(f"    cocycle length: {len(cc)}")
        if len(cc) > 0:
            first = cc[0]
            print(f"    first element: type={type(first)}, value={first if not isinstance(first, np.ndarray) else first.shape}")
    # If scipy sparse
    if hasattr(cc, 'nnz'):
        print(f"    nnz: {cc.nnz}")
    if hasattr(cc, 'nonzero'):
        nz = cc.nonzero()
        print(f"    nonzero indices: {nz[0][:20]}...")

# ── Build the edge-to-vertex mapping using distance matrix ──
# For cocycle reconstruction: each simplex index maps to an edge.
# Ripser uses a fixed enumeration: all edges up to thresh, sorted by distance.
# We can enumerate them ourselves.

print("\n\n=== Reconstructing edge enumeration ===")
# Extract all edges within thresh, sorted by distance
print("Building edge list from distance matrix...")
t0 = time.time()

# We only need edges within thresh for mapping
mask = (dists_full <= R_THRESH) & (np.tri(n, k=-1, dtype=bool))
iu, iv = np.where(mask)
dvals = dists_full[iu, iv]
# Sort by distance (Ripser's order uses smallest distance first)
edge_order = np.argsort(dvals)
iu_sorted = iu[edge_order]
iv_sorted = iv[edge_order]
d_sorted = dvals[edge_order]

print(f"  Edges within thresh: {len(iu_sorted)} ({time.time()-t0:.1f}s)")

# Now we have: edge_index → (iu_sorted[edge_index], iv_sorted[edge_index])

# ── Decode top cocycles ──
from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

print(f"\nDecoding top {N_TOP} cocycles...")
cycle_results = []

for rank, full_i in enumerate(top_full[:N_TOP]):
    cc = h1_coc[full_i]
    birth = float(h1_dgm[full_i, 0])
    death = float(h1_dgm[full_i, 1])
    p = float(death - birth)
    
    # Extract edge indices from cocycle
    edge_indices = []
    if isinstance(cc, list):
        # list of (coefficient, edge_index) tuples
        edge_indices = [item[1] if isinstance(item, tuple) else item for item in cc if item != 0]
    elif hasattr(cc, 'nonzero'):
        nz = cc.nonzero()
        edge_indices = nz[0].tolist() if len(nz) > 0 and len(nz[0]) > 0 else []
    elif isinstance(cc, np.ndarray):
        edge_indices = np.nonzero(cc)[0].tolist()
    
    if len(edge_indices) == 0:
        continue
    
    # Map to vertex pairs
    valid_edges = []
    vertices_in_cycle = set()
    for ei in edge_indices:
        if ei < len(iu_sorted):
            u, v = int(iu_sorted[ei]), int(iv_sorted[ei])
            valid_edges.append((u, v))
            vertices_in_cycle.add(u)
            vertices_in_cycle.add(v)
    
    if len(valid_edges) < 2:
        continue
    
    # Build graph from these edges
    adj = {v: [] for v in vertices_in_cycle}
    for u, v in valid_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Try to find an Eulerian cycle or just trace the edges
    # For a 1-cycle cocycle, the edges should form one or more cycles
    # Extract connected components and find cycles
    
    visited_edges = set()
    all_cycles = []
    
    for start_v in vertices_in_cycle:
        if all(deg == 0 for deg in [len(adj[v]) for v in vertices_in_cycle]):
            break
        path = [start_v]
        # Simple DFS to find a cycle
        stack = [(start_v, -1, [start_v], set())]
        while stack:
            v, parent, path, used_edges = stack.pop()
            for nb in adj.get(v, []):
                edge = (min(v, nb), max(v, nb))
                if edge in used_edges:
                    continue
                if nb == parent:
                    continue
                if nb in path:
                    # Found a cycle
                    cycle_start = path.index(nb)
                    cycle_verts = path[cycle_start:]
                    cycle_edges = []
                    for k in range(len(cycle_verts)):
                        a = cycle_verts[k]
                        b = cycle_verts[(k+1) % len(cycle_verts)]
                        cycle_edges.append((min(a,b), max(a,b)))
                    all_cycles.append(cycle_verts)
                    break
                else:
                    new_used = used_edges | {edge}
                    stack.append((nb, v, path + [nb], new_used))
    
    if len(all_cycles) == 0:
        continue
    
    # Use the longest cycle found
    best_cycle = max(all_cycles, key=len)
    vert_seq = best_cycle
    
    if len(vert_seq) < 3:
        continue
    
    # Map vertices to tokens
    token_ids = [int(token_sub[v]) for v in vert_seq]
    # Try to decode with spaces between individual tokens
    tokens = [tok.decode([tid]) for tid in token_ids]
    decoded_clean = ' '.join(tokens)
    
    cycle_results.append({
        'rank': rank,
        'birth': birth,
        'death': death,
        'persistence': p,
        'n_vertices': len(vert_seq),
        'vertices': vert_seq,
        'tokens': tokens,
        'decoded': decoded_clean,
    })
    
    if rank < 20:
        print(f"\n  Cycle {rank}: {len(vert_seq)} vertices, pers={p:.4f}")
        print(f"    tokens: {' '.join(tokens[:30])}{'...' if len(tokens) > 30 else ''}")

# ── Save ──
with open(DATA_DIR / 'cocycle_sequences.pkl', 'wb') as f:
    pickle.dump(cycle_results, f)
print(f"\nSaved {len(cycle_results)} decoded cocycle sequences")

# Save readable text
txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_order.txt')
with open(txt_path, 'w') as f:
    for c in cycle_results:
        f.write(f"Cycle {c['rank']}: pers={c['persistence']:.4f} | "
                f"{' '.join(c['tokens'])}\n")

print(f"Saved to {txt_path}")
print("\nDone!")
PYEOF
