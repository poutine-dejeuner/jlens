#!/usr/bin/env bash
#SBATCH --job-name=rip-cyc
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/rip_cyc_%j.out
#SBATCH --error=logs/rip_cyc_%j.err

# Recover actual homology 1-cycles from Ripser cocycles.
# Algorithm: for each H1 class born at edge e=(u,v):
#   1. Identify e as the cocycle edge with largest distance (closest to birth)
#   2. Find shortest path u→v in graph of edges with distance < d(e)
#   3. Path + edge = the homology cycle
set -euo pipefail
echo "=== Homology Cycle Recovery ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, json, pickle
from pathlib import Path
from ripser import ripser
from collections import deque

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10; R_THRESH = 1.2
TOP_N = 2000  # top cycles to analyze

print("Loading...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = manifest['token_ids'][:X_raw.shape[0]]
N = X_raw.shape[0]
print(f"  {N} tokens")

print("Preprocessing...")
t0 = time.time()
X_c = X_raw - X_raw.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
print(f"  Done ({time.time()-t0:.1f}s)")

print(f"\nRipser(maxdim=1, thresh={R_THRESH}, do_cocycles=True)...")
t0 = time.time()
result = ripser(X, maxdim=1, thresh=R_THRESH, do_cocycles=True)
print(f"  Done: {time.time()-t0:.0f}s")
h1 = result['dgms'][1]; h1c = result['cocycles'][1]
pers = h1[:,1] - h1[:,0]; fin = np.isfinite(h1[:,1])
fin_idx = np.where(fin)[0]
top_order = np.argsort(-pers[fin])
top_i = fin_idx[top_order[:TOP_N]]
print(f"  H1={len(h1)}, cocycles={len(h1c)}, top={len(top_i)}")

from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

# ── Precompute pairwise distances for efficient path finding ──
# We need to find paths between vertices using only edges < birth_distance.
# For efficiency, precompute sorted edge lists per vertex.
# But full O(N^2) is infeasible. Use on-demand distance queries instead.

def shortest_path(u, v, max_dist, max_vertices=1000):
    """BFS from u to v using only edges with distance < max_dist."""
    if u == v: return [u]
    visited = {u: -1}  # vertex → parent
    queue = deque([u])
    while queue:
        cur = queue.popleft()
        # Find neighbors: any vertex within max_dist of cur
        # For efficiency, check against a subset
        dists_to_cur = np.linalg.norm(X - X[cur], axis=1)
        neighbors = np.where(dists_to_cur < max_dist)[0]
        for nb in neighbors:
            if nb == cur: continue
            if nb not in visited:
                visited[nb] = cur
                if nb == v:
                    # Reconstruct path
                    path = [v]
                    while path[-1] != u:
                        path.append(visited[path[-1]])
                    return path[::-1]
                if len(visited) < max_vertices:
                    queue.append(nb)
    return None

print(f"\nRecovering homology cycles for top {TOP_N} H1 classes...")
out = []
t0 = time.time()

for rank, full_i in enumerate(top_i):
    cc = h1c[full_i]
    birth = float(h1[full_i, 0]); death = float(h1[full_i, 1]); p = death - birth
    
    if not hasattr(cc, 'ndim') or cc.ndim < 2 or len(cc) < 1:
        continue
    
    # Get cocycle edges and their distances
    edges_with_dists = []
    for row in cc:
        u, v = int(row[0]), int(row[1])
        d = np.linalg.norm(X[u] - X[v])
        edges_with_dists.append((u, v, d))
    
    if not edges_with_dists:
        continue
    
    # Find the birth edge: the one with largest distance
    # (closest to birth radius — the edge that creates the cycle)
    u_birth, v_birth, d_birth = max(edges_with_dists, key=lambda x: x[2])
    
    # Build graph of edges STRICTLY before birth
    # (edges with distance < d_birth, excluding the birth edge itself)
    pre_edges = [(u,v) for u,v,d in edges_with_dists if d < d_birth]
    
    # Find path from u_birth to v_birth using pre-edges
    # First try using the pre-edges graph
    path = None
    
    # Build adjacency from pre-edges for fast path finding
    if pre_edges:
        adj = {}
        for u,v in pre_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        
        # BFS in the pre-edge graph
        if u_birth in adj and v_birth in adj:
            visited = {u_birth: -1}
            queue = deque([u_birth])
            while queue:
                cur = queue.popleft()
                for nb in adj.get(cur, []):
                    if nb not in visited:
                        visited[nb] = cur
                        if nb == v_birth:
                            path = [v_birth]
                            while path[-1] != u_birth:
                                path.append(visited[path[-1]])
                            path = path[::-1]
                            break
                if path: break
    
    # If pre-edge graph doesn't connect, use distance-based path
    if path is None:
        # Use all edges with distance < d_birth (not just cocycle edges)
        path = shortest_path(u_birth, v_birth, d_birth * 0.999)
    
    if path is None or len(path) < 2:
        continue
    
    # The cycle is: path from u to v, plus the birth edge back
    # Remove the last vertex if it's the same as the first
    cycle_verts = path  # path already goes u→v
    
    tokens = [tok.decode([int(all_tok[v])]) for v in cycle_verts]
    
    out.append({
        'rank': rank, 'birth': birth, 'death': death, 'persistence': p,
        'n_verts': len(cycle_verts),
        'birth_edge_dist': d_birth,
        'n_pre_edges': len(pre_edges),
        'tokens': tokens,
        'decoded': ' '.join(tokens),
    })
    
    if rank % 200 == 0 and rank > 0:
        elapsed = time.time() - t0
        rate = rank / elapsed
        eta = (TOP_N - rank) / rate / 60 if rate > 0 else 0
        print(f"  {rank}/{TOP_N}: {len(out)} cycles [{elapsed:.0f}s, ~{eta:.0f}min left]")

print(f"\nDone: {len(out)} cycles in {time.time()-t0:.0f}s")

# Save
pkl_path = DATA_DIR / 'ripser_homology_cycles.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(out, f)
print(f"Saved {pkl_path} ({pkl_path.stat().st_size/1024/1024:.1f} MB)")

txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_order.txt')
with open(txt_path, 'w') as f:
    for c in out:
        f.write(f"pers={c['persistence']:.4f} | n={c['n_verts']:2d} | "
                f"birth_edge={c['birth_edge_dist']:.4f} | {c['decoded']}\n")
print(f"Saved {txt_path} ({len(out)} lines)")

if out:
    n_v = [c['n_verts'] for c in out]
    p_arr = [c['persistence'] for c in out]
    print(f"\nStats: {len(out)} cycles")
    print(f"  Verts: min={min(n_v)}, max={max(n_v)}, mean={np.mean(n_v):.1f}, median={np.median(n_v):.0f}")
    print(f"  Persistence: {min(p_arr):.4f} - {max(p_arr):.4f}")

# Print top cycles sorted by size (look for large interpretable cycles)
print(f"\n=== Top 30 by persistence ===")
for c in sorted(out, key=lambda x: x['persistence'], reverse=True)[:30]:
    ns = min(20, c['n_verts'])
    print(f"  p={c['persistence']:.4f} n={c['n_verts']:3d} edge={c['birth_edge_dist']:.4f} | "
          f"{' '.join(c['tokens'][:ns])}{'...' if c['n_verts'] > ns else ''}")

# Also show largest cycles
print(f"\n=== Largest cycles ===")
for c in sorted(out, key=lambda x: x['n_verts'], reverse=True)[:20]:
    ns = min(40, c['n_verts'])
    print(f"  p={c['persistence']:.4f} n={c['n_verts']:3d} edge={c['birth_edge_dist']:.4f} | "
          f"{' '.join(c['tokens'][:ns])}{'...' if c['n_verts'] > ns else ''}")

print("\nDone!")
PYEOF
