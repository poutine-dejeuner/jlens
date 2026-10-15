#!/usr/bin/env bash
#SBATCH --job-name=rip-2d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/rip_2d_%j.out
#SBATCH --error=logs/rip_2d_%j.err

# 2D projection + full homology cycle extraction.
# Lower dimension → sparser graph at birth → longer cycles.
set -euo pipefail
echo "=== 2D Homology Cycles ==="
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
PROJ_DIM = 2  # 2D for sparser graph
R_THRESH = 1.5  # on 2D sphere, max pairwise ~2.0
TOP_N = 5000

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
print(f"  H1={len(h1)}, cocycles={len(h1c)}")
print(f"  Persistence: [{pers[fin].min():.4f}, {pers[fin].max():.4f}]")
b = h1[fin, 0]; d = h1[fin, 1]
print(f"  Birth: [{b.min():.4f}, {b.max():.4f}]")
print(f"  Death: [{d.min():.4f}, {d.max():.4f}]")

from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

def recover_cycle(cc, birth, top_n_pre=5000):
    """Recover homology 1-cycle from cocycle."""
    if not hasattr(cc, 'ndim') or cc.ndim < 2 or len(cc) < 1:
        return None
    
    edges_with_dists = []
    for row in cc:
        u, v = int(row[0]), int(row[1])
        d = np.linalg.norm(X[u] - X[v])
        edges_with_dists.append((u, v, d))
    
    if not edges_with_dists:
        return None
    
    # Birth edge = edge with largest distance
    u_birth, v_birth, d_birth = max(edges_with_dists, key=lambda x: x[2])
    
    # Build graph of ALL edges with distance < d_birth (not just cocycle)
    # Sample edges efficiently: for each vertex, find closest neighbors
    adj = {}
    
    # Include cocycle edges
    for u, v, d in edges_with_dists:
        if d < d_birth:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
    
    # Find a few closest neighbors for u_birth and v_birth to ensure connectivity
    for seed in [u_birth, v_birth]:
        if seed not in adj:
            adj[seed] = []
        dists = np.linalg.norm(X - X[seed], axis=1)
        nearby = np.where(dists < d_birth * 0.999)[0]
        for nb in nearby[:100]:  # cap per-vertex neighbors
            if nb != seed and nb not in adj[seed]:
                adj[seed].append(nb)
                adj.setdefault(nb, []).append(seed)
    
    # BFS from u_birth to v_birth
    if u_birth not in adj or v_birth not in adj:
        return None
    
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
                    return path[::-1], d_birth, len(adj)
                queue.append(nb)
    return None, d_birth, len(adj)

print(f"\nRecovering cycles for top {TOP_N}...")
out = []
t0 = time.time()

for rank, full_i in enumerate(top_i):
    cc = h1c[full_i]
    birth = float(h1[full_i, 0]); death = float(h1[full_i, 1]); p = death - birth
    
    result_rc = recover_cycle(cc, birth)
    if result_rc is None: continue
    cycle_verts, d_birth, n_adj = result_rc
    if cycle_verts is None or len(cycle_verts) < 3: continue
    
    tokens = [tok.decode([int(all_tok[v])]) for v in cycle_verts]
    out.append({
        'rank': rank, 'birth': birth, 'death': death, 'persistence': p,
        'n_verts': len(cycle_verts), 'birth_edge_dist': d_birth,
        'tokens': tokens, 'decoded': ' '.join(tokens),
    })
    
    if rank % 500 == 0 and rank > 0:
        elapsed = time.time() - t0
        rate = rank / elapsed
        eta = (TOP_N - rank) / rate / 60 if rate > 0 else 0
        n_max = max([c['n_verts'] for c in out]) if out else 0
        print(f"  {rank}/{TOP_N}: {len(out)} cycles (max_len={n_max}) "
              f"[{elapsed:.0f}s, ~{eta:.0f}min left]")

print(f"\nDone: {len(out)} cycles in {time.time()-t0:.0f}s")

pkl_path = DATA_DIR / 'ripser_homology_2d.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(out, f)
print(f"Saved {pkl_path} ({pkl_path.stat().st_size/1024/1024:.1f} MB)")

txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_order_2d.txt')
with open(txt_path, 'w') as f:
    for c in out:
        f.write(f"pers={c['persistence']:.4f} | n={c['n_verts']:3d} | "
                f"edge={c['birth_edge_dist']:.4f} | {c['decoded']}\n")

if out:
    n_v = [c['n_verts'] for c in out]
    p_arr = [c['persistence'] for c in out]
    print(f"\nStats: {len(out)} cycles, max_len={max(n_v)}")
    print(f"  Verts: mean={np.mean(n_v):.1f}, median={np.median(n_v):.0f}, "
          f" 90th={np.percentile(n_v, 90):.0f}, 99th={np.percentile(n_v, 99):.0f}")
    print(f"  Persistence: [{min(p_arr):.4f}, {max(p_arr):.4f}]")

print(f"\n=== Top 20 by persistence ===")
for c in sorted(out, key=lambda x: x['persistence'], reverse=True)[:20]:
    ns = min(30, c['n_verts'])
    print(f"  p={c['persistence']:.4f} n={c['n_verts']:3d} | "
          f"{' '.join(c['tokens'][:ns])}{'...' if c['n_verts'] > ns else ''}")

print(f"\n=== Largest cycles ===")
for c in sorted(out, key=lambda x: x['n_verts'], reverse=True)[:25]:
    ns = min(50, c['n_verts'])
    print(f"  p={c['persistence']:.4f} n={c['n_verts']:3d} | "
          f"{' '.join(c['tokens'][:ns])}{'...' if c['n_verts'] > ns else ''}")

print("\nDone!")
PYEOF
