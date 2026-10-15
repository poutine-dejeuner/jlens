#!/usr/bin/env bash
#SBATCH --job-name=rip-dbg2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=logs/rip_dbg2_%j.out
#SBATCH --error=logs/rip_dbg2_%j.err
set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, h5py, json
from pathlib import Path
from ripser import ripser

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:5000].astype(np.float64)
X_c = X_raw - X_raw.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:10].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

result = ripser(X, maxdim=1, thresh=1.2, do_cocycles=True)
h1 = result['dgms'][1]; h1c = result['cocycles'][1]
pers = h1[:,1] - h1[:,0]; fin = np.isfinite(h1[:,1])
top_i = np.where(fin)[0][np.argsort(-pers[fin])[:5]]

def find_cycle(edges):
    verts = set(); [verts.update([u,v]) for u,v in edges]
    if len(verts) < 3: return None
    adj = {v: [] for v in verts}
    for u,v in edges: adj[u].append(v); adj[v].append(u)
    parent = {}; depth = {}
    for start in sorted(verts):
        if start in parent: continue
        queue = [start]; parent[start] = -1; depth[start] = 0
        while queue:
            v = queue.pop(0)
            for nb in adj[v]:
                if nb == parent[v]: continue
                if nb in parent:
                    cycle = []
                    a, b = v, nb
                    path_a = [a]
                    while parent[a] != -1: a = parent[a]; path_a.append(a)
                    path_b = [b]
                    while parent[b] != -1: b = parent[b]; path_b.append(b)
                    while path_a and path_b and path_a[-1] == path_b[-1]:
                        lca = path_a.pop(); path_b.pop()
                    cycle = path_a + [lca] + path_b[::-1]
                    if len(cycle) >= 3: return cycle
                else:
                    parent[nb] = v; depth[nb] = depth[v] + 1
                    queue.append(nb)
    return None

for rank, full_i in enumerate(top_i):
    cc = h1c[full_i]
    birth = float(h1[full_i, 0])
    print(f"\n[{rank}] idx={full_i} pers={pers[full_i]:.4f} birth={birth:.4f}")
    print(f"  cc shape={cc.shape}")
    
    # Direct
    seed_edges = [(int(r[0]), int(r[1])) for r in cc]
    seed_edges = list(set((min(u,v), max(u,v)) for u,v in seed_edges))
    c1 = find_cycle(seed_edges)
    print(f"  Direct: edges={len(seed_edges)}, cycle={c1}")
    
    # Birth-expanded
    seed_verts = set()
    for u,v in seed_edges: seed_verts.add(u); seed_verts.add(v)
    sv = list(seed_verts)
    exp = list(seed_edges)
    for i in range(len(sv)):
        for j in range(i+1, len(sv)):
            u,v = sv[i], sv[j]
            if np.linalg.norm(X[u] - X[v]) < birth:
                exp.append((u,v))
    exp = list(set((min(u,v), max(u,v)) for u,v in exp))
    c2 = find_cycle(exp)
    print(f"  Expanded: edges={len(exp)}, verts={len(sv)}, cycle={c2}")
    
    degs = {}
    for u,v in exp: degs[u] = degs.get(u,0)+1; degs[v] = degs.get(v,0)+1
    deg_counts = {}
    for d in degs.values(): deg_counts[d] = deg_counts.get(d,0)+1
    print(f"  Degrees: {dict(sorted(deg_counts.items()))}")
PYEOF