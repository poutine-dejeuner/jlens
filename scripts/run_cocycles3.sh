#!/usr/bin/env bash
#SBATCH --job-name=rip-coc3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rip_coc3_%j.out
#SBATCH --error=logs/rip_coc3_%j.err

set -euo pipefail
echo "=== Cocycle Decode v3 ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, json, pickle
from pathlib import Path
from ripser import ripser
from scipy.sparse import coo_matrix

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10; R_THRESH = 1.2; N_SUB = 10000; N_TOP = 50

print("Loading...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = manifest['token_ids'][:X_raw.shape[0]]

rng = np.random.RandomState(42)
idx_sub = rng.choice(len(X_raw), N_SUB, replace=False)
X_sub = X_raw[idx_sub]; tok_sub = [all_tok[i] for i in idx_sub]

print(f"N={N_SUB}, preprocessing...")
X_c = X_sub - X_sub.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

print("Ripser...")
t0 = time.time()
result = ripser(X, maxdim=1, thresh=R_THRESH, do_cocycles=True)
print(f"  Done: {time.time()-t0:.1f}s")

dgms = result['dgms']; coc = result['cocycles']
h1 = dgms[1]; h1c = coc[1]
pers = h1[:,1] - h1[:,0]; fin = np.isfinite(h1[:,1])
top_i = np.where(fin)[0][np.argsort(-pers[fin])[:N_TOP]]

print("\nDecoding top", N_TOP, "...")
t0 = time.time()

# Inspect format
cc0 = h1c[0]
print(f"h1c[0] type: {type(cc0)}, shape: {cc0.shape}, dtype: {cc0.dtype}")
print(f"  first row: [{cc0[0,0]}, {cc0[0,1]}, {cc0[0,2]}]")

from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

out = []
for rank, full_i in enumerate(top_i):
    cc = h1c[full_i]
    birth = float(h1[full_i, 0]); death = float(h1[full_i, 1]); p = death - birth
    
    print(f"\n  [{rank}] idx={full_i} pers={p:.4f} cc_type={type(cc)}")
    if hasattr(cc, 'shape'):
        print(f"       shape={cc.shape}, ndim={cc.ndim}")
    
    # Get edges directly from cocycle: each row is [v_a, v_b, coeff]
    if not hasattr(cc, 'ndim') or cc.ndim < 2:
        print(f"       SKIP: no ndim or ndim<2")
        continue
    if len(cc) == 0:
        print(f"       SKIP: empty")
        continue
    
    edges = []
    verts = set()
    for row in cc:
        u, v = int(row[0]), int(row[1])
        edges.append((u, v))
        verts.add(u); verts.add(v)
    print(f"       edges={len(edges)}, verts={len(verts)}")
    
    # Deduplicate edges
    edges = list(set((min(u,v), max(u,v)) for u,v in edges))
    
    # Build adjacency
    adj = {v: [] for v in verts}
    for u, v in edges:
        adj[u].append(v); adj[v].append(u)
    
    # Find connected components and extract the largest one
    visited = set()
    components = []
    for v in verts:
        if v in visited:
            continue
        # BFS to find component
        comp = []
        queue = [v]
        visited.add(v)
        while queue:
            node = queue.pop(0)
            comp.append(node)
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
    
    largest_comp = max(components, key=len)
    print(f"       components={len(components)}, largest={len(largest_comp)}")
    
    if len(largest_comp) < 3:
        print(f"       SKIP: largest component too small")
        continue
    
    
    # For the largest component, extract a simple cycle via DFS
    comp_set = set(largest_comp)
    comp_edges = [(u,v) for u,v in edges if u in comp_set and v in comp_set]
    comp_adj = {v: [] for v in largest_comp}
    for u, v in comp_edges:
        comp_adj[u].append(v); comp_adj[v].append(u)
    
    # DFS to find a cycle
    path = []
    visited2 = set()
    parent = {}
    
    def dfs_find(v, p):
        visited2.add(v)
        parent[v] = p
        for nb in comp_adj.get(v, []):
            if nb == p:
                continue
            if nb in visited2:
                # Back edge found — extract cycle
                cycle = [nb, v]
                cur = v
                while cur != nb and cur in parent:
                    cur = parent[cur]
                    if cur != nb:
                        cycle.append(cur)
                return cycle
            result = dfs_find(nb, v)
            if result:
                return result
        return None
    
    cycle = dfs_find(largest_comp[0], -1)
    
    if cycle is None:
        print(f"       SKIP: no cycle found in component")
        continue
    
    print(f"       cycle found: {len(cycle)} vertices")
    
    # Remove duplicate endpoint
    if cycle[0] == cycle[-1]:
        cycle = cycle[:-1]
    
    # Map to tokens
    tokens = []
    for v in cycle:
        tid = int(tok_sub[v])
        tokens.append(tok.decode([tid]))
    
    out.append({
        'rank': rank, 'birth': birth, 'death': death, 'persistence': p,
        'n_verts': len(cycle), 'tokens': tokens,
        'decoded': ' '.join(tokens),
        'decoded_clean': tok.decode([int(tok_sub[v]) for v in cycle]),
    })
    
    if rank < 30:
        n_show = min(20, len(tokens))
        print(f"  [{rank:2d}] pers={p:.4f} | {len(cycle):2d}v | {' '.join(tokens[:n_show])}{'...' if len(tokens)>n_show else ''}")

# ── Save ──
with open(DATA_DIR / 'cocycle_sequences.pkl', 'wb') as f:
    pickle.dump(out, f)
print(f"\nSaved {len(out)} sequences")

txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_order.txt')
with open(txt_path, 'w') as f:
    for c in out:
        f.write(f"Cycle {c['rank']}: pers={c['persistence']:.4f} | {c['decoded']}\n")
        f.write(f"  clean: {c['decoded_clean']}\n\n")
print(f"Saved to {txt_path}")
print("Done!")
PYEOF
