#!/usr/bin/env bash
#SBATCH --job-name=rip-iter
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rip_iter_%j.out
#SBATCH --error=logs/rip_iter_%j.err

# Iterative: for each top H1 cycle, re-Ripser its birth-radius neighborhood
# with do_cocycles=True to get the vertex cycle sequence.
set -euo pipefail
echo "=== Iterative Neighborhood Cocycles ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, json, pickle
from pathlib import Path
from ripser import ripser

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10; R_THRESH = 1.2
TOP_N = 5000  # top N cycles to extract

print("Loading...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = manifest['token_ids'][:X_raw.shape[0]]
print(f"  {X_raw.shape[0]} tokens")

# Load the precomputed diagram (from job 7031)
print("Loading precomputed diagram...")
with open(DATA_DIR / 'ripser_h1_neighborhoods.pkl', 'rb') as f:
    saved = pickle.load(f)
h1 = np.array(saved['ripser_dgms'][1])
pers = h1[:,1] - h1[:,0]
fin = np.isfinite(h1[:,1]); fin_idx = np.where(fin)[0]
top_order = np.argsort(-pers[fin])
top_i = fin_idx[top_order[:TOP_N]]
print(f"  Top {TOP_N} cycles, persistence range: [{pers[fin][top_order[TOP_N-1]]:.4f}, {pers[fin][top_order[0]]:.4f}]")

# Preprocess projection
print("Preprocessing...")
t0 = time.time()
X_c = X_raw - X_raw.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
print(f"  Done ({time.time()-t0:.1f}s)")

from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

def extract_cycle_from_edges(edges):
    verts = set()
    for u, v in edges: verts.add(u); verts.add(v)
    if len(verts) < 3: return None
    
    adj = {v: [] for v in verts}
    for u, v in edges: adj[u].append(v); adj[v].append(u)
    
    # BFS cycle detection
    for start in sorted(verts):
        queue = [(start, -1, [start])]
        visited_local = {start}
        while queue:
            v, parent, path = queue.pop(0)
            for nb in adj[v]:
                if nb == parent: continue
                if nb in path:
                    idx = path.index(nb)
                    return path[idx:]
                if nb not in visited_local:
                    visited_local.add(nb)
                    queue.append((nb, v, path + [nb]))
    return None

print(f"\nProcessing {TOP_N} cycles via neighborhood Ripser...")
out = []
t_start = time.time()

for rank, full_i in enumerate(top_i):
    birth = float(h1[full_i, 0])
    death = float(h1[full_i, 1])
    p = death - birth
    
    # Birth-radius neighborhood
    seed = 0  # arbitrary seed point
    dists = np.linalg.norm(X - X[seed], axis=1)
    radius = death * 1.05  # slightly larger than death to capture the cycle
    nearby = np.where(dists < radius)[0]
    
    if len(nearby) < 10 or len(nearby) > 1000:
        # Too small or too large neighborhood
        nearby = np.where(dists < death)[0]
        if len(nearby) < 10 or len(nearby) > 1000:
            continue
    
    # Subsample if too large
    if len(nearby) > 500:
        nearby = np.random.choice(nearby, 500, replace=False)
    
    X_local = X[nearby]
    
    # Ripser on local neighborhood with do_cocycles
    try:
        result = ripser(X_local, maxdim=1, thresh=R_THRESH, do_cocycles=True)
        local_h1 = result['dgms'][1]
        local_coc = result['cocycles'][1]
    except Exception as e:
        continue
    
    if len(local_coc) == 0:
        continue
    
    # Find the most persistent H1 cycle in this local diagram
    local_pers = local_h1[:,1] - local_h1[:,0]
    local_fin = np.isfinite(local_h1[:,1])
    if not local_fin.any():
        continue
    
    local_best = np.where(local_fin)[0][np.argmax(local_pers[local_fin])]
    cc = local_coc[local_best]
    
    if not hasattr(cc, 'ndim') or cc.ndim < 2 or len(cc) < 2:
        continue
    
    # Extract edges (local vertex indices)
    local_edges = [(int(row[0]), int(row[1])) for row in cc]
    local_edges = list(set((min(u,v), max(u,v)) for u,v in local_edges))
    
    local_cycle = extract_cycle_from_edges(local_edges)
    if local_cycle is None or len(local_cycle) < 3:
        continue
    
    # Map local vertices → global vertices → tokens
    global_verts = [int(nearby[v]) for v in local_cycle]
    tokens = [tok.decode([int(all_tok[v])]) for v in global_verts]
    
    out.append({
        'rank': rank, 'birth': birth, 'death': death, 'persistence': p,
        'n_verts': len(local_cycle), 'tokens': tokens,
        'decoded': ' '.join(tokens),
    })
    
    if (rank + 1) % 500 == 0:
        elapsed = time.time() - t_start
        rate = (rank + 1) / elapsed
        eta = (TOP_N - rank - 1) / rate / 60
        print(f"  {rank+1}/{TOP_N} ({len(out)} cycles, {elapsed:.0f}s, "
              f"~{eta:.0f}min remaining)")

print(f"\nDone: {len(out)} cycles in {time.time()-t_start:.0f}s")

# ── Save ──
pkl_path = DATA_DIR / 'ripser_cocycle_cycles.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(out, f)
print(f"Saved {pkl_path} ({pkl_path.stat().st_size/1024/1024:.1f} MB)")

txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_order.txt')
with open(txt_path, 'w') as f:
    for c in out:
        f.write(f"pers={c['persistence']:.4f} | n={c['n_verts']:2d} | {c['decoded']}\n")
print(f"Saved {txt_path} ({len(out)} lines)")

# Stats
if out:
    n_v = [c['n_verts'] for c in out]
    p_arr = [c['persistence'] for c in out]
    print(f"\nStats: {len(out)} cycles")
    print(f"  Verts: min={min(n_v)}, max={max(n_v)}, mean={np.mean(n_v):.1f}, median={np.median(n_v):.0f}")
    print(f"  Persistence: {min(p_arr):.4f} - {max(p_arr):.4f}")

# Top 30
print(f"\n=== Top 30 ===")
for c in sorted(out, key=lambda x: x['persistence'], reverse=True)[:30]:
    n_s = min(15, c['n_verts'])
    print(f"  pers={c['persistence']:.4f} n={c['n_verts']:2d} | "
          f"{' '.join(c['tokens'][:n_s])}{'...' if c['n_verts'] > n_s else ''}")

print("\nDone!")
PYEOF
