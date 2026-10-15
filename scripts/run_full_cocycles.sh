#!/usr/bin/env bash
#SBATCH --job-name=rip-full
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=logs/rip_full_%j.out
#SBATCH --error=logs/rip_full_%j.err

# Full 42K Ripser with cocycles + cycle extraction + token mapping
set -euo pipefail
echo "=== Full Cocycle Decode ==="
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

print("Loading...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = manifest['token_ids'][:X_raw.shape[0]]
print(f"  {X_raw.shape[0]} tokens")

print("Preprocessing...")
t0 = time.time()
X_c = X_raw - X_raw.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
print(f"  Done ({time.time()-t0:.1f}s)")

print(f"Ripser(maxdim=1, thresh={R_THRESH}, do_cocycles=True)...")
t0 = time.time()
result = ripser(X, maxdim=1, thresh=R_THRESH, do_cocycles=True)
print(f"  Done: {time.time()-t0:.1f}s")

dgms = result['dgms']; h1c = result['cocycles'][1]
h1 = dgms[1]; pers = h1[:,1] - h1[:,0]
fin = np.isfinite(h1[:,1]); fin_idx = np.where(fin)[0]
top_order = np.argsort(-pers[fin])  # all finite, sorted by persistence
top_i = fin_idx[top_order]

print(f"H1 cycles: {len(h1)} (finite: {fin.sum()})")
print(f"Persistence range: [{pers[fin].min():.4f}, {pers[fin].max():.4f}]")
print(f"Cocycles: {len(h1c)}")

from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

# ── Cycle extraction ──
def extract_cycle(edges):
    """Given list of (u,v) edges, find a simple cycle via DFS."""
    verts = set()
    for u, v in edges: verts.add(u); verts.add(v)
    
    adj = {}
    for v in verts: adj[v] = []
    for u, v in edges: adj[u].append(v); adj[v].append(u)
    
    # Find largest connected component
    visited = set()
    components = []
    for v in verts:
        if v in visited: continue
        comp = []; queue = [v]; visited.add(v)
        while queue:
            node = queue.pop(0); comp.append(node)
            for nb in adj[node]:
                if nb not in visited: visited.add(nb); queue.append(nb)
        components.append(comp)
    
    if not components: return None
    largest = max(components, key=len)
    if len(largest) < 3: return None
    
    # DFS cycle detection on largest component
    comp_set = set(largest)
    comp_adj = {v: [n for n in adj[v] if n in comp_set] for v in largest}
    
    parent = {}
    visited2 = set()
    
    def dfs(v, p):
        visited2.add(v); parent[v] = p
        for nb in comp_adj.get(v, []):
            if nb == p: continue
            if nb in visited2:
                # Extract cycle from back edge
                cycle = [nb]
                cur = v
                while cur != nb:
                    cycle.append(cur)
                    cur = parent.get(cur, nb)
                return cycle
            r = dfs(nb, v)
            if r: return r
        return None
    
    for start in largest:
        if start not in visited2:
            cycle = dfs(start, -1)
            if cycle: return cycle
    return None

print("\nExtracting cycles from cocycles...")
out = []
n_total = len(top_i)
report_every = 5000
t0 = time.time()

for rank, full_i in enumerate(top_i):
    cc = h1c[full_i]
    birth = float(h1[full_i, 0]); death = float(h1[full_i, 1]); p = death - birth
    
    if not hasattr(cc, 'ndim') or cc.ndim < 2 or len(cc) == 0:
        continue
    
    # Extract edges
    edges = [(int(row[0]), int(row[1])) for row in cc]
    edges = list(set((min(u,v), max(u,v)) for u,v in edges))
    
    if len(edges) < 2:
        continue
    
    cycle = extract_cycle(edges)
    if cycle is None or len(cycle) < 3:
        continue
    
    # Map to tokens
    tokens = [tok.decode([int(all_tok[v])]) for v in cycle]
    
    out.append({
        'rank': rank, 'birth': birth, 'death': death, 'persistence': p,
        'n_verts': len(cycle), 'tokens': tokens,
        'decoded': ' '.join(tokens),
    })
    
    if rank % report_every == 0 and rank > 0:
        print(f"  {rank}/{n_total} ({len(out)} cycles found, {time.time()-t0:.0f}s)")

print(f"\nDone: {len(out)} cycles extracted from {n_total} total")

# ── Save ──
pkl_path = DATA_DIR / 'ripser_cocycle_cycles.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(out, f)
print(f"Saved {pkl_path} ({pkl_path.stat().st_size/1024/1024:.1f} MB)")

# Save text (top 5000 by persistence)
txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_order.txt')
with open(txt_path, 'w') as f:
    for c in out:
        f.write(f"pers={c['persistence']:.4f} | n={c['n_verts']:2d} | {c['decoded']}\n")
print(f"Saved {txt_path} ({len(out)} lines)")

# ── Quick stats ──
n_verts_arr = np.array([c['n_verts'] for c in out])
pers_arr = np.array([c['persistence'] for c in out])
print(f"\nStats:")
print(f"  Cycles: {len(out)}")
print(f"  Verts/cycle: min={n_verts_arr.min()}, max={n_verts_arr.max()}, "
      f"mean={n_verts_arr.mean():.1f}, median={np.median(n_verts_arr):.0f}")
print(f"  Persistence: min={pers_arr.min():.4f}, max={pers_arr.max():.4f}, "
      f"mean={pers_arr.mean():.4f}")

# Print most persistent ones
print(f"\n=== Top 30 cycles ===")
for c in sorted(out, key=lambda x: x['persistence'], reverse=True)[:30]:
    n_show = min(15, c['n_verts'])
    print(f"  pers={c['persistence']:.4f} n={c['n_verts']:2d} | "
          f"{' '.join(c['tokens'][:n_show])}{'...' if c['n_verts'] > n_show else ''}")

print("\nDone!")
PYEOF
