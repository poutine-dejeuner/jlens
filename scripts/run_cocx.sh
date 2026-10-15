#!/usr/bin/env bash
#SBATCH --job-name=rip-cocx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/rip_cocx_%j.out
#SBATCH --error=logs/rip_cocx_%j.err

# Full 42K Ripser with do_cocycles=True — extract vertex sequences for all H1 cycles.
# Single-pass: Ripser → parse cocycles → map to tokens → save.
set -euo pipefail
echo "=== Full Cocycle Extraction ==="
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

print(f"Ripser(maxdim=1, thresh={R_THRESH}, do_cocycles=True) on {X.shape[0]} points...")
t0 = time.time()
result = ripser(X, maxdim=1, thresh=R_THRESH, do_cocycles=True)
print(f"  Done: {time.time()-t0:.1f}s")

h1 = result['dgms'][1]
h1c = result['cocycles'][1]
pers = h1[:,1] - h1[:,0]
fin = np.isfinite(h1[:,1]); fin_idx = np.where(fin)[0]
top_order = np.argsort(-pers[fin])
top_i = fin_idx[top_order]  # all sorted by persistence

print(f"H1: {len(h1)} total, {fin.sum()} finite")
print(f"Cocycles: {len(h1c)}")
print(f"Persistence: [{pers[fin].min():.4f}, {pers[fin].max():.4f}]")

from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

# Cycle extraction from edges
def find_cycle(edges):
    verts = set(); [verts.update([u,v]) for u,v in edges]
    if len(verts) < 3: return None
    adj = {v: [] for v in verts}
    for u, v in edges: adj[u].append(v); adj[v].append(u)
    for start in sorted(verts):
        queue = [(start, -1, [start])]
        seen = {start}
        while queue:
            v, p, path = queue.pop(0)
            for nb in adj[v]:
                if nb == p: continue
                if nb in path:
                    idx = path.index(nb)
                    return path[idx:]
                if nb not in seen:
                    seen.add(nb)
                    queue.append((nb, v, path + [nb]))
    return None

print("Extracting cycles...")
out = []
t_start = time.time()
n_parsed = 0

for rank, full_i in enumerate(top_i):
    cc = h1c[full_i]
    birth = float(h1[full_i, 0]); death = float(h1[full_i, 1]); p = death - birth
    
    if not hasattr(cc, 'ndim') or cc.ndim < 2 or len(cc) < 2:
        continue
    
    edges = [(int(r[0]), int(r[1])) for r in cc]
    edges = list(set((min(u,v), max(u,v)) for u,v in edges))
    
    cycle = find_cycle(edges)
    if cycle is None or len(cycle) < 3:
        continue
    
    tokens = [tok.decode([int(all_tok[v])]) for v in cycle]
    
    out.append({
        'rank': rank, 'birth': birth, 'death': death, 'persistence': p,
        'n_verts': len(cycle), 'tokens': tokens,
        'decoded': ' '.join(tokens),
    })
    n_parsed += 1
    
    if rank % 10000 == 0 and rank > 0:
        elapsed = time.time() - t_start
        print(f"  {rank}/{len(top_i)} parsed, {len(out)} cycles ({elapsed:.0f}s)")

print(f"\nDone parsing: {len(out)} cycles from {n_parsed} valid cocycles ({time.time()-t_start:.0f}s)")

# Save
pkl_path = DATA_DIR / 'ripser_cocycle_cycles.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(out, f)
print(f"Saved {pkl_path} ({pkl_path.stat().st_size/1024/1024:.1f} MB)")

txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_order.txt')
with open(txt_path, 'w') as f:
    for c in out:
        f.write(f"pers={c['persistence']:.4f} | n={c['n_verts']:2d} | {c['decoded']}\n")
print(f"Saved {txt_path} ({len(out)} lines)")

if out:
    n_v = [c['n_verts'] for c in out]
    p_arr = [c['persistence'] for c in out]
    print(f"\nStats: {len(out)} cycles")
    print(f"  Verts: min={min(n_v)}, max={max(n_v)}, mean={np.mean(n_v):.1f}")
    print(f"  Persistence: {min(p_arr):.4f} - {max(p_arr):.4f}")

print(f"\n=== Top 30 ===")
for c in sorted(out, key=lambda x: x['persistence'], reverse=True)[:30]:
    n_s = min(20, c['n_verts'])
    print(f"  p={c['persistence']:.4f} n={c['n_verts']:2d} | "
          f"{' '.join(c['tokens'][:n_s])}{'...' if c['n_verts'] > n_s else ''}")

print("\nDone!")
PYEOF
