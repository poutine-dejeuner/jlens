#!/usr/bin/env bash
#SBATCH --job-name=rip-cocf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/rip_cocf_%j.out
#SBATCH --error=logs/rip_cocf_%j.err

# Full 42K Ripser with do_cocycles=True.
# Uses cocycle seed edges + birth-radius graph expansion to find cycles.
set -euo pipefail
echo "=== Full Cocycle Extraction v2 ==="
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
MIN_VERTICES = 3
MAX_TOP = 10000  # extract top 10K by persistence

print("Loading...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = manifest['token_ids'][:X_raw.shape[0]]
N = X_raw.shape[0]
print(f"  {N} tokens")

print("Preprocessing...")
t0_p = time.time()
X_c = X_raw - X_raw.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
print(f"  Done ({time.time()-t0_p:.1f}s)")

print(f"\nRipser(maxdim=1, thresh={R_THRESH}, do_cocycles=True)...")
t0_r = time.time()
result = ripser(X, maxdim=1, thresh=R_THRESH, do_cocycles=True)
rip_elapsed = time.time() - t0_r
h1 = result['dgms'][1]; h1c = result['cocycles'][1]
pers = h1[:,1] - h1[:,0]; fin = np.isfinite(h1[:,1])
fin_idx = np.where(fin)[0]
top_order = np.argsort(-pers[fin])
top_i = fin_idx[top_order[:MAX_TOP]]
print(f"  Done: {rip_elapsed:.0f}s, H1={len(h1)}, cocycles={len(h1c)}")

from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

def find_longest_cycle(edges, max_search=5000):
    """Find a SIMPLE cycle, preferring longer ones. Uses DFS with backtracking."""
    verts = set(); [verts.update([u,v]) for u,v in edges]
    if len(verts) < 3: return None
    adj = {v: [] for v in verts}
    for u,v in edges: adj[u].append(v); adj[v].append(u)
    
    best = None
    searched = 0
    
    for start in sorted(verts, key=lambda v: len(adj[v]), reverse=True)[:10]:
        if searched > max_search: break
        # DFS to find a cycle
        stack = [(start, -1, [start])]
        while stack and searched < max_search:
            v, parent, path = stack.pop()
            searched += 1
            for nb in adj[v]:
                if nb == parent: continue
                if nb in path:
                    idx = path.index(nb)
                    cycle = path[idx:]
                    if best is None or len(cycle) > len(best):
                        best = cycle
                    if len(cycle) >= 15:  # good enough
                        return cycle
                elif len(path) < 30:  # limit depth
                    stack.append((nb, v, path + [nb]))
    return best

print(f"\nProcessing top {MAX_TOP} cycles...")
out = []
t_s = time.time()
n_processed = 0; n_birth_expanded = 0; n_direct = 0; n_skipped = 0

for rank, full_i in enumerate(top_i):
    cc = h1c[full_i]
    birth = float(h1[full_i, 0]); death = float(h1[full_i, 1]); p = death - birth
    
    if not hasattr(cc, 'ndim') or cc.ndim < 2:
        n_skipped += 1; continue
    
    # STEP 1: Try direct cocycle edges
    seed_edges = [(int(r[0]), int(r[1])) for r in cc]
    seed_edges = list(set((min(u,v), max(u,v)) for u,v in seed_edges))
    
    cycle = find_longest_cycle(seed_edges)
    
    if cycle is not None and len(cycle) >= MIN_VERTICES:
        n_direct += 1
    else:
        # STEP 2: Expand using birth-radius graph among seed vertices
        seed_verts = set()
        for u,v in seed_edges: seed_verts.add(u); seed_verts.add(v)
        sv_list = list(seed_verts)
        
        # Add edges between seed vertices if distance < birth
        expanded_edges = list(seed_edges)
        for i in range(len(sv_list)):
            for j in range(i+1, len(sv_list)):
                u, v = sv_list[i], sv_list[j]
                if np.linalg.norm(X[u] - X[v]) < birth:
                    expanded_edges.append((u, v))
        
        expanded_edges = list(set((min(u,v), max(u,v)) for u,v in expanded_edges))
        
        if len(expanded_edges) > len(seed_edges):
            n_birth_expanded += 1
            cycle = find_longest_cycle(expanded_edges)
    
    if cycle is None or len(cycle) < MIN_VERTICES:
        n_skipped += 1; continue
    
    tokens = [tok.decode([int(all_tok[v])]) for v in cycle]
    out.append({
        'rank': rank, 'birth': birth, 'death': death,
        'persistence': p, 'n_verts': len(cycle),
        'tokens': tokens, 'decoded': ' '.join(tokens),
    })
    n_processed += 1
    
    if rank % 1000 == 0 and rank > 0:
        elapsed = time.time() - t_s
        rate = rank / elapsed
        eta = (MAX_TOP - rank) / rate / 60 if rate > 0 else 0
        print(f"  {rank}/{MAX_TOP}: {len(out)} cycles "
              f"(direct={n_direct}, expanded={n_birth_expanded}, skip={n_skipped}) "
              f"[{elapsed:.0f}s, ~{eta:.0f}min left]")

elapsed = time.time() - t_s
print(f"\nDone: {len(out)} cycles in {elapsed:.0f}s")
print(f"  direct={n_direct}, birth-expanded={n_birth_expanded}, skipped={n_skipped}")

# Save
pkl_path = DATA_DIR / 'ripser_cocycle_cycles.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(out, f)
mb = pkl_path.stat().st_size / 1024 / 1024
print(f"Saved {pkl_path} ({mb:.1f} MB)")

txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_order.txt')
with open(txt_path, 'w') as f:
    for c in out:
        f.write(f"pers={c['persistence']:.4f} | n={c['n_verts']:2d} | {c['decoded']}\n")
print(f"Saved {txt_path} ({len(out)} lines)")

if out:
    n_v = [c['n_verts'] for c in out]
    p_arr = [c['persistence'] for c in out]
    print(f"\nStats: {len(out)} cycles")
    print(f"  Verts: min={min(n_v)}, max={max(n_v)}, mean={np.mean(n_v):.1f}, median={np.median(n_v):.0f}")
    print(f"  Persistence: {min(p_arr):.4f} - {max(p_arr):.4f}")

print("\n=== Top 40 ===")
for c in sorted(out, key=lambda x: x['persistence'], reverse=True)[:40]:
    ns = min(20, c['n_verts'])
    print(f"  p={c['persistence']:.4f} n={c['n_verts']:2d} | "
          f"{' '.join(c['tokens'][:ns])}{'...' if c['n_verts'] > ns else ''}")

total_time = time.time() - t0_p
print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")
print("Done!")
PYEOF
