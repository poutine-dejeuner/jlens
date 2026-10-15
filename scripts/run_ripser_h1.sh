#!/usr/bin/env bash
#SBATCH --job-name=rip-h1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rip_h1_%j.out
#SBATCH --error=logs/rip_h1_%j.err

# Ripser-only H1 persistence on full contextual 42K set.
# Uses finite threshold (1.2) — all H1 cycles close within this range.
# Then runs Dionysus on smaller representative subsets for cycle extraction.

set -euo pipefail
echo "=== Ripser H1 + Token Mapping: Qwen3.6-27B contextual ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, pickle, json, warnings, sys, io
from pathlib import Path
from ripser import ripser
import dionysus as d
warnings.filterwarnings('ignore')

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10
R_THRESH = 1.2
D_THRESH = 1.2

# ── Load ──
print("Loading contextual representations...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
print(f"  {X_raw.shape}")

with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_token_ids = manifest['token_ids'][:X_raw.shape[0]]

# ── Preprocess ──
print("Preprocessing...")
t0 = time.time()
X_c = X_raw - X_raw.mean(axis=0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X_proj = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X_proj = X_proj / (np.linalg.norm(X_proj, axis=1, keepdims=True) + 1e-8)
print(f"  Done ({time.time()-t0:.1f}s)")

# ── Ripser ──
print(f"\nRipser(maxdim=1, thresh={R_THRESH}) on {X_proj.shape[0]} points...")
t0 = time.time()
result = ripser(X_proj, maxdim=1, thresh=R_THRESH)
dgms = result['dgms']
elapsed = time.time() - t0
print(f"  Done: {elapsed:.1f}s, H0={len(dgms[0])}, H1={len(dgms[1])}")

# ── Save Ripser results ──
ripser_out = {
    'n_tokens': X_proj.shape[0],
    'proj_dim': PROJ_DIM,
    'thresh': R_THRESH,
    'time_s': elapsed,
    'h0_diagram': dgms[0].tolist(),
    'h1_diagram': dgms[1].tolist(),
    'token_ids': all_token_ids,
}
with open(DATA_DIR / 'ripser_h1_contextual.pkl', 'wb') as f:
    pickle.dump(ripser_out, f)
print(f"  Saved H1 diagram ({len(dgms[1])} cycles)")

# ── Top persistent cycles: localize points ──
# For each of the top-100 most persistent H1 cycles, find which points are nearby
# and run Dionysus on just that sub-cloud.
persistence = dgms[1][:,1] - dgms[1][:,0]
# Filter infinite deaths
fin_idx = np.isfinite(dgms[1][:,1])
pers_fin = persistence[fin_idx]
top_n = min(100, len(pers_fin))
top_idx = np.argsort(-pers_fin)[:top_n]

print(f"\nTop {top_n} H1 cycles: persistence range "
      f"[{pers_fin[top_idx].min():.4f}, {pers_fin[top_idx].max():.4f}]")

# For each top cycle, find candidate vertices at its birth radius level
cycles = []
from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

# Strategy: for each top cycle with birth radius r, sample up to N_LOCAL=2000
# points within radius r*2 of a seed point, run Dionysus on those
N_LOCAL = 2000
MAX_CYCLES = 50  # limit to avoid long runtime

for cycle_rank, idx in enumerate(top_idx[:MAX_CYCLES]):
    birth = dgms[1][idx, 0]
    death = dgms[1][idx, 1]
    pers = death - birth
    print(f"\n  Cycle {cycle_rank}: birth={birth:.4f}, death={death:.4f}, pers={pers:.4f}")
    
    # Find a seed point: pick random point, build local neighborhood
    seed = np.random.randint(0, len(X_proj))
    dists = np.linalg.norm(X_proj - X_proj[seed], axis=1)
    # Points within 2*birth of seed
    local_mask = dists < 2.0 * max(birth, 0.2)
    n_local = local_mask.sum()
    
    if n_local < 5 or n_local > N_LOCAL:
        # Try larger/smaller radius
        for mult in [1.5, 2.5, 3.0, 0.5]:
            local_mask = dists < mult * max(birth, 0.2)
            n_local = local_mask.sum()
            if 5 <= n_local <= N_LOCAL:
                break
    
    if n_local < 5 or n_local > N_LOCAL:
        local_mask = dists < 0.5
        n_local = local_mask.sum()
        if n_local < 5:
            # Just take the N_LOCAL nearest
            local_idx = np.argsort(dists)[:min(N_LOCAL, len(X_proj))]
            local_mask = np.zeros(len(X_proj), dtype=bool)
            local_mask[local_idx] = True
            n_local = local_mask.sum()
    
    print(f"    Local neighborhood: {n_local} points (radius={dists[local_mask].max():.3f})")
    
    X_local = X_proj[local_mask]
    local_indices = np.where(local_mask)[0]
    
    # Run Dionysus on local neighborhood
    # Suppress verbose C++ simplex output
    import os
    old_stderr = os.dup(2)
    os.close(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        f = d.fill_rips(X_local, 2, D_THRESH)
    except Exception as e:
        print(f"    Dionysus fill_rips failed: {e}")
        os.close(devnull)
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        continue
    os.close(devnull)
    os.dup2(old_stderr, 2)
    os.close(old_stderr)
    
    try:
        m = d.homology_persistence(f)
        dgms_d = d.init_diagrams(m, f)
    except Exception as e:
        print(f"    Dionysus persistence failed: {e}")
        continue
    
    # Find cycles matching our birth/death
    for pt in dgms_d[1]:
        if np.isinf(pt.death):
            continue
        if abs(pt.birth - birth) > 0.01:
            continue
        try:
            chain = m[pt.data]
        except:
            continue
        if chain is None:
            continue
        
        # Extract vertices
        edges = []
        verts_set = set()
        for entry in chain:
            idx = entry.index
            try:
                vs = [v for v in f[idx] if v.dimension() == 1]
            except:
                continue
            for v in vs:
                bdry = f[v.index]
                if bdry.dimension() == 1:
                    vv = [s.index for s in bdry.boundary(f)]
                    edges.append(tuple(sorted(vv)))
                    verts_set.update(vv)
        
        if not edges:
            continue
        
        # Map back to global token indices
        vert_token_idx = [int(local_indices[v]) for v in sorted(verts_set)]
        vert_token_ids = [int(all_token_ids[ti]) for ti in vert_token_idx]
        decoded = tok.decode(vert_token_ids)
        
        cycles.append({
            'rank': cycle_rank,
            'birth': float(birth),
            'death': float(death),
            'persistence': float(pers),
            'n_vertices': len(verts_set),
            'n_edges': len(edges),
            'vertex_indices': vert_token_idx,
            'token_ids': vert_token_ids,
            'decoded': decoded,
        })
        print(f"    → {len(verts_set)} vertices: {decoded[:150]}")
        break  # one cycle per rank

print(f"\nExtracted {len(cycles)} token-mapped cycles")

# ── Save ──
results = {
    'source': 'contextual_last_layer_full',
    'model_id': 'Qwen/Qwen3.6-27B',
    'n_tokens': X_proj.shape[0],
    'proj_dim': PROJ_DIM,
    'thresh': R_THRESH,
    'ripser_time_s': elapsed,
    'h1_total': len(dgms[1]),
    'h1_finite': int(np.isfinite(dgms[1][:,1]).sum()),
    'cycles': cycles,
}

outfile = DATA_DIR / 'ripser_h1_cycles.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved to {outfile}")
print(f"Size: {outfile.stat().st_size / 1024:.1f} KB")

print("\n=== Top Cycles ===")
for c in cycles[:15]:
    print(f"  Rank {c['rank']}: {c['n_vertices']} vertices, pers={c['persistence']:.4f}")
    print(f"    {c['decoded'][:200]}")

print("\nDone!")
PYEOF

echo ""
echo "=== Done ==="
echo "Date: $(date)"
