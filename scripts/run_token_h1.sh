#!/usr/bin/env bash
#SBATCH --job-name=token-h1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1:30:00
#SBATCH --output=logs/token_h1_%j.out
#SBATCH --error=logs/token_h1_%j.err

# Full-token H1 persistence with cycle representatives on contextual last-layer embeddings.
# Uses Ripser for fast H1 diagram, Dionysus for cycle representatives (edge chains).

set -euo pipefail
echo "=== Token H1 Persistence: Qwen3.6-27B contextual last layer ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, pickle, json, warnings
from pathlib import Path
from ripser import ripser
import dionysus as d
warnings.filterwarnings('ignore')

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10
R_THRESH = 1.2   # Ripser threshold (all H1 deaths < 1.0 for this data)
D_THRESH = 1.2   # Dionysus Rips threshold (finite, same as Ripser)

# ── Load contextual representations (last layer only) ──
print("Loading contextual representations...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
print(f"  Loaded: {X_raw.shape} ({X_raw.nbytes / 1024**2:.0f} MB)")

# ── Load token manifest ──
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_token_ids = manifest['token_ids'][:X_raw.shape[0]]
print(f"  Token IDs: {len(all_token_ids)}")

# ── Preprocess: center → PCA → L2 normalize ──
print("Preprocessing (center → PCA 10D → L2)...")
t0 = time.time()
X_c = X_raw - X_raw.mean(axis=0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X_proj = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X_proj = X_proj / (np.linalg.norm(X_proj, axis=1, keepdims=True) + 1e-8)
print(f"  Preprocessed: {X_proj.shape} ({time.time()-t0:.1f}s)")
print(f"  Top 5 singular values: {s[:5]}")

# ── Ripser: H1 persistence (fast with finite threshold) ──
print(f"\nRipser(maxdim=1, thresh={R_THRESH}) on {X_proj.shape[0]} points...")
t0 = time.time()
result = ripser(X_proj, maxdim=1, thresh=R_THRESH)
dgms = result['dgms']
elapsed = time.time() - t0
print(f"  Done: {elapsed:.1f}s, H0={len(dgms[0])}, H1={len(dgms[1])}")
print(f"  Persistence range: birth=[{dgms[1][:,0].min():.3f}, {dgms[1][:,0].max():.3f}], "
      f"death=[{dgms[1][:,1].min():.3f}, {dgms[1][:,1].max():.3f}]")

# Filter to top most persistent cycles
persistence = dgms[1][:,1] - dgms[1][:,0]
top_n = min(100, len(persistence))
top_idx = np.argsort(-persistence)[:top_n]
print(f"  Top {top_n} persistence range: [{persistence[top_idx].min():.3f}, {persistence[top_idx].max():.3f}]")

# ── Dionysus: build complex, get cycle representatives ──
# Only do this for a subset of the most persistent cycles
print(f"\nDionysus Rips(maxdim=2, thresh={D_THRESH}) on {X_proj.shape[0]} points...")
t0 = time.time()
pts = [tuple(float(x) for x in row) for row in X_proj]
# Note: fill_rips is verbose (C++ prints). Redirect stdout.
import sys, io
old_stdout = sys.stdout
sys.stdout = io.StringIO()
f = d.fill_rips(pts, 2, D_THRESH)
sys.stdout = old_stdout
elapsed_build = time.time() - t0
n_simplices = sum(1 for _ in f)
print(f"  Complex built: {elapsed_build:.1f}s, {n_simplices} simplices")

t0 = time.time()
m = d.homology_persistence(f)
dgms_d = d.init_diagrams(m, f)
elapsed = time.time() - t0
print(f"  Persistence computed: {elapsed:.1f}s")
print(f"  H1 cycles: {len(dgms_d[1])}")

# Extract cycle representatives for persistent cycles
print("\nExtracting cycle representatives...")
cycle_data = []
for i, pt in enumerate(dgms_d[1]):
    birth, death = pt.birth, pt.death
    pers = death - birth
    
    if np.isinf(death):
        continue
    
    try:
        chain = m[pt.data]
    except Exception:
        continue
    
    if chain is None:
        continue
    
    # Collect edges (1-simplices) from the chain
    edges = []
    vertices_set = set()
    for entry in chain:
        idx = entry.index
        try:
            vs = [d for d in f[idx] if d.dimension() == 1]
        except:
            continue
        for v in vs:
            # vertex v is a 1-simplex (edge): get its boundary (0-simplices)
            bdry = f[v.index]
            if bdry.dimension() == 1:
                verts = [s.index for s in bdry.boundary(f)]
                edges.append(tuple(sorted(verts)))
                vertices_set.update(verts)
    
    if not edges:
        continue
    
    # Map vertex indices to original token IDs
    token_indices = sorted(vertices_set)
    token_ids = [int(all_token_ids[ti]) for ti in token_indices]
    
    cycle_data.append({
        'birth': float(birth),
        'death': float(death),
        'persistence': float(pers),
        'n_vertices': len(vertices_set),
        'n_edges': len(edges),
        'vertex_indices': token_indices,
        'token_ids': token_ids,
    })

# Sort by persistence
cycle_data.sort(key=lambda x: -x['persistence'])
print(f"  Extracted {len(cycle_data)} cycle representatives")

# ── Save ──
results = {
    'source': 'contextual_last_layer',
    'model_id': 'Qwen/Qwen3.6-27B',
    'n_tokens': X_proj.shape[0],
    'proj_dim': PROJ_DIM,
    'ripser_thresh': R_THRESH,
    'dionysus_thresh': D_THRESH,
    'ripser_h0_h1': [len(dgms[0]), len(dgms[1])],
    'ripser_time_s': elapsed,
    'dionysus_h1': len(dgms_d[1]),
    'dionysus_build_time_s': elapsed_build,
    'dionysus_time_s': elapsed,
    'cycles': cycle_data,
    'master_token_ids': all_token_ids,
}

outfile = DATA_DIR / 'token_h1_contextual_last.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved to {outfile}")
print(f"File size: {outfile.stat().st_size / 1024 / 1024:.1f} MB")

# ── Show top cycles with decoded tokens ──
print("\n=== Top 10 most persistent H1 cycles ===")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
for i, c in enumerate(cycle_data[:10]):
    tokens = tok.decode(c['token_ids'])
    print(f"\nCycle {i}: {c['n_vertices']} vertices, {c['n_edges']} edges, "
          f"persistence={c['persistence']:.4f}")
    print(f"  Tokens: {tokens[:200]}")

print("\nDone!")
PYEOF

echo ""
echo "=== Done ==="
echo "Date: $(date)"
