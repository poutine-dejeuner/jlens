#!/usr/bin/env bash
#SBATCH --job-name=rip-gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=1:30:00
#SBATCH --output=logs/rip_gen_%j.out
#SBATCH --error=logs/rip_gen_%j.err

# Ripser H1 persistence + cocycle generators on full contextual 42K set.
# Uses return_generators=True for cocycle representatives (vertex indices).

set -euo pipefail
echo "=== Ripser H1 + Generators: Qwen3.6-27B contextual ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, pickle, json, warnings
from pathlib import Path
from ripser import ripser
warnings.filterwarnings('ignore')

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10
R_THRESH = 1.2

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

# ── Ripser with generators ──
print(f"\nRipser(maxdim=1, thresh={R_THRESH}, do_cocycles=True) on {X_proj.shape[0]} points...")
t0 = time.time()
result = ripser(X_proj, maxdim=1, thresh=R_THRESH, do_cocycles=True)
elapsed = time.time() - t0
dgms = result['dgms']
gens = result.get('cocycles')
print(f"  Done: {elapsed:.1f}s, H0={len(dgms[0])}, H1={len(dgms[1])}")
if gens is not None:
    print(f"  Cocycles: H0={len(gens[0]) if gens[0] is not None else 'None'}, "
          f"H1={len(gens[1]) if gens[1] is not None else 'None'}")
else:
    print("  No cocycles returned")

# ── Analyze top generators ──
persistence = dgms[1][:,1] - dgms[1][:,0]
fin_mask = np.isfinite(dgms[1][:,1])
pers_fin = persistence[fin_mask]
top_n = min(200, len(pers_fin))
top_order = np.argsort(-pers_fin)[:top_n]
# Map from finite index to full index
fin_indices = np.where(fin_mask)[0]
top_full_idx = fin_indices[top_order]

print(f"\nTop {top_n} cycles (persistence range: [{pers_fin[top_order].min():.4f}, "
      f"{pers_fin[top_order].max():.4f}])")

# Extract vertices from generators
from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

# Ripser generators are cocycle representatives: each generator is a list of
# (simplex, coefficient) pairs. For H1, simplices are edges (vertex pairs).
# For cocycles, generators are lists of edges indexed by vertex IDs.
print("\nExtracting vertex indices from H1 generators...")

cycles = []
for rank, full_idx in enumerate(top_full_idx[:100]):  # top 100
    birth = dgms[1][full_idx, 0]
    death = dgms[1][full_idx, 1]
    pers = death - birth
    
    if gens[1] is None or full_idx >= len(gens[1]):
        continue
    
    gen = gens[1][full_idx]
    if gen is None or len(gen) == 0:
        continue
    
    # gen is a sparse array of coefficients for edges
    # Each nonzero entry corresponds to an edge
    if hasattr(gen, 'nonzero'):
        coords = gen.nonzero()
        edge_indices = np.unique(coords[0]) if len(coords) > 0 else np.array([])
    elif hasattr(gen, 'items'):
        edge_indices = np.array([k for k, v in gen.items() if v != 0])
    else:
        edge_indices = np.array(gen).flatten()
    
    if len(edge_indices) == 0:
        continue
    
    # For Ripser, the generator edges are indexed within the coface data structure.
    # We need to extract actual vertex pairs. For maxdim=1, we can use the
    # distance matrix to find vertex pairs at the birth radius.
    # Simpler: use Ripser's internal sparse distance matrix structure.
    
    # Actually, Ripser generators for H1 (cocycles) are indexed differently.
    # Let's try: gen is typically a 1D sparse array where each index corresponds
    # to a specific edge in the complex. We need to map these to vertex pairs.
    # This requires accessing Ripser's internal edge list, which ripser.py doesn't expose.
    
    cycles.append({
        'rank': rank,
        'birth': float(birth),
        'death': float(death),
        'persistence': float(pers),
        'generator_size': len(edge_indices),
        'note': 'generator_indices_not_mapped_to_vertices_yet',
    })

print(f"\nExtracted {len(cycles)} generator entries")
print(f"Note: Ripser's return_generators gives cocycle indices, not direct vertex pairs.")
print(f"These need further processing to map to actual tokens.")

# ── Alternative: extract vertex neighborhoods from the birth radius ──
# For each persistent cycle, get all points within its birth radius
# from a reference point. This gives interpretable token clusters.
print("\n\n=== Alternative: birth-radius neighborhoods ===")
print("(Simpler approach to token-interpretable cycles)")

# For top cycles, find all points within birth radius of a seed point
token_cycles = []
for rank, full_idx in enumerate(top_full_idx[:50]):
    birth = dgms[1][full_idx, 0]
    death = dgms[1][full_idx, 1]
    pers = death - birth
    
    # Pick a seed point
    seed = np.random.randint(0, len(X_proj))
    
    # Find all points within birth radius
    dists = np.linalg.norm(X_proj - X_proj[seed], axis=1)
    nearby = dists < birth
    n_nearby = nearby.sum()
    
    if n_nearby < 3 or n_nearby > 100:
        continue
    
    vert_idx = np.where(nearby)[0]
    token_ids = [int(all_token_ids[i]) for i in vert_idx]
    decoded = tok.decode(token_ids)
    
    token_cycles.append({
        'rank': rank,
        'birth': float(birth),
        'death': float(death),
        'persistence': float(pers),
        'n_tokens': n_nearby,
        'token_ids': token_ids,
        'decoded': decoded,
    })
    
    print(f"\n  Cycle {rank} (pers={pers:.4f}, birth={birth:.4f}): {n_nearby} tokens")
    print(f"    {decoded[:200]}")

# ── Save ──
results = {
    'source': 'contextual_last_layer_full',
    'n_tokens': X_proj.shape[0],
    'proj_dim': PROJ_DIM,
    'thresh': R_THRESH,
    'ripser_time_s': elapsed,
    'h0_count': int(len(dgms[0])),
    'h1_count': int(len(dgms[1])),
    'h1_finite': int(fin_mask.sum()),
    'generators': cycles,
    'token_neighborhoods': token_cycles,
    'ripser_dgms': [dgms[0].tolist(), dgms[1].tolist()],
}

outfile = DATA_DIR / 'ripser_h1_neighborhoods.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved to {outfile}")
print(f"Size: {outfile.stat().st_size / 1024:.1f} KB")

print("\nDone!")
PYEOF

echo ""
echo "=== Done ==="
echo "Date: $(date)"
