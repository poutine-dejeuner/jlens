#!/usr/bin/env bash
#SBATCH --job-name=cycle-diam
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=logs/cycle_diam_%j.out
#SBATCH --error=logs/cycle_diam_%j.err
set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import pickle, h5py, json, numpy as np
from pathlib import Path
from numpy.linalg import norm
from collections import defaultdict

DATA_DIR = Path('/data/jlens/qwen3.6-27b')

# Load embeddings (L2-normalized, 5120D)
print("Loading contextual embeddings...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    reps = f['representations'][:]  # 42265 x 5120
print(f"  reps: {reps.shape}")

# Load cycles with token indices
with open(DATA_DIR / 'ripser_h1_full_nothresh.pkl', 'rb') as f:
    ripser_data = pickle.load(f)

cocycles = ripser_data['h1_cocycles']
pg = ripser_data['h1']  # H1 persistence diagram

# Each cocycle may have multiple entries; each is (n_edges, 3) [a,b,coeff]
# We need unique vertices per cycle
print(f"  H1 bars: {len(pg)}, cocycles: {len(cocycles)}")

# Load cycle_tokens_full for the token strings
with open(DATA_DIR / 'cycle_tokens_full.pkl', 'rb') as f:
    cycles = pickle.load(f)
print(f"  cycles with tokens: {len(cycles)}")

# For each cycle, compute:
# 1. Diameter (max pairwise L2 distance in 5120D)
# 2. Mean pairwise distance
# 3. Compare with random points of same size
print("\nComputing diameters...")

def clean(t):
    return t.split('(')[0].strip()

def compute_diameter(vertices):
    """Max pairwise Euclidean distance among vertices."""
    pts = reps[vertices]
    # Use squared distances for efficiency, take sqrt at end
    # ||a-b||^2 = 2 - 2 * dot(a,b) since vectors are L2-normalized
    dots = pts @ pts.T  # n x n
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 0)
    return float(np.sqrt(sq_dists.max()))

def compute_mean_dist(vertices):
    pts = reps[vertices]
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    n = len(vertices)
    # Upper triangle only
    triu = sq_dists[np.triu_indices(n, k=1)]
    return float(np.sqrt(triu.mean()))

results = []
for c in cycles:
    if c['n_vertices'] < 4:
        continue
    
    # Get original vertex indices from the cocycle
    vset = set()
    cc_idx = c.get('cocycle_idx', c['idx'])
    cc = cocycles[cc_idx] if cc_idx < len(cocycles) else None
    if cc is None or not isinstance(cc, np.ndarray) or len(cc) == 0:
        continue
    
    for row in cc:
        vset.add(int(row[0]))
        vset.add(int(row[1]))
    
    vertices = sorted(vset)
    if len(vertices) < 4:
        continue
    
    diam = compute_diameter(vertices)
    mean_d = compute_mean_dist(vertices)
    birth = c['birth']
    death = c['death']
    persistence = c['persistence']
    
    results.append({
        'idx': c['idx'],
        'n_vertices': len(vertices),
        'diameter': diam,
        'mean_dist': mean_d,
        'birth': birth,
        'persistence': persistence,
        'tokens': c['tokens'],
        'vertices': vertices,
    })

results.sort(key=lambda r: -r['diameter'])
print(f"Computed diameters for {len(results)} cycles")

# Report top cycles by diameter
print("\n" + "="*80)
print("TOP 40 CYCLES BY DIAMETER (spread across embedding space)")
print("="*80)
for i, r in enumerate(results[:40]):
    token_preview = ', '.join([clean(t) for t in r['tokens'][:12]])
    print(f"\n{i+1:3d}. idx={r['idx']:5d}  diam={r['diameter']:.4f}  mean_dist={r['mean_dist']:.4f}  "
          f"verts={r['n_vertices']:4d}  birth={r['birth']:.4f}  pers={r['persistence']:.4f}")
    print(f"    tokens: {token_preview}")

# Also: random baseline
print("\n" + "="*80)
print("RANDOM BASELINE: diameters of random point sets")
print("="*80)

N_total = reps.shape[0]
rng = np.random.RandomState(42)
sizes = [5, 10, 20, 50, 100, 200, 500]
for sz in sizes:
    diams = []
    for _ in range(100):
        verts = rng.choice(N_total, sz, replace=False)
        diams.append(compute_diameter(verts))
    diams = np.array(diams)
    print(f"  n={sz:4d}: diam={diams.mean():.4f}±{diams.std():.4f} "
          f"[{diams.min():.4f}, {diams.max():.4f}]")

# Also: for each semantic category, what's the diameter?
print("\n" + "="*80)
print("SEMANTIC CATEGORY DIAMETERS: direct lookup (not from cycles)")
print("="*80)

# Get token -> index mapping
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
# manifest format: {id: "token string (id=N)"}
tok_to_idx = {}
for k, v in manifest.items():
    tok = v.split('(')[0].strip().lower()
    tok_to_idx[tok] = int(k)

categories = {
    'months': ['january','february','march','april','may','june','july',
               'august','september','october','november','december'],
    'days': ['monday','tuesday','wednesday','thursday','friday','saturday','sunday'],
    'directions': ['north','south','east','west'],
    'colors': ['red','blue','green','yellow','orange','purple','pink','brown',
               'black','white','gray'],
    'numbers_1_10': ['one','two','three','four','five','six','seven','eight','nine','ten'],
    'seasons': ['spring','summer','autumn','winter'],
}

for cat, words in categories.items():
    indices = []
    for w in words:
        # Try exact match, then with Ġ prefix
        for variant in [w, 'Ġ'+w, 'Ġ'+w.capitalize()]:
            if variant in tok_to_idx:
                indices.append(tok_to_idx[variant])
                break
    
    if len(indices) >= 3:
        diam = compute_diameter(indices)
        mean_d = compute_mean_dist(indices)
        pts = reps[indices]
        all_dists = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                all_dists.append(float(norm(pts[i] - pts[j])))
        all_dists = np.array(all_dists)
        print(f"\n  {cat} ({len(indices)}/{len(words)} matched):")
        print(f"    diameter={diam:.4f}, mean_dist={mean_d:.4f}")
        print(f"    min_dist={all_dists.min():.4f}, max_dist={all_dists.max():.4f}")
        print(f"    pairs: ")
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                d = float(norm(pts[i] - pts[j]))
                print(f"      {words[i]:>10} <-> {words[j]:<10}: {d:.4f}")

# Also: what's the overall pairwise distance distribution of all tokens?
print("\n" + "="*80)
print("GLOBAL PAIRWISE DISTANCE STATISTICS (L2-normalized sphere)")
print("="*80)
# Sample
n_sample = 5000
sample = rng.choice(N_total, n_sample, replace=False)
pts = reps[sample]
dots = pts @ pts.T
sq_dists = 2.0 - 2.0 * dots
triu = sq_dists[np.triu_indices(n_sample, k=1)]
all_dists = np.sqrt(triu)
print(f"  n={n_sample}: mean={all_dists.mean():.4f} std={all_dists.std():.4f} "
      f"min={all_dists.min():.4f} max={all_dists.max():.4f}")
print(f"  On L2-normalized sphere: max possible = sqrt(2) = {np.sqrt(2):.4f}")

# Report bottom cycles (tightest clusters)
print("\n" + "="*80)
print("TOP 20 TIGHTEST CYCLES (smallest diameter)")
print("="*80)
for i, r in enumerate(results[-20:]):
    token_preview = ', '.join([clean(t) for t in r['tokens'][:12]])
    print(f"\n{i+1:3d}. idx={r['idx']:5d}  diam={r['diameter']:.4f}  mean_dist={r['mean_dist']:.4f}  "
          f"verts={r['n_vertices']:4d}  birth={r['birth']:.4f}  pers={r['persistence']:.4f}")
    print(f"    tokens: {token_preview}")

# Ratio: diameter / mean_dist (how "spread" is the cycle?)
print("\n" + "="*80)
print("CYCLES WITH HIGHEST DIAMETER/MEAN_DIST RATIO (most 'spread out')")
print("="*80)
for r in results:
    r['spread_ratio'] = r['diameter'] / max(r['mean_dist'], 0.01)
results_by_spread = sorted(results, key=lambda r: -r['spread_ratio'])
for i, r in enumerate(results_by_spread[:30]):
    token_preview = ', '.join([clean(t) for t in r['tokens'][:12]])
    print(f"\n{i+1:3d}. idx={r['idx']:5d}  ratio={r['spread_ratio']:.2f}  diam={r['diameter']:.4f}  "
          f"mean={r['mean_dist']:.4f}  verts={r['n_vertices']:4d}")
    print(f"    tokens: {token_preview}")

print("\nDone!")
PYEOF
