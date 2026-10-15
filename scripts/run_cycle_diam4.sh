#!/usr/bin/env bash
#SBATCH --job-name=cycle-diam4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:45:00
#SBATCH --output=logs/cycle_diam4_%j.out
#SBATCH --error=logs/cycle_diam4_%j.err
set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import pickle, h5py, json, numpy as np
from pathlib import Path

DATA_DIR = Path('/data/jlens/qwen3.6-27b')

print("Loading contextual embeddings...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    reps = f['representations'][:]

# Check normalization
norms = np.linalg.norm(reps, axis=1)
print(f"  reps shape: {reps.shape}, dtype: {reps.dtype}")
print(f"  norms: mean={norms.mean():.4f}, std={norms.std():.4f}, "
      f"min={norms.min():.4f}, max={norms.max():.4f}")

# L2-normalize (handle zero-norm rows)
nz = norms > 0
print(f"  zero-norm rows: {(~nz).sum()}")
reps = reps.copy()
reps[nz] = reps[nz] / norms[nz, np.newaxis]
reps[~nz] = 0  # zero vectors stay zero
norms2 = np.linalg.norm(reps, axis=1)
print(f"  after normalize: norms mean={norms2[nz].mean():.4f}, min={norms2[nz].min():.4f}")
N_total = reps.shape[0]

print("Loading cycles...")
with open(DATA_DIR / 'ripser_h1_full_nothresh.pkl', 'rb') as f:
    rip = pickle.load(f)
    cocycles = rip['h1_cocycles']
    h1 = rip['h1']

with open(DATA_DIR / 'cycle_tokens_full.pkl', 'rb') as f:
    cycles = pickle.load(f)

def clean(t):
    return t.split('(')[0].strip()

def compute_diameter(vertices):
    pts = reps[vertices]
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    np.fill_diagonal(sq_dists, 0)
    # Clamp tiny negatives
    sq_dists = np.clip(sq_dists, 0, None)
    return float(np.sqrt(sq_dists.max()))

def compute_mean_dist(vertices):
    pts = reps[vertices]
    dots = pts @ pts.T
    sq_dists = 2.0 - 2.0 * dots
    sq_dists = np.clip(sq_dists, 0, None)
    n = len(vertices)
    triu = sq_dists[np.triu_indices(n, k=1)]
    return float(np.sqrt(triu.mean()))

print("Computing diameters...")
results = []
for c in cycles:
    idx = c['idx']
    cc = cocycles[idx]
    if not isinstance(cc, np.ndarray) or len(cc) == 0:
        continue
    
    vset = set()
    for row in cc:
        vset.add(int(row[0]))
        vset.add(int(row[1]))
    vertices = sorted(vset)
    
    if len(vertices) < 4:
        continue
    
    diam = compute_diameter(vertices)
    mean_d = compute_mean_dist(vertices)
    
    results.append({
        'idx': idx,
        'n_vertices': len(vertices),
        'diameter': diam,
        'mean_dist': mean_d,
        'birth': c['birth'],
        'death': c['death'],
        'persistence': c['persistence'],
        'tokens': c['tokens'],
    })

results.sort(key=lambda r: -r['diameter'])
print(f"Computed diameters for {len(results)} cycles")

# Report
print("\n" + "="*80)
print("TOP 50 CYCLES BY DIAMETER")
print("="*80)
for i, r in enumerate(results[:50]):
    token_preview = ', '.join([clean(t) for t in r['tokens'][:15]])
    print(f"\n{i+1:3d}. idx={r['idx']:5d}  diam={r['diameter']:.4f}  mean={r['mean_dist']:.4f}  "
          f"verts={r['n_vertices']:4d}  birth={r['birth']:.4f}  pers={r['persistence']:.4f}")
    print(f"    {token_preview}")

# Random baseline (L2-normalized)
print("\n" + "="*80)
print("RANDOM BASELINE (L2-normalized sphere)")
print("="*80)
rng = np.random.RandomState(42)
for sz in [5, 10, 20, 50, 100, 200, 500, 1000]:
    diams = []
    for _ in range(100):
        verts = rng.choice(N_total, min(sz, N_total), replace=False)
        diams.append(compute_diameter(verts))
    diams = np.array(diams)
    print(f"  n={sz:4d}: diam={diams.mean():.4f}±{diams.std():.4f}  "
          f"[{diams.min():.4f}, {diams.max():.4f}]")

# Semantic categories direct lookup
print("\n" + "="*80)
print("SEMANTIC CATEGORY DIAMETERS (direct lookup)")
print("="*80)

with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
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
    'numbers': ['one','two','three','four','five','six','seven','eight','nine','ten'],
    'seasons': ['spring','summer','autumn','winter'],
}

for cat, words in categories.items():
    indices = []
    found_words = []
    for w in words:
        for variant in [w, 'Ġ'+w, 'Ġ'+w.capitalize()]:
            if variant in tok_to_idx:
                indices.append(tok_to_idx[variant])
                found_words.append(w)
                break
    if len(indices) >= 3:
        pts = reps[indices]
        all_pairs = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                all_pairs.append(d)
        all_pairs = np.array(all_pairs)
        diam = compute_diameter(indices)
        mean_d = compute_mean_dist(indices)
        print(f"\n  {cat} ({len(indices)}/{len(words)}): diam={diam:.4f}, mean={mean_d:.4f}")
        print(f"    min={all_pairs.min():.4f}, max={all_pairs.max():.4f}, "
              f"max_possible=sqrt(2)={np.sqrt(2):.4f}")
        # Print all pairs sorted by distance
        pairs = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                pairs.append((found_words[i], found_words[j], float(np.linalg.norm(pts[i] - pts[j]))))
        pairs.sort(key=lambda x: -x[2])
        for w1, w2, d in pairs:
            print(f"      {w1:>10} <-> {w2:<10}: {d:.4f}")

# Global stats
print("\n" + "="*80)
print("GLOBAL CONTEXT")
print("="*80)
n_sample = 5000
sample = rng.choice(N_total, n_sample, replace=False)
pts_sample = reps[sample]
dots = pts_sample @ pts_sample.T
sq_dists = 2.0 - 2.0 * dots
sq_dists = np.clip(sq_dists, 0, None)
triu = sq_dists[np.triu_indices(n_sample, k=1)]
all_dists = np.sqrt(triu)
print(f"  All pairs ({n_sample} sample): mean={all_dists.mean():.4f}±{all_dists.std():.4f}")
print(f"  Max possible = sqrt(2) = {np.sqrt(2):.4f}")

# Cycle diameter distribution
di = np.array([r['diameter'] for r in results])
print(f"\n  Cycle diameters: mean={di.mean():.4f}±{di.std():.4f}, "
      f"min={di.min():.4f}, max={di.max():.4f}")
nv = [r['n_vertices'] for r in results]
print(f"  Cycle n_vertices: mean={np.mean(nv):.1f}, min={min(nv)}, max={max(nv)}")

# How many cycles have diameter > 1.0 (large spread)?
large = sum(1 for r in results if r['diameter'] > 1.0)
print(f"  Cycles with diam > 1.0: {large}/{len(results)} ({100*large/len(results):.1f}%)")

# Top cycles by spread_ratio
print("\n" + "="*80)
print("TOP 30 BY SPREAD RATIO (diam/mean_dist)")
print("="*80)
for r in results:
    r['spread_ratio'] = r['diameter'] / max(r['mean_dist'], 0.001)
results_by_spread = sorted(results, key=lambda r: -r['spread_ratio'])
for i, r in enumerate(results_by_spread[:30]):
    token_preview = ', '.join([clean(t) for t in r['tokens'][:12]])
    print(f"\n{i+1:3d}. idx={r['idx']:5d}  ratio={r['spread_ratio']:.2f}  "
          f"diam={r['diameter']:.4f}  mean={r['mean_dist']:.4f}  "
          f"verts={r['n_vertices']:4d}")
    print(f"    {token_preview}")

print("\nDone!")
PYEOF
