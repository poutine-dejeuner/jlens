#!/usr/bin/env bash
#SBATCH --job-name=day-rips
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/day_rips_%j.out
#SBATCH --error=logs/day_rips_%j.err

# Local homology: compute Ripser on the ball neighborhood containing
# the 7 days-of-week tokens (contextual, full 5120D → L2 norm).
# Check if the 7-day cycle is captured as an H1 feature.
set -euo pipefail
echo "=== Local Homology: Days of Week Ball ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, json, h5py, pickle, time
from pathlib import Path
from numpy.linalg import norm
from transformers import AutoTokenizer

DATA_DIR = Path('/data/jlens/qwen3.6-27b')

print("Loading contextual data...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)

with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = np.array(manifest['token_ids'][:X_raw.shape[0]])

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
id_to_token = {v: k for k, v in tok.get_vocab().items()}

# ── Find day tokens ──
days_canon = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
day_positions = []
for day in days_canon:
    for i, tid in enumerate(all_tok):
        clean = id_to_token[int(tid)].lstrip('Ġ').lower()
        if clean == day:
            day_positions.append(i)
            break
print(f"Found {len(day_positions)}/7 canonical day tokens")

# ── L2 normalize ──
X_norm = X_raw / norm(X_raw, axis=1, keepdims=True)
day_norm = X_norm[day_positions]
center = day_norm.mean(axis=0)
radii = norm(day_norm - center, axis=1)
r_max = radii.max()
print(f"Day ball: center_norm={norm(center):.3f}, max_radius={r_max:.4f}")

# ── Extract ball neighborhood at 2× radius ──
MARGIN = 2.0
dists = norm(X_norm - center, axis=1)
ball_mask = dists <= r_max * MARGIN
ball_indices = np.where(ball_mask)[0]
print(f"Ball ({MARGIN}x radius): {len(ball_indices)} tokens")

if len(ball_indices) < 10:
    print("Too few tokens — aborting")
    exit(1)

X_ball = X_norm[ball_indices]  # (N, 5120)

# ── PCA to manageable dimension ──
# Ripser on 5120D is impossible. Use PCA to d=10.
print("PCA to 10D...")
X_ctr = X_ball - X_ball.mean(axis=0)
_, S, Vt = np.linalg.svd(X_ctr, full_matrices=False)
X_10d = X_ctr @ Vt[:10].T
# Re-normalize after PCA
X_10d = X_10d / norm(X_10d, axis=1, keepdims=True)
print(f"  PCA 10D shape: {X_10d.shape}")

# ── Compute distance matrix ──
print("Computing distance matrix...")
from scipy.spatial.distance import pdist, squareform
dm = squareform(pdist(X_10d, 'euclidean'))

# Find threshold: max distance between any two day tokens
day_local_indices = [np.where(ball_indices == dp)[0][0] for dp in day_positions]
day_dm = dm[np.ix_(day_local_indices, day_local_indices)]
max_day_dist = day_dm.max()
print(f"Max distance between day tokens: {max_day_dist:.4f}")
print(f"Thresh for Ripser: {max_day_dist * 1.2:.4f}")

# ── Run Ripser ──
print("Running Ripser (maxdim=2)...")
from ripser import ripser
t0 = time.time()
diagrams = ripser(
    dm,
    maxdim=2,
    thresh=max_day_dist * 1.2,
    distance_matrix=True,
    do_cocycles=True,
)
dt = time.time() - t0
print(f"Ripser done in {dt:.1f}s")

# ── Analyze ──
for dim in [0, 1, 2]:
    dgm = diagrams['dgms'][dim]
    if len(dgm) == 0:
        print(f"H{dim}: no features")
    else:
        lifetimes = dgm[:, 1] - dgm[:, 0]
        finite = lifetimes < np.inf
        real = dgm[finite]
        print(f"H{dim}: {len(real)} features "
              f"(max lifetime={lifetimes[finite].max():.4f}, "
              f"total={len(dgm)})")
        if dim == 1 and len(real) > 0:
            # Show top 5 by persistence
            top5 = real[np.argsort(-lifetimes[finite])[:min(5, len(real))]]
            print(f"  Top 5 H1 (birth, death, persistence):")
            for b, d in top5:
                print(f"    ({b:.4f}, {d:.4f}, {d-b:.4f})")

# ── Extract cocycles for top H1 features ──
if 'cocycles' in diagrams and 1 in diagrams['cocycles']:
    cocycles = diagrams['cocycles'][1]
    finite_mask = (diagrams['dgms'][1][:, 1] < np.inf)
    finite_cocycles = [c for c, m in zip(cocycles, finite_mask) if m]
    lifetimes = diagrams['dgms'][1][finite_mask][:, 1] - diagrams['dgms'][1][finite_mask][:, 0]
    
    if finite_cocycles:
        # Sort by persistence
        sorted_idx = np.argsort(-lifetimes)
        print(f"\nCocycles for top H1 features:")
        for rank in range(min(5, len(finite_cocycles))):
            idx = sorted_idx[rank]
            cc = finite_cocycles[idx]
            lt = lifetimes[idx]
            print(f"\n  Rank {rank+1}: persistence={lt:.4f}")
            print(f"  Cocycle shape: {cc.shape}")
            # Show first 10 edges
            for i in range(min(10, len(cc))):
                a, b, coeff = cc[i]
                tok_a = id_to_token[int(all_tok[ball_indices[a]])]
                tok_b = id_to_token[int(all_tok[ball_indices[b]])]
                print(f"    {a}→{b} ({tok_a} → {tok_b}) coeff={coeff}")

# ── Show ball contents ──
print(f"\nToken types in ball:")
from collections import Counter
ball_tokens = [id_to_token[int(all_tok[ball_indices[i]])] for i in range(len(ball_indices))]
ball_strings = [t.lstrip('Ġ').lower() for t in ball_tokens]
counter = Counter(ball_strings)
for token_str, count in counter.most_common(30):
    print(f"  {token_str}: {count}")

# Save
result = {
    'ball_indices': ball_indices,
    'ball_tokens': ball_tokens,
    'diagrams': {dim: diagrams['dgms'][dim] for dim in [0,1,2]},
    'day_local_indices': day_local_indices,
    'r_max': r_max,
    'margin': MARGIN,
    'n_ball': len(ball_indices),
    'max_day_dist': max_day_dist,
}
out_path = DATA_DIR / 'day_local_homology.pkl'
with open(out_path, 'wb') as f:
    pickle.dump(result, f)
print(f"\nSaved {out_path} ({out_path.stat().st_size/1024/1024:.1f} MB)")
print("Done!")
PYEOF
