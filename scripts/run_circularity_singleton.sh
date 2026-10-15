#!/usr/bin/env bash
#SBATCH --job-name=circ-sng
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/circ_sng_%j.out
#SBATCH --error=logs/circ_sng_%j.err

# Circularity sweep on singleton (248K token) PCA projections.
# For each decoded token string, check if its 2D PCA projection forms a circle.
set -euo pipefail
echo "=== Circularity Sweep — Singleton ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, json, pickle, h5py, time
from pathlib import Path

DATA_DIR = Path('/data/jlens/qwen3.6-27b')

print("Loading singleton PCA data...")
with h5py.File(DATA_DIR / 'singleton' / 'manifold.h5', 'r') as f:
    X_pca = f['eigenvectors'][:].astype(np.float64)  # (248320, 50)
print(f"  Shape: {X_pca.shape}")

print("Loading manifest...")
with open(DATA_DIR / 'singleton' / 'token_manifest.json') as f:
    manifest = json.load(f)
decoded = manifest['decoded']
print(f"  {len(decoded)} entries")

# ── Build concept index ──
# Group token IDs by their decoded string (strip BPE Ġ prefix, lowercase)
print("Building concept index...")
concept_tids = {}
for tid, d in enumerate(decoded):
    key = d.lstrip('Ġ').lower()
    if key:
        if key not in concept_tids:
            concept_tids[key] = []
        concept_tids[key].append(tid)

# Filter: concepts with at least 3 tokens (need >=3 points for circle)
MIN_POINTS = 3
candidates = sorted(
    [(s, tids) for s, tids in concept_tids.items() if len(tids) >= MIN_POINTS],
    key=lambda x: -len(x[1])
)
print(f"  {len(candidates)} concepts with >= {MIN_POINTS} tokens")
print(f"  Max tokens/concept: {len(candidates[0][1]) if candidates else 0}")

# Use first 2 PCs globally
X_2d_global = X_pca[:, :2]

def circularity_score(angles):
    """Measure circularity from sorted angles.
    
    Returns: R (mean resultant length), spread (fraction of circle covered),
             n_large_gaps, combined_score
    """
    n = len(angles)
    if n < 3:
        return 0.0, 0.0, 0, 0.0
    
    # Sort angles
    angles = np.sort(angles)
    
    # Mean resultant length R
    R = np.abs(np.mean(np.exp(1j * angles)))
    
    # Angular gaps
    gaps = np.diff(angles)
    wrap_gap = 2*np.pi - (angles[-1] - angles[0])
    all_gaps = np.append(gaps, wrap_gap)
    
    max_gap = all_gaps.max()
    spread = 1.0 - max_gap / (2*np.pi)
    
    # Large gaps > 90 degrees
    n_large = (all_gaps > np.pi/2).sum()
    
    # Combined: high R, high spread, few large gaps
    score = R * spread * (1.0 - 0.5 * min(n_large / max(n, 1), 1.0))
    
    return float(R), float(spread), int(n_large), float(score)

print(f"\nComputing circularity for {len(candidates)} concepts...")
t0 = time.time()
results = []

for idx, (token_str, tids) in enumerate(candidates):
    # Get 2D coordinates
    xy = X_2d_global[tids]
    angles = np.arctan2(xy[:, 1], xy[:, 0])
    R, spread, n_large, score = circularity_score(angles)
    
    # Get angular ordering
    order = np.argsort(angles)
    ordered_tids = [tids[o] for o in order]
    ordered_strings = [decoded[tid] for tid in ordered_tids]
    
    results.append({
        'token': token_str,
        'count': len(tids),
        'score': score,
        'R': R,
        'spread': spread,
        'n_large_gaps': n_large,
        'ordered_tokens': ordered_strings,
    })
    
    if (idx + 1) % 5000 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(candidates) - idx) / rate / 60
        top = max(r['score'] for r in results)
        print(f"  {idx+1}/{len(candidates)} [{elapsed:.0f}s, ~{eta:.1f}min] "
              f"best={top:.3f}")

elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s ({elapsed/60:.1f}min)")

# Sort
results.sort(key=lambda x: -x['score'])

# ── Output ──
print(f"\n{'='*80}")
print(f"Top 100 most circular concepts (Singleton PCA, 2D)")
print(f"{'='*80}")
print(f"{'Rank':<6}{'Score':<8}{'R':<8}{'Spread':<8}{'N':<6}{'Token'}{'':<10}Ordered")
print("-"*100)
for i, r in enumerate(results[:100]):
    ordered = ' → '.join(r['ordered_tokens'][:8])
    print(f"{i+1:<6}{r['score']:<8.3f}{r['R']:<8.3f}{r['spread']:<8.3f}"
          f"{r['count']:<6}{r['token']:<12}{ordered}")

# Top 20 detail
print(f"\n{'='*80}")
print(f"Top 20 detail")
print(f"{'='*80}")
for r in results[:20]:
    print(f"\n  {r['token']} (score={r['score']:.3f}, R={r['R']:.3f}, "
          f"spread={r['spread']:.3f}, n={r['count']}):")
    print(f"    Ordered: {' → '.join(r['ordered_tokens'])}")

# Save
pkl_path = DATA_DIR / 'circularity_singleton.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved {pkl_path} ({pkl_path.stat().st_size/1024/1024:.1f} MB)")

# Text
txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/circularity_singleton.txt')
with open(txt_path, 'w') as f:
    f.write("Circularity Sweep — Singleton (Qwen 27B, 248K tokens, 2D PCA)\n")
    f.write("="*70 + "\n")
    f.write(f"Method: global 2D PCA, mean resultant length R × angular spread\n\n")
    for i, r in enumerate(results):
        ordered = ' → '.join(r['ordered_tokens'][:10])
        f.write(f"{i+1:<6}{r['score']:<8.3f}{r['count']:<8}"
                f"{r['token']:<16}{ordered}\n")
print(f"Saved {txt_path}")
print("Done!")
PYEOF
