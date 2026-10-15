#!/usr/bin/env bash
#SBATCH --job-name=circ-sweep2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/circ_sweep2_%j.out
#SBATCH --error=logs/circ_sweep2_%j.err

# Large-scale circularity sweep v2.
# For each token string (grouped by decoded string, ignoring BPE prefix),
# measure how well its embeddings form a circle using 2D PCA + circular statistics.
set -euo pipefail
echo "=== Circularity Sweep v2 ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, json, pickle, h5py
from pathlib import Path
from transformers import AutoTokenizer
import time

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
MIN_TOKENS = 4  # minimum 4 points needed to define a circle

print("Loading vocabulary...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
id_to_token = {v: k for k, v in tok.get_vocab().items()}

print("Loading contextual embeddings...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = np.array(manifest['token_ids'][:X_raw.shape[0]])

# ── Build concept index ──
print("Building concept index...")
concept_positions = {}
for i, tid in enumerate(all_tok):
    s = id_to_token[int(tid)].lstrip('Ġ').lower()
    if s:
        concept_positions.setdefault(s, []).append(i)

candidates = [(s, positions) for s, positions in concept_positions.items()
              if len(positions) >= MIN_TOKENS]
candidates.sort(key=lambda x: -len(x[1]))
print(f"  {len(candidates)} concepts with >= {MIN_TOKENS} tokens")
print(f"  Max occurrences: {len(candidates[0][1]) if candidates else 0}")

def circularity_score(X):
    """
    Measure how circular a set of points is.
    
    Method:
    1. PCA to 2D
    2. Convert to angles: theta_i = arctan2(y_i, x_i)
    3. Compute mean resultant length R = |mean(exp(i*theta_i))|
       R = 1 means perfect circle (points evenly spread around 0-2π)
       R = 0 means random scatter (no circular structure)
    4. Also compute angular spread: how much of the circle is covered
    5. Combined: R * spread
    
    Returns: score in [0, 1], plus diagnostics
    """
    n = len(X)
    if n < 4:
        return 0.0, {'R': 0, 'spread': 0, 'max_gap': 0, 'n_large_gaps': 0, 'n': n}
    
    # PCA to 2D
    X_ctr = X - X.mean(axis=0)
    if np.linalg.matrix_rank(X_ctr) < 2:
        return 0.0, {'R': 0, 'spread': 0, 'max_gap': 0, 'n_large_gaps': 0, 'n': n}
    
    _, _, Vt = np.linalg.svd(X_ctr, full_matrices=False)
    X_2d = X_ctr @ Vt[:2].T
    
    # Angles
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    angles = np.sort(angles)
    
    # Mean resultant length: |1/n * sum(exp(i*theta))|
    R = np.abs(np.mean(np.exp(1j * angles)))
    
    # Angular spread: fraction of [0, 2π) covered by sorted angles
    # Include wrap-around gap
    gaps = np.diff(angles)
    wrap_gap = 2*np.pi - (angles[-1] - angles[0])
    all_gaps = np.append(gaps, wrap_gap)
    
    # Normalize gaps
    max_gap = all_gaps.max()
    spread = 1.0 - max_gap / (2*np.pi)  # 1 = evenly spread, 0 = all in one cluster
    
    # Count large gaps (> 90 degrees = pi/2)
    n_large_gaps = (all_gaps > np.pi/2).sum()
    
    # Combined score:
    # R measures circular symmetry
    # spread measures full-circle coverage
    # Penalize having large gaps
    score = R * spread * (1.0 - 0.5 * min(n_large_gaps / n, 1.0))
    
    return float(score), {
        'R': float(R), 
        'spread': float(spread),
        'max_gap': float(max_gap),
        'n_large_gaps': n_large_gaps,
        'n': n,
    }

print(f"\nComputing circularity for {len(candidates)} concepts...")
t0 = time.time()
results = []

for idx, (token_str, positions) in enumerate(candidates):
    X_tokens = X_raw[positions]
    score, diag = circularity_score(X_tokens)
    
    # Get angular ordering
    X_ctr = X_tokens - X_tokens.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_ctr, full_matrices=False)
    X_2d = X_ctr @ Vt[:2].T
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    order_idx = np.argsort(angles)
    
    # Get context around each occurrence (for interpretability)
    contexts = []
    for pos in positions:
        ctx_start = max(0, pos - 2)
        ctx_end = min(len(all_tok), pos + 3)
        ctx_tokens = [id_to_token[int(all_tok[i])] for i in range(ctx_start, ctx_end)]
        contexts.append(''.join(ctx_tokens))
    
    results.append({
        'token': token_str,
        'count': len(positions),
        'score': score,
        'R': diag['R'],
        'spread': diag['spread'],
        'max_gap': diag['max_gap'],
        'n_large_gaps': diag['n_large_gaps'],
        'positions': positions,
        'contexts': contexts,
    })
    
    if (idx + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (len(candidates) - idx) / rate / 60
        top = max(r['score'] for r in results)
        print(f"  {idx+1}/{len(candidates)} [{elapsed:.0f}s, ~{eta:.0f}min] "
              f"best={top:.3f}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

# Sort by score
results.sort(key=lambda x: -x['score'])

# ── Output ──
print(f"\n{'='*80}")
print(f"Top 100 most circular concepts")
print(f"{'='*80}")
print(f"{'Rank':<6}{'Score':<8}{'R':<8}{'Spread':<8}{'N':<6}{'Token'}")
print("-"*60)
for i, r in enumerate(results[:100]):
    print(f"{i+1:<6}{r['score']:<8.3f}{r['R']:<8.3f}{r['spread']:<8.3f}"
          f"{r['count']:<6}{r['token']}")

# Show context for top 20
print(f"\n{'='*80}")
print(f"Top 20 concepts with context")
print(f"{'='*80}")
for r in results[:20]:
    print(f"\n  {r['token']} (score={r['score']:.3f}, R={r['R']:.3f}, "
          f"spread={r['spread']:.3f}, n={r['count']}):")
    
    # Re-sort contexts by angular order
    positions = r['positions']
    X_tokens = X_raw[positions]
    X_ctr = X_tokens - X_tokens.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_ctr, full_matrices=False)
    X_2d = X_ctr @ Vt[:2].T
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    order_idx = np.argsort(angles)
    
    for j, oi in enumerate(order_idx):
        ctx = r['contexts'][oi]
        print(f"    {j}: ...{ctx}...")

# Save
pkl_path = DATA_DIR / 'circularity_sweep.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved {pkl_path} ({pkl_path.stat().st_size/1024/1024:.1f} MB)")

# Text file
txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/circularity_sweep.txt')
with open(txt_path, 'w') as f:
    f.write("Circularity Sweep — Qwen 27B Last-Layer Contextual Embeddings\n")
    f.write("="*70 + "\n")
    f.write(f"Method: 2D PCA → mean resultant length R * angular spread\n")
    f.write(f"Concepts: {len(results)} token strings with >= {MIN_TOKENS} occurrences\n\n")
    f.write(f"{'Rank':<6}{'Score':<8}{'R':<8}{'Spread':<8}{'N':<6}{'Token'}\n")
    f.write("-"*50 + "\n")
    for i, r in enumerate(results):
        f.write(f"{i+1:<6}{r['score']:<8.3f}{r['R']:<8.3f}{r['spread']:<8.3f}"
                f"{r['count']:<6}{r['token']}\n")
    
    # Top 50 with context
    f.write(f"\n\nTop 50 with Angular Context\n")
    f.write("="*70 + "\n")
    for r in results[:50]:
        f.write(f"\n{r['token']} (score={r['score']:.3f}, R={r['R']:.3f}, "
                f"spread={r['spread']:.3f}, n={r['count']}):\n")
        
        positions = r['positions']
        X_tokens = X_raw[positions]
        X_ctr = X_tokens - X_tokens.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_ctr, full_matrices=False)
        X_2d = X_ctr @ Vt[:2].T
        angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
        order_idx = np.argsort(angles)
        
        for j, oi in enumerate(order_idx):
            ctx = r['contexts'][oi]
            f.write(f"  {j}: ...{ctx}...\n")

print(f"Saved {txt_path}")
print("Done!")
PYEOF
