#!/usr/bin/env bash
#SBATCH --job-name=circ-sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=logs/circ_sweep_%j.out
#SBATCH --error=logs/circ_sweep_%j.err

# Large-scale circularity sweep: test 1000+ concepts from Qwen vocabulary.
# For each concept, subsample tokens N times, measure angular ordering
# consistency. Rank by self-consistency score.
set -euo pipefail
echo "=== Circularity Sweep ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, json, pickle, h5py
from pathlib import Path
from transformers import AutoTokenizer
from collections import Counter
import time

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
N_SUBSAMPLES = 10
SUBSAMPLE_SIZE = 10  # how many tokens per subsample

print("Loading vocabulary...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
id_to_token = {v: k for k, v in tok.get_vocab().items()}

print(f"Loading contextual embeddings...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = np.array(manifest['token_ids'][:X_raw.shape[0]])

# ── Build concept index ──
# Group by decoded token string (strip BPE prefix Ġ, lowercase)
# A "concept" = all positions where the decoded string matches.
print("Building concept index...")
concept_positions = {}  # string → list of positions
for i, tid in enumerate(all_tok):
    s = id_to_token[int(tid)].lstrip('Ġ').lower()
    if s not in concept_positions:
        concept_positions[s] = []
    concept_positions[s].append(i)

# Filter: need at least SUBSAMPLE_SIZE occurrences
candidates = [(s, len(positions)) for s, positions in concept_positions.items()
              if len(positions) >= SUBSAMPLE_SIZE]
candidates.sort(key=lambda x: -x[1])

print(f"  {len(candidates)} candidate concepts (appearing >= {SUBSAMPLE_SIZE} times)")

# Limit to top N by frequency
N_CONCEPTS = min(2000, len(candidates))
candidates = candidates[:N_CONCEPTS]
print(f"  Using top {N_CONCEPTS} concepts")

def circularity_score(X_tokens):
    """
    Measure how well a set of token embeddings forms a circle.
    Returns (score, consistency_stdev).
    
    Method: subsample N_SUBSAMPLES times, compute angular ordering each time.
    For each pair of consecutive concept-tokens in the full ordering,
    measure how often they appear consecutively across subsamples.
    """
    n = len(X_tokens)
    if n < SUBSAMPLE_SIZE:
        return 0.0, 0.0
    
    # Get the "ground truth" ordering from ALL tokens
    X_ctr = X_tokens - X_tokens.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_ctr, full_matrices=False)
    X_2d = X_ctr @ Vt[:2].T
    angles_full = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    full_order = np.argsort(angles_full)
    
    # Subsample and measure consistency
    pair_counts = {}
    total_pairs = 0
    
    for _ in range(N_SUBSAMPLES):
        indices = np.random.choice(n, min(SUBSAMPLE_SIZE, n), replace=False)
        X_sub = X_tokens[indices]
        
        X_ctr_sub = X_sub - X_sub.mean(axis=0)
        if np.linalg.matrix_rank(X_ctr_sub) < 2:
            continue  # degenerate, skip
        
        _, _, Vt_sub = np.linalg.svd(X_ctr_sub, full_matrices=False)
        X_2d_sub = X_ctr_sub @ Vt_sub[:2].T
        angles_sub = np.arctan2(X_2d_sub[:, 1], X_2d_sub[:, 0])
        sub_order = indices[np.argsort(angles_sub)]
        
        # Record consecutive pairs in this subsample ordering
        for i in range(len(sub_order)):
            a, b = sub_order[i], sub_order[(i+1) % len(sub_order)]
            pair = (min(a,b), max(a,b))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            total_pairs += 1
    
    if total_pairs == 0:
        return 0.0, 0.0
    
    # Score: mean consistency = average count/total_subsamples
    # The score is: for each pair that should be adjacent, how often is it?
    # Actually: the max count represents the most consistent adjacency.
    # We also want variance across pairs.
    if not pair_counts:
        return 0.0, 0.0
    
    counts = np.array(list(pair_counts.values()))
    max_possible = N_SUBSAMPLES
    mean_consistency = counts.mean() / max_possible
    
    # Bonus: if all pairs are equally consistent (low variance), the circle is real
    # Penalize high variance (some pairs always together, others never)
    cv = counts.std() / (counts.mean() + 1e-8)  # coefficient of variation
    
    # Combined score: high mean consistency, low cv
    score = mean_consistency * (1.0 - min(cv, 1.0) * 0.5)
    
    return score, cv

print(f"\nComputing circularity for {N_CONCEPTS} concepts...")
print(f"  ({N_SUBSAMPLES} subsamples of {SUBSAMPLE_SIZE} tokens each)")
print()

results = []
t0 = time.time()

for idx, (token_str, cnt) in enumerate(candidates):
    positions = concept_positions[token_str]
    if len(positions) < SUBSAMPLE_SIZE:
        continue
    
    X_tokens = X_raw[positions]
    
    score, cv = circularity_score(X_tokens)
    
    results.append({
        'token': token_str,
        'count': cnt,
        'score': score,
        'cv': cv,
        'positions': positions,
    })
    
    if (idx + 1) % 200 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        eta = (N_CONCEPTS - idx) / rate / 60
        print(f"  {idx+1}/{N_CONCEPTS} [{elapsed:.0f}s, ~{eta:.0f}min remaining] "
              f"best_so_far={results[-1]['score']:.3f}")

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

# Sort by score
results.sort(key=lambda x: -x['score'])

# ── Output ──
print(f"\n{'='*70}")
print(f"Top 50 most circular concepts")
print(f"{'='*70}")
for i, r in enumerate(results[:50]):
    print(f"  {i+1:3d}. score={r['score']:.3f} cv={r['cv']:.2f} "
          f"({r['count']:4d}×) | {r['token']}")

# Also output the full angular order for top concepts
print(f"\n{'='*70}")
print(f"Angular orderings for top 10")
print(f"{'='*70}")
for r in results[:10]:
    positions = r['positions']
    X_tokens = X_raw[positions]
    X_ctr = X_tokens - X_tokens.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_ctr, full_matrices=False)
    X_2d = X_ctr @ Vt[:2].T
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    order_idx = np.argsort(angles)
    
    # Show the token string at each position — but these are all the same token!
    # Instead show context words near each occurrence
    contexts = []
    for pos in positions[order_idx][:15]:
        # Get 2 tokens before and after
        ctx_start = max(0, pos - 2)
        ctx_end = min(len(all_tok), pos + 3)
        ctx_tokens = [id_to_token[int(all_tok[i])] for i in range(ctx_start, ctx_end)]
        contexts.append(''.join(ctx_tokens))
    
    print(f"\n  {r['token']} (score={r['score']:.3f}, {r['count']}×):")
    for c in contexts:
        print(f"    ...{c}...")

# Save
pkl_path = DATA_DIR / 'circularity_sweep.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved {pkl_path} ({pkl_path.stat().st_size/1024/1024:.1f} MB)")

# Text summary
txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/circularity_sweep.txt')
with open(txt_path, 'w') as f:
    f.write("Circularity Sweep Results\n")
    f.write("="*70 + "\n")
    f.write(f"Method: 2D PCA angular ordering, {N_SUBSAMPLES} subsamples of {SUBSAMPLE_SIZE} tokens\n")
    f.write(f"Concepts tested: {N_CONCEPTS}\n\n")
    f.write(f"{'Rank':<6}{'Score':<8}{'CV':<8}{'Count':<8}{'Token'}\n")
    f.write("-"*50 + "\n")
    for i, r in enumerate(results[:200]):
        f.write(f"{i+1:<6}{r['score']:<8.3f}{r['cv']:<8.2f}{r['count']:<8}{r['token']}\n")
    
    # Also angular orderings for top 20
    f.write(f"\n\nAngular Orderings (top 20)\n")
    f.write("="*70 + "\n")
    for r in results[:20]:
        positions = r['positions']
        X_tokens = X_raw[positions]
        X_ctr = X_tokens - X_tokens.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_ctr, full_matrices=False)
        X_2d = X_ctr @ Vt[:2].T
        angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
        order_idx = np.argsort(angles)
        
        f.write(f"\n{r['token']} (score={r['score']:.3f}, {r['count']}×):\n")
        for pos in positions[order_idx][:20]:
            ctx_start = max(0, pos - 3)
            ctx_end = min(len(all_tok), pos + 4)
            ctx_tokens = [id_to_token[int(all_tok[i])] for i in range(ctx_start, ctx_end)]
            f.write(f"  ...{''.join(ctx_tokens)}...\n")

print(f"Saved {txt_path}")
print("Done!")
PYEOF
