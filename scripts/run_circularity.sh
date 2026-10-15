#!/usr/bin/env bash
#SBATCH --job-name=circ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/circ_%j.out
#SBATCH --error=logs/circ_%j.err

# Direct circularity test: measure if known cyclic concepts form circles
# in embedding space. No homology needed — just distance geometry.
set -euo pipefail
echo "=== Circularity Analysis ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, json, pickle
from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = Path('/data/jlens/qwen3.6-27b')

print("Loading data...")
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = np.array(manifest['token_ids'])

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
vocab = tok.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}

# Define cyclic concepts with their natural order
CYCLIC_CONCEPTS = {
    "days_of_week": {
        "order": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
        "words": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                  "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
                  "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
                  "mon", "tue", "wed", "thu", "fri", "sat", "sun"],
    },
    "months": {
        "order": ["january", "february", "march", "april", "may", "june",
                   "july", "august", "september", "october", "november", "december"],
        "words": ["January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December",
                  "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    },
    "cardinal_directions": {
        "order": ["north", "east", "south", "west"],
        "words": ["north", "south", "east", "west",
                  "North", "South", "East", "West"],
    },
    "seasons": {
        "order": ["spring", "summer", "autumn", "winter"],
        "words": ["spring", "summer", "autumn", "winter", "fall",
                  "Spring", "Summer", "Autumn", "Winter", "Fall"],
    },
    "colors_rainbow": {
        "order": ["red", "orange", "yellow", "green", "blue", "indigo", "violet"],
        "words": ["red", "orange", "yellow", "green", "blue", "indigo", "violet",
                  "Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet",
                  "purple", "pink", "white", "black", "brown", "gray", "grey",
                  "Purple", "Pink", "White", "Black", "Brown", "Gray", "Grey"],
    },
    "greek_letters": {
        "order": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
        "words": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
                  "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"],
    },
    "planets": {
        "order": ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune"],
        "words": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    },
}

def get_token_embeddings(all_tok, id_to_token, vocab, X, target_word, max_per=10):
    """Find all token positions matching target_word (substring in decoded form)."""
    target_lower = target_word.lower()
    positions = []
    for i, tid in enumerate(all_tok):
        if len(positions) >= max_per:
            break
        token_str = id_to_token[int(tid)]
        if target_lower in token_str.lower():
            positions.append(i)
    return positions

def circularity_score(X_sub, natural_order_indices):
    """
    Measure how well a set of points forms a circle in the natural order.
    Returns: circularity metric (0=not circular, 1=perfect circle with natural order).
    
    Method: compute the "cumulative angle" around the 2D PCA projection.
    If the points form a circle in natural order, their angular progression
    should be monotonic.
    """
    n = len(X_sub)
    if n < 4: return 0.0
    
    # PCA to 2D
    X_c = X_sub - X_sub.mean(0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    X_2d = X_c @ Vt[:2].T
    
    # Compute angles
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    # For points with natural_order_indices, check if angles are monotonic
    # (allowing wrap-around)
    ordered_angles = angles[natural_order_indices]
    
    # Normalize to [0, 2π)
    ordered_angles = ordered_angles % (2 * np.pi)
    
    # Compute angular differences between consecutive elements
    diffs = np.diff(ordered_angles)
    # Adjust for wrap-around
    diffs = np.where(diffs > np.pi, diffs - 2*np.pi, diffs)
    diffs = np.where(diffs < -np.pi, diffs + 2*np.pi, diffs)
    
    # All diffs should have the same sign (monotonic)
    signs = np.sign(diffs)
    n_pos = (signs > 0).sum()
    n_neg = (signs < 0).sum()
    
    # Circularity = proportion of diffs going in the dominant direction
    agreement = max(n_pos, n_neg) / len(diffs) if len(diffs) > 0 else 0
    
    # Also check: total angular span should be close to 2π
    total_span = np.abs(diffs.sum())
    span_score = min(total_span, 2*np.pi) / (2*np.pi)
    
    # Combined score
    return (agreement * 0.5 + span_score * 0.5)

def best_matching_order(X_sub, word_indices, word_names):
    """
    Find the ordering of word_indices that best matches the circular structure.
    Returns: (ordered_word_names, circularity)
    """
    # PCA to 2D
    X_c = X_sub - X_sub.mean(0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    X_2d = X_c @ Vt[:2].T
    
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    
    # Sort by angle
    sorted_idx = np.argsort(angles)
    ordered_names = [word_names[i] for i in sorted_idx]
    
    return ordered_names

print(f"\n{'='*70}")
print("Loading embeddings per layer...")
print(f"{'='*70}")

# Load raw embeddings for last layer
with __import__('h5py').File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)

results = {}

for name, concept in CYCLIC_CONCEPTS.items():
    natural_order = concept['order']
    
    print(f"\n{'─'*60}")
    print(f"{name}")
    print(f"  Natural order: {' → '.join(natural_order)}")
    
    # For each layer (just last layer for now), extract embeddings
    X = X_raw
    
    # Find token positions for each word in natural order
    word_positions = {}
    for w in natural_order:
        positions = get_token_embeddings(all_tok, id_to_token, vocab, X, w, max_per=5)
        if positions:
            word_positions[w] = positions
    
    all_positions = []
    point_word_map = []
    for w, positions in word_positions.items():
        for p in positions:
            all_positions.append(p)
            point_word_map.append(w)
    
    if len(all_positions) < 3:
        print(f"  Only {len(all_positions)} tokens found, skipping")
        continue
    
    # Get embeddings
    X_sub = X[all_positions]
    
    # PCA to 2D
    X_c = X_sub - X_sub.mean(0)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    X_2d = X_c @ Vt[:2].T
    
    # Compute circular ordering by angle
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    sorted_idx = np.argsort(angles)
    
    ordered_words = [point_word_map[i] for i in sorted_idx]
    
    # Deduplicate consecutive identical words
    dedup = []
    for w in ordered_words:
        if not dedup or w != dedup[-1]:
            dedup.append(w)
    
    print(f"  Found: {', '.join(set(point_word_map))}")
    print(f"  Circular order: {' → '.join(dedup)}")
    
    # Score: how many pairs in natural order appear consecutively in circular order?
    natural_pairs = [(natural_order[i], natural_order[(i+1) % len(natural_order)])
                     for i in range(len(natural_order))]
    
    circular_pairs = [(dedup[i], dedup[(i+1) % len(dedup)]) 
                      for i in range(len(dedup))]
    
    matches = 0
    for np in natural_pairs:
        if np in circular_pairs or (np[1], np[0]) in circular_pairs:
            matches += 1
    
    # Also check reverse order
    reverse_pairs = [(dedup[(i+1) % len(dedup)], dedup[i]) for i in range(len(dedup))]
    rev_matches = 0
    for np in natural_pairs:
        if np in reverse_pairs or (np[1], np[0]) in reverse_pairs:
            rev_matches += 1
    
    total_expected = len(natural_pairs)
    match_score = max(matches, rev_matches) / total_expected if total_expected > 0 else 0
    
    print(f"  Match score: {match_score:.2f} ({max(matches, rev_matches)}/{total_expected} consecutive pairs)")
    
    results[name] = {
        'natural_order': natural_order,
        'circular_order': dedup,
        'found_words': list(set(point_word_map)),
        'match_score': match_score,
        'n_tokens': len(all_positions),
    }

# Summary
print(f"\n{'='*70}")
print(f"Summary")
print(f"{'='*70}")
for name, r in sorted(results.items()):
    score_str = f"{r['match_score']:.2f}" 
    marker = "✓" if r['match_score'] > 0.5 else ("·" if r['match_score'] > 0.2 else "✗")
    print(f"  {marker} {name:25s} score={score_str}  order={' → '.join(r['circular_order'][:8])}")
    print(f"    natural: {' → '.join(r['natural_order'])}")

pkl_path = DATA_DIR / 'circularity.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved {pkl_path}")

print("\nDone!")
PYEOF
