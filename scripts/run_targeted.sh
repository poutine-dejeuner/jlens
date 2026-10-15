#!/usr/bin/env bash
#SBATCH --job-name=rip-targ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rip_targ_%j.out
#SBATCH --error=logs/rip_targ_%j.err

# Targeted homology: compute H1 cycles on semantic token subsets
# (days, months, colors, directions, etc.)
# Then extract ordered token sequences from the persistence diagram.
set -euo pipefail
echo "=== Targeted Homology Cycles ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, json, pickle
from pathlib import Path
from ripser import ripser
from collections import deque
from transformers import AutoTokenizer

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10

print("Loading data...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = np.array(manifest['token_ids'][:X_raw.shape[0]])

tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
vocab = tok.get_vocab()
id_to_token = {v: k for k, v in vocab.items()}

# Preprocess embeddings
X_c = X_raw - X_raw.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

# ── Candidate semantic cycles ──
# Define sets of words/tokens that should form cycles
CANDIDATES = {
    "days_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                      "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "months": ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December",
               "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "colors": ["red", "blue", "green", "yellow", "orange", "purple", "pink",
               "white", "black", "brown", "gray", "grey", "cyan", "magenta",
               "Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Pink",
               "White", "Black", "Brown", "Gray", "Grey"],
    "directions": ["north", "south", "east", "west",
                   "North", "South", "East", "West",
                   "northeast", "northwest", "southeast", "southwest",
                   "left", "right", "up", "down",
                   "Left", "Right", "Up", "Down"],
    "seasons": ["spring", "summer", "autumn", "winter",
                "Spring", "Summer", "Autumn", "Winter", "Fall", "fall"],
    "numbers_small": ["zero", "one", "two", "three", "four", "five", "six", "seven",
                      "eight", "nine", "ten",
                      "Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
                      "Eight", "Nine", "Ten"],
    "continents": ["Africa", "Europe", "Asia", "America", "Australia", "Antarctica"],
    "planets": ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"],
    "musical_notes": ["C", "D", "E", "F", "G", "A", "B", "do", "re", "mi", "fa", "sol", "la", "ti"],
    "greek_letters": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"],
    "metals": ["gold", "silver", "bronze", "iron", "copper", "steel", "lead", "zinc", "tin",
               "Gold", "Silver", "Bronze", "Iron", "Copper", "Steel", "Lead", "Zinc", "Tin"],
    "sizes": ["tiny", "small", "medium", "large", "huge", "giant", "enormous",
              "Tiny", "Small", "Medium", "Large", "Huge", "Giant", "Enormous"],
    "emotions": ["happy", "sad", "angry", "fear", "surprise", "disgust", "joy", "love",
                 "Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Joy", "Love",
                 "happiness", "sadness", "anger"],
    "family": ["mother", "father", "son", "daughter", "brother", "sister",
               "Mother", "Father", "Son", "Daughter", "Brother", "Sister",
               "aunt", "uncle", "cousin", "grandmother", "grandfather"],
}

print(f"\n{'='*80}")
print(f"Targeted homology on {len(X)} contextual tokens (last layer, 10D PCA+L2 norm)")
print(f"{'='*80}")

results = {}

for name, words in CANDIDATES.items():
    # Find token IDs for these words
    word_ids = []
    for w in words:
        tid = vocab.get(w)
        if tid is not None:
            # Find all occurrences in the contextual token list
            positions = np.where(all_tok == tid)[0]
            if len(positions) > 0:
                word_ids.extend(positions[:20])  # cap per word
    
    word_ids = list(set(word_ids))
    
    if len(word_ids) < 3:
        print(f"\n{name}: only {len(word_ids)} tokens found, skipping")
        continue
    
    # Extract embeddings for these tokens
    X_sub = X[word_ids]
    n = len(word_ids)
    
    print(f"\n{name}: {len(words)} words → {n} token occurrences")
    
    # Compute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dists[i,j] = dists[j,i] = np.linalg.norm(X_sub[i] - X_sub[j])
    
    # Check circularity: compute mean distance to nearest neighbor
    # for each point (should be lower than average pairwise distance)
    nn_dists = []
    for i in range(n):
        others = [dists[i,j] for j in range(n) if j != i]
        nn_dists.append(min(others))
    
    avg_pairwise = dists.sum() / (n * (n-1))
    avg_nn = np.mean(nn_dists)
    
    print(f"  Avg pairwise dist: {avg_pairwise:.4f}, Avg NN dist: {avg_nn:.4f}")
    
    # Run Ripser on this subset
    try:
        t0 = time.time()
        result = ripser(X_sub, maxdim=1, do_cocycles=True)
        h1 = result['dgms'][1]
        h1c = result['cocycles'][1]
        pers = h1[:,1] - h1[:,0]
        fin = np.isfinite(h1[:,1])
        n_fin = fin.sum()
        
        print(f"  Ripser: {time.time()-t0:.1f}s, H0={len(result['dgms'][0])}, H1={len(h1)}, "
              f"finite H1={n_fin}")
        
        if n_fin > 0:
            pers_fin = pers[fin]
            print(f"  Persistence: [{pers_fin.min():.4f}, {pers_fin.max():.4f}]")
        
        # Extract cycles for top 10 most persistent
        top_idx = np.where(fin)[0][np.argsort(-pers[fin])[:10]]
        cycles = []
        
        for rank, idx in enumerate(top_idx):
            cc = h1c[idx]
            birth = float(h1[idx, 0]); death = float(h1[idx, 1]); p = death - birth
            
            if not hasattr(cc, 'ndim') or cc.ndim < 2 or len(cc) < 1:
                continue
            
            edges_with_dists = [(int(r[0]), int(r[1]), 
                                np.linalg.norm(X_sub[int(r[0])] - X_sub[int(r[1])]))
                               for r in cc]
            
            u_birth, v_birth, d_birth = max(edges_with_dists, key=lambda x: x[2])
            
            # Build sub-birthedge graph
            adj = {}
            for u, v, d in edges_with_dists:
                if d < d_birth:
                    adj.setdefault(u, []).append(v)
                    adj.setdefault(v, []).append(u)
            
            # BFS
            path = None
            if u_birth in adj and v_birth in adj:
                visited = {u_birth: -1}
                queue = deque([u_birth])
                while queue:
                    cur = queue.popleft()
                    for nb in adj.get(cur, []):
                        if nb not in visited:
                            visited[nb] = cur
                            if nb == v_birth:
                                path = [v_birth]
                                while path[-1] != u_birth:
                                    path.append(visited[path[-1]])
                                path = path[::-1]
                                break
                    if path: break
            
            if path and len(path) >= 3:
                # Map back to original tokens
                tokens = [id_to_token[int(all_tok[word_ids[v]])] for v in path]
                cycles.append({'rank': rank, 'persistence': p, 'path': path,
                               'tokens': tokens, 'decoded': ' → '.join(tokens)})
        
        results[name] = {
            'n_tokens': n, 'n_h1': len(h1), 'n_finite': n_fin,
            'avg_pairwise': avg_pairwise, 'avg_nn': avg_nn,
            'cycles': cycles
        }
        
        if cycles:
            print(f"  Cycles extracted: {len(cycles)}")
            for c in cycles[:5]:
                print(f"    p={c['persistence']:.4f} n={len(c['path'])}: {c['decoded']}")
    
    except Exception as e:
        print(f"  ERROR: {e}")
        results[name] = {'error': str(e)}

# ── Summary ──
print(f"\n{'='*80}")
print(f"Summary")
print(f"{'='*80}")
for name, r in results.items():
    if 'error' in r:
        print(f"  {name}: ERROR - {r['error']}")
    else:
        cyc_str = f", {len(r['cycles'])} cycles" if r['cycles'] else ", no cycles"
        print(f"  {name}: {r['n_tokens']} tokens, {r['n_finite']} H1{cyc_str}")
        if r['cycles']:
            best = r['cycles'][0]
            print(f"    Best: p={best['persistence']:.4f} {best['decoded']}")

# Save
pkl_path = DATA_DIR / 'targeted_homology.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved {pkl_path}")

# Text output
txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/targeted_cycles.txt')
with open(txt_path, 'w') as f:
    for name, r in sorted(results.items()):
        if 'cycles' not in r: continue
        f.write(f"\n{'='*60}\n{name}: {r['n_tokens']} tokens, {r['n_finite']} H1\n"
                f"avg_pairwise={r['avg_pairwise']:.4f}, avg_nn={r['avg_nn']:.4f}\n")
        for c in r['cycles'][:10]:
            f.write(f"  p={c['persistence']:.4f} n={len(c['path'])}: {c['decoded']}\n")
print(f"Saved {txt_path}")

print("\nDone!")
PYEOF
