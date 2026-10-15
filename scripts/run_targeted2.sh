#!/usr/bin/env bash
#SBATCH --job-name=rip-tar2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rip_tar2_%j.out
#SBATCH --error=logs/rip_tar2_%j.err

# V2: Fix cycle extraction + add BPE sub-word search.
set -euo pipefail
echo "=== Targeted Homology v2 ==="
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
id_to_token = {v: k for k, v in tok.get_vocab().items()}

# Preprocess embeddings
X_c = X_raw - X_raw.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

# ── Candidate semantic cycles ──
CANDIDATES = {
    "days_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
                      "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun",
                      "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    "months": ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December",
               "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "colors": ["red", "blue", "green", "yellow", "orange", "purple", "pink",
               "white", "black", "brown", "gray", "grey", "cyan", "magenta",
               "violet", "indigo", "crimson", "scarlet", "amber"],
    "directions": ["north", "south", "east", "west",
                   "northeast", "northwest", "southeast", "southwest",
                   "left", "right", "up", "down", "forward", "backward"],
    "seasons": ["spring", "summer", "autumn", "winter", "fall"],
    "numbers": ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve",
                "first", "second", "third", "fourth", "fifth",
                "sixth", "seventh", "eighth", "ninth", "tenth"],
}

def find_tokens_by_substring(all_tok, id_to_token, substring, max_per_word=10):
    """Find tokens where decoded string contains the substring."""
    positions = []
    for i, tid in enumerate(all_tok):
        token_str = id_to_token[int(tid)]
        if substring.lower() in token_str.lower():
            positions.append(i)
    return positions[:max_per_word * 10]  # overall cap

def find_tokens_by_exact(all_tok, vocab, word, max_per_word=20):
    """Find tokens by exact vocabulary match."""
    tid = vocab.get(word)
    if tid is None:
        # Try lowercased
        tid = vocab.get(word.lower())
    if tid is None:
        return []
    positions = np.where(all_tok == tid)[0][:max_per_word]
    return list(positions)

def recover_cycle(X_sub, h1c, idx, h1):
    """Recover homology cycle from cocycle. Returns (path, d_birth) or (None, None)."""
    cc = h1c[idx]
    birth = float(h1[idx, 0])
    
    if not hasattr(cc, 'ndim') or cc.ndim < 2 or len(cc) < 1:
        return None, None
    
    edges_with_dists = []
    for row in cc:
        u, v = int(row[0]), int(row[1])
        d = np.linalg.norm(X_sub[u] - X_sub[v])
        edges_with_dists.append((u, v, d))
    
    u_birth, v_birth, d_birth = max(edges_with_dists, key=lambda x: x[2])
    
    # Build graph of edges with distance < d_birth
    adj = {}
    # Include ALL cocycle edges with d < d_birth
    for u, v, d in edges_with_dists:
        if d < d_birth:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
    
    # Also add nearest neighbors based on distance matrix
    for seed in [u_birth, v_birth]:
        dists = np.linalg.norm(X_sub - X_sub[seed], axis=1)
        nearby = np.where(dists < d_birth * 0.999)[0]
        for nb in nearby[:50]:
            if nb != seed:
                adj.setdefault(seed, []).append(nb)
                adj.setdefault(nb, []).append(seed)
    
    if u_birth not in adj or v_birth not in adj:
        return None, d_birth
    
    # BFS
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
                    return path[::-1], d_birth
                queue.append(nb)
    
    return None, d_birth

print(f"\n{'='*80}")
print(f"Targeted homology on {len(X)} tokens (10D PCA+L2 norm)")
print(f"Using both exact vocab match AND substring search")
print(f"{'='*80}")

results = {}

for name, words in CANDIDATES.items():
    word_ids = []
    
    # Exact vocab match
    for w in words:
        word_ids.extend(find_tokens_by_exact(all_tok, tok.get_vocab(), w))
    
    # Substring search for multi-token words (e.g., "Wednesday" might be split)
    for w in words:
        if len(w) > 5:  # only for longer words
            word_ids.extend(find_tokens_by_substring(all_tok, id_to_token, w[:5], max_per_word=5))
    
    word_ids = list(set(word_ids))
    
    if len(word_ids) < 3:
        print(f"\n{name}: only {len(word_ids)} tokens, skipping")
        continue
    
    X_sub = X[word_ids]
    n = len(word_ids)
    
    # Show what tokens we found
    sample_tokens = [id_to_token[int(all_tok[i])] for i in word_ids[:20]]
    
    print(f"\n{name}: {len(words)} words → {n} tokens")
    print(f"  Sample: {', '.join(sample_tokens[:15])}")
    
    # Distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dists[i,j] = dists[j,i] = np.linalg.norm(X_sub[i] - X_sub[j])
    
    avg_pairwise = dists.sum() / (n * (n-1))
    nn_dists = [min([dists[i,j] for j in range(n) if j != i]) for i in range(n)]
    avg_nn = np.mean(nn_dists)
    print(f"  Dist: avg={avg_pairwise:.4f}, nn={avg_nn:.4f}")
    
    # Ripser
    try:
        t0 = time.time()
        result = ripser(X_sub, maxdim=1, do_cocycles=True)
        h1 = result['dgms'][1]; h1c = result['cocycles'][1]
        pers = h1[:,1] - h1[:,0]; fin = np.isfinite(h1[:,1])
        n_fin = fin.sum()
        print(f"  Ripser: {time.time()-t0:.1f}s, H1={len(h1)}, fin={n_fin}")
        
        if n_fin > 0:
            print(f"  Persistence: [{pers[fin].min():.4f}, {pers[fin].max():.4f}]")
        
        # Extract top 20 cycles
        top_idx = np.where(fin)[0][np.argsort(-pers[fin])[:20]]
        cycles = []
        
        for rank, idx in enumerate(top_idx):
            path, d_birth = recover_cycle(X_sub, h1c, idx, h1)
            if path is None:
                continue
            tokens = [id_to_token[int(all_tok[word_ids[v]])] for v in path]
            cycles.append({
                'rank': rank, 'persistence': float(h1[idx,1] - h1[idx,0]),
                'birth': float(h1[idx,0]), 'death': float(h1[idx,1]),
                'path': path, 'tokens': tokens,
                'decoded': ' → '.join(tokens),
                'birth_edge_dist': d_birth,
            })
        
        results[name] = {'n_tokens': n, 'n_h1': len(h1), 'n_finite': n_fin,
                         'avg_pairwise': avg_pairwise, 'avg_nn': avg_nn,
                         'cycles': cycles, 'tokens': sample_tokens}
        
        if cycles:
            print(f"  Cycles: {len(cycles)}")
            for c in cycles[:5]:
                print(f"    {c['decoded']}")
        else:
            print(f"  No cycles extracted")
    
    except Exception as e:
        import traceback
        print(f"  ERROR: {e}")
        traceback.print_exc()

# Summary
print(f"\n{'='*80}")
for name, r in results.items():
    if 'error' in r:
        print(f"  {name}: ERROR")
    else:
        cstr = f", {len(r['cycles'])} cycles" if r['cycles'] else ", 0 cycles"
        print(f"  {name}: {r['n_tokens']} tokens, {r['n_finite']} H1{cstr}")
        if r['cycles']:
            for c in r['cycles'][:3]:
                print(f"    {c['decoded']}")

pkl_path = DATA_DIR / 'targeted_homology_v2.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved {pkl_path}")

txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/targeted_cycles.txt')
with open(txt_path, 'w') as f:
    for name, r in sorted(results.items()):
        if 'cycles' not in r: continue
        f.write(f"\n{'='*60}\n{name}: {r['n_tokens']} tokens, {r['n_finite']} H1\n")
        f.write(f"  tokens: {', '.join(r['tokens'][:30])}\n")
        for c in r['cycles']:
            f.write(f"  p={c['persistence']:.4f} n={len(c['path']):2d} | {c['decoded']}\n")
print(f"Saved {txt_path}")

print("\nDone!")
PYEOF
