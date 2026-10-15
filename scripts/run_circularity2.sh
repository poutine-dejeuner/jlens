#!/usr/bin/env bash
#SBATCH --job-name=circ2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/circ2_%j.out
#SBATCH --error=logs/circ2_%j.err

# Direct circularity test v2: measure if known cyclic concepts form circles
# in embedding space (last layer, contextual).
set -euo pipefail
echo "=== Circularity Analysis v2 ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, json, pickle, h5py
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

with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)

# Cyclic concepts with natural order
CONCEPTS = {
    "days": {
        "order": ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
        "search": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday",
                   "Mon","Tue","Wed","Thu","Fri","Sat","Sun",
                   "monday","tuesday","wednesday","thursday","friday","saturday","sunday"],
        "canonical": {"mon":"monday","tue":"tuesday","wed":"wednesday",
                       "thu":"thursday","fri":"friday","sat":"saturday","sun":"sunday"},
    },
    "months": {
        "order": ["january","february","march","april","may","june",
                  "july","august","september","october","november","december"],
        "search": ["January","February","March","April","May","June","July",
                   "August","September","October","November","December",
                   "Jan","Feb","Mar","Apr","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        "canonical": {"jan":"january","feb":"february","mar":"march","apr":"april",
                       "may":"may","jun":"june","jul":"july","aug":"august",
                       "sep":"september","oct":"october","nov":"november","dec":"december"},
    },
    "directions": {
        "order": ["north","east","south","west"],
        "search": ["north","south","east","west",
                   "North","South","East","West"],
        "canonical": {},
    },
    "seasons": {
        "order": ["spring","summer","autumn","winter"],
        "search": ["spring","summer","autumn","winter","fall",
                   "Spring","Summer","Autumn","Winter","Fall"],
    },
    "greek": {
        "order": ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta"],
        "search": ["alpha","beta","gamma","delta","epsilon","zeta","eta","theta"],
    },
    "planets": {
        "order": ["mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"],
        "search": ["Mercury","Venus","Earth","Mars","Jupiter","Saturn","Uranus","Neptune"],
    },
}

def find_token_positions(all_tok, id_to_token, word, max_per=5):
    """Find token positions where decoded token contains the word."""
    wl = word.lower()
    positions = []
    for i, tid in enumerate(all_tok):
        if wl in id_to_token[int(tid)].lower():
            positions.append(i)
            if len(positions) >= max_per:
                break
    return positions

results = {}

for name, concept in CONCEPTS.items():
    natural = concept['order']
    
    # First, find all tokens matching ANY search word
    all_positions = []
    point_word_map = []
    for w in concept['search']:
        positions = find_token_positions(all_tok, id_to_token, w, max_per=3)
        for p in positions:
            all_positions.append(p)
            point_word_map.append(w)
    
    all_positions = list(set(all_positions))  # deduplicate positions
    
    if len(all_positions) < 3:
        print(f"\n{name}: only {len(all_positions)} tokens, skipping")
        results[name] = {'error': f'only {len(all_positions)} tokens'}
        continue
    
    # Map each position back to the best matching natural-order word
    canonical_map = concept.get('canonical', {})
    pos_to_natural = {}
    for p in all_positions:
        token_str = id_to_token[int(all_tok[p])].lower()
        # Remove BPE prefix if present
        clean = token_str.lstrip('Ġ').lower()
        best_match = None
        for nw in natural:
            if nw in clean:
                best_match = nw
                break
        if best_match is None:
            # Try search words
            for sw in concept['search']:
                if sw.lower() in clean:
                    best_match = canonical_map.get(sw.lower(), sw.lower())
                    break
        pos_to_natural[p] = best_match or clean
    
    # Get embeddings
    positions_list = list(all_positions)
    X_sub = X_raw[positions_list]
    
    # PCA to 2D → compute angular ordering
    X_ctr = X_sub - X_sub.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_ctr, full_matrices=False)
    X_2d = X_ctr @ Vt[:2].T
    angles = np.arctan2(X_2d[:, 1], X_2d[:, 0])
    sorted_idx = np.argsort(angles)
    
    ordered = [pos_to_natural[positions_list[i]] for i in sorted_idx]
    
    # Deduplicate consecutive
    dedup = []
    for w in ordered:
        if not dedup or w != dedup[-1]:
            dedup.append(w)
    
    # Match score: count natural-order consecutive pairs
    natural_pairs = set()
    for i in range(len(natural)):
        natural_pairs.add((natural[i], natural[(i+1) % len(natural)]))
    
    found_pairs = set()
    for i in range(len(dedup)):
        found_pairs.add((dedup[i], dedup[(i+1) % len(dedup)]))
    
    matches = len(natural_pairs & found_pairs)
    # Also check reverse
    rev_matches = len(natural_pairs & {(b,a) for a,b in found_pairs})
    
    match_score = max(matches, rev_matches) / len(natural_pairs)
    
    print(f"\n{name}:")
    print(f"  Natural: {' → '.join(natural)}")
    print(f"  Found: {', '.join(sorted(set(dedup)))}")
    print(f"  Circle: {' → '.join(dedup)}")
    print(f"  Score: {match_score:.2f} ({max(matches, rev_matches)}/{len(natural_pairs)} pairs)")
    
    results[name] = {
        'natural': natural, 'circular': dedup,
        'found': sorted(set(dedup)),
        'score': match_score, 'matches': max(matches, rev_matches),
        'n_pairs': len(natural_pairs), 'n_tokens': len(positions_list),
    }

# Summary
print(f"\n{'='*60}")
print(f"{'Concept':<20} {'Score':>6}  {'Order'}")
print(f"{'='*60}")
for name in ["days","months","directions","seasons","greek","planets"]:
    r = results.get(name, {})
    if 'error' in r:
        print(f"  {name:<18} ERROR: {r['error']}")
    else:
        s = r.get('score', -1)
        m = '✓' if s > 0.5 else ('·' if s > 0.2 else '✗')
        order_str = ' → '.join(r.get('circular', [])[:8])
        nat_str = ' → '.join(r.get('natural', []))
        print(f"  {m} {name:<18} {s:.2f}    {order_str}")
        print(f"    natural: {nat_str}")

pkl_path = DATA_DIR / 'circularity.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nSaved {pkl_path}")
print("Done!")
PYEOF
