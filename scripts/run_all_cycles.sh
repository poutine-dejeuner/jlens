#!/usr/bin/env bash
#SBATCH --job-name=cyc-all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cyc_all_%j.out
#SBATCH --error=logs/cyc_all_%j.err

# Extract ALL 53,250 H1 cycles using birth-radius neighborhoods.
# Data from job 7031 is already saved — we only recompute token decoding.

set -euo pipefail
echo "=== Extract All H1 Cycles: Qwen3.6-27B contextual ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, time, h5py, pickle, json
from pathlib import Path

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10

# ── Load pre-saved data ──
print("Loading pre-saved data...")
with open(DATA_DIR / 'ripser_h1_neighborhoods.pkl', 'rb') as f:
    saved = pickle.load(f)

dgms_h1 = np.array(saved['ripser_dgms'][1])
fin_mask = np.isfinite(dgms_h1[:,1])
fin_idx = np.where(fin_mask)[0]

# Sort by persistence (descending)
pers = dgms_h1[fin_idx, 1] - dgms_h1[fin_idx, 0]
order = np.argsort(-pers)
sorted_idx = fin_idx[order]
sorted_pers = pers[order]

n_total = len(sorted_idx)
print(f"Total H1 cycles: {len(dgms_h1)} (finite: {n_total})")
print(f"Persistence range: [{sorted_pers.min():.4f}, {sorted_pers.max():.4f}]")

# ── Recompute projected data (needed for birth-radius neighborhoods) ──
print("\nLoading contextual representations...")
with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)

# Preprocess
print("Preprocessing...")
t0 = time.time()
X_c = X_raw - X_raw.mean(axis=0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X_proj = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X_proj = X_proj / (np.linalg.norm(X_proj, axis=1, keepdims=True) + 1e-8)

with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_token_ids = manifest['token_ids'][:X_raw.shape[0]]
print(f"  Done ({time.time()-t0:.1f}s)")

# ── Extract all cycle neighborhoods ──
from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)

print(f"\nExtracting birth-radius neighborhoods for all {n_total} cycles...")
all_cycles = []
n_skipped = 0

for idx, (full_idx, pers_val) in enumerate(zip(sorted_idx, sorted_pers)):
    birth = dgms_h1[full_idx, 0]
    
    # Seed point: pick the first point whose distance to others is reasonable
    # For efficiency, use a fixed seed per cycle (index 0)
    seed = idx % len(X_proj)
    dists = np.linalg.norm(X_proj - X_proj[seed], axis=1)
    nearby = dists < birth
    n_nearby = int(nearby.sum())
    
    if n_nearby < 1 or n_nearby > 500:
        n_skipped += 1
        continue
    
    vert_idx = np.where(nearby)[0]
    token_ids = [int(all_token_ids[i]) for i in vert_idx]
    decoded = tok.decode(token_ids)
    
    all_cycles.append({
        'rank': int(idx),
        'persistence': float(pers_val),
        'birth': float(birth),
        'death': float(dgms_h1[full_idx, 1]),
        'n_tokens': n_nearby,
        'decoded': decoded,
    })
    
    if (idx + 1) % 5000 == 0:
        print(f"  {idx+1}/{n_total} ({n_skipped} skipped)...")

print(f"\nDone: {len(all_cycles)} cycles extracted, {n_skipped} skipped")

# ── Save text file (one cycle per line) ──
txt_path = Path('/mnt/home/jacob/vl/repos/jlens/figures/cycle_tokens_all.txt')
with open(txt_path, 'w') as f:
    for c in all_cycles:
        f.write(c['decoded'] + '\n')

print(f"Wrote {len(all_cycles)} cycles to {txt_path}")

# ── Save detailed pickle ──
pkl_path = DATA_DIR / 'ripser_h1_all_cycles.pkl'
with open(pkl_path, 'wb') as f:
    pickle.dump(all_cycles, f)

print(f"Saved detailed data to {pkl_path}")
print(f"Size: {pkl_path.stat().st_size / 1024 / 1024:.1f} MB")

# ── Stats ──
n_tokens_arr = np.array([c['n_tokens'] for c in all_cycles])
pers_arr = np.array([c['persistence'] for c in all_cycles])
print(f"\nStats:")
print(f"  Tokens per cycle: min={n_tokens_arr.min()}, max={n_tokens_arr.max()}, "
      f"mean={n_tokens_arr.mean():.1f}, median={np.median(n_tokens_arr):.0f}")
print(f"  Persistence: min={pers_arr.min():.4f}, max={pers_arr.max():.4f}, "
      f"mean={pers_arr.mean():.4f}")

print("\nDone!")
PYEOF
