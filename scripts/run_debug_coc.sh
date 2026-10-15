#!/usr/bin/env bash
#SBATCH --job-name=rip-dbg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=logs/rip_dbg_%j.out
#SBATCH --error=logs/rip_dbg_%j.err

set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, h5py, json
from pathlib import Path
from ripser import ripser

DATA_DIR = Path('/data/jlens/qwen3.6-27b')
PROJ_DIM = 10; R_THRESH = 1.2

with h5py.File(DATA_DIR / 'step0000' / 'manifold.h5', 'r') as f:
    X_raw = f['representations'][:].astype(np.float64)
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
all_tok = manifest['token_ids'][:X_raw.shape[0]]

X_c = X_raw - X_raw.mean(0).astype(np.float64)
_, s, Vt = np.linalg.svd(X_c, full_matrices=False)
X = (X_c @ Vt[:PROJ_DIM].T).astype(np.float64)
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

# Use N=1000 for faster debug
idx = np.random.choice(len(X), 1000, replace=False)
X_sub = X_raw[:5000]  # first 5000 tokens (not random subset)

X_c = X_sub - X_sub.mean(0).astype(np.float64)
_, s_sub, Vt_sub = np.linalg.svd(X_c, full_matrices=False)
X_sub_proj = (X_c @ Vt_sub[:10].T).astype(np.float64)
X_sub_proj = X_sub_proj / (np.linalg.norm(X_sub_proj, axis=1, keepdims=True) + 1e-8)

print("Ripser on 5000 points...")
result = ripser(X_sub_proj, maxdim=1, thresh=R_THRESH, do_cocycles=True)

h1c = result['cocycles'][1]
print(f"cocycles: {len(h1c)}")

valid = sum(1 for c in h1c if hasattr(c, 'ndim') and c.ndim >= 2 and len(c) >= 2)
ndim1 = sum(1 for c in h1c if hasattr(c, 'ndim') and c.ndim == 1)
ndim0 = sum(1 for c in h1c if hasattr(c, 'ndim') and c.ndim == 0)
not_ndarray = sum(1 for c in h1c if not hasattr(c, 'ndim'))
empty = sum(1 for c in h1c if hasattr(c, 'ndim') and (c.ndim < 2 or len(c) < 2))
print(f"Valid (ndim>=2, len>=2): {valid}")
print(f"ndim=1: {ndim1}, ndim=0: {ndim0}, not ndarray: {not_ndarray}")
print(f"Total empty/invalid: {empty}")

# Sample first 10
for i in range(min(10, len(h1c))):
    c = h1c[i]
    nd = c.ndim if hasattr(c, 'ndim') else -1
    sh = c.shape if hasattr(c, 'shape') else None
    ln = len(c) if hasattr(c, '__len__') else '?'
    print(f"  [{i}] ndim={nd} shape={sh} len={ln}")
    if nd == 1:
        print(f"       content: {c}")
PYEOF
