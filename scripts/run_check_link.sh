#!/usr/bin/env bash
#SBATCH --job-name=check-link
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:05:00
#SBATCH --output=logs/check_link_%j.out
#SBATCH --error=logs/check_link_%j.err
set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import pickle, json, numpy as np
from pathlib import Path

DATA_DIR = Path('/data/jlens/qwen3.6-27b')

with open(DATA_DIR / 'ripser_h1_full_nothresh.pkl', 'rb') as f:
    rip = pickle.load(f)
with open(DATA_DIR / 'cycle_tokens_full.pkl', 'rb') as f:
    cyc = pickle.load(f)

print(f"ripser data: h1={rip['h1'].shape}, cocycles={len(rip['h1_cocycles'])}")
print(f"cycle_tokens: {len(cyc)} cycles")
print(f"  first cycle keys: {list(cyc[0].keys())}")
print(f"  first cycle idx={cyc[0]['idx']}, cocycle_idx={cyc[0].get('cocycle_idx', 'N/A')}")
print(f"  second cycle idx={cyc[1]['idx']}, cocycle_idx={cyc[1].get('cocycle_idx', 'N/A')}")
print(f"  last cycle idx={cyc[-1]['idx']}, cocycle_idx={cyc[-1].get('cocycle_idx', 'N/A')}")

# Check if cycle['idx'] matches h1_cocycles index
# Try a few
for i in [0, 5, 100, 1000]:
    c = cyc[i]
    cc_idx = c.get('cocycle_idx', c['idx'])
    print(f"\n  cycle[{i}]: idx={c['idx']}, cocycle_idx={c.get('cocycle_idx','N/A')}")
    cc = rip['h1_cocycles'][cc_idx]
    if isinstance(cc, np.ndarray):
        print(f"    cocycle: shape={cc.shape}, dtype={cc.dtype}")
        print(f"    first edge: {cc[0]}")
    else:
        print(f"    cocycle: {type(cc)}")
PYEOF
