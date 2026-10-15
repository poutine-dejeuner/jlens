#!/usr/bin/env bash
#SBATCH --job-name=cycle-diam2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=logs/cycle_diam2_%j.out
#SBATCH --error=logs/cycle_diam2_%j.err
set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import pickle, h5py, json, numpy as np
from pathlib import Path

DATA_DIR = Path('/data/jlens/qwen3.6-27b')

# Inspect the pickle structure
with open(DATA_DIR / 'ripser_h1_full_nothresh.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
if isinstance(data, dict):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"  {k}: list of {len(v)} items, first type={type(v[0]) if v else 'empty'}")
        else:
            print(f"  {k}: {type(v)}")
PYEOF
