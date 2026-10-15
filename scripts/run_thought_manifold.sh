#!/usr/bin/env bash
#SBATCH --job-name=tm-full
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=logs/tm_full_%j.out
#SBATCH --error=logs/tm_full_%j.err
#
# Full thought manifold computation: all 248,320 vocabulary tokens
# on Qwen3.6-27B.  Saves to results/step0000/manifold.h5.

set -euo pipefail

echo "=== Thought Manifold Full Run ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /mnt/home/jacob/vl/repos/jlens

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

uv run python -c "
from __future__ import annotations
import logging, time, sys
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('thought_manifold.full')

logger.info('=== Qwen3.6-27B full thought manifold ===')

# ---- Load model ----
from transformers import AutoModelForCausalLM
t0 = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.6-27B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
logger.info(f'Model loaded in {time.time()-t0:.1f}s')

# ---- Wrap ----
from jlens._bridge import get_upstream
upstream = get_upstream()
lens_model = upstream.from_hf(hf_model, None)
logger.info(f'Layout: path={lens_model.layout.path}, n_layers={lens_model.n_layers}, d_model={lens_model.d_model}')

# ---- Run pipeline ----
from jlens.thought_manifold import thought_manifold_pipeline

t1 = time.time()
result = thought_manifold_pipeline(
    lens_model,
    k=50,
    batch_size=512,
    center=True,
    max_token_id=248320,  # full vocab
)
dt = time.time() - t1
logger.info(f'Pipeline done in {dt:.1f}s ({dt/60:.1f}m)')

# ---- Save ----
from jlens.storage import save_thought_manifold
results_dir = Path('/data/jlens/qwen3.6-27b')
save_thought_manifold(results_dir, 0, result)

# ---- Report ----
logger.info(f'V={result[\"V\"]}, d={result[\"d\"]}, centered={result[\"centered\"]}')
logger.info(f'Top-10 eigenvalues: {result[\"eigenvalues\"][:10]}')
logger.info(f'Explained variance:')
for i in range(10):
    logger.info(f'  λ_{i+1}: {result[\"eigenvalues\"][i]:.2f} ({100*result[\"explained_variance_ratio\"][i]:.2f}%)')
logger.info(f'Cumulative (top-50): {100*result[\"explained_variance_ratio\"].sum():.2f}%')
logger.info(f'Spectral gap λ₁/λ₂: {result[\"eigenvalues\"][0]/result[\"eigenvalues\"][1]:.2f}')
logger.info(f'Effective rank: {result[\"singular_values\"].sum()**2 / (result[\"singular_values\"]**2).sum():.2f}')

del lens_model, hf_model
torch.cuda.empty_cache()
logger.info('Full run complete!')
"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
