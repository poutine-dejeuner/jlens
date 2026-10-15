#!/usr/bin/env bash
#SBATCH --job-name=tm-context
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/tm_context_%j.out
#SBATCH --error=logs/tm_context_%j.err

# Option B: Contextualized thought manifold
# Phase 1: Scan Wikipedia EN for token occurrences (CPU)
# Phase 2: Run forward passes & Gram+eigh (GPU)

set -euo pipefail

echo "=== Thought Manifold Contextual Run ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

uv run python -c "
from __future__ import annotations
import logging, time, sys, json
import numpy as np
from collections import Counter
from datasets import load_dataset
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('thought_manifold.phase1')

# -----------------------------------------------------------------------
# Phase 1: Scan Wikipedia EN for token occurrence counts
# -----------------------------------------------------------------------
logger.info('=== Phase 1: Scanning Wikipedia EN for token occurrences ===')

# Load tokenizer (lightweight, no GPU needed)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
V = 248320
logger.info(f'Tokenizer loaded, vocab_size={V}')

# Stream Wikipedia and count token occurrences
ds = load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True)
counter = Counter()
n_articles = 0
t0 = time.time()

for article_idx, row in enumerate(ds):
    text = row['text']
    # Use the tokenizer's own encoding for accuracy
    ids = tok.encode(text, add_special_tokens=False)
    counter.update(ids)
    n_articles += 1

    if article_idx % 500 == 0:
        elapsed = time.time() - t0
        n_unique = sum(1 for tid, cnt in counter.items() if cnt >= 100)
        logger.info(
            '  article %d: %d unique tokens in corpus, '
            '%d with >=100 occurrences (%.1fs)',
            article_idx, len(counter), n_unique, elapsed)

    # Early stop: once entire vocab is enumerated
    if article_idx >= 100000 and len(counter) >= 90000:
        break

elapsed = time.time() - t0
n_qualifying = sum(1 for tid, cnt in counter.items() if cnt >= 100)
logger.info(f'Scan complete: {n_articles} articles in {elapsed:.1f}s')
logger.info(f'  Unique tokens in corpus: {len(counter)}')
logger.info(f'  Tokens with >=100 occurrences: {n_qualifying}')

# Save the qualifying token list
qualifying = [tid for tid, cnt in counter.items() if cnt >= 100]
qualifying.sort()

manifest = {
    'n_articles_scanned': n_articles,
    'vocab_size': V,
    'unique_in_corpus': len(counter),
    'qualifying_tokens': n_qualifying,
    'token_ids': qualifying,
    'scan_time_s': elapsed,
}
outdir = Path('/data/jlens/qwen3.6-27b')
outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / 'contextual_token_manifest.json', 'w') as f:
    json.dump(manifest, f)
logger.info(f'Saved manifest to {outdir / \"contextual_token_manifest.json\"}')
del counter, ds
" || { echo 'Phase 1 failed'; exit 1; }

echo "Phase 1 complete at $(date)"

# -----------------------------------------------------------------------
# Phase 2: Load model and run contextual forward passes
# -----------------------------------------------------------------------
echo "=== Phase 2: Contextual forward passes ==="

uv run python -c "
from __future__ import annotations
import logging, time, json
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('thought_manifold.phase2')

logger.info('=== Qwen3.6-27B contextual thought manifold (Option B) ===')

# ---- Load manifest ----
results_dir = Path('/data/jlens/qwen3.6-27b')
with open(results_dir / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
token_ids = manifest['token_ids']
logger.info(f'Loaded manifest: {len(token_ids)} qualifying tokens')

# ---- Load model ----
from transformers import AutoModelForCausalLM
t0 = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.6-27B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
logger.info(f'Model loaded in {time.time() - t0:.1f}s')

# ---- Run contextual pipeline ----
from jlens.thought_manifold import thought_manifold_contextual_pipeline_fast

t1 = time.time()
result = thought_manifold_contextual_pipeline_fast(
    hf_model,
    token_ids=token_ids,
    k=50,
    n_contexts=500_000,
    sent_batch_size=16,
)
dt = time.time() - t1
logger.info(f'Pipeline done in {dt:.1f}s ({dt / 60:.1f}m)')

# ---- Save ----
from jlens.storage import save_thought_manifold
save_thought_manifold(results_dir, 0, result)

# ---- Report ----
logger.info(f'V={result[\"V\"]}, d={result[\"d\"]}, centered={result[\"centered\"]}')
logger.info(f'Top-10 eigenvalues: {result[\"eigenvalues\"][:10]}')
logger.info('Explained variance:')
for i in range(10):
    logger.info(
        f'  λ_{i + 1}: {result[\"eigenvalues\"][i]:.2f} '
        f'({100 * result[\"explained_variance_ratio\"][i]:.2f}%)'
    )
cum_var = 100 * result['explained_variance_ratio'].sum()
logger.info(f'Cumulative (top-50): {cum_var:.2f}%')
gap = result['eigenvalues'][0] / result['eigenvalues'][1]
logger.info(f'Spectral gap λ₁/λ₂: {gap:.2f}')
sv = result['singular_values']
eff_rank = sv.sum() ** 2 / (sv ** 2).sum()
logger.info(f'Effective rank: {eff_rank:.2f}')

del hf_model
torch.cuda.empty_cache()
logger.info('Contextual run complete!')
"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
