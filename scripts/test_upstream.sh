#!/usr/bin/env bash
#SBATCH --job-name=jlens-upstream-test
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/upstream_test_%j.out
#SBATCH --error=logs/upstream_test_%j.err
#SBATCH --partition=batch

set -e
cd /mnt/home/jacob/vl/repos/jlens

# Clear any stale bytecode
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

mkdir -p logs
echo "=== Starting upstream validation ==="
echo "Node: $(hostname)"
echo "Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

uv run python << 'PYEOF'
import os, time, torch, logging
os.environ['TRITON_BUILD_WITHOUT_CC'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'
logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')

from jlens._bridge import get_upstream
from jlens.checkpoint_manager import load_checkpoint
from jlens.prompts import load_prompts_text
from jlens.accumulator import JacobianAccumulator
from jlens.spectra import analyze_checkpoint

print('=== Setup ===', flush=True)
t0 = time.time()
hf_model, tokenizer = load_checkpoint('EleutherAI/pythia-160m-deduped', 143000, dtype=torch.float16)
up = get_upstream()
layout = up.Layout(path='gpt_neox', layers='layers', norm='final_layer_norm', embed='embed_in', lm_head='lm_head')
lens_model = up.from_hf(hf_model, tokenizer, layout=layout)
prompts = load_prompts_text(tokenizer, n_prompts=3, max_seq_len=128, seed=42)
print(f'Setup done in {time.time()-t0:.1f}s, {len(prompts)} prompts (n_layers={lens_model.n_layers}, d_model={lens_model.d_model})', flush=True)

layers = [0, 5]
accum = JacobianAccumulator(lens_model.n_layers, lens_model.d_model, dtype=torch.float32)

for pi, text in enumerate(prompts):
    t0 = time.time()
    print(f'Prompt {pi+1}/{len(prompts)} (len={len(text)} chars)...', flush=True, end=' ')
    jacobians, seq_len, n_valid = up.fitting.jacobian_for_prompt(
        lens_model, text, source_layers=layers,
        dim_batch=128, max_seq_len=128, skip_first=16,
    )
    dt = time.time() - t0
    accum.update(jacobians)
    print(f'{dt:.1f}s  seq_len={seq_len}  valid_tokens={n_valid}', flush=True)

stats = accum.get_all_stats()
spectral = analyze_checkpoint(stats)

print()
print('=== SPECTRAL RESULTS (upstream estimator) ===')
print(f'N prompts: {stats["n"]}')
for li in layers:
    coh = spectral["coherence"][li]
    qe = spectral["q_eff"][li]
    a_val = spectral["a_coeff"][li]
    sp = spectral["n_spikes"][li]
    er = spectral["effective_rank"][li]
    if coh is not None:
        print(f'  Layer {li}: coh={coh:.4f}  q_eff={qe:.4f}  a={a_val:.3f}  spikes={int(sp)}  eff_rk={er:.0f}')
    else:
        print(f'  Layer {li}: NO DATA')

print()
print('=== VALIDATION COMPLETE ===')
PYEOF

echo ""
echo "=== Done at $(date) ==="
