#!/usr/bin/env bash
#SBATCH --job-name=jlens-bench
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err
#SBATCH --partition=batch

set -e
cd /mnt/home/jacob/vl/repos/jlens
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

mkdir -p logs
export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1
export PYTHONPATH="$(pwd):$PYTHONPATH"

uv run python << 'PYEOF'
import os, time, torch
os.environ['TRITON_BUILD_WITHOUT_CC'] = '1'
os.environ['TORCH_COMPILE_DISABLE'] = '1'

from jlens._bridge import get_upstream
from jlens.checkpoint_manager import load_checkpoint
from jlens.prompts import load_prompts_text

print("Setup...", flush=True)
hf_model, tokenizer = load_checkpoint('EleutherAI/pythia-160m-deduped', 143000, dtype=torch.float16)
up = get_upstream()
layout = up.Layout(path='gpt_neox', layers='layers', norm='final_layer_norm', embed='embed_in', lm_head='lm_head')
lens_model = up.from_hf(hf_model, tokenizer, layout=layout)
prompts = load_prompts_text(tokenizer, n_prompts=10, max_seq_len=128, seed=42)

all_layers = list(range(11))  # 0..10 (n_layers=12, target=11)

times = []
for pi, text in enumerate(prompts[:5]):
    t0 = time.time()
    jacobians, seq_len, n_valid = up.fitting.jacobian_for_prompt(
        lens_model, text, source_layers=all_layers,
        dim_batch=128, max_seq_len=128, skip_first=16,
    )
    dt = time.time() - t0
    times.append(dt)
    print(f"  Prompt {pi+1}: {dt:.1f}s  seq={seq_len}  valid={n_valid}", flush=True)

avg = sum(times) / len(times)
print(f"\nAverage: {avg:.1f}s/prompt for {len(all_layers)} layers")
est_1000 = avg * 1000
est_per_ckpt = est_1000 / 3600
print(f"Estimated: {est_1000:.0f}s per checkpoint = {est_per_ckpt:.1f}h")
print(f"153 checkpoints: {est_per_ckpt * 153:.1f}h total")
print(f"8 GPUs: {est_per_ckpt * 153 / 8:.1f}h")
print(f"16 GPUs: {est_per_ckpt * 153 / 16:.1f}h")
PYEOF
