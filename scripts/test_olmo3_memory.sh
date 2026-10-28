#!/bin/bash
#SBATCH --job-name=vincent-jl-olmo-memtest3
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vincent-jl-olmo-memtest3_%j.out
#SBATCH --error=logs/vincent-jl-olmo-memtest3_%j.err

set -e

echo "=== OLMo3-32B memory test 3: detailed ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

uv run python -c "
import os, torch, gc
os.environ.setdefault('TRITON_BUILD_WITHOUT_CC', '1')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')

import logging
logging.basicConfig(level=logging.WARNING)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = 'allenai/Olmo-3-1125-32B'
revision = 'stage1-step10000'

print('Loading...')
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    revision=revision,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

from jlens._bridge import get_upstream
up = get_upstream()
lens_model = up.from_hf(model, tokenizer)
print(f'Model: {lens_model.n_layers} layers, d_model={lens_model.d_model}')

layers = list(range(lens_model.n_layers - 1))
print(f'Requested {len(layers)} source layers: {layers[:5]}...{layers[-5:]}')

# Check what _check_layer_indices does
from jlens._bridge import get_upstream as _gup2
_fitting_mod = _gup2().fitting
checked, target = _fitting_mod._check_layer_indices(layers, None, lens_model.n_layers)
print(f'After check: {len(checked)} source layers, target={target}')
print(f'Checked layers: min={min(checked)}, max={max(checked)}')

# Now do 1 prompt with dim_batch=32
text = 'The economic impact of the Industrial Revolution was not uniform across all sectors. While textile manufacturing saw dramatic increases in productivity due to mechanization, other industries such as agriculture experienced more gradual changes. The introduction of the steam engine by James Watt in 1769 revolutionized transportation and manufacturing, leading to the development of railways and steamships that connected distant markets. This period also witnessed significant social changes, including the growth of urban centers and the emergence of a new middle class of factory owners and merchants. The working conditions in early factories were often harsh, with long hours and dangerous machinery, which eventually led to the formation of labor unions and the push for workers rights.'

torch.cuda.reset_peak_memory_stats()
gc.collect()
torch.cuda.empty_cache()

jacobians, seq_len, n_valid = up.fitting.jacobian_for_prompt(
    lens_model, text,
    source_layers=layers,
    dim_batch=32,
    max_seq_len=128,
    skip_first=16,
)

mem = torch.cuda.max_memory_allocated() / 1e9
curr = torch.cuda.memory_allocated() / 1e9
print(f'Results: {len(jacobians)} layers, seq_len={seq_len}, n_valid={n_valid}')
if len(jacobians) > 0:
    sample_layer = list(jacobians.keys())[0]
    print(f'Sample key: {sample_layer}, shape: {jacobians[sample_layer].shape}')
    print(f'All layer keys: {sorted(jacobians.keys())}')
print(f'Peak memory: {mem:.1f} GB, current: {curr:.1f} GB')
print('Test complete.')
" 2>&1

echo "End: $(date)"
echo "=== Done ==="
