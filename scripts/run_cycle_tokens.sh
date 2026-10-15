#!/usr/bin/env bash
#SBATCH --job-name=cycle-tokens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=logs/cycle_tokens_%j.out
#SBATCH --error=logs/cycle_tokens_%j.err

# Map cycle vertex indices back to original token IDs for singleton layers.
# Loads existing cycle data, re-runs forward passes to get point_idx mapping,
# then saves per-cycle token lists.

set -euo pipefail
echo "=== Cycle Token Mapping: Qwen3.6-27B ==="
echo "Date: $(date)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, json, time, pickle, warnings
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')

MODEL_ID = 'Qwen/Qwen3.6-27B'
DATA_DIR = Path('/data/jlens/qwen3.6-27b')
N_WITNESS = 225
PROJ_DIM = 10
BATCH_SIZE = 128
N_TOKENS = 5000

# ── Load existing cycle data ──
with open(DATA_DIR / 'cycle_representatives.pkl', 'rb') as f:
    cycle_data = pickle.load(f)
print(f'Loaded cycle data: {len(cycle_data["singleton"])} singleton layers, {len(cycle_data["contextual"])} contextual layers')

# ── Load model ──
t0 = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
n_layers = len(hf_model.model.layers)
device = hf_model.device
text_model = hf_model.model
print(f'Model loaded: {n_layers} layers, {time.time()-t0:.1f}s')

# ── Token list ──
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
token_ids = manifest['token_ids'][:N_TOKENS]
print(f'Token IDs: {len(token_ids)}')

# ── Helper: preprocess (same as original) ──
def preprocess(X_raw):
    X_c = X_raw - X_raw.mean(axis=0)
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    proj = X_c @ Vt[:PROJ_DIM].T
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)
    norms = np.linalg.norm(proj, axis=1)
    idx = np.argpartition(-norms, N_WITNESS - 1)[:N_WITNESS]
    return proj[idx, :PROJ_DIM].astype(np.float64), idx

# ── Singleton forward passes ──
print('\n=== Singleton forward passes ===')
sng_accum = {l: [] for l in range(n_layers)}

def make_sng_hook(layer_idx):
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        sng_accum[layer_idx].append(hs.squeeze(1).detach().cpu().float().numpy())
    return hook

hook_handles = []
for layer_idx in range(n_layers):
    h = text_model.layers[layer_idx].register_forward_hook(make_sng_hook(layer_idx))
    hook_handles.append(h)

t0_fwd = time.time()
for start in range(0, len(token_ids), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(token_ids))
    batch = [[tid] for tid in token_ids[start:end]]
    input_ids = torch.tensor(batch, dtype=torch.long, device=device)
    with torch.no_grad():
        _ = text_model(input_ids)
    if (start // BATCH_SIZE) % 10 == 0:
        print(f'  {end}/{len(token_ids)} tokens ({time.time()-t0_fwd:.1f}s)')

for h in hook_handles:
    h.remove()
print(f'Forward passes done: {time.time()-t0_fwd:.1f}s')

# ── Concatenate ──
sng_hidden = {}
for l in range(n_layers):
    sng_hidden[l] = np.concatenate(sng_accum[l], axis=0)
del sng_accum

# ── For each singleton layer: compute point_idx, map cycles to tokens ──
print('\n=== Mapping cycles to tokens ===')
sng_token_cycles = {}

for layer_idx in range(n_layers):
    points, point_idx = preprocess(sng_hidden[layer_idx][:N_TOKENS])
    # point_idx[i] = original token index for sub-cloud vertex i

    cycles = cycle_data['singleton'][layer_idx]['cycles']
    layer_cycles = []
    for cycle in cycles:
        # Map vertex indices to original token IDs
        token_indices = [int(point_idx[v]) for v in cycle['vertices']]
        token_ids_in_cycle = [int(token_ids[ti]) for ti in token_indices]
        layer_cycles.append({
            'birth': cycle['birth'],
            'death': cycle['death'],
            'persistence': cycle['persistence'],
            'n_vertices': cycle['n_vertices'],
            'token_indices': token_indices,      # index into master token list
            'token_ids': token_ids_in_cycle,     # actual HF token IDs
        })

    sng_token_cycles[layer_idx] = layer_cycles
    if layer_idx % 10 == 0:
        print(f'  Layer {layer_idx}: mapped {len(layer_cycles)} cycles')

del sng_hidden  # free memory

# ── Save ──
results = {
    'config': cycle_data['config'],
    'master_token_ids': token_ids,  # the full token list
    'singleton_cycles': sng_token_cycles,
}

outfile = DATA_DIR / 'cycle_tokens.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(results, f)
print(f'\nSaved to {outfile}')
print(f'File size: {outfile.stat().st_size / 1024 / 1024:.1f} MB')

# ── Show examples ──
print('\n=== Example: Layer 0, first 3 cycles ===')
for i, c in enumerate(sng_token_cycles[0][:3]):
    nv = c['n_vertices']
    pers = c['persistence']
    tokens = tok.decode(c['token_ids'])
    print(f'  Cycle {i}: {nv} vertices, persistence={pers:.3f}')
    print(f'    Tokens: {tokens[:120]}')

print('\n=== Layer 59 outlier, first 3 cycles ===')
for i, c in enumerate(sng_token_cycles[59][:3]):
    nv = c['n_vertices']
    pers = c['persistence']
    tokens = tok.decode(c['token_ids'])
    print(f'  Cycle {i}: {nv} vertices, persistence={pers:.3f}')
    print(f'    Tokens: {tokens[:120]}')

print('\nDone!')
PYEOF

echo ""
echo "=== Done ==="
echo "Date: $(date)"
