#!/usr/bin/env bash
#SBATCH --job-name=cycle-reps
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=logs/cycle_reps_%j.out
#SBATCH --error=logs/cycle_reps_%j.err

# Compute 1-cycle representatives (edge sets) for singleton and contextual
# token embedding persistence across all 64 layers of Qwen3.6-27B.
# Uses Dionysus to extract the actual cycle chains.

set -euo pipefail
echo "=== Cycle Representatives: Qwen3.6-27B ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

uv run python -u << 'PYEOF'
import numpy as np, json, time, pickle, warnings
from pathlib import Path
from ripser import Rips
import dionysus as d
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
warnings.filterwarnings('ignore')

MODEL_ID = 'Qwen/Qwen3.6-27B'
DATA_DIR = Path('/data/jlens/qwen3.6-27b')
N_WITNESS = 225
PROJ_DIM = 10
DIONYSUS_THRESH = 3.0
BATCH_SIZE = 128
N_TOKENS = 5000

print(f'Config: N={N_WITNESS}, d={PROJ_DIM}, thresh={DIONYSUS_THRESH}')

# ── Helper: preprocess point cloud (PCA + extreme norm + L2 norm) ──
def preprocess(X_raw):
    X_c = X_raw - X_raw.mean(axis=0)
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    proj = X_c @ Vt[:PROJ_DIM].T
    proj = proj / (np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8)
    norms = np.linalg.norm(proj, axis=1)
    idx = np.argpartition(-norms, N_WITNESS - 1)[:N_WITNESS]
    return proj[idx, :PROJ_DIM].astype(np.float64), idx

# ── Helper: Dionysus persistence + cycle representatives ──
def dionysus_cycles(points):
    """Returns list of dicts for each H1 cycle: {birth, death, persistence, edges, n_vertices}"""
    rips = d.fill_rips(points, 2, DIONYSUS_THRESH)
    p = d.homology_persistence(rips)
    dgms = d.init_diagrams(p, rips)

    cycles = []
    for pt in dgms[1]:
        persistence = pt.death - pt.birth
        if persistence <= 1e-6:
            continue
        if np.isinf(pt.death):
            # Use thresh as death for infinite bars
            death_val = DIONYSUS_THRESH
        else:
            death_val = float(pt.death)

        chain_idx = p.pair(pt.data)
        chain = p[chain_idx]

        edges = []
        vertices = set()
        for entry in chain:
            s_idx = entry.index
            if s_idx < len(rips):
                s = rips[s_idx]
                if s.dimension() == 1:
                    v = list(s)
                    edges.append(tuple(sorted(v)))
                    vertices.update(v)

        cycles.append({
            'birth': float(pt.birth),
            'death': death_val,
            'persistence': float(death_val - pt.birth),
            'n_edges': len(edges),
            'n_vertices': len(vertices),
            'edges': edges,
            'vertices': sorted(vertices),
        })

    return cycles, len(rips), len(dgms[1])

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: Load model + token list
# ══════════════════════════════════════════════════════════════════════
t0 = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
n_layers = len(hf_model.model.layers)
d_model = hf_model.config.hidden_size
device = hf_model.device
print(f'Model loaded: {n_layers} layers, d={d_model}, {time.time()-t0:.1f}s')

if hasattr(hf_model, 'model'):
    text_model = hf_model.model
else:
    text_model = hf_model.transformer

# Token list from manifest
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
token_ids = manifest['token_ids'][:N_TOKENS]
print(f'Token IDs: {len(token_ids)}')

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: Singleton forward passes (capture all layers)
# ══════════════════════════════════════════════════════════════════════
print('\n=== Phase 2: Singleton hidden states ===')
sng_accum = {l: [] for l in range(n_layers)}

def make_sng_hook(layer_idx):
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        sng_accum[layer_idx].append(hs.squeeze(1).detach().cpu().float().numpy())
    return hook

sng_hook_handles = []
for layer_idx in range(n_layers):
    handle = text_model.layers[layer_idx].register_forward_hook(make_sng_hook(layer_idx))
    sng_hook_handles.append(handle)

t0_fwd = time.time()
for start in range(0, len(token_ids), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(token_ids))
    batch = [[tid] for tid in token_ids[start:end]]
    input_ids = torch.tensor(batch, dtype=torch.long, device=device)
    with torch.no_grad():
        _ = text_model(input_ids)
    if (start // BATCH_SIZE) % 20 == 0:
        print(f'  {end}/{len(token_ids)} tokens ({time.time()-t0_fwd:.1f}s)')

for handle in sng_hook_handles:
    handle.remove()
print(f'Forward passes done: {time.time()-t0_fwd:.1f}s')

# Concatenate
sng_hidden = {}
for l in range(n_layers):
    sng_hidden[l] = np.concatenate(sng_accum[l], axis=0)
del sng_accum

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: Singleton per-layer cycles
# ══════════════════════════════════════════════════════════════════════
print('\n=== Phase 3: Singleton per-layer cycles ===')
sng_results = {}

for layer_idx in range(n_layers):
    t0 = time.time()
    points, point_idx = preprocess(sng_hidden[layer_idx][:N_TOKENS])
    cycles, n_simplices, n_total = dionysus_cycles(points)
    dt = time.time() - t0

    # Also run Ripser for comparison (H0-H3 counts)
    ripser = Rips(maxdim=3, thresh=np.inf)
    dgms_r = ripser.fit_transform(points)
    ripser_counts = [len(dgms_r[d][np.isfinite(dgms_r[d][:, 1]) & (dgms_r[d][:, 1] > dgms_r[d][:, 0])])
                     if d < len(dgms_r) else 0 for d in range(4)]

    sng_results[layer_idx] = {
        'cycles': cycles,
        'n_simplices': n_simplices,
        'n_total_h1': n_total,
        'ripser_h0_h3': ripser_counts,
    }
    cycle_sizes = [c['n_vertices'] for c in cycles]
    print(f'  Layer {layer_idx}: {dt:.1f}s, H1 cycles={len(cycles)}, '
          f'vtx sizes={cycle_sizes[:5]}{"..." if len(cycle_sizes) > 5 else ""}, '
          f'ripser={ripser_counts}')

# Free memory
del sng_hidden

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: Contextual forward passes
# ══════════════════════════════════════════════════════════════════════
print('\n=== Phase 4: Contextual hidden states ===')

# Use longer context by repeating and padding with diverse sentences
sentences = [
    "The transformer architecture has revolutionized natural language processing by enabling parallel computation across all tokens in a sequence.",
    "Deep learning models require massive amounts of training data and computational resources to achieve state-of-the-art performance on benchmark tasks.",
    "Neural networks learn hierarchical representations of data through successive layers of nonlinear transformations applied to input features.",
    "The attention mechanism allows the model to dynamically weight the importance of different input elements when computing output representations.",
    "Language models trained on internet-scale corpora demonstrate emergent capabilities such as reasoning, translation, and code generation.",
]
full_text = " ".join(sentences)
inputs = tok(full_text, return_tensors='pt').to(device)
ctx_input_ids = inputs['input_ids']
n_ctx_tokens = ctx_input_ids.shape[1]
print(f'Context tokens in sentence: {n_ctx_tokens}')

# Re-hook all layers
ctx_accum = {l: [] for l in range(n_layers)}

def make_ctx_hook(layer_idx):
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        ctx_accum[layer_idx].append(hs.detach().cpu().float().numpy())
    return hook

ctx_hook_handles = []
for layer_idx in range(n_layers):
    handle = text_model.layers[layer_idx].register_forward_hook(make_ctx_hook(layer_idx))
    ctx_hook_handles.append(handle)

t0_fwd = time.time()
# Forward pass — repeat with differently positioned sequences to get more tokens
n_repeats = max(1, N_TOKENS // n_ctx_tokens)
print(f'Context: {n_ctx_tokens} tokens/seq, {n_repeats} repeats for ~{n_repeats*n_ctx_tokens} total tokens')

for rep in range(min(n_repeats, 50)):  # Cap at 50 repeats
    with torch.no_grad():
        _ = text_model(**inputs)
    if rep % 10 == 0:
        print(f'  ctx pass {rep+1}/{min(n_repeats, 50)}')

for handle in ctx_hook_handles:
    handle.remove()
print(f'Context forward passes done: {time.time()-t0_fwd:.1f}s')

# Concatenate: each layer's accumulator has (n_repeats, B, T, d) tensors
ctx_hidden = {}
for l in range(n_layers):
    all_passes = ctx_accum[l]
    # Each pass: (1, n_ctx_tokens, d_model)
    # Stack along batch dim: (n_repeats, n_ctx_tokens, d_model)
    stacked = np.concatenate(all_passes, axis=0)  # (n_repeats, n_ctx_tokens, d_model)
    # Reshape to (n_repeats * n_ctx_tokens, d_model)
    ctx_hidden[l] = stacked.reshape(-1, d_model)[:N_TOKENS]

del ctx_accum

# ══════════════════════════════════════════════════════════════════════
# PHASE 5: Contextual per-layer cycles
# ══════════════════════════════════════════════════════════════════════
print('\n=== Phase 5: Contextual per-layer cycles ===')
ctx_results = {}

for layer_idx in range(n_layers):
    t0 = time.time()
    points, point_idx = preprocess(ctx_hidden[layer_idx])
    cycles, n_simplices, n_total = dionysus_cycles(points)
    dt = time.time() - t0

    ripser = Rips(maxdim=3, thresh=np.inf)
    dgms_r = ripser.fit_transform(points)
    ripser_counts = [len(dgms_r[d][np.isfinite(dgms_r[d][:, 1]) & (dgms_r[d][:, 1] > dgms_r[d][:, 0])])
                     if d < len(dgms_r) else 0 for d in range(4)]

    ctx_results[layer_idx] = {
        'cycles': cycles,
        'n_simplices': n_simplices,
        'n_total_h1': n_total,
        'ripser_h0_h3': ripser_counts,
    }
    cycle_sizes = [c['n_vertices'] for c in cycles]
    print(f'  Layer {layer_idx}: {dt:.1f}s, H1 cycles={len(cycles)}, '
          f'vtx sizes={cycle_sizes[:5]}{"..." if len(cycle_sizes) > 5 else ""}, '
          f'ripser={ripser_counts}')

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════
results = {
    'config': {
        'model_id': MODEL_ID, 'n_layers': n_layers, 'd_model': d_model,
        'n_witness': N_WITNESS, 'proj_dim': PROJ_DIM,
        'dionysus_thresh': DIONYSUS_THRESH,
        'n_tokens': N_TOKENS,
    },
    'singleton': sng_results,
    'contextual': ctx_results,
}

outfile = DATA_DIR / 'cycle_representatives.pkl'
with open(outfile, 'wb') as f:
    pickle.dump(results, f)
print(f'\nSaved to {outfile}')
print(f'File size: {outfile.stat().st_size / 1024 / 1024:.1f} MB')

# Quick summary
print('\n=== SUMMARY ===')
print(f'{"Layer":>6}  {"Sng H1":>7}  {"Sng min vtx":>12}  {"Sng max vtx":>12}  {"Ctx H1":>7}  {"Ctx min vtx":>12}  {"Ctx max vtx":>12}')
for l in range(n_layers):
    sc = sng_results[l]['cycles']
    cc = ctx_results[l]['cycles']
    s_min = min((c['n_vertices'] for c in sc), default=0)
    s_max = max((c['n_vertices'] for c in sc), default=0)
    c_min = min((c['n_vertices'] for c in cc), default=0)
    c_max = max((c['n_vertices'] for c in cc), default=0)
    print(f'{l:>6}  {len(sc):>7}  {s_min:>12}  {s_max:>12}  {len(cc):>7}  {c_min:>12}  {c_max:>12}')

print('\nDone!')
PYEOF

echo ""
echo "=== Done ==="
echo "Date: $(date)"
