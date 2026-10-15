#!/usr/bin/env bash
#SBATCH --job-name=layer-persist
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=logs/layer_persist_%j.out
#SBATCH --error=logs/layer_persist_%j.err

# Phase 3-6 only: contextual + random per-layer persistence
# Singleton results were captured in previous run (job 6944).

set -euo pipefail
echo "=== Layer-wise Persistence: Contextual + Random ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

uv pip install ripser 2>&1 | tail -1

uv run python << 'PYEOF'
import numpy as np, time, h5py, logging, json, torch
from pathlib import Path
from ripser import Rips
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

MODEL_ID = 'Qwen/Qwen3.6-27B'
DATA_DIR = Path('/data/jlens/qwen3.6-27b')
N_WITNESS = 225
MD = 3
N_RANDOM_RUNS = 5
BATCH_SIZE = 128

# ===========================================================================
# Singleton results from previous run (hardcoded from job 6944 logs)
# ===========================================================================
singleton_persistence = {
    0: [224,47,5,0], 1:[224,47,4,0], 2:[224,42,1,0], 3:[224,49,0,0],
    4:[224,45,4,0], 5:[224,53,8,1], 6:[224,52,2,0], 7:[224,49,2,0],
    8:[224,50,2,0], 9:[224,46,0,0], 10:[224,44,1,0], 11:[224,41,0,0],
    12:[224,40,1,0], 13:[224,41,1,0], 14:[224,38,2,1], 15:[224,37,5,0],
    16:[224,41,4,1], 17:[224,35,2,0], 18:[224,39,1,0], 19:[224,37,1,0],
    20:[224,48,2,0], 21:[224,41,1,0], 22:[224,35,3,0], 23:[224,41,3,0],
    24:[224,36,2,0], 25:[224,34,2,0], 26:[224,34,3,0], 27:[224,43,3,0],
    28:[224,44,4,0], 29:[224,43,5,0], 30:[224,42,5,1], 31:[224,38,3,0],
    32:[224,50,3,0], 33:[224,49,3,0], 34:[224,49,4,0], 35:[224,53,5,0],
    36:[224,60,9,1], 37:[224,68,7,0], 38:[224,65,6,2], 39:[224,65,8,0],
    40:[224,63,6,0], 41:[224,64,4,0], 42:[224,56,4,0], 43:[224,69,14,3],
    44:[224,71,15,2], 45:[224,67,16,3], 46:[224,75,14,0], 47:[224,85,23,2],
    48:[224,79,24,5], 49:[224,84,13,1], 50:[224,92,20,7], 51:[224,86,20,0],
    52:[224,73,12,2], 53:[224,62,6,0], 54:[224,54,4,0], 55:[224,51,3,0],
    56:[224,52,4,1], 57:[224,46,6,0], 58:[224,55,0,0], 59:[224,118,41,6],
    60:[224,52,2,0], 61:[224,90,13,0], 62:[224,79,16,1], 63:[224,59,7,2],
}
n_layers = 64
d_model = 5120

def pick_extreme(U, n):
    norms = np.linalg.norm(U, axis=1)
    idx = np.argpartition(-norms, n-1)[:n]
    return U[idx, :10]

def project_norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

def random_cloud(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (50000, 50))
    return project_norm(pick_extreme(X, n))

def ripser_h0_h3(points, maxdim=3):
    if len(points) < 2:
        return [0] * (maxdim+1)
    rips = Rips(maxdim=maxdim, thresh=np.inf)
    dgms = rips.fit_transform(points)
    result = []
    for dim in range(maxdim+1):
        if dim < len(dgms):
            dgm = dgms[dim]
            finite = dgm[np.isfinite(dgm[:, 1]) & (dgm[:, 1] > dgm[:, 0])]
            result.append(len(finite))
        else:
            result.append(0)
    return result

# ===========================================================================
# PHASE 3: Contextual hidden states (fixed approach)
# ===========================================================================
log.info('=== Phase 3: Per-layer contextual hidden states ===')

from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
log.info(f'Model loaded in {time.time()-t0:.1f}s')

# Get text model
if hasattr(hf_model, 'model'):
    text_model = hf_model.model
else:
    text_model = hf_model.transformer

# Load contextual token list
with open(DATA_DIR / 'contextual_token_manifest.json') as f:
    manifest = json.load(f)
ctx_token_ids = manifest['token_ids']
N_CTX = min(len(ctx_token_ids), 5000)
log.info(f'Contextual tokens: {N_CTX}')

# Per-layer accumulator: dictionary of lists of numpy arrays
# Key fix: accumulate all batches, then np.concatenate at the end
ctx_accum = {layer_idx: [] for layer_idx in range(n_layers)}

def make_ctx_hook(layer_idx):
    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # hs is (B, 1, d_model) for singleton forward passes
        ctx_accum[layer_idx].append(hs.squeeze(1).detach().cpu().float().numpy())
    return hook

for layer_idx in range(n_layers):
    text_model.layers[layer_idx].register_forward_hook(make_ctx_hook(layer_idx))

# Run singleton forward passes for contextual tokens
device = hf_model.device
t0_fwd = time.time()
for start in range(0, N_CTX, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N_CTX)
    batch_ids = [[ctx_token_ids[i]] for i in range(start, end)]
    input_ids = torch.tensor(batch_ids, dtype=torch.long, device=device)

    with torch.no_grad():
        _ = text_model(input_ids)

    if (start // BATCH_SIZE) % 20 == 0:
        dt = time.time() - t0_fwd
        log.info(f'  {end}/{N_CTX} tokens ({dt:.1f}s)')

# Clear hooks
text_model._forward_hooks.clear()

# Concatenate all batches per layer
log.info(f'Forward passes done ({time.time()-t0_fwd:.1f}s), concatenating...')
ctx_hidden_full = {}
for layer_idx in range(n_layers):
    ctx_hidden_full[layer_idx] = np.concatenate(ctx_accum[layer_idx], axis=0)
    log.info(f'  Layer {layer_idx}: {ctx_hidden_full[layer_idx].shape}')

del ctx_accum, hf_model, text_model, tok
torch.cuda.empty_cache()

# ===========================================================================
# PHASE 4: Per-layer contextual persistence
# ===========================================================================
log.info('')
log.info('=== Phase 4: Per-layer contextual persistence ===')

contextual_persistence = {}
for layer_idx in range(n_layers):
    X = ctx_hidden_full[layer_idx][:N_CTX]
    X_c = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    proj = X_c @ Vt[:10].T
    points = project_norm(pick_extreme(proj, N_WITNESS))
    t0 = time.time()
    try:
        result = ripser_h0_h3(points, MD)
        dt = time.time() - t0
        contextual_persistence[layer_idx] = result
        log.info(f'  Layer {layer_idx}: {dt:.1f}s, H0-H3={result}')
    except Exception as e:
        log.error(f'  Layer {layer_idx}: Ripser failed: {e}')
        contextual_persistence[layer_idx] = [-1] * (MD+1)

del ctx_hidden_full

# ===========================================================================
# PHASE 5: Random baseline
# ===========================================================================
log.info('')
log.info('=== Phase 5: Random baseline ===')

random_persistence = []
for run in range(N_RANDOM_RUNS):
    points = random_cloud(N_WITNESS, 70000 + run)
    result = ripser_h0_h3(points, MD)
    random_persistence.append(result)
    log.info(f'  Run {run}: H0-H3={result}')

# ===========================================================================
# PHASE 6: Compute stats and print tables
# ===========================================================================
rand_stats = {}
for dim in range(MD+1):
    vals = [r[dim] for r in random_persistence]
    rand_stats[dim] = (np.mean(vals), np.std(vals))

log.info('')
log.info(f'Random baseline: H0={rand_stats[0][0]:.0f}±{rand_stats[0][1]:.0f}, H1={rand_stats[1][0]:.0f}±{rand_stats[1][1]:.0f}, H2={rand_stats[2][0]:.0f}±{rand_stats[2][1]:.0f}, H3={rand_stats[3][0]:.0f}±{rand_stats[3][1]:.0f}')

# ===========================================================================
# FINAL COMPARATIVE TABLES
# ===========================================================================
print()
print('=' * 110)
print('PER-LAYER PERSISTENCE (H0-H3): Singleton vs Contextual vs Random')
print(f'  Model: {MODEL_ID}, N={N_WITNESS}, {n_layers} layers')
print(f'  Random: H0={rand_stats[0][0]:.0f}±{rand_stats[0][1]:.0f}, H1={rand_stats[1][0]:.0f}±{rand_stats[1][1]:.0f}, H2={rand_stats[2][0]:.0f}±{rand_stats[2][1]:.0f}, H3={rand_stats[3][0]:.0f}±{rand_stats[3][1]:.0f}')
print('=' * 110)

print()
print('--- SINGLETON ---')
print(f'{"Layer":>6}  {"H0":>5}  {"H1":>5}  {"H2":>5}  {"H3":>5}  {"H1 z":>8}  {"H2 z":>8}  {"H3 z":>8}')
for layer_idx in range(n_layers):
    sp = singleton_persistence[layer_idx]
    z1 = (sp[1] - rand_stats[1][0]) / rand_stats[1][1] if rand_stats[1][1] > 0 else 0
    z2 = (sp[2] - rand_stats[2][0]) / rand_stats[2][1] if rand_stats[2][1] > 0 else 0
    z3 = (sp[3] - rand_stats[3][0]) / rand_stats[3][1] if rand_stats[3][1] > 0 else 0
    print(f'{layer_idx:>6}  {sp[0]:>5}  {sp[1]:>5}  {sp[2]:>5}  {sp[3]:>5}  {z1:+8.1f}  {z2:+8.1f}  {z3:+8.1f}')

print()
print('--- CONTEXTUAL ---')
print(f'{"Layer":>6}  {"H0":>5}  {"H1":>5}  {"H2":>5}  {"H3":>5}  {"H1 z":>8}  {"H2 z":>8}  {"H3 z":>8}')
for layer_idx in range(n_layers):
    cp = contextual_persistence[layer_idx]
    z1 = (cp[1] - rand_stats[1][0]) / rand_stats[1][1] if rand_stats[1][1] > 0 else 0
    z2 = (cp[2] - rand_stats[2][0]) / rand_stats[2][1] if rand_stats[2][1] > 0 else 0
    z3 = (cp[3] - rand_stats[3][0]) / rand_stats[3][1] if rand_stats[3][1] > 0 else 0
    print(f'{layer_idx:>6}  {cp[0]:>5}  {cp[1]:>5}  {cp[2]:>5}  {cp[3]:>5}  {z1:+8.1f}  {z2:+8.1f}  {z3:+8.1f}')

print()
print('--- H1 Evolution Across Layers ---')
print(f'{"Layer":>6}  {"Singleton":>10}  {"Contextual":>10}  {"Ratio ctx/sng":>14}')
for layer_idx in range(n_layers):
    s_h1 = singleton_persistence[layer_idx][1]
    c_h1 = contextual_persistence[layer_idx][1]
    ratio = c_h1 / s_h1 if s_h1 > 0 else float('inf')
    print(f'{layer_idx:>6}  {s_h1:>10}  {c_h1:>10}  {ratio:>14.3f}')

# Save
results = {
    'config': {
        'model_id': MODEL_ID, 'n_layers': n_layers, 'd_model': d_model,
        'n_witness': N_WITNESS, 'maxdim': MD, 'n_random_runs': N_RANDOM_RUNS,
    },
    'singleton': singleton_persistence,
    'contextual': contextual_persistence,
    'random': random_persistence,
}
import pickle
with open(DATA_DIR / 'persistence_per_layer.pkl', 'wb') as f:
    pickle.dump(results, f)
log.info(f'Saved to {DATA_DIR / "persistence_per_layer.pkl"}')

PYEOF

echo ""
echo "=== Done ==="
echo "Date: $(date)"
