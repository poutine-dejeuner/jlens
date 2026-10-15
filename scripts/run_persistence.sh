#!/usr/bin/env bash
#SBATCH --job-name=tm-persist
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/tm_persist_%j.out
#SBATCH --error=logs/tm_persist_%j.err

# Persistence diagrams of token embeddings:
#   1) Re-run Option A (singleton) to /data/jlens/qwen3.6-27b/step0000_singleton/
#   2) Compute H0/H1 persistence for both Option A and Option B

set -euo pipefail

echo "=== Persistence Diagrams ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# ======================================================================
# Step 1: Re-run Option A (singleton) thought manifold
# ======================================================================
echo "=== Step 1: Option A singleton manifold ==="

uv run python -c "
from __future__ import annotations
import logging, time, json
import torch
import numpy as np
import h5py
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('thought_manifold.singleton')

logger.info('=== Qwen3.6-27B singleton thought manifold (Option A) ===')

from transformers import AutoModelForCausalLM, AutoTokenizer
t0 = time.time()
hf_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.6-27B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)
logger.info(f'Model loaded in {time.time()-t0:.1f}s')

from jlens._bridge import get_upstream
from jlens.thought_manifold import thought_manifold_pipeline

upstream = get_upstream()
lens_model = upstream.from_hf(hf_model, None)
logger.info(f'Layout: path={lens_model.layout.path}, n_layers={lens_model.n_layers}, d_model={lens_model.d_model}')

t1 = time.time()
result = thought_manifold_pipeline(
    lens_model,
    k=50,
    batch_size=512,
    center=True,
)
dt = time.time() - t1
logger.info(f'Pipeline done in {dt:.1f}s ({dt/60:.1f}m)')

from jlens.storage import save_thought_manifold

# Save singleton to a separate directory from contextual
singleton_dir = Path('/data/jlens/qwen3.6-27b/singleton')
singleton_dir.mkdir(parents=True, exist_ok=True)
with h5py.File(singleton_dir / 'manifold.h5', 'w') as f:
    f.create_dataset('eigenvectors', data=result['eigenvectors'])
    f.create_dataset('eigenvalues', data=result['eigenvalues'])
    f.create_dataset('explained_variance_ratio', data=result['explained_variance_ratio'])
    f.attrs['V'] = result.get('V', result['eigenvectors'].shape[0])
    f.attrs['d'] = result.get('d', 0)
    f.attrs['centered'] = result.get('centered', True)
    f.attrs['step'] = 0
import json
with open(singleton_dir / 'metadata.json', 'w') as mf:
    json.dump({'step': 0, 'model_id': 'Qwen/Qwen3.6-27B',
               'V': int(result.get('V', 0)), 'd': int(result.get('d', 0)),
               'k': 50, 'centered': True}, mf)

results_dir = singleton_dir

logger.info(f'V={result[\"V\"]}, d={result[\"d\"]}, centered={result[\"centered\"]}')
logger.info(f'Top-10 eigenvalues: {result[\"eigenvalues\"][:10]}')
logger.info('Explained variance:')
for i in range(10):
    logger.info(f'  l_{i+1}: {result[\"eigenvalues\"][i]:.2f} '
                f'({100*result[\"explained_variance_ratio\"][i]:.2f}%)')
cum_var = 100 * result['explained_variance_ratio'].sum()
logger.info(f'Cumulative (top-50): {cum_var:.2f}%')
gap = result['eigenvalues'][0] / result['eigenvalues'][1]
logger.info(f'Spectral gap l1/l2: {gap:.2f}')
sv = result['singular_values']
eff_rank = sv.sum() ** 2 / (sv ** 2).sum()
logger.info(f'Effective rank: {eff_rank:.2f}')

del hf_model
torch.cuda.empty_cache()
logger.info('Singleton run complete!')
"

echo "Step 1 complete at $(date)"

# ======================================================================
# Step 2: Compute persistence diagrams for both manifolds
# ======================================================================
echo "=== Step 2: Persistence diagrams ==="

uv run python -c "
from __future__ import annotations
import logging, time
import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict, deque
from itertools import combinations
import h5py
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)

# -----------------------------------------------------------------------
# Persistence algorithms
# -----------------------------------------------------------------------

def persistence_h0(dist_matrix):
    '''0-dim persistence from single-linkage dendrogram (MST).'''
    n = dist_matrix.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    edges = [(dist_matrix[i, j], i, j) for i, j in zip(tri_i, tri_j)]
    edges.sort()

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    deaths = np.full(n, np.inf)
    for d, i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            if rank[ri] < rank[rj]:
                deaths[ri] = d
                parent[ri] = rj
            else:
                deaths[rj] = d
                parent[rj] = ri
                if rank[ri] == rank[rj]:
                    rank[ri] += 1

    max_d = edges[-1][0] if edges else 0
    deaths[np.isinf(deaths)] = max_d
    return np.stack([np.zeros(n), deaths], axis=1)


def persistence_h1_mst(dist_matrix):
    '''Approximate H1 persistence: for each non-MST edge, the shortest
    cycle it creates via the MST path gives a 1-cycle birth/death pair.'''
    n = dist_matrix.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    edges = [(dist_matrix[i, j], i, j) for i, j in zip(tri_i, tri_j)]
    edges.sort()

    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst_edges = set()
    adj = defaultdict(list)
    for d, i, j in edges:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            mst_edges.add((min(i, j), max(i, j)))
            adj[i].append((j, d))
            adj[j].append((i, d))

    h1_pairs = []
    for d, i, j in edges:
        if (min(i, j), max(i, j)) in mst_edges:
            continue
        # BFS from i to j through MST to find max edge on unique path
        visited = {i: (None, 0.0)}
        q = deque([i])
        while q:
            node = q.popleft()
            if node == j:
                break
            for nb, w in adj[node]:
                if nb not in visited:
                    visited[nb] = (node, max(visited[node][1], w))
                    q.append(nb)
        if j in visited:
            birth = visited[j][1]  # max MST edge on path
            death = d
            if death > birth + 1e-10:
                h1_pairs.append((birth, death))

    return np.array(h1_pairs) if h1_pairs else np.zeros((0, 2))


# -----------------------------------------------------------------------
# Load manifolds
# -----------------------------------------------------------------------

def load_manifold(path):
    with h5py.File(path, 'r') as f:
        return {
            'eigenvectors': f['eigenvectors'][:],
            'eigenvalues': f['eigenvalues'][:],
            'explained_variance_ratio': f['explained_variance_ratio'][:],
        }

logging.info('Loading Option A (singleton) manifold...')
ma = load_manifold('/data/jlens/qwen3.6-27b/step0000_singleton/manifold.h5')
U_single = ma['eigenvectors']
ev_single = ma['explained_variance_ratio']

logging.info('Loading Option B (contextualized) manifold...')
mb = load_manifold('/data/jlens/qwen3.6-27b/step0000/manifold.h5')
U_context = mb['eigenvectors']
ev_context = mb['explained_variance_ratio']

logging.info(f'Option A: V={U_single.shape[0]}, d={U_single.shape[1]}, '
             f'top-10 var={ev_single[:10].sum()*100:.1f}%')
logging.info(f'Option B: V={U_context.shape[0]}, d={U_context.shape[1]}, '
             f'top-10 var={ev_context[:10].sum()*100:.1f}%')

# -----------------------------------------------------------------------
# Compute persistence (H0 + H1) for both
# -----------------------------------------------------------------------
N_SUBSAMPLE = 5000
N_PROJ = 20

def pick_extreme(U, n):
    '''Pick n tokens with highest L2 norm in PC space.'''
    norms = np.linalg.norm(U, axis=1)
    idx = np.argpartition(-norms, n - 1)[:n]
    return idx, U[idx]

def compute_both(U_full, label):
    logging.info(f'--- {label}: subsample={N_SUBSAMPLE}, proj={N_PROJ} ---')
    idx, X_full = pick_extreme(U_full, N_SUBSAMPLE)
    X = X_full[:, :N_PROJ]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    logging.info(f'{label}: computing {N_SUBSAMPLE}x{N_SUBSAMPLE} distance matrix...')
    t0 = time.time()
    dm = squareform(pdist(X, metric='euclidean'))
    logging.info(f'{label}: DM done in {time.time()-t0:.1f}s, '
                 f'range=[{dm.min():.3f}, {dm.max():.3f}]')

    t0 = time.time()
    h0 = persistence_h0(dm)
    logging.info(f'{label}: H0 done in {time.time()-t0:.1f}s, {len(h0)} bars')

    t0 = time.time()
    h1 = persistence_h1_mst(dm)
    logging.info(f'{label}: H1 done in {time.time()-t0:.1f}s, {len(h1)} bars')

    return h0, h1, dm

h0_single, h1_single, dm_single = compute_both(U_single, 'Option A')
h0_context, h1_context, dm_context = compute_both(U_context, 'Option B')

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
logging.info('')
logging.info('=' * 60)
logging.info('PERSISTENCE DIAGRAM COMPARISON')
logging.info('=' * 60)

for name, h0, h1, dm in [
    ('Option A (singleton)', h0_single, h1_single, dm_single),
    ('Option B (contextual)', h0_context, h1_context, dm_context),
]:
    ls_h0 = h0[:, 1] - h0[:, 0]
    logging.info(f'')
    logging.info(f'{name}:')
    logging.info(f'  H0 (connected components): {len(h0)} bars')
    logging.info(f'    Mean lifespan:  {np.mean(ls_h0):.4f}')
    logging.info(f'    Median lifespan: {np.median(ls_h0):.4f}')
    logging.info(f'    90th pct:       {np.percentile(ls_h0, 90):.4f}')
    logging.info(f'    Max lifespan:   {np.max(ls_h0):.4f}')
    logging.info(f'    >0.1:           {np.sum(ls_h0 > 0.1)} / {len(ls_h0)}')

    logging.info(f'')
    logging.info(f'  H1 (1-cycles): {len(h1)} bars')
    if len(h1):
        ls_h1 = h1[:, 1] - h1[:, 0]
        logging.info(f'    Mean lifespan:  {np.mean(ls_h1):.4f}')
        logging.info(f'    Median lifespan: {np.median(ls_h1):.4f}')
        logging.info(f'    90th pct:       {np.percentile(ls_h1, 90):.4f}')
        logging.info(f'    Max lifespan:   {np.max(ls_h1):.4f}')
        logging.info(f'    >0.05:          {np.sum(ls_h1 > 0.05)} / {len(ls_h1)}')

    logging.info(f'')
    logging.info(f'  Distance matrix stats:')
    logging.info(f'    Min:         {dm.min():.4f}')
    logging.info(f'    Mean:        {dm.mean():.4f}')
    logging.info(f'    Max:         {dm.max():.4f}')
    logging.info(f'    % in [0, 1]: {100 * np.mean((dm >= 0) & (dm <= 1)):.1f}%')

# -----------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------
outdir = Path('/data/jlens/qwen3.6-27b')
np.savez(
    outdir / 'persistence_diagrams.npz',
    h0_single=h0_single,
    h1_single=h1_single,
    h0_context=h0_context,
    h1_context=h1_context,
)

logging.info('')
logging.info(f'Saved to {outdir / \"persistence_diagrams.npz\"}')
logging.info('Done!')
"

echo ""
echo "=== All Done ==="
echo "Date: $(date)"
