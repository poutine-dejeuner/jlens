#!/usr/bin/env bash
#SBATCH --job-name=tm-persist3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/tm_persist3_%j.out
#SBATCH --error=logs/tm_persist3_%j.err

set -euo pipefail
echo "=== Higher-order persistence ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Install ripser and persim on the compute node
uv pip install ripser persim 2>&1 | tail -5

uv run python -c "
from __future__ import annotations
import logging, time
import numpy as np
from scipy.spatial.distance import pdist, squareform
import h5py
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

# -----------------------------------------------------------------------
# Load manifolds
# -----------------------------------------------------------------------
logging.info('Loading manifolds...')

with h5py.File('/data/jlens/qwen3.6-27b/singleton/manifold.h5', 'r') as f:
    U_single = f['eigenvectors'][:]
logging.info(f'Option A: V={U_single.shape[0]}, d={U_single.shape[1]}')

with h5py.File('/data/jlens/qwen3.6-27b/step0000/manifold.h5', 'r') as f:
    U_context = f['eigenvectors'][:]
logging.info(f'Option B: V={U_context.shape[0]}, d={U_context.shape[1]}')

# Project to top PC dimensions (use 10D — enough to capture topology
# without the curse of dimensionality making all points equidistant)
N_PROJ = 10

def pick_extreme(U, n):
    norms = np.linalg.norm(U, axis=1)
    idx = np.argpartition(-norms, n - 1)[:n]
    return U[idx]

# -----------------------------------------------------------------------
# Option A: varying subsample sizes for H0/H1/H2
# -----------------------------------------------------------------------
from ripser import ripser

for label, U_full, color in [
    ('Option A (singleton)', U_single, 'blue'),
    ('Option B (contextual)', U_context, 'red'),
]:
    logging.info(f'')
    logging.info(f'=== {label} ===')

    for n_pts in [300, 500, 800]:
        logging.info(f'--- n={n_pts}, d_proj={N_PROJ} ---')

        X_raw = pick_extreme(U_full, n_pts)
        X = X_raw[:, :N_PROJ]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        t0 = time.time()
        # maxdim=2 gives H0, H1, H2
        # thresh=inf computes full filtration
        # metric='euclidean' uses the distance matrix
        dgms = ripser(X, maxdim=2, thresh=np.inf, distance_matrix=False)
        dt = time.time() - t0
        logging.info(f'  Ripser done in {dt:.1f}s')

        for dim, dgm in enumerate(dgms['dgms']):
            if len(dgm) == 0:
                logging.info(f'  H{dim}: 0 bars')
                continue
            # Remove infinite bars for stats
            finite = dgm[dgm[:, 1] < np.inf]
            lifespans = finite[:, 1] - finite[:, 0]
            n_inf = len(dgm) - len(finite)

            if len(finite) > 0:
                logging.info(f'  H{dim}: {len(dgm)} bars ({n_inf} inf), '
                             f'mean lifespan={np.mean(lifespans):.4f}, '
                             f'median={np.median(lifespans):.4f}, '
                             f'max={np.max(lifespans):.4f}, '
                             f'p90={np.percentile(lifespans, 90):.4f}')
            else:
                logging.info(f'  H{dim}: {len(dgm)} bars ({n_inf} inf), all infinite')

        # Compute persistent entropy (measure of topological complexity)
        for dim, dgm in enumerate(dgms['dgms']):
            finite = dgm[dgm[:, 1] < np.inf]
            if len(finite) > 0:
                lifespans = finite[:, 1] - finite[:, 0]
                total = lifespans.sum()
                if total > 0:
                    p = lifespans / total
                    entropy = -np.sum(p * np.log(p + 1e-10))
                    logging.info(f'  H{dim} persistent entropy: {entropy:.4f}')

logging.info('')
logging.info('Done!')
"
echo "Done at $(date)"
