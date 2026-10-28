#!/usr/bin/env bash
#SBATCH --job-name=circle-probe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=logs/circle_probe_%j.out
#SBATCH --error=logs/circle_probe_%j.err

# Circle probe on Qwen3.6-27B (see jlens/circle_probe.py, PROJECT1.md).
# 1. GPU: templated sentences for cyclic concepts + controls, capture residual,
#    normed residual, attention value/head outputs, DeltaNet head outputs
#    at the concept word, all layers.
# 2. CPU: per (concept, rep, layer, head) circularity scores + permutation nulls.
# 3. Figures.
# 27B bf16 fits on one B300; extraction is ~1.2K short sentences -> 1 GPU.
set -euo pipefail
echo "=== Circle probe: Qwen3.6-27B ==="
echo "Date: $(date)  Host: $(hostname)"

cd /mnt/home/jacob/vl/repos/jlens
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
# forked scoring workers: keep BLAS single-threaded
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

OUT=/data/jlens/qwen3.6-27b/circle_probe
mkdir -p "$OUT"

if [ ! -f "$OUT/probe.h5" ]; then
    uv run python -m jlens.circle_probe extract --out "$OUT/probe.h5"
fi
uv run python -m jlens.circle_probe score --probe "$OUT/probe.h5" \
    --out "$OUT/scores.csv" --n-perm 200 --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
uv run python scripts/plot_circle_probe.py --scores "$OUT/scores.csv" \
    --probe "$OUT/probe.h5" --out-dir figures

echo "=== Done: $(date) ==="
