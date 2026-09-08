#!/bin/bash
#SBATCH --job-name=wmt_eval
#SBATCH --output=logs/wmt_eval_%A_%a.out
#SBATCH --error=logs/wmt_eval_%A_%a.err
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=batch

# Evaluate Pythia-160M-deduped on WMT14 fr-en
# Direct inference — no lm-eval overhead
# 8 slices × ~19 checkpoints each
# ~60s/checkpoint: model load + 3003 generations + sacrebleu.

set -euo pipefail

echo "=== WMT14 fr-en eval ==="
echo "Slice: ${SLURM_ARRAY_TASK_ID} / 8"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================="

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs results/wmt_eval

uv run python scripts/eval_wmt.py \
    --slice "${SLURM_ARRAY_TASK_ID}" \
    --num-slices 8 \
    --device cuda \
    --output-dir results/wmt_eval

echo "=== Done slice ${SLURM_ARRAY_TASK_ID} ==="
