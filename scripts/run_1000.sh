#!/bin/bash
#SBATCH --job-name=jspace-143k
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jspace_143k_%j.out
#SBATCH --error=logs/jspace_143k_%j.err

# jlens: Full Jacobian lens analysis on step 143000
# 1000 prompts, blockwise Jacobian computation
# Estimated: ~1.6h runtime

set -e

echo "=== jlens step 143000 (1000 prompts) ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

uv run python -m jlens.cli run \
    --model-id "EleutherAI/pythia-160m-deduped" \
    --dtype float16 \
    --n-prompts 1000 \
    --max-seq-len 128 \
    --results-dir results \
    --checkpoints "143000"

echo "End: $(date)"
echo "=== Done ==="
