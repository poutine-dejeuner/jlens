#!/bin/bash
#SBATCH --job-name=jspace-test
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/jspace_test_%j.out
#SBATCH --error=logs/jspace_test_%j.err

# Quick test: validate Jacobian computation on 1 checkpoint with 10 prompts

set -e

echo "=== Test job $SLURM_JOB_ID ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

# Disable Triton JIT compilation (compute nodes lack C compiler)
export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

# Test with a single late checkpoint (most interesting) and 10 prompts
uv run python -m jlens.cli run \
    --model-id "EleutherAI/pythia-160m-deduped" \
    --dtype float16 \
    --n-prompts 10 \
    --max-seq-len 128 \
    --results-dir results_test \
    --checkpoints "143000"

echo "End: $(date)"
echo "=== Done ==="
