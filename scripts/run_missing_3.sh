#!/bin/bash
#SBATCH --job-name=jlens-m3
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jlens_miss_%j.out
#SBATCH --error=logs/jlens_miss_%j.err

set -e
echo "=== jlens missing chunk 3: 11 steps ==="
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
    --checkpoints "81000,82000,83000,84000,85000,86000,87000,88000,89000,90000,98000" \
    --prompt-seed 42
echo "End: $(date)"
echo "=== Done ==="
