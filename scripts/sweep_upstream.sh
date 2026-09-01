#!/usr/bin/env bash
#SBATCH --job-name=jlens
#SBATCH --partition=batch
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/jlens_%j_%a.out
#SBATCH --error=logs/jlens_%j_%a.err

cd /mnt/home/jacob/vl/repos/jlens
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
find . -name '*.pyc' -delete 2>/dev/null; true
mkdir -p logs

echo "=== jlens sweep ==="
echo "Node: $(hostname)  Task: ${SLURM_ARRAY_TASK_ID}  Date: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1
export PYTHONPATH="$(pwd):$PYTHONPATH"

uv run scripts/run_slice.py --task-id "${SLURM_ARRAY_TASK_ID:-0}" --num-tasks 8

echo "Done: $(date)"
