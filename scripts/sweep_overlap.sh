#!/usr/bin/env bash
#SBATCH --job-name=jlens-ov
#SBATCH --partition=batch
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/jlens_ov_%A_%a.out
#SBATCH --error=logs/jlens_ov_%A_%a.err

cd /mnt/home/jacob/vl/repos/jlens
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
find . -name '*.pyc' -delete 2>/dev/null; true
mkdir -p logs

echo "=== jlens overlap sweep ==="
echo "Node: $(hostname)  Task: ${SLURM_ARRAY_TASK_ID}  Date: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1
export PYTHONPATH="$(pwd):$PYTHONPATH"

uv run scripts/run_slice.py \
  --task-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --num-tasks 8 \
  --overlap \
  --n-prompts 1000 \
  --results-dir results

echo "Done: $(date)"
