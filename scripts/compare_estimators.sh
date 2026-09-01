#!/usr/bin/env bash
#SBATCH --job-name=jlens-compare
#SBATCH --gres=gpu:b300:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/compare_%j.out
#SBATCH --error=logs/compare_%j.err
#SBATCH --partition=batch

set -e
cd /mnt/home/jacob/vl/repos/jlens

find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

mkdir -p logs
echo "=== Upstream vs Blockwise comparison ==="
echo "Node: $(hostname)  Date: $(date)"

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1
export PYTHONPATH="$(pwd):$PYTHONPATH"

uv run python scripts/compare_estimators.py

echo "Done at $(date)"
