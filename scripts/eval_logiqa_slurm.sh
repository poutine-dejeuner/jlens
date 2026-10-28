#!/bin/bash
#SBATCH --job-name=jl-logiqa
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/eval_logiqa_%j.out
#SBATCH --error=logs/eval_logiqa_%j.err

set -e

cd /mnt/home/jacob/vl/repos/jlens

echo "=== LogiQA eval ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Args: $@"
echo

if [ $# -eq 0 ]; then
    # Evaluate all missing steps
    uv run python scripts/eval_logiqa.py --missing
else
    # Evaluate a specific step
    uv run python scripts/eval_logiqa.py "$1"
fi

echo
echo "=== Done at $(date) ==="
