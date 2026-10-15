#!/bin/bash
#SBATCH --job-name=jlens-plot
#SBATCH --output=logs/jlens-plot_%j.out
#SBATCH --error=logs/jlens-plot_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --nodelist=b300-nsu4-0-37x9xf

set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens

echo "=== Plotting individual figures ==="
echo "Start: $(date)"

source .venv/bin/activate
export HF_HOME=/mnt/home/jacob/.cache/huggingface

uv run python scripts/plot_individual.py

echo "=== Done ==="
echo "End: $(date)"
