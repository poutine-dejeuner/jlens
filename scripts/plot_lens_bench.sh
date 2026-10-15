#!/bin/bash
#SBATCH --job-name=vincent-jl-plot-lens
#SBATCH --partition=batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=logs/jlens-plot-lens_%j.out
#SBATCH --error=logs/jlens-plot-lens_%j.err

cd /mnt/home/jacob/vl/repos/jlens
echo "=== Lens benchmark plots ==="
echo "Start: $(date)"
echo

uv run python scripts/plot_lens_bench.py

echo
echo "End: $(date)"
