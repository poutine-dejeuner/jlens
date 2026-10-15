#!/bin/bash
#SBATCH --job-name=jlens-plot-12b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/jlens-plot-12b_%j.out
#SBATCH --error=logs/jlens-plot-12b_%j.err

set -e
cd /mnt/home/jacob/vl/repos/jlens
mkdir -p figures/pythia-12b logs

echo "=== Plotting 12B evolution ==="
echo "Start: $(date)"

uv run python scripts/plot_evolution.py \
    --results-dir /data/jlens/pythia-12b-deduped \
    --output-dir figures/pythia-12b

echo "End: $(date)"
echo "==="
ls -lh figures/pythia-12b/
echo "=== Done ==="
