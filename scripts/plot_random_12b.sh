#!/bin/bash
#SBATCH --job-name=vincent-jl-rndplot
#SBATCH --partition=batch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=logs/jlens-plot-random_%j.out
#SBATCH --error=logs/jlens-plot-random_%j.err

set -e
cd /mnt/home/jacob/vl/repos/jlens
mkdir -p figures/pythia-12b-random logs

echo "=== Random vs Wikitext comparison ==="
echo "Start: $(date)"

.venv/bin/python -u scripts/plot_random_comparison.py \
    --wikitext-dir /data/jlens/pythia-12b-deduped \
    --random-dir /data/jlens/pythia-12b-deduped-random \
    --output-dir figures/pythia-12b-random

echo "End: $(date)"
echo "==="
ls -lh figures/pythia-12b-random/
echo "=== Done ==="
