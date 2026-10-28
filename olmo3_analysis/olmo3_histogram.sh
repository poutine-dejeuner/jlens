#!/bin/bash
#SBATCH --job-name=olmo3-hist-wikitext
#SBATCH --partition=batch
#SBATCH --nodelist=b300-nsu4-0-37x9xy
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/home/jacob/vl/repos/jlens/logs/olmo3_hist_%A.out
#SBATCH --error=/mnt/home/jacob/vl/repos/jlens/logs/olmo3_hist_%A.err

set -euo pipefail

echo "=== Olmo3-32B wikitext spectrum histogram dashboard ==="
echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}  Start: $(date)"

WORK_DIR=/mnt/home/jacob/vl/repos/jlens/olmo3_analysis
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p figures logs

python3 plot_spectrum_histogram.py \
    --results-dir /data/jlens/olmo3-32b \
    --label wikitext \
    --model-label "Olmo-3-1125-32B" \
    --output figures/spectra_histogram_wikitext.html \
    --bins 90

echo "End: $(date)"
echo "=== Olmo3-32B wikitext spectrum histogram dashboard complete ==="
