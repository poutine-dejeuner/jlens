#!/usr/bin/env bash
#SBATCH --job-name=pythia-skew-grid
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pythia_skewnormal_heatmap_%j.out
#SBATCH --error=logs/pythia_skewnormal_heatmap_%j.err

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

uv run \
    --with numpy \
    --with scipy \
    --with h5py \
    --with matplotlib \
    python scripts/plot_pythia_skewnormal_heatmap.py \
    --results-dir /data/jlens/pythia-12b-deduped \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --csv docs/paper/model_comparison_grid_pythia12b_logskewnormal.csv \
    --figure figures/pythia_skewnormal_heatmap.png
