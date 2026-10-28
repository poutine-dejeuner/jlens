#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pythia_%x_%j.out
#SBATCH --error=logs/pythia_%x_%j.err

set -euo pipefail

model="${1:?usage: sbatch --job-name=pythia-MODEL scripts/sweep_pythia_spectral_model.sh MODEL}"
cd "${SLURM_SUBMIT_DIR:?This script must be submitted with sbatch.}"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

uv run \
    --with numpy \
    --with scipy \
    --with h5py \
    --with matplotlib \
    python scripts/plot_pythia_skewnormal_heatmap.py \
    --results-dir /data/jlens/pythia-12b-deduped \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --models "$model" \
    --csv "results/pythia_${model}_model_grid.csv"
