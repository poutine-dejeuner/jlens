#!/usr/bin/env bash
#SBATCH --job-name=pythia-skew
#SBATCH --array=0-47%48
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=logs/pythia_skew_%A_%a.out
#SBATCH --error=logs/pythia_skew_%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?This script must be submitted with sbatch.}"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

mapfile -t steps < <(
    find /data/jlens/pythia-12b-deduped -maxdepth 1 -type d -name 'step*' \
        -printf '%f\n' |
        sed 's/^step//' |
        sort -n
)
step="${steps[SLURM_ARRAY_TASK_ID]:?Missing checkpoint for array index ${SLURM_ARRAY_TASK_ID}}"

uv run \
    --with numpy \
    --with scipy \
    --with h5py \
    --with matplotlib \
    python scripts/plot_pythia_skewnormal_heatmap.py \
    --results-dir /data/jlens/pythia-12b-deduped \
    --workers 1 \
    --models logskewnormal \
    --steps "$step" \
    --csv "results/pythia_logskewnormal_parts/step_${step}.csv"
