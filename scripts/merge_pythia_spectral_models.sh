#!/usr/bin/env bash
#SBATCH --job-name=pythia-model-merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --output=logs/pythia_model_merge_%j.out
#SBATCH --error=logs/pythia_model_merge_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?This script must be submitted with sbatch.}"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob
skew_parts=(results/pythia_logskewnormal_parts/*.csv)
expected_rows=$(( $(wc -l < results/pythia_mp_model_grid.csv) - 1 ))
actual_rows=0
for part in "${skew_parts[@]}"; do
    actual_rows=$((actual_rows + $(($(wc -l < "$part") - 1))))
done
if [[ "$actual_rows" -ne "$expected_rows" ]]; then
    echo "Expected $expected_rows skew-normal rows, found $actual_rows." >&2
    exit 1
fi
{
    head -n 1 "${skew_parts[0]}"
    for part in "${skew_parts[@]}"; do
        tail -n +2 "$part"
    done
} > results/pythia_logskewnormal_model_grid.csv

uv run \
    --with numpy \
    --with scipy \
    --with h5py \
    --with matplotlib \
    python scripts/combine_pythia_model_comparison.py \
    --mp-csv results/pythia_mp_model_grid.csv \
    --logskewnormal-csv results/pythia_logskewnormal_model_grid.csv \
    --semicircle-csv results/pythia_semicircle_model_grid.csv \
    --output-csv docs/paper/model_comparison_grid_pythia12b_logskewnormal.csv \
    --figure figures/pythia_skewnormal_heatmap.png
