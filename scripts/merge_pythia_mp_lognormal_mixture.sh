#!/usr/bin/env bash
#SBATCH --job-name=pythia-mp-ln-merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --output=logs/pythia_mp_ln_merge_%j.out
#SBATCH --error=logs/pythia_mp_ln_merge_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?This script must be submitted with sbatch.}"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

shopt -s nullglob
parts=(results/pythia_mp_lognormal_mixture_parts/*.csv)
expected_rows=$(( $(wc -l < docs/paper/model_comparison_grid_pythia12b_logskewnormal.csv) - 1 ))
actual_rows=0
for part in "${parts[@]}"; do
    actual_rows=$((actual_rows + $(($(wc -l < "$part") - 1))))
done
if [[ "$actual_rows" -ne "$expected_rows" ]]; then
    echo "Expected $expected_rows mixture rows, found $actual_rows." >&2
    exit 1
fi
{
    head -n 1 "${parts[0]}"
    for part in "${parts[@]}"; do
        tail -n +2 "$part"
    done
} > results/pythia_mp_lognormal_mixture_grid.csv

uv run \
    --with numpy \
    --with scipy \
    --with h5py \
    --with matplotlib \
    python scripts/add_mp_lognormal_mixture_to_comparison.py \
    --base-csv docs/paper/model_comparison_grid_pythia12b_logskewnormal.csv \
    --mixture-csv results/pythia_mp_lognormal_mixture_grid.csv
