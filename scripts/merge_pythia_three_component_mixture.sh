#!/usr/bin/env bash
#SBATCH --job-name=pythia-3mix-merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --output=logs/pythia_3mix_merge_%j.out
#SBATCH --error=logs/pythia_3mix_merge_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch.}"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
shopt -s nullglob
parts=(results/pythia_3mix_parts/*.csv)
if [[ "${#parts[@]}" -ne 48 ]]; then
    echo "Expected 48 mixture shards, found ${#parts[@]}." >&2
    exit 1
fi
{
    head -n 1 "${parts[0]}"
    for part in "${parts[@]}"; do tail -n +2 "$part"; done
} > docs/paper/pythia_three_component_mixture.csv
uv run --with numpy --with scipy --with h5py --with matplotlib python scripts/plot_three_component_mixture_heatmap.py --csv docs/paper/pythia_three_component_mixture.csv --output figures/pythia_three_component_mixture_composition_heatmap.png
