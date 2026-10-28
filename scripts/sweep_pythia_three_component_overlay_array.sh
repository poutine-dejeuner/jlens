#!/usr/bin/env bash
#SBATCH --job-name=pythia-3mix-overlay
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --output=logs/pythia_3mix_overlay_%A_%a.out
#SBATCH --error=logs/pythia_3mix_overlay_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch.}"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
steps=(1 47000 97000 143000)
step="${steps[SLURM_ARRAY_TASK_ID]}"
uv run --with numpy --with scipy --with h5py --with matplotlib python scripts/fit_pythia_three_component_mixture.py --results-dir /data/jlens/pythia-12b-deduped --steps "$step" --csv "results/pythia_3mix_overlay_parts/step_${step}.csv"
