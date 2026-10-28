#!/usr/bin/env bash
#SBATCH --job-name=pythia-3mix-overlay-merge
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:15:00
#SBATCH --output=logs/pythia_3mix_overlay_merge_%j.out
#SBATCH --error=logs/pythia_3mix_overlay_merge_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch.}"
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
uv run --with numpy --with scipy --with h5py --with matplotlib python scripts/plot_three_component_mixture_overlays.py --results-dir /data/jlens/pythia-12b-deduped --fit-dir results/pythia_3mix_overlay_parts --steps 1 47000 97000 143000 --output-dir figures
