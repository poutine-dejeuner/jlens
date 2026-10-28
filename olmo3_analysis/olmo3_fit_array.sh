#!/bin/bash
#SBATCH --job-name=olmo3-3mix-fit
#SBATCH --partition=batch
#SBATCH --nodelist=b300-nsu4-0-37x9xy
#SBATCH --array=0-60%40
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/home/jacob/vl/repos/jlens/logs/olmo3_fit_%A_%a.out
#SBATCH --error=/mnt/home/jacob/vl/repos/jlens/logs/olmo3_fit_%A_%a.err

set -euo pipefail

# Usage: sbatch --array=0-$((N_STEPS-1))%40 olmo3_fit_array.sh [OUT_DIR] [SEED_DIR]
#   N_STEPS = number of step* dirs with spectra.h5 in /data/jlens/olmo3-32b (default array: 61)
#   pass 1: sbatch olmo3_fit_array.sh results/olmo3_3mix_pass1
#   pass 2: sbatch --dependency=afterok:<pass1> olmo3_fit_array.sh results/olmo3_3mix_parts results/olmo3_3mix_pass1
OUT_DIR="${1:-results/olmo3_3mix_parts}"
SEED_DIR="${2:-}"

echo "=== Olmo3-32B 3-component fit array task ${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname)  CPUs: ${SLURM_CPUS_PER_TASK}  Start: $(date)"

WORK_DIR=/mnt/home/jacob/vl/repos/jlens/olmo3_analysis
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR"

# Build the sorted list of available steps (mirrors plot_logskewnormal_fits.checkpoint_paths).
mapfile -t STEPS < <(python3 - <<'PY'
from pathlib import Path
from plot_pythia_skewnormal_heatmap import checkpoint_paths
steps = sorted(checkpoint_paths(Path("/data/jlens/olmo3-32b")).keys())
print("\n".join(str(s) for s in steps))
PY
)
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#STEPS[@]}" ]; then
    echo "task ${SLURM_ARRAY_TASK_ID} >= ${#STEPS[@]} available steps" >&2
    exit 1
fi
STEP="${STEPS[$SLURM_ARRAY_TASK_ID]}"
OUT_CSV="${OUT_DIR}/step_${STEP}.csv"

if [ -f "$OUT_CSV" ]; then
    echo "[fit] step ${STEP}: CSV already exists, skipping"
    exit 0
fi

python3 fit_pythia_three_component_mixture.py \
    --results-dir /data/jlens/olmo3-32b \
    --steps "$STEP" \
    --csv "$OUT_CSV" \
    ${SEED_DIR:+--seed-dir "$SEED_DIR"}

echo "[fit] step ${STEP}: wrote $OUT_CSV  End: $(date)"
