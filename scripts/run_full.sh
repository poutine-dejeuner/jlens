#!/bin/bash
#SBATCH --job-name=jspace-full
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jspace_full_%A_%a.out
#SBATCH --error=logs/jspace_full_%A_%a.err
#SBATCH --array=0-7

# jlens: Full Jacobian lens spectral evolution across Pythia checkpoints
# 8 GPU job array, each processes ~19 checkpoints
# Total: 153 checkpoints, 1000 prompts each
# Estimated: ~30h per job array element

set -e

echo "=== jlens full pipeline ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs results

# Get all checkpoint steps (skip step0)
ALL_STEPS=$(uv run python -c "
from jlens.checkpoint_manager import get_checkpoint_steps
steps = get_checkpoint_steps(1, 143000)
print(','.join(map(str, steps)))
")

# Convert to array
IFS=',' read -ra STEP_ARRAY <<< "$ALL_STEPS"
TOTAL=${#STEP_ARRAY[@]}

# Split across 8 array tasks
CHUNK_SIZE=$(( (TOTAL + 7) / 8 ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
if [ $END -gt $TOTAL ]; then
    END=$TOTAL
fi

# Build comma-separated list for this task
MY_STEPS=""
for (( i=$START; i<$END; i++ )); do
    if [ -z "$MY_STEPS" ]; then
        MY_STEPS="${STEP_ARRAY[$i]}"
    else
        MY_STEPS="$MY_STEPS,${STEP_ARRAY[$i]}"
    fi
done

NUM_STEPS=$(( END - START ))
echo "Total checkpoints: ${TOTAL}"
echo "This task processes: ${NUM_STEPS} checkpoints (${START} to ${END-1})"
echo "First: ${STEP_ARRAY[$START]}, Last: ${STEP_ARRAY[$((END-1))]}"
echo "Steps: ${MY_STEPS}"

uv run python -m jlens.cli run \
    --model-id "EleutherAI/pythia-160m-deduped" \
    --dtype float16 \
    --n-prompts 1000 \
    --max-seq-len 128 \
    --results-dir results \
    --checkpoints "${MY_STEPS}" \
    --prompt-seed 42

echo "End: $(date)"
echo "=== Done ==="
