#!/bin/bash
#SBATCH --job-name=jlens-miss
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jlens_miss_%j.out
#SBATCH --error=logs/jlens_miss_%j.err

# jlens: Catch up missing checkpoints
# Processes checkpoints not already in results/

set -e

echo "=== jlens catch-up ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

# Get all expected steps
ALL_STEPS=$(uv run python -c "
from jlens.checkpoint_manager import get_checkpoint_steps
steps = get_checkpoint_steps(1, 143000)
print(','.join(str(s) for s in steps))
")

IFS=',' read -ra STEP_ARRAY <<< "$ALL_STEPS"

# Filter out steps that already have results
MISSING=""
for step in "${STEP_ARRAY[@]}"; do
    if [ ! -d "results/step${step}" ] && [ ! -d "results/step$(printf '%07d' $step)" ]; then
        if [ -z "$MISSING" ]; then
            MISSING="$step"
        else
            MISSING="$MISSING,$step"
        fi
    fi
done

NUM_MISSING=$(echo "$MISSING" | tr ',' '\n' | wc -l)
echo "Missing: $NUM_MISSING checkpoints"
echo "First: $(echo $MISSING | cut -d',' -f1)"
echo "Last: $(echo $MISSING | rev | cut -d',' -f1 | rev)"

uv run python -m jlens.cli run \
    --model-id "EleutherAI/pythia-160m-deduped" \
    --dtype float16 \
    --n-prompts 1000 \
    --max-seq-len 128 \
    --results-dir results \
    --checkpoints "$MISSING" \
    --prompt-seed 42

echo "End: $(date)"
echo "=== Done ==="
