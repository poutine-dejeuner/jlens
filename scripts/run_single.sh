#!/bin/bash
#SBATCH --job-name=jspace-lens
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/jspace_%j_%a.out
#SBATCH --error=logs/jspace_%j_%a.err
#SBATCH --array=0-152  # 153 checkpoints total, adjust based on --array size

# jspace-lens SLURM job array
# Each array task processes one checkpoint

set -e

echo "=== Job $SLURM_JOB_ID Task $SLURM_ARRAY_TASK_ID ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

# Navigate to project
cd /mnt/home/jacob/vl/repos/jlens

# Get checkpoint step from the full list
# We generate the list of steps and pick the one at index SLURM_ARRAY_TASK_ID
ALL_STEPS=(1 2 4 8 16 32 64 128 256 512 1000 $(seq 2000 1000 143000))
STEP=${ALL_STEPS[$SLURM_ARRAY_TASK_ID]}

echo "Processing step $STEP"

# Run with uv
uv run python -m jlens.cli run \
    --model-id "EleutherAI/pythia-160m-deduped" \
    --dtype bfloat16 \
    --n-prompts 1000 \
    --max-seq-len 256 \
    --results-dir results \
    --cache-dir /tmp/pythia-cache \
    --checkpoints "$STEP"

echo "End: $(date)"
echo "=== Done ==="
