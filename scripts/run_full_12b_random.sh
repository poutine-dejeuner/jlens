#!/bin/bash
#SBATCH --job-name=vincent-jl-12b-rnd
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=96:00:00
#SBATCH --output=logs/vincent-jl-12b-rnd_%A_%a.out
#SBATCH --error=logs/vincent-jl-12b-rnd_%A_%a.err
#SBATCH --array=0-7

# jlens: Jacobian lens spectral evolution — Pythia-12B-deduped (RANDOM TOKENS)
# 8 GPU job array, each processes ~7 checkpoints
# Total: ~56 checkpoints (subsampled from 154), 1000 random-token prompts each
# Disk: ~11 GB/checkpoint, ~620 GB total on NFS
#
# This is the PROJECT2 experiment: jlens averaged over random token sequences
# instead of wikitext, to test whether spectral metrics capture linguistic
# structure or just weight-matrix evolution.

set -e

echo "=== jlens Pythia-12B RANDOM-TOKEN pipeline ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

# Results on NFS, temporary writes on local SSD
RESULTS_DIR="/data/jlens/pythia-12b-deduped-random"
SCRATCH_DIR="/tmp/jlens-12b-rnd-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RESULTS_DIR" "$SCRATCH_DIR"

# Get subsampled checkpoint steps (~56 from 154) — same as wikitext baseline
ALL_STEPS=$(uv run python -c "
from jlens.checkpoint_manager import get_checkpoint_steps
import numpy as np
steps = get_checkpoint_steps(1, 143000)
# Subsample: keep powers of 2 up to 1024, then every ~3000 after
keep = set()
v = 1
while v <= 143000:
    keep.add(v)
    v *= 2
# Add evenly spaced after 1024
late = [s for s in steps if s > 1024]
n_late = 56 - len(keep)
idx = np.linspace(0, len(late)-1, n_late, dtype=int)
for i in idx:
    keep.add(late[i])
result = sorted(keep)
print(','.join(map(str, result)))
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
echo "This task processes: ${NUM_STEPS} checkpoints (indices ${START}-$((END-1)))"
echo "First: ${STEP_ARRAY[$START]}, Last: ${STEP_ARRAY[$((END-1))]}"

uv run python -m jlens.cli run \
    --model-id "EleutherAI/pythia-12b-deduped" \
    --dtype float16 \
    --n-prompts 1000 \
    --max-seq-len 64 \
    --dim-batch 128 \
    --results-dir "$RESULTS_DIR" \
    --scratch-dir "$SCRATCH_DIR" \
    --cache-dir /mnt/home/jacob/.cache/huggingface/hub \
    --checkpoints "$MY_STEPS" \
    --prompt-seed 42 \
    --random-tokens

# Cleanup scratch
rm -rf "$SCRATCH_DIR"

echo "End: $(date)"
echo "=== Done ==="
