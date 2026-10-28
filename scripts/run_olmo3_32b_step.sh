#!/bin/bash
#SBATCH --job-name=jl-olmo3-step
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/jl-olmo3-step_%j.out
#SBATCH --error=logs/jl-olmo3-step_%j.err

# Single OLMo3-32B checkpoint: sbatch --export=ALL,STEP=80000 scripts/run_olmo3_32b_step.sh
# Used to run the tail of a chunk in parallel instead of sequentially.

set -e
cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

RESULTS_DIR="/data/jlens/olmo3-32b"
SCRATCH_DIR="/tmp/jlens-olmo3-step-${STEP}"
mkdir -p "$RESULTS_DIR" "$SCRATCH_DIR"

if [ -f "${RESULTS_DIR}/step$(printf '%04d' ${STEP})/metadata.json" ]; then
    echo "Step ${STEP} already done"
    exit 0
fi

uv run python -m jlens.cli run \
    --model-id "allenai/Olmo-3-1125-32B" \
    --revision-prefix "stage1-step" \
    --dtype float16 \
    --n-prompts 1000 \
    --max-seq-len 128 \
    --dim-batch 64 \
    --results-dir "$RESULTS_DIR" \
    --scratch-dir "$SCRATCH_DIR" \
    --cache-dir /mnt/home/jacob/.cache/huggingface/hub \
    --checkpoints "$STEP" \
    --prompt-seed 42

rm -rf "$SCRATCH_DIR"
