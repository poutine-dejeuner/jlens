#!/bin/bash
#SBATCH --job-name=vincent-jl-12b-rnd-test
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vincent-jl-12b-rnd-test_%j.out
#SBATCH --error=logs/vincent-jl-12b-rnd-test_%j.err

# jlens: Single-checkpoint test for Pythia-12B-deduped RANDOM TOKENS.
# Validates the random-token pipeline before launching the full sweep.

set -e

echo "=== jlens Pythia-12B RANDOM-TOKEN single-checkpoint test ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

RESULTS_DIR="/data/jlens/pythia-12b-deduped-random"
SCRATCH_DIR="/tmp/jlens-12b-rnd-test"
mkdir -p "$RESULTS_DIR" "$SCRATCH_DIR"

uv run python -m jlens.cli run \
    --model-id "EleutherAI/pythia-12b-deduped" \
    --dtype float16 \
    --n-prompts 10 \
    --max-seq-len 64 \
    --dim-batch 128 \
    --results-dir "$RESULTS_DIR" \
    --scratch-dir "$SCRATCH_DIR" \
    --cache-dir /mnt/home/jacob/.cache/huggingface/hub \
    --checkpoints 143000 \
    --prompt-seed 42 \
    --random-tokens

rm -rf "$SCRATCH_DIR"

echo "End: $(date)"
echo "=== Done ==="
