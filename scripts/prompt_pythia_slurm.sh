#!/usr/bin/env bash
# Launch Pythia on a single GPU via vLLM, then run prompts.
# Usage: sbatch scripts/prompt_pythia_slurm.sh
#
# Override defaults:
#   MODEL=EleutherAI/pythia-1.4b-deduped PROMPTS="Hello world" sbatch scripts/prompt_pythia_slurm.sh

#SBATCH --job-name=pythia-vllm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/pythia_vllm_%j.out
#SBATCH --error=logs/pythia_vllm_%j.err
#SBATCH --partition=b300

set -e

MODEL="${MODEL:-EleutherAI/pythia-160m-deduped}"
PORT="${PORT:-8000}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMP="${TEMP:-0.0}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
GPU_MEM="${GPU_MEM:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPTS="${PROMPTS:-}"

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

VLLM_PYTHON="${VLLM_PYTHON:-/mnt/home/jacob/vllm-env/bin/python}"

# Build args
ARGS=(
    --model "$MODEL"
    --port "$PORT"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMP"
    --num-prompts "$NUM_PROMPTS"
    --launch
    --gpu-memory-utilization "$GPU_MEM"
)

if [ -n "$MAX_MODEL_LEN" ]; then
    ARGS+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [ -n "$PROMPT_FILE" ]; then
    ARGS+=(--prompt-file "$PROMPT_FILE")
fi
if [ -n "$PROMPTS" ]; then
    ARGS+=(--prompts $PROMPTS)
fi

echo "=== vLLM Pythia Prompt ==="
echo "Model: $MODEL"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Args: ${ARGS[*]}"
echo "=========================="

export VLLM_PYTHON="$VLLM_PYTHON"
exec uv run python scripts/prompt_pythia.py "${ARGS[@]}"
