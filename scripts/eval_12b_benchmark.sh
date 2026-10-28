#!/bin/bash
#SBATCH --job-name=jl-eval-12b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_12b_%j.out
#SBATCH --error=logs/eval_12b_%j.err

# Evaluate a Pythia-12B-deduped checkpoint on 7 benchmarks via lm_eval.
# Excluding logiqa (evaluated separately via scripts/eval_logiqa.py).
#
# Usage:
#   sbatch scripts/eval_12b_benchmark.sh <step>
#
# Results saved to eval_data/12b_zero_shot/step{step}.json

set -e

STEP="${1:?Usage: $0 <step>}"
echo "=== Evaluating Pythia-12B step ${STEP} ==="
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs eval_data/12b_zero_shot

OUTPUT_FILE="eval_data/12b_zero_shot/step${STEP}.json"

# Skip if already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "  [skip] step ${STEP} — already exists at ${OUTPUT_FILE}"
    exit 0
fi

# Run lm_eval on 7 core benchmarks (logiqa excluded)
TASKS="arc_easy,arc_challenge,piqa,winogrande,sciq,lambada_openai,wsc"
echo "  tasks: ${TASKS}"
echo "  step: ${STEP}"

if uv run lm_eval run \
    --model hf \
    --model_args "pretrained=EleutherAI/pythia-12b-deduped,revision=step${STEP},dtype=float16,trust_remote_code=True,pretrained_tokenizer=EleutherAI/pythia-12b-deduped" \
    --tasks "${TASKS}" \
    --output_path "eval_data/12b_zero_shot/step${STEP}" \
    --log_samples; then

    echo "lm_eval succeeded, extracting results..."
    uv run python scripts/extract_lm_eval_results.py "${STEP}"
else
    echo "ERROR: lm_eval failed for step ${STEP}"
    exit 1
fi

echo "End: $(date)"
echo "=== Done ==="
