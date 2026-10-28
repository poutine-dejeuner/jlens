#!/bin/bash
#SBATCH --job-name=jl-olmo3-rerun2
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=168:00:00
#SBATCH --output=logs/jl-olmo3-rerun_%A_%a.out
#SBATCH --error=logs/jl-olmo3-rerun_%A_%a.err
#SBATCH --array=0,1,2,3,4,5,6

# Rerun for tasks that were killed mid-checkpoint.
# Skips steps that already have metadata.json.
# Uses the same step split as the original run_olmo3_32b.sh.

set -e

echo "=== jlens OLMo3-32B rerun ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

MODEL_ID="allenai/Olmo-3-1125-32B"
RESULTS_DIR="/data/jlens/olmo3-32b"
SCRATCH_DIR="/tmp/jlens-olmo3-rerun-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RESULTS_DIR" "$SCRATCH_DIR"

ALL_STEPS=(0 10000 20000 30000 40000 50000 60000 70000 80000 90000 \
           100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 \
           200000 210000 220000 230000 240000 250000 260000 270000 280000 290000 \
           300000 310000 320000 330000 340000 350000 360000 370000 380000 390000 \
           400000 410000 420000 430000 440000 450000 460000 470000 480000 490000 \
           500000 510000 520000 530000 540000 550000 560000 570000 580000 590000 \
           600000 610000 620000 630000 640000 650000)
TOTAL=${#ALL_STEPS[@]}

# Same split as original
CHUNK_SIZE=$(( (TOTAL + 7) / 8 ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
if [ $END -gt $TOTAL ]; then
    END=$TOTAL
fi

echo "Total checkpoints: ${TOTAL}"
echo "This task processes: $((END - START)) checkpoints (indices ${START}-$((END-1)))"
echo "Steps: ${ALL_STEPS[@]:$START:$((END-START))}"

FAILED=0
for (( i=$START; i<$END; i++ )); do
    STEP="${ALL_STEPS[$i]}"
    STEP_DIR="${RESULTS_DIR}/step$(printf '%06d' ${STEP})"

    # Skip if already completed
    if [ -f "${STEP_DIR}/metadata.json" ]; then
        echo "[$((i-START+1))/$((END-START))] Step ${STEP} already done, skipping"
        continue
    fi

    echo ""
    echo "=============================================="
    echo "[$((i-START+1))/$((END-START))] Processing step ${STEP} (stage1-step${STEP})"
    echo "=============================================="
    echo "Start: $(date)"

    T0=$(date +%s)

    if uv run python -m jlens.cli run \
        --model-id "$MODEL_ID" \
        --revision-prefix "stage1-step" \
        --dtype float16 \
        --n-prompts 1000 \
        --max-seq-len 128 \
        --dim-batch 64 \
        --results-dir "$RESULTS_DIR" \
        --scratch-dir "$SCRATCH_DIR" \
        --cache-dir /mnt/home/jacob/.cache/huggingface/hub \
        --checkpoints "$STEP" \
        --prompt-seed 42; then

        T1=$(date +%s)
        ELAPSED=$(( T1 - T0 ))
        echo "Step ${STEP} completed in $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m"
    else
        echo "Step ${STEP} FAILED"
        FAILED=$((FAILED + 1))
    fi

    # Cleanup scratch only — do NOT touch HF cache (stale file handles)
    rm -rf "$SCRATCH_DIR"/* 2>/dev/null || true
    uv run python -c "import torch, gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true

    echo "End for step ${STEP}: $(date)"
done

rm -rf "$SCRATCH_DIR"

echo ""
echo "=== Array task ${SLURM_ARRAY_TASK_ID} complete ==="
echo "Failed: ${FAILED}"
echo "End: $(date)"
