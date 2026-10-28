#!/bin/bash
#SBATCH --job-name=vincent-jl-olmo3
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --output=logs/vincent-jl-olmo3_%A_%a.out
#SBATCH --error=logs/vincent-jl-olmo3_%A_%a.err
#SBATCH --array=0-7

# jlens: Jacobian lens spectral evolution — OLMo3-32B Stage 1
# 8 GPU array job, each processes 8-9 checkpoints sequentially.
# Total: 66 checkpoints (step0, 10000, 20000, ..., 650000)
#
# Per-checkpoint: download ~64 GB → compute JLens (~10h) → save HDF5 → delete HF cache
# Estimated: ~5h per checkpoint (dim_batch=32 needs 640/2=320 passes vs 80, more passes but less OOM),
#   ~45h per array task (8-9 ckpts)
# Results: ~10 GB/checkpoint, ~660 GB total on /data
# GPU memory: Model 64.5 GB + dim_batch=64 peak 208.8 GB — ~59 GB headroom

set -e

echo "=== jlens OLMo3-32B pipeline ==="
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Set HF token for faster downloads
export HF_TOKEN="${HF_TOKEN:-}"

export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs

MODEL_ID="allenai/Olmo-3-1125-32B"
RESULTS_DIR="/data/jlens/olmo3-32b"
SCRATCH_DIR="/tmp/jlens-olmo3-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RESULTS_DIR" "$SCRATCH_DIR"

# 66 stage1 checkpoints at 10k steps (step0 through step650000)
ALL_STEPS=(0 10000 20000 30000 40000 50000 60000 70000 80000 90000 \
           100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 \
           200000 210000 220000 230000 240000 250000 260000 270000 280000 290000 \
           300000 310000 320000 330000 340000 350000 360000 370000 380000 390000 \
           400000 410000 420000 430000 440000 450000 460000 470000 480000 490000 \
           500000 510000 520000 530000 540000 550000 560000 570000 580000 590000 \
           600000 610000 620000 630000 640000 650000)
TOTAL=${#ALL_STEPS[@]}

# Split across 8 array tasks
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

    # Resume: skip if already completed
    if [ -f "${STEP_DIR}/metadata.json" ]; then
        echo "[$((i-START+1))/$((END-START))] Step ${STEP} already done, skipping"
        continue
    fi

    echo ""
    echo "=============================================="
    echo "[$((i-START+1))/$((END-START))] Processing step ${STEP} (stage1-step${STEP})"
    echo "=============================================="
    echo "Start: $(date)"

    # Record start time
    T0=$(date +%s)

    # Download, compute, save, then delete
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

    # Cleanup: delete OLMo3 HF cache blobs to free disk space
    # (only the refs/snapshots dir, not the whole module — keeps config cached)
    HF_MODEL_CACHE="/mnt/home/jacob/.cache/huggingface/hub/models--allenai--Olmo-3-1125-32B"
    if [ -d "$HF_MODEL_CACHE" ]; then
        # Remove snapshot dirs (download blobs)
        rm -rf "$HF_MODEL_CACHE/blobs/" 2>/dev/null || true
        rm -rf "$HF_MODEL_CACHE/snapshots/" 2>/dev/null || true
        echo "Cleaned HF cache for olmo3-32b"
    fi

    # Clean scratch
    rm -rf "$SCRATCH_DIR"/* 2>/dev/null || true

    # Clear GPU memory between runs
    uv run python -c "import torch, gc; gc.collect(); torch.cuda.empty_cache()" 2>/dev/null || true

    echo "End for step ${STEP}: $(date)"
done

rm -rf "$SCRATCH_DIR"

echo ""
echo "=== Array task ${SLURM_ARRAY_TASK_ID} complete ==="
echo "Failed: ${FAILED}"
echo "End: $(date)"
