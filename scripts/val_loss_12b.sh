#!/bin/bash
#SBATCH --job-name=jlens-val
#SBATCH --output=logs/jlens-val_%j.out
#SBATCH --error=logs/jlens-val_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --nodelist=b300-nsu4-0-37x9xf

set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens

echo "=== Validation loss for 12B checkpoints ==="
echo "Start: $(date)"

source .venv/bin/activate
export HF_HOME=/mnt/home/jacob/.cache/huggingface

uv run python scripts/compute_val_loss.py \
    --results-dir /data/jlens/pythia-12b-deduped \
    --model EleutherAI/pythia-12b-deduped \
    --max-samples 200

echo "=== Done ==="
echo "End: $(date)"
