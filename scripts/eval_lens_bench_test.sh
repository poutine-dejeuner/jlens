#!/bin/bash
#SBATCH --job-name=jlens-eval-test
#SBATCH --output=logs/jlens-eval-test_%j.out
#SBATCH --error=logs/jlens-eval-test_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=b300-nsu4-0-37x9xf

set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens

echo "=== Testing lens bench eval on step 143000 ==="
echo "Start: $(date)"

source .venv/bin/activate
export HF_HOME=/mnt/home/jacob/.cache/huggingface

uv run python scripts/eval_lens_bench.py --step 143000

echo "=== Done ==="
echo "End: $(date)"
