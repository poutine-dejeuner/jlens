#!/bin/bash
#SBATCH --job-name=vincent-jl-lb-rd
#SBATCH --output=logs/jlens-lb-random_%j.out
#SBATCH --error=logs/jlens-lb-random_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens

echo "=== Lens benchmark sweep on Pythia-12B with random-token JLens ==="
echo "Start: $(date)"
echo ""

source .venv/bin/activate
export HF_HOME=/mnt/home/jacob/.cache/huggingface

uv run python scripts/eval_lens_bench.py \
    --results-dir /data/jlens/pythia-12b-deduped-random \
    --all

echo ""
echo "=== Done ==="
echo "End: $(date)"
