#!/bin/bash
#SBATCH --job-name=vincent-jl-fixlb
#SBATCH --output=logs/jlens-lb-fix_%j.out
#SBATCH --error=logs/jlens-lb-fix_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --nodelist=b300-nsu4-0-37x9xq

set -euo pipefail
cd /mnt/home/jacob/vl/repos/jlens

echo "=== Re-running lens bench evals with fixed scoring ==="
echo "Start: $(date)"

source .venv/bin/activate
export HF_HOME=/mnt/home/jacob/.cache/huggingface

# Delete all existing JSONs so they get recomputed with fixed scoring
echo "Deleting existing wikitext results..."
rm -f /data/jlens/pythia-12b-deduped/lens_bench/step*.json
rm -f /data/jlens/pythia-12b-deduped/lens_bench/all_results.json

echo "Deleting existing random results..."
rm -f /data/jlens/pythia-12b-deduped-random/lens_bench/step*.json
rm -f /data/jlens/pythia-12b-deduped-random/lens_bench/all_results.json

echo ""
echo "--- Wikitext ---"
uv run python scripts/eval_lens_bench.py \
    --results-dir /data/jlens/pythia-12b-deduped \
    --all

echo ""
echo "--- Random ---"
uv run python scripts/eval_lens_bench.py \
    --results-dir /data/jlens/pythia-12b-deduped-random \
    --all

echo ""
echo "=== Done ==="
echo "End: $(date)"
