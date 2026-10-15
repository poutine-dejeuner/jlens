#!/bin/bash
#SBATCH --job-name=jlens-bench-12b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/jlens-bench-12b_%j.out
#SBATCH --error=logs/jlens-bench-12b_%j.err

set -e
cd /mnt/home/jacob/vl/repos/jlens
mkdir -p figures/pythia-12b logs

echo "=== 12B Benchmark Pipeline ==="
echo "Start: $(date)"

echo ""
echo "Step 1: Download eval data + cross-join..."
uv run python scripts/prepare_eval_data_12b.py

echo ""
echo "Step 2: Plot benchmark correlations..."
uv run python scripts/plot_benchmark_corr_12b.py

echo ""
echo "End: $(date)"
echo "Output:"
ls -lh figures/pythia-12b/
echo "=== Done ==="
