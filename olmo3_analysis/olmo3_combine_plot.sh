#!/bin/bash
#SBATCH --job-name=olmo3-3mix-combine-plot
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/mnt/home/jacob/vl/repos/jlens/logs/olmo3_combine_plot_%A.out
#SBATCH --error=/mnt/home/jacob/vl/repos/jlens/logs/olmo3_combine_plot_%A.err
#SBATCH --nodelist=b300-nsu4-0-37x9xy

set -euo pipefail

echo "=== Olmo3-32B combine + plot ==="
echo "Node: $(hostname)  Start: $(date)"

WORK_DIR=/mnt/home/jacob/vl/repos/jlens/olmo3_analysis
cd "$WORK_DIR"
export PYTHONPATH="$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p results figures logs

python3 olmo3_driver.py \
    --results-dir /data/jlens/olmo3-32b \
    --parts-dir results/olmo3_3mix_parts \
    --figures-dir figures \
    --skip-fit \
    --overlay-steps 0 1000 2000 5000 100000 300000 650000

echo "End: $(date)"
echo "=== Olmo3-32B combine + plot complete ==="
