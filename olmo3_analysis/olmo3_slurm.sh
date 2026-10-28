#!/bin/bash
#SBATCH --job-name=olmo3-3mix-fit-plot
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/home/jacob/vl/repos/jlens/logs/olmo3_3mix_%A.out
#SBATCH --error=/mnt/home/jacob/vl/repos/jlens/logs/olmo3_3mix_%A.err

set -euo pipefail

echo "=== Olmo3-32B three-component mixture fit + plot ==="
echo "Array/job id: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-1}"
echo "Start: $(date)"

WORK_DIR=/mnt/home/jacob/vl/repos/jlens/olmo3_analysis
cd "$WORK_DIR"

# The copied reference modules live in this directory; python picks up cwd on sys.path[0].
export PYTHONPATH="$WORK_DIR${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p results/olmo3_3mix_parts figures logs

python3 olmo3_driver.py \
    --results-dir /data/jlens/olmo3-32b \
    --parts-dir results/olmo3_3mix_parts \
    --figures-dir figures \
    --overlay-steps 0 100000 300000 650000

echo "End: $(date)"
echo "=== Olmo3-32B three-component mixture fit + plot complete ==="
