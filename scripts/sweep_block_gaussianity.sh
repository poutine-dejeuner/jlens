#!/usr/bin/env bash
#SBATCH --job-name=jl-blockgauss
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/block_gauss_%j.out
#SBATCH --error=logs/block_gauss_%j.err

cd /mnt/home/jacob/vl/repos/jlens
mkdir -p logs
export TRITON_BUILD_WITHOUT_CC=1
export TORCH_COMPILE_DISABLE=1
export PYTHONPATH="$(pwd):$PYTHONPATH"
uv run python scripts/block_jacobian_gaussianity.py "$@"
