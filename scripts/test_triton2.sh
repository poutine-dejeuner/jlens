#!/bin/bash
#SBATCH --job-name=triton2
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/triton2_%j.out
#SBATCH --error=logs/triton2_%j.err

cd /mnt/home/jacob/vl/repos/jlens
uv run python scripts/test_triton_fix2.py
