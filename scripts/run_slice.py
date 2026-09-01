#!/usr/bin/env python3
"""Run a slice of checkpoints via the jlens CLI.

Usage:
    uv run scripts/run_slice.py --task-id 0 --num-tasks 8
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jlens.checkpoint_manager import get_checkpoint_steps
from jlens.storage import is_checkpoint_done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--num-tasks", type=int, required=True)
    parser.add_argument("--n-prompts", type=int, default=1000)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    # Get all steps
    all_steps = get_checkpoint_steps(1, 143000)

    # Split into slices
    slice_size = (len(all_steps) + args.num_tasks - 1) // args.num_tasks
    start = args.task_id * slice_size
    end = min(start + slice_size, len(all_steps))
    my_steps = all_steps[start:end]

    # Filter out already-done checkpoints (resume support)
    todo = [s for s in my_steps if not is_checkpoint_done(args.results_dir, s)]
    skipped = len(my_steps) - len(todo)

    print(f"Task {args.task_id}: {len(my_steps)} total, {len(todo)} pending, {skipped} already done")
    if not todo:
        print("Nothing to do.")
        return

    steps_str = ",".join(str(s) for s in todo)
    print(f"Steps: {steps_str[:200]}...")
    print(f"First: {todo[0]}, Last: {todo[-1]}")

    # Run via CLI
    cmd = [
        sys.executable, "-m", "jlens.cli", "run",
        "--model-id", "EleutherAI/pythia-160m-deduped",
        "--dtype", "float16",
        "--n-prompts", str(args.n_prompts),
        "--max-seq-len", "128",
        "--results-dir", args.results_dir,
        "--checkpoints", steps_str,
        "--prompt-seed", "42",
    ]
    print(f"Running: {' '.join(cmd[:6])} ...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
