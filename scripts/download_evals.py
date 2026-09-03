#!/usr/bin/env python3
"""Download Pythia-160M-deduped evaluation JSONs from EleutherAI/pythia GitHub.

Usage:
    python scripts/download_evals.py

Downloads 27 zero-shot eval JSONs to eval_data/zero_shot/.
"""

import json
import os
import urllib.request

REPO_BASE = "https://raw.githubusercontent.com/EleutherAI/pythia/main"
EVAL_PATH = "evals/pythia-v1/pythia-160m-deduped/zero-shot"

# All 27 checkpoints evaluated by EleutherAI
STEPS = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 3000,
    13000, 23000, 33000, 43000, 53000, 63000, 73000, 83000,
    93000, 103000, 113000, 123000, 133000, 143000,
]


def main():
    out_dir = "eval_data/zero_shot"
    os.makedirs(out_dir, exist_ok=True)

    for step in STEPS:
        filename = f"160m-deduped_step{step}.json"
        url = f"{REPO_BASE}/{EVAL_PATH}/{filename}"
        out_path = os.path.join(out_dir, f"step{step}.json")

        if os.path.exists(out_path):
            print(f"  [skip] step {step} (already downloaded)")
            continue

        print(f"  downloading step {step}...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(url) as resp:
                data = json.load(resp)
            with open(out_path, "w") as f:
                json.dump(data, f)
            print("ok")
        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
