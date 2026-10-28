#!/usr/bin/env python3
"""Extract simplified evaluation results from lm_eval output directory.

Usage:
    python3 scripts/extract_lm_eval_results.py <step>
    
Reads results from eval_data/12b_zero_shot/step{step}/**/results*.json
Writes simplified JSON to eval_data/12b_zero_shot/step{step}.json
"""

import json
import os
import glob
import shutil
import sys


def main():
    step = int(sys.argv[1])
    results_dir = f"eval_data/12b_zero_shot/step{step}"
    
    results_files = glob.glob(os.path.join(results_dir, "**", "results*.json"), recursive=True)
    if not results_files:
        results_files = sorted(glob.glob(os.path.join(results_dir, "results*.json")))
    
    if not results_files:
        print(f"WARNING: no results files found in {results_dir}")
        sys.exit(1)
    
    rf = results_files[0]
    with open(rf) as f:
        data = json.load(f)
    
    out = {"results": {}}
    for task, res in data.get("results", {}).items():
        acc = res.get("acc,none", None)
        acc_norm = res.get("acc_norm,none", None)
        if acc is None:
            acc = res.get("acc", None)
        if acc_norm is None:
            acc_norm = res.get("acc_norm", None)
        out["results"][task] = {
            "acc": acc,
            "acc_norm": acc_norm,
        }
    
    out_path = f"eval_data/12b_zero_shot/step{step}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(out['results'])} tasks to {out_path}")
    
    for task, r in sorted(out["results"].items()):
        acc = r.get("acc", r.get("acc_norm", "N/A"))
        print(f"  {task}: acc={acc}")
    
    shutil.rmtree(results_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
