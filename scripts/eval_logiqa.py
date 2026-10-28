#!/usr/bin/env python3
"""Evaluate LogiQA on all 12B checkpoints using direct PyTorch evaluation.

Usage (SLURM):
    sbatch --gpus=1 --cpus-per-task=8 --mem=64G -J jl-logiqa \\
        --wrap "uv run python scripts/eval_logiqa.py <STEP>"

For all missing steps:
    uv run python scripts/eval_logiqa.py --missing
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-12b-deduped"
PARQUET_DIR = "eval_data/logiqa_local"
OUT_DIR = "eval_data/12b_zero_shot"
BENCHMARKS = ["arc_easy", "arc_challenge", "piqa", "winogrande", "sciq", "lambada_openai", "wsc"]

# All 48 J-lens 12B steps
JLENS_STEPS = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    2000, 5000, 9000,
    13000, 17000, 21000, 24000, 28000, 32000, 36000, 40000, 43000,
    47000, 51000, 55000, 59000, 62000, 66000, 70000, 74000, 78000,
    82000, 85000, 89000, 93000, 97000,
    101000, 104000, 108000, 112000, 116000, 120000, 123000,
    127000, 131000, 135000, 139000, 143000,
]


def load_logiqa(split: str = "test") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(PARQUET_DIR, f"{split}.parquet"))


def extract_score(step: int) -> float | None:
    """Extract logiqa score from an existing eval JSON (if available from EleutherAI)."""
    fname = os.path.join(OUT_DIR, f"step{step}.json")
    if not os.path.exists(fname):
        return None
    with open(fname) as f:
        data = json.load(f)
    results = data.get("results", {})
    if "logiqa" in results:
        return results["logiqa"].get("acc", None)
    # Check agieval
    if "agieval_logiqa_en" in results:
        return results["agieval_logiqa_en"].get("acc", None)
    return None


def eval_step(step: int, model=None, tokenizer=None):
    """Evaluate Pythia-12B at given step on LogiQA test set."""
    step_str = f"step{step:04d}"
    
    # Load model
    if model is None or tokenizer is None:
        print(f"Loading model at step {step}...", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            revision=step_str,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).cuda()
        model.eval()

    # Load data
    df = load_logiqa("test")
    label_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    
    correct = 0
    total = 0

    for idx, row in df.iterrows():
        question = row["question"]
        options = list(row["options"])
        label = label_map[row["label"].strip().lower()]
        
        # Compute log-likelihood for each option
        option_scores = []
        for opt_idx, opt in enumerate(options):
            # Format: "Question: {question}\nAnswer: {option}"
            prefix = f"Question: {question}\nAnswer:"
            full_text = f"{prefix} {opt}"

            inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            
            # Get loss only over the answer part
            prefix_len = len(tokenizer(prefix, return_tensors="pt")["input_ids"][0]) - 1
            logits = outputs.logits[0]
            labels = inputs["input_ids"][0]
            
            # Compute log-likelihood of the answer tokens
            shift_logits = logits[prefix_len-1:-1]
            shift_labels = labels[prefix_len:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction='sum'
            )
            option_scores.append(-loss.item())

        if np.argmax(option_scores) == label:
            correct += 1
        total += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(df)} ({100*correct/total:.1f}%)", file=sys.stderr)

    acc = correct / total
    print(f"LogiQA acc for step {step}: {acc:.4f} ({correct}/{total})", file=sys.stderr)
    return acc


def save_logiqa_score(step: int, acc: float):
    """Merge logiqa score into existing eval JSON or create new one."""
    fname = os.path.join(OUT_DIR, f"step{step}.json")
    if os.path.exists(fname):
        with open(fname) as f:
            data = json.load(f)
    else:
        data = {"results": {}}
    
    # Create logiqa entry (matching EleutherAI format pattern)
    data["results"]["logiqa"] = {
        "acc": acc,
        "acc_norm": acc,
    }
    
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved logiqa score for step {step} to {fname}", file=sys.stderr)


def find_missing():
    """Find steps that have J-lens data but are missing logiqa scores."""
    missing = []
    for step in JLENS_STEPS:
        if step == 0:
            continue
        score = extract_score(step)
        if score is None:
            missing.append(step)
    return missing


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/eval_logiqa.py <STEP>")
        print("       uv run python scripts/eval_logiqa.py --missing")
        sys.exit(1)

    if sys.argv[1] == "--missing":
        missing = find_missing()
        print(f"Steps missing logiqa scores ({len(missing)}): {missing}")
        for step in missing:
            print(f"\n{'='*60}")
            print(f"Evaluating step {step}")
            print(f"{'='*60}")
            acc = eval_step(step)
            save_logiqa_score(step, acc)
    else:
        step = int(sys.argv[1])
        acc = eval_step(step)
        save_logiqa_score(step, acc)


if __name__ == "__main__":
    main()
