#!/usr/bin/env python3
"""Download LogiQA dataset from original source and save as local Parquet files.

Usage:
    uv run python scripts/prepare_logiqa.py
"""

import os
import re
import urllib.request
import pandas as pd

OUT_DIR = "eval_data/logiqa_local"
BASE_URL = "https://raw.githubusercontent.com/lgw863/LogiQA-dataset/master"
SPLIT_MAP = {"Train.txt": "train", "Eval.txt": "validation", "Test.txt": "test"}


def parse_logiqa_file(text: str) -> list[dict]:
    """Parse a LogiQA text file into records."""
    lines = text.strip().splitlines()
    records = []
    i = 0
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue

        # Line is the label (a, b, c, d)
        label = lines[i].strip().lower()
        if label not in ("a", "b", "c", "d"):
            i += 1
            continue

        # Next non-empty line(s) are the passage + question
        i += 1
        passage_parts = []
        while i < len(lines) and not re.match(r'^[A-E]\.', lines[i].strip()):
            if lines[i].strip():
                passage_parts.append(lines[i].strip())
            i += 1

        # Next lines starting with A., B., C., D. are the options
        options = []
        while i < len(lines) and re.match(r'^[A-E]\.', lines[i].strip()):
            opt_text = re.sub(r'^[A-E]\.\s*', '', lines[i].strip())
            options.append(opt_text)
            i += 1

        if len(options) < 2:
            continue

        # Split passage and question - last sentence is usually the question
        full_passage = " ".join(passage_parts)
        
        # Try to find the question: usually ends with "?" or contains "which of the following"
        question = ""
        context = full_passage
        
        # Simple heuristic: if there's a "?" use everything after the last period before "?"
        if "?" in full_passage:
            # Split at the question mark and take the question part
            parts = re.split(r'(?<=[。！])', full_passage)
            if len(parts) > 1:
                question_part = parts[-1].strip()
                if question_part:
                    question = question_part
                    context = "".join(parts[:-1])
                else:
                    question = full_passage
                    context = ""
            else:
                question = full_passage
                context = ""
        else:
            # No question mark - use the whole thing as context
            context = full_passage
            question = ""

        records.append({
            "label": label,
            "context": context.strip(),
            "question": question.strip() if question else full_passage.strip(),
            "options": options,
        })

    return records


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for fname, split in SPLIT_MAP.items():
        url = f"{BASE_URL}/{fname}"
        out_path = os.path.join(OUT_DIR, f"{split}.parquet")

        if os.path.exists(out_path):
            print(f"  {split} exists, skipping")
            continue

        print(f"  downloading {split} from {url}...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(url) as resp:
                text = resp.read().decode("utf-8")
            records = parse_logiqa_file(text)
            df = pd.DataFrame(records)
            df.to_parquet(out_path, index=False)
            print(f"{len(df)} records saved to {out_path}")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nDone. Files in {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        df = pd.read_parquet(os.path.join(OUT_DIR, f))
        print(f"  {f}: {len(df)} rows")


if __name__ == "__main__":
    main()
