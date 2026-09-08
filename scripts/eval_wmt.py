#!/usr/bin/env python
"""Evaluate Pythia-160M-deduped checkpoints on WMT14 fr-en.

Direct inference with transformers + sacrebleu — no lm-eval dependency.
"""

import json
import os
import sys
import time
from pathlib import Path

import click
import numpy as np
import sacrebleu
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# All 153 Pythia-160M checkpoint steps
_LOG_SPACED = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
_EVENLY_SPACED = list(range(1000, 144000, 1000))
ALL_STEPS = sorted(_LOG_SPACED + _EVENLY_SPACED)

MODEL_ID = "EleutherAI/pythia-160m-deduped"


def load_wmt14_test():
    """Load WMT14 fr-en test set (3003 examples)."""
    ds = load_dataset("wmt/wmt14", "fr-en", split="test")
    sources = []  # French
    references = []  # English
    for ex in ds:
        sources.append(ex["translation"]["fr"])
        references.append(ex["translation"]["en"])
    return sources, references


def evaluate_checkpoint(step: int, sources: list[str], references: list[str], device: str) -> dict:
    """Run WMT14 fr-en evaluation on a single checkpoint."""
    t0 = time.time()

    print(f"  Loading model step{step}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=f"step{step}",
        dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=f"step{step}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded ({time.time() - t0:.0f}s). Generating...", flush=True)

    # Build prompts: "French: {fr}\nEnglish:"
    prompts = [f"French: {src}\nEnglish:" for src in sources]

    # Tokenize and generate in batches
    batch_size = 8
    predictions = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"  step{step}", leave=False):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        for j, out_seq in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            gen_ids = out_seq[input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # Take first line (until newline)
            gen_text = gen_text.split("\n")[0].strip()
            predictions.append(gen_text)

    # Compute metrics
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    ter = sacrebleu.corpus_ter(predictions, [references])

    elapsed = time.time() - t0
    print(f"  BLEU={bleu.score:.1f}, chrF={chrf.score:.1f}, TER={ter.score:.1f} [{elapsed:.0f}s]", flush=True)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        "step": step,
        "bleu": bleu.score,
        "bleu_str": str(bleu),
        "chrf": chrf.score,
        "chrf_str": str(chrf),
        "ter": ter.score,
        "ter_str": str(ter),
        "elapsed_s": elapsed,
    }


@click.command()
@click.option("--step", type=int, default=None, help="Single step to evaluate")
@click.option("--all", is_flag=True, help="Evaluate all 153 steps")
@click.option("--slice", type=int, default=None, help="Slice index for SLURM array job")
@click.option("--num-slices", type=int, default=8, help="Number of slices")
@click.option("--output-dir", default="results/wmt_eval", help="Output directory")
@click.option("--device", default="cuda", help="Device")
@click.option("--limit", type=int, default=0, help="Limit test examples (0 = all 3003)")
def main(step, all, slice, num_slices, output_dir, device, limit):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if slice is not None:
        steps_to_eval = ALL_STEPS[slice::num_slices]
        print(f"Slice {slice}/{num_slices}: {len(steps_to_eval)} steps", flush=True)
    elif all:
        steps_to_eval = list(ALL_STEPS)
    elif step is not None:
        steps_to_eval = [step]
    else:
        click.echo("Must specify --step, --all, or --slice")
        sys.exit(1)

    # Load dataset once
    print("Loading WMT14 fr-en test set...", flush=True)
    sources, references = load_wmt14_test()
    if limit > 0:
        sources = sources[:limit]
        references = references[:limit]
    print(f"  {len(sources)} test examples loaded.", flush=True)

    errors = []
    for i, s in enumerate(steps_to_eval):
        result_file = out / f"wmt14_fr-en_step{s}.json"
        if result_file.exists():
            with open(result_file) as f:
                prev = json.load(f)
            print(f"[{i+1}/{len(steps_to_eval)}] Step {s}: BLEU={prev.get('bleu','?')} (cached)")
            continue

        print(f"[{i+1}/{len(steps_to_eval)}] Step {s}:", flush=True)
        try:
            metrics = evaluate_checkpoint(s, sources, references, device)
            with open(result_file, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            errors.append((s, str(e)))

    if errors:
        print(f"\n{len(errors)} errors:")
        for s, err in errors:
            print(f"  step {s}: {err}")

    # Collect to all_results.json
    all_results = {}
    for jf in sorted(out.glob("wmt14_fr-en_step*.json")):
        step_str = jf.stem.replace("wmt14_fr-en_step", "")
        try:
            step_num = int(step_str)
        except ValueError:
            continue
        with open(jf) as f:
            data = json.load(f)
        all_results[str(step_num)] = {
            "bleu": data["bleu"],
            "chrf": data["chrf"],
            "ter": data["ter"],
            "elapsed_s": data.get("elapsed_s", 0),
        }

    if all_results:
        all_file = out / "all_results.json"
        with open(all_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Collected {len(all_results)} results → {all_file}")


if __name__ == "__main__":
    main()
