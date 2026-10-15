#!/usr/bin/env python3
"""Compute validation loss on WikiText-2 for Pythia checkpoints.

Usage:
    python compute_val_loss.py --results-dir /data/jlens/pythia-12b-deduped --model EleutherAI/pythia-12b-deduped

For each completed checkpoint in results_dir, loads the model, runs a forward
pass on WikiText-2 validation (first N samples), and writes loss to a JSON file.
"""
import argparse, json, os, sys, time, torch
from pathlib import Path
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_loss(model_id: str, revision: str, max_samples: int = 200) -> dict:
    """Compute average cross-entropy loss on WikiText-2 validation."""
    print(f"  Loading {model_id} @ {revision}...", flush=True)
    t0 = time.time()
    model = GPTNeoXForCausalLM.from_pretrained(
        model_id, revision=revision,
        torch_dtype=torch.float16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    t_load = time.time() - t0

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in dataset["text"] if len(t.strip()) > 0]
    n = min(max_samples, len(texts))

    total_loss = 0.0
    total_tokens = 0
    t_infer = time.time()

    for i, text in enumerate(texts[:n]):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        total_loss += out.loss.item() * enc["input_ids"].numel()
        total_tokens += enc["input_ids"].numel()

    avg_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(avg_loss)).item())
    elapsed = time.time() - t_infer

    del model
    torch.cuda.empty_cache()

    return {
        "step": revision,
        "loss": round(avg_loss, 6),
        "ppl": round(ppl, 4),
        "tokens": total_tokens,
        "samples": n,
        "load_time_s": round(t_load, 1),
        "infer_time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                        help="Path to results directory (e.g. /data/jlens/pythia-12b-deduped)")
    parser.add_argument("--model", default="EleutherAI/pythia-12b-deduped")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max WikiText-2 validation samples per checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if val_loss.json exists")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    step_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("step")],
        key=lambda d: int(d.name.replace("step", ""))
    )

    print(f"Found {len(step_dirs)} step directories")
    results = []

    for d in step_dirs:
        step = int(d.name.replace("step", ""))
        out_file = d / "val_loss.json"

        if out_file.exists() and not args.overwrite:
            with open(out_file) as f:
                entry = json.load(f)
            print(f"[{len(results)+1}/{len(step_dirs)}] step{step}: cached loss={entry['loss']:.4f} ppl={entry['ppl']:.2f}")
            results.append(entry)
            continue

        # Map step to HF revision
        revision = f"step{step}"

        try:
            entry = compute_loss(args.model, revision, args.max_samples)
            print(f"[{len(results)+1}/{len(step_dirs)}] step{step}: loss={entry['loss']:.4f} ppl={entry['ppl']:.2f} "
                  f"({entry['load_time_s']:.0f}s load, {entry['infer_time_s']:.0f}s infer)")
            results.append(entry)

            with open(out_file, "w") as f:
                json.dump(entry, f)
        except Exception as e:
            print(f"[{len(results)+1}/{len(step_dirs)}] step{step}: ERROR - {e}")

    # Save aggregate
    agg_file = results_dir / "val_loss_all.json"
    with open(agg_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {agg_file}")


if __name__ == "__main__":
    main()
